import random
from typing import Optional, Callable, Any
import numpy as np
import torch
import torch.utils.checkpoint
import jax
import jax.numpy as jnp
import optax
from datasets import load_dataset, Dataset
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxUNet2DConditionModel,
)
from diffusers.utils import check_min_version
from flax import jax_utils
from flax.training import train_state
from torchvision import transforms
from transformers import CLIPTokenizer, FlaxCLIPTextModel, set_seed
import pytest
import sys
from iree.jax import (
    like,
    kernel,
    IREE,
    Program,
)
from iree import runtime as iree_rt
from tempfile import TemporaryDirectory
import os

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


def create_optimizer(learning_rate=1e-4,
                     adam_beta1=0.9,
                     adam_beta2=0.999,
                     adam_epsilon=1e-08,
                     adam_weight_decay=1e-2,
                     max_grad_norm=1.0):
    constant_scheduler = optax.constant_schedule(learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=adam_beta1,
        b2=adam_beta2,
        eps=adam_epsilon,
        weight_decay=adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adamw,
    )
    return optimizer


class TrainState:

    def __init__(self,
                 optimizer,
                 pretrained_model_name_or_path: str,
                 rng,
                 weight_dtype=jnp.float32):
        # Load models and create wrapper for stable diffusion
        self.text_encoder = FlaxCLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            dtype=weight_dtype)
        self.vae, self.vae_params = FlaxAutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", dtype=weight_dtype)
        self.unet, self.unet_params = FlaxUNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            dtype=weight_dtype)
        self.optimizer = optimizer
        self.unet_optimizer_state = optimizer.init(self.unet_params)
        self.noise_scheduler = FlaxDDPMScheduler(beta_start=0.00085,
                                                 beta_end=0.012,
                                                 beta_schedule="scaled_linear",
                                                 num_train_timesteps=1000)
        self.noise_scheduler_state = self.noise_scheduler.create_state()
        self.rng = rng


def make_train_step_pure_fn(train_state: TrainState):

    def train_step(batch, unet_optimizer_state, unet_params, rng):
        sample_rng, new_rng = jax.random.split(rng, 2)

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = train_state.vae.apply(
                {"params": train_state.vae_params},
                batch["pixel_values"],
                deterministic=True,
                method=train_state.vae.encode)
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz, ),
                0,
                train_state.noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = train_state.noise_scheduler.add_noise(
                train_state.noise_scheduler_state, latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = train_state.text_encoder(
                batch["input_ids"],
                params=train_state.text_encoder.params,
                train=False,
            )[0]

            # Predict the noise residual and compute loss
            model_pred = train_state.unet.apply({
                "params": params
            },
                                                noisy_latents,
                                                timesteps,
                                                encoder_hidden_states,
                                                train=True).sample

            # Get the target for loss depending on the prediction type
            if train_state.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif train_state.noise_scheduler.config.prediction_type == "v_prediction":
                target = train_state.noise_scheduler.get_velocity(
                    train_state.noise_scheduler_state, latents, noise,
                    timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {train_state.noise_scheduler.config.prediction_type}"
                )

            loss = (target - model_pred)**2
            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(unet_params)

        unet_param_updates, new_unet_optimizer_state = train_state.optimizer.update(
            grad, unet_optimizer_state, unet_params)
        new_unet_params = optax.apply_updates(unet_params, unet_param_updates)

        metrics = {"loss": loss}

        return new_unet_optimizer_state, new_unet_params, metrics, new_rng

    return train_step


class JaxTrainer:

    def __init__(self, train_state: TrainState):
        self.train_state = train_state
        #self.jitted_train_step = jax.jit(make_train_step_pure_fn(train_state))
        self.jitted_train_step = make_train_step_pure_fn(train_state)

    def train_step(self, batch):
        self.train_state.unet_optimizer_state, self.train_state.unet_params, metrics, self.train_state.rng = self.jitted_train_step(
            batch, self.train_state.unet_optimizer_state,
            self.train_state.unet_params, self.train_state.rng)
        return metrics


def jax_train(pretrained_model_name_or_path: str, rng_seed: int):
    # Train for 1 step and return Unet train state

    set_seed(rng_seed)

    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    dataloader = create_dataloader(
        dataset,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        seed=rng_seed)

    rng = jax.random.PRNGKey(seed=rng_seed)

    train_state = TrainState(
        optimizer=create_optimizer(),
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        rng=rng,
        weight_dtype=jnp.float32)
    trainer = JaxTrainer(train_state)

    for batch in dataloader:
        trainer.train_step(batch)

    return train_state.unet_optimizer_state, train_state.unet_params


def create_iree_jax_module(train_state: TrainState, example_batch):
    train_step_fn = make_train_step_pure_fn(train_state)

    class IreeJaxStableDiffusionModule(Program):
        _untet_optimizer_state = train_state.unet_optimizer_state
        _untet_params = train_state.unet_params
        _rng = train_state.rng

        def get_unet_optimizer_state(self):
            return self._untet_optimizer_state

        def get_unet_params(self):
            return self._untet_params

        def train_step(self, batch=like(example_batch)):
            self._untet_optimizer_state, self._untet_params, metrics, self._rng = self._train_step(
                batch, self._untet_optimizer_state, self._untet_params,
                self._rng)
            return metrics

        @kernel
        def _train_step(batch, untet_optimizer_state, unet_params, rng):
            new_untet_optimizer_state, new_unet_params, metrics, new_rng = train_step_fn(
                batch, untet_optimizer_state, unet_params, rng)
            return new_untet_optimizer_state, new_unet_params, metrics, new_rng

    return IreeJaxStableDiffusionModule()


def build_iree_module(artifacts_dir: str,
                      train_state: TrainState,
                      sample_batch,
                      backend: str = "llvm-cpu",
                      runtime: str = "local-task"):
    module = create_iree_jax_module(train_state, sample_batch)
    with open(os.path.join(artifacts_dir, "stable_diffusion.mlir"), "wb") as f:
        Program.get_mlir_module(module).operation.print(f, binary=True)
    binary = IREE.compile_program(module, backends=[backend], runtime=runtime)
    iree_vmfb_path = os.path.join(artifacts_dir, "stable_diffusion.vmfb")
    with open(iree_vmfb_path, "wb") as f:
        f.write(binary.compiled_artifact)
    loaded_module = iree_rt.system_api.load_vm_flatbuffer_file(iree_vmfb_path,
                                                               driver=runtime)
    return loaded_module


def iree_train(pretrained_model_name_or_path: str, rng_seed: int):
    # Train for 1 step and return Unet train state

    set_seed(rng_seed)

    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    dataloader = create_dataloader(
        dataset,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        seed=rng_seed)

    rng = jax.random.PRNGKey(seed=rng_seed)

    train_state = TrainState(
        optimizer=create_optimizer(),
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        rng=rng,
        weight_dtype=jnp.float32)
    with TemporaryDirectory() as tmp_dir:
        for batch in dataloader:
            sample_batch = batch
            break
        module = build_iree_module(train_state=train_state,
                                   sample_batch=sample_batch,
                                   artifacts_dir=tmp_dir,
                                   backend="cuda",
                                   runtime="cuda")
        module.train_step(sample_batch)

    return module.get_unet_optimizer_state(), module.get_unet_params()


def create_dataloader(
    dataset: Dataset,
    pretrained_model_name_or_path: str,
    image_column: str = "image",
    caption_column: str = "text",
    max_train_samples: int = 1,
    resolution: int = 512,
    center_crop: int = True,
    random_flip: int = True,
    seed: Optional[int] = None,
    train_batch_size: int = 1,
) -> torch.utils.data.DataLoader:
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if image_column not in column_names:
        raise ValueError(
            f"image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}"
        )
    if caption_column not in column_names:
        raise ValueError(
            f"caption_column' value '{caption_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(
                    random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions,
                           max_length=tokenizer.model_max_length,
                           padding="do_not_pad",
                           truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    train_transforms = transforms.Compose([
        transforms.Resize(resolution,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution)
        if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip()
        if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [
            train_transforms(image) for image in images
        ]
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    if max_train_samples is not None:
        train_dataset = dataset["train"].shuffle(seed=seed).select(
            range(max_train_samples))
    else:
        train_dataset = dataset["train"]
    # Set the training transforms
    train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack(
            [example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(
            memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]

        padded_tokens = tokenizer.pad({"input_ids": input_ids},
                                      padding="max_length",
                                      max_length=tokenizer.model_max_length,
                                      return_tensors="pt")
        batch = {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
        }
        batch = {k: v.numpy() for k, v in batch.items()}

        return batch

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   shuffle=True,
                                                   collate_fn=collate_fn,
                                                   batch_size=train_batch_size,
                                                   drop_last=True)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,
                                              subfolder="tokenizer")

    return train_dataloader


def assert_array_almost_equal(a, b):
    np_a = np.asarray(a)
    np_b = np.asarray(b)
    # Test for absolute error.
    np.testing.assert_array_almost_equal(np_a, np_b, decimal=5)
    # Test for relative error while ignoring false positives from
    # catastrophic cancellation.
    np.testing.assert_array_almost_equal_nulp(np.abs(np_a - np_b) + 10**-7,
                                              np.zeros_like(np_a),
                                              nulp=10**8)


def assert_array_list_equal(
    a,
    b,
    array_compare_fn: Callable[[Any, Any],
                               None] = np.testing.assert_array_equal):
    assert (len(a) == len(b))
    for x, y in zip(a, b):
        array_compare_fn(x, y)


def assert_array_list_almost_equal(a, b):
    assert_array_list_equal(a, b, assert_array_almost_equal)


def test_training_with_iree_jax():
    rng_seed = 12345

    pretrained_model_name_or_path = "duongna/stable-diffusion-v1-4-flax"

    iree_unet_optimizer_state, iree_unet_params = iree_train(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        rng_seed=rng_seed)
    return
    jax_unet_optimizer_state, jax_unet_params = jax_train(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        rng_seed=rng_seed)
    assert_array_list_almost_equal(jax_unet_state, iree_unet_state)


if __name__ == "__main__":
    pytest.main(sys.argv)
