import random
from typing import Optional, Callable, List
import os
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
from torchvision import transforms
from transformers.models.clip.modeling_flax_clip import FlaxCLIPTextModule
from transformers import CLIPTokenizer, FlaxCLIPTextModel, set_seed, FlaxCLIPTextPreTrainedModel, CLIPTextConfig
from iree.jax import (
    like,
    kernel,
    IREE,
    Program,
)
from jax.tree_util import tree_flatten
from iree import runtime as iree_rt
from tempfile import TemporaryDirectory
import logging
import iree.compiler.tools

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


def create_optimizer(learning_rate=1e-5,
                     adam_beta1=0.9,
                     adam_beta2=0.999,
                     adam_epsilon=1e-08,
                     adam_weight_decay=1e-2,
                     max_grad_norm=1.0):
    #return optax.sgd(learning_rate)

    adamw = optax.adamw(
        learning_rate=learning_rate,
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
                 text_encoder,
                 vae,
                 vae_params,
                 unet,
                 unet_params,
                 unet_optimizer_state,
                 noise_scheduler,
                 noise_scheduler_state,
                 rng,
                 output_gradient=False):
        self.optimizer = optimizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.vae_params = vae_params
        self.unet = unet
        self.unet_params = unet_params
        self.unet_optimizer_state = unet_optimizer_state
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_state = noise_scheduler_state
        self.rng = rng
        self.output_gradient = output_gradient


def load_train_state(optimizer,
                     pretrained_model_name_or_path: str,
                     rng,
                     output_gradient=False,
                     weight_dtype=jnp.float32) -> TrainState:
    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        dtype=weight_dtype)
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", dtype=weight_dtype)
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", dtype=weight_dtype)
    unet_optimizer_state = optimizer.init(unet_params)
    noise_scheduler = FlaxDDPMScheduler(beta_start=0.00085,
                                        beta_end=0.012,
                                        beta_schedule="scaled_linear",
                                        num_train_timesteps=1000)
    noise_scheduler_state = noise_scheduler.create_state()
    train_state = TrainState(optimizer=optimizer,
                             text_encoder=text_encoder,
                             vae=vae,
                             vae_params=vae_params,
                             unet=unet,
                             unet_params=unet_params,
                             unet_optimizer_state=unet_optimizer_state,
                             noise_scheduler=noise_scheduler,
                             noise_scheduler_state=noise_scheduler_state,
                             rng=rng,
                             output_gradient=output_gradient)
    logger.debug(
        f"Stable Diffusion train state loaded from \"{pretrained_model_name_or_path}\""
    )
    return train_state


def create_small_model_train_state(optimizer,
                                   seed,
                                   weight_dtype=jnp.float32,
                                   output_gradient=False) -> TrainState:
    rng = jax.random.PRNGKey(seed)
    text_encoder = FlaxCLIPTextModel(config=CLIPTextConfig(
        architectures=["CLIPTextModel"],
        attention_dropout=0.0,
        bos_token_id=0,
        dropout=0.0,
        eos_token_id=2,
        hidden_act="gelu",
        hidden_size=1,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=1,
        layer_norm_eps=1e-05,
        max_position_embeddings=77,
        model_type="clip_text_model",
        num_attention_heads=1,
        num_hidden_layers=1,
        pad_token_id=1,
        projection_dim=1,
        torch_dtype=weight_dtype.dtype.name,
        transformers_version="4.25.1",
        vocab_size=49408),
                                     dtype=weight_dtype,
                                     seed=seed)
    vae = FlaxAutoencoderKL(in_channels=3,
                            out_channels=3,
                            down_block_types=['DownEncoderBlock2D'],
                            up_block_types=['UpDecoderBlock2D'],
                            block_out_channels=[2],
                            layers_per_block=1,
                            act_fn='silu',
                            latent_channels=1,
                            norm_num_groups=2,
                            sample_size=48,
                            dtype=weight_dtype)
    vae_params = vae.init_weights(rng=rng)
    unet = FlaxUNet2DConditionModel(
        sample_size=6,
        in_channels=1,
        out_channels=1,
        down_block_types=['CrossAttnDownBlock2D', 'DownBlock2D'],
        up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D'],
        only_cross_attention=False,
        block_out_channels=[32, 32],
        layers_per_block=1,
        attention_head_dim=[1, 1],
        cross_attention_dim=1,
        dropout=0.0,
        use_linear_projection=True,
        dtype=weight_dtype,
        flip_sin_to_cos=True,
        freq_shift=0)
    unet_params = unet.init_weights(rng=rng)
    unet_optimizer_state = optimizer.init(unet_params)
    noise_scheduler = FlaxDDPMScheduler(beta_start=0.00085,
                                        beta_end=0.012,
                                        beta_schedule="scaled_linear",
                                        num_train_timesteps=1000)
    noise_scheduler_state = noise_scheduler.create_state()
    return TrainState(optimizer=optimizer,
                      text_encoder=text_encoder,
                      vae=vae,
                      vae_params=vae_params,
                      unet=unet,
                      unet_params=unet_params,
                      unet_optimizer_state=unet_optimizer_state,
                      noise_scheduler=noise_scheduler,
                      noise_scheduler_state=noise_scheduler_state,
                      rng=rng,
                      output_gradient=output_gradient)


def create_full_model_train_state(optimizer,
                                  seed,
                                  weight_dtype=jnp.float32) -> TrainState:
    rng = jax.random.PRNGKey(seed)
    text_encoder = FlaxCLIPTextModel(config=CLIPTextConfig(
        architectures=["CLIPTextModel"],
        attention_dropout=0.0,
        bos_token_id=0,
        dropout=0.0,
        eos_token_id=2,
        hidden_act="gelu",
        hidden_size=1024,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=4096,
        layer_norm_eps=1e-05,
        max_position_embeddings=77,
        model_type="clip_text_model",
        num_attention_heads=16,
        num_hidden_layers=23,
        pad_token_id=1,
        projection_dim=512,
        torch_dtype=weight_dtype.dtype.name,
        transformers_version="4.25.1",
        vocab_size=49408),
                                     dtype=weight_dtype,
                                     seed=seed)
    vae = FlaxAutoencoderKL(in_channels=3,
                            out_channels=3,
                            down_block_types=[
                                'DownEncoderBlock2D', 'DownEncoderBlock2D',
                                'DownEncoderBlock2D', 'DownEncoderBlock2D'
                            ],
                            up_block_types=[
                                'UpDecoderBlock2D', 'UpDecoderBlock2D',
                                'UpDecoderBlock2D', 'UpDecoderBlock2D'
                            ],
                            block_out_channels=[128, 256, 512, 512],
                            layers_per_block=2,
                            act_fn='silu',
                            latent_channels=4,
                            norm_num_groups=32,
                            sample_size=768,
                            dtype=weight_dtype)
    vae_params = vae.init_weights(rng=rng)
    unet = FlaxUNet2DConditionModel(sample_size=96,
                                    in_channels=4,
                                    out_channels=4,
                                    down_block_types=[
                                        'CrossAttnDownBlock2D',
                                        'CrossAttnDownBlock2D',
                                        'CrossAttnDownBlock2D', 'DownBlock2D'
                                    ],
                                    up_block_types=[
                                        'UpBlock2D', 'CrossAttnUpBlock2D',
                                        'CrossAttnUpBlock2D',
                                        'CrossAttnUpBlock2D'
                                    ],
                                    only_cross_attention=False,
                                    block_out_channels=[320, 640, 1280, 1280],
                                    layers_per_block=2,
                                    attention_head_dim=[5, 10, 20, 20],
                                    cross_attention_dim=1024,
                                    dropout=0.0,
                                    use_linear_projection=True,
                                    dtype=weight_dtype,
                                    flip_sin_to_cos=True,
                                    freq_shift=0)
    unet_params = unet.init_weights(rng=rng)
    unet_optimizer_state = optimizer.init(unet_params)
    noise_scheduler = FlaxDDPMScheduler(beta_start=0.00085,
                                        beta_end=0.012,
                                        beta_schedule="scaled_linear",
                                        num_train_timesteps=1000)
    noise_scheduler_state = noise_scheduler.create_state()
    return TrainState(optimizer=optimizer,
                      text_encoder=text_encoder,
                      vae=vae,
                      vae_params=vae_params,
                      unet=unet,
                      unet_params=unet_params,
                      unet_optimizer_state=unet_optimizer_state,
                      noise_scheduler=noise_scheduler,
                      noise_scheduler_state=noise_scheduler_state,
                      rng=rng)


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

        if train_state.output_gradient:
            metrics = {"loss": loss, "gradient": grad}
        else:
            metrics = {"loss": loss}

        return new_unet_optimizer_state, new_unet_params, metrics, new_rng

    return train_step


def make_apply_gradient_pure_fn(train_state: TrainState):

    def apply_gradient(gradinet, unet_optimizer_state, unet_params):

        unet_param_updates, new_unet_optimizer_state = train_state.optimizer.update(
            gradinet, unet_optimizer_state, unet_params)
        new_unet_params = optax.apply_updates(unet_params, unet_param_updates)

        return new_unet_optimizer_state, new_unet_params

    return apply_gradient


class JaxTrainer:

    def __init__(self, train_state: TrainState):
        self.train_state = train_state
        #self._train_step = jax.jit(make_train_step_pure_fn(train_state))
        self._train_step = make_train_step_pure_fn(train_state)
        self._apply_gradient = make_apply_gradient_pure_fn(train_state)

    def train_step(self, batch):
        self.train_state.unet_optimizer_state, self.train_state.unet_params, metrics, self.train_state.rng = self._train_step(
            batch, self.train_state.unet_optimizer_state,
            self.train_state.unet_params, self.train_state.rng)
        return metrics

    def apply_gradient(self, gradient):
        self.train_state.unet_optimizer_state, self.train_state.unet_params = self._apply_gradient(
            gradient, self.train_state.unet_optimizer_state,
            self.train_state.unet_params)


def train_jax_moduel(trainer: JaxTrainer,
                     dataloader: torch.utils.data.DataLoader):
    metrics = []
    for batch in dataloader:
        metrics.append(trainer.train_step(batch))
    return metrics


def create_iree_jax_program(train_state: TrainState, example_batch) -> Program:
    train_step_fn = make_train_step_pure_fn(train_state)
    apply_gradient_fn = make_apply_gradient_pure_fn(train_state)

    class IreeJaxStableDiffusionModule(Program):
        _unet_optimizer_state = train_state.unet_optimizer_state
        _unet_params = train_state.unet_params
        _rng = train_state.rng

        def get_unet_optimizer_state(self):
            return self._unet_optimizer_state

        def set_unet_optimizer_state(self,
                                     val=like(
                                         train_state.unet_optimizer_state)):
            self._unet_optimizer_state = val

        def get_unet_params(self):
            return self._unet_params

        def train_step(self, batch=like(example_batch)):
            self._unet_optimizer_state, self._unet_params, metrics, self._rng = self._train_step(
                batch, self._unet_optimizer_state, self._unet_params,
                self._rng)
            return metrics

        @kernel
        def _train_step(batch, unet_optimizer_state, unet_params, rng):
            new_unet_optimizer_state, new_unet_params, metrics, new_rng = train_step_fn(
                batch, unet_optimizer_state, unet_params, rng)
            return new_unet_optimizer_state, new_unet_params, metrics, new_rng

        def apply_gradient(self, gradient=like(train_state.unet_params)):
            self._unet_optimizer_state, self._unet_params = self._apply_gradient(
                gradient, self._unet_optimizer_state, self._unet_params)

        @kernel
        def _apply_gradient(gradient, unet_optimizer_state, unet_params):
            new_unet_optimizer_state, new_unet_params = apply_gradient_fn(
                gradient, unet_optimizer_state, unet_params)
            return new_unet_optimizer_state, new_unet_params

    return IreeJaxStableDiffusionModule()


def build_mlir_module(module: Program, path: str):
    with open(path, "wb") as f:
        Program.save(module, f)


def build_iree_module(get_iree_jax_program: Callable[[], Program],
                      mlir_module_path: str,
                      iree_module_path: str,
                      use_cache: bool = True,
                      iree_backend: str = "llvm-cpu",
                      iree_runtime: str = "local-task"):
    # Train for 1 step and return Unet train state

    should_make_mlir = not use_cache or not os.path.exists(mlir_module_path)
    if should_make_mlir:
        iree_jax_program = get_iree_jax_program()
        build_mlir_module(module=iree_jax_program, path=mlir_module_path)
        del iree_jax_program
        logger.debug(f"MLIR written to \"{mlir_module_path}\".")

    should_make_iree_module = not use_cache or not os.path.exists(
        iree_module_path)
    if should_make_iree_module:
        iree.compiler.tools.compile_file(input_file=mlir_module_path,
                                         output_file=iree_module_path,
                                         target_backends=[iree_backend],
                                         input_type="mhlo")
        logger.debug(f"File \"{iree_module_path}\" compiled.")

    module = iree_rt.system_api.load_vm_flatbuffer_file(iree_module_path,
                                                        driver=iree_runtime)
    return module


def create_default_iree_jax_program_getter(sample_batch,
                                           pretrained_model_name_or_path,
                                           rng,
                                           weight_dtype=jnp.float32):

    def get_iree_jax_program():
        optimizer = create_optimizer()
        train_state = load_train_state(
            optimizer=optimizer,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            rng=rng,
            weight_dtype=weight_dtype)
        iree_jax_program = create_iree_jax_program(train_state, sample_batch)
        return iree_jax_program

    return get_iree_jax_program


def train_iree_module(dataloader: torch.utils.data.DataLoader,
                      module: iree_rt.system_api.BoundModule):
    metrics = []
    for batch in dataloader:
        args = tree_flatten(batch)[0]
        args[0] = np.array(args[0], dtype=np.int32)
        #np.savez_compressed("batch.npz", *args)
        metrics.append(call_iree_function(module.train_step, *args))
    return metrics


def create_dataloader(
    dataset: Dataset,
    tokenizer: CLIPTokenizer,
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

    return train_dataloader


def clone_device_array_list_to_numpy(
        array_list: List[iree_rt.DeviceArray]) -> np.ndarray:
    res = [None] * len(array_list)
    while (len(array_list)):
        arr = array_list[-1].to_host()
        res[len(array_list) - 1] = np.copy(arr)
        del array_list[-1]
    return res


def call_iree_function(fn: Callable, *args):
    res_tuple = fn(*args)
    res = list(res_tuple)
    del res_tuple
    return clone_device_array_list_to_numpy(res)
