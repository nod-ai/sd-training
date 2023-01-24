from typing import Callable
import torch
import torch.utils.checkpoint
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import CLIPTokenizer, set_seed
from iree.jax import Program
from jax.tree_util import tree_flatten, tree_unflatten
import os
import logging
from nod_stable_diffusion_training.testing import (assert_array_list_equal,
                                                   assert_array_list_allclose,
                                                   print_array_list_diff, main,
                                                   args as testing_args)
from nod_stable_diffusion_training.iree_jax import (
    create_optimizer, load_train_state, JaxTrainer, train_jax_moduel,
    build_iree_module, train_iree_module, create_small_model_train_state,
    create_dataloader, create_iree_jax_program, create_full_model_train_state,
    clone_device_array_list_to_numpy, call_iree_function)
from copy import deepcopy
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def jax_train_pretrained(dataloader: torch.utils.data.DataLoader,
                         pretrained_model_name_or_path: str, rng_seed: int):
    # Train for 1 step and return Unet train state

    set_seed(rng_seed)

    rng = jax.random.PRNGKey(seed=rng_seed)

    train_state = load_train_state(
        optimizer=create_optimizer(),
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        rng=rng,
        weight_dtype=jnp.float32)
    trainer = JaxTrainer(train_state)

    logger.debug("Jax model created.")
    train_jax_moduel(trainer, dataloader)
    logger.debug("Jax train step done.")

    return train_state.unet_optimizer_state, train_state.unet_params


def build_iree_module_in_dir(
    dataloader: torch.utils.data.DataLoader,
    iree_backend,
    iree_runtime,
    get_iree_jax_program: Callable[[], Program],
    artifacts_dir: str,
    use_cache: bool = True,
):
    module = build_iree_module(
        get_iree_jax_program=get_iree_jax_program,
        mlir_module_path=os.path.join(artifacts_dir,
                                      f"stable_diffusion_training.mlir"),
        iree_module_path=os.path.join(
            artifacts_dir, f"stable_diffusion_training_{iree_backend}.vmfb"),
        use_cache=use_cache,
        iree_backend=iree_backend,
        iree_runtime=iree_runtime,
    )
    return module


def test_training_with_iree_jax_pretrained():
    """Test that training with IREE Jax produces the same result as Jax.
    The model tested is a full pretrained version."""
    seed = 12345
    set_seed(seed)
    pretrained_model_name_or_path = "flax/stable-diffusion-2-1"

    rng = jax.random.PRNGKey(seed)
    jax_train_state = load_train_state(
        optimizer=create_optimizer(),
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        rng=rng,
        weight_dtype=jnp.float32)
    iree_train_state = deepcopy(jax_train_state)

    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,
                                              subfolder="tokenizer")
    dataloader = create_dataloader(dataset, tokenizer=tokenizer, seed=seed)

    sample_batch = None
    for batch in dataloader:
        sample_batch = batch
        break
    assert (sample_batch is not None)

    def get_iree_jax_program():
        return create_iree_jax_program(iree_train_state, sample_batch)

    iree_module = build_iree_module_in_dir(
        dataloader=dataloader,
        get_iree_jax_program=get_iree_jax_program,
        artifacts_dir=os.getcwd(),
        use_cache=True,
        iree_backend=testing_args.target_backend,
        iree_runtime=testing_args.driver)
    train_iree_module(module=iree_module, dataloader=dataloader)
    logger.debug("Iree train step done.")
    iree_unet_optimizer_state = iree_module.get_unet_optimizer_state()
    iree_unet_params = iree_module.get_unet_params()

    jax_trainer = JaxTrainer(jax_train_state)
    train_jax_moduel(jax_trainer, dataloader)
    logger.debug("Jax train step done.")
    jax_unet_optimizer_state = jax_train_state.unet_optimizer_state
    jax_unet_params = jax_train_state.unet_params
    del jax_train_state

    assert_array_list_allclose(
        tree_flatten(iree_unet_optimizer_state)[0],
        tree_flatten(jax_unet_optimizer_state)[0])
    assert_array_list_allclose(
        tree_flatten(iree_unet_params)[0],
        tree_flatten(jax_unet_params)[0])


def test_training_with_iree_jax_small_model():
    """Test that training with IREE Jax produces the same result as Jax.
    The model tested is as small as possible."""
    seed = 12345
    set_seed(seed)
    optimizer = create_optimizer()
    jax_train_state = create_small_model_train_state(optimizer=optimizer,
                                                     seed=seed,
                                                     output_gradient=True)
    iree_train_state = deepcopy(jax_train_state)
    # jax_train_state_orig = deepcopy(jax_train_state)

    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    pretrained_model_name_or_path = "flax/stable-diffusion-2-1"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,
                                              subfolder="tokenizer")
    dataloader = create_dataloader(dataset,
                                   tokenizer=tokenizer,
                                   seed=seed,
                                   resolution=8)

    sample_batch = None
    for batch in dataloader:
        sample_batch = batch
        break
    assert (sample_batch is not None)

    def get_iree_jax_program():
        return create_iree_jax_program(iree_train_state, sample_batch)

    iree_module = build_iree_module_in_dir(
        dataloader=dataloader,
        get_iree_jax_program=get_iree_jax_program,
        artifacts_dir=os.getcwd(),
        use_cache=True,
        iree_backend=testing_args.target_backend,
        iree_runtime=testing_args.driver)

    #np.savez_compressed("jax_unet_optimizer_state.npz", *tree_flatten(jax_train_state.unet_optimizer_state)[0])

    iree_unet_optimizer_state = call_iree_function(
        iree_module.get_unet_optimizer_state)
    assert_array_list_equal(
        iree_unet_optimizer_state,
        tree_flatten(jax_train_state.unet_optimizer_state)[0])
    iree_unet_params = call_iree_function(iree_module.get_unet_params)
    assert_array_list_equal(iree_unet_params,
                            tree_flatten(jax_train_state.unet_params)[0])

    jax_trainer = JaxTrainer(jax_train_state)
    jax_metrics = train_jax_moduel(jax_trainer, dataloader)[0]
    jax_loss = jax_metrics["loss"]
    jax_gradient = jax_metrics["gradient"]
    logger.debug("Jax train step done.")
    jax_unet_optimizer_state = jax_train_state.unet_optimizer_state
    jax_unet_params = jax_train_state.unet_params

    iree_metrics = train_iree_module(module=iree_module,
                                     dataloader=dataloader)[0]
    iree_gradient = iree_metrics[:len(iree_metrics) - 1]
    iree_loss = iree_metrics[len(iree_metrics) - 1:]
    logger.debug("Iree train step done.")
    # iree_module.apply_gradient(*tree_flatten(jax_gradient)[0])
    iree_unet_optimizer_state = call_iree_function(
        iree_module.get_unet_optimizer_state)
    iree_unet_params = call_iree_function(iree_module.get_unet_params)

    # jax_train_state.unet_params = deepcopy(jax_train_state_orig.unet_params)
    # jax_train_state.unet_optimizer_state = deepcopy(jax_train_state_orig.unet_optimizer_state)
    # jax_gradient_flattented, jax_gradient_tree = tree_flatten(jax_gradient)
    # iree_gradient_unflattened = tree_unflatten(jax_gradient_tree, iree_gradient)
    # jax_trainer.apply_gradient(iree_gradient_unflattened)
    # jax_unet_optimizer_state = jax_train_state.unet_optimizer_state
    # jax_unet_params = jax_train_state.unet_params

    print_array_list_diff(iree_gradient,
                          tree_flatten(jax_gradient)[0], "unet_gradinet")
    print_array_list_diff(iree_unet_params,
                          tree_flatten(jax_unet_params)[0], "unet_params")
    print_array_list_diff(iree_unet_optimizer_state,
                          tree_flatten(jax_unet_optimizer_state)[0],
                          "unet_optimizer_state")

    assert_array_list_allclose(iree_metrics, tree_flatten(jax_metrics)[0])
    assert_array_list_allclose(iree_unet_optimizer_state,
                               tree_flatten(jax_unet_optimizer_state)[0])
    assert_array_list_allclose(
        iree_unet_params,
        tree_flatten(jax_unet_params)[0],
    )


def test_training_with_iree_jax_full_model():
    """Test that training with IREE Jax produces the same result as Jax.
    The model tested is as small as possible."""
    seed = 12345
    set_seed(seed)
    optimizer = create_optimizer()
    jax_train_state = create_full_model_train_state(optimizer=optimizer,
                                                    seed=seed)
    iree_train_state = deepcopy(jax_train_state)

    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    pretrained_model_name_or_path = "flax/stable-diffusion-2-1"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,
                                              subfolder="tokenizer")
    dataloader = create_dataloader(dataset, tokenizer=tokenizer, seed=seed)

    sample_batch = None
    for batch in dataloader:
        sample_batch = batch
        break
    assert (sample_batch is not None)

    def get_iree_jax_program():
        return create_iree_jax_program(iree_train_state, sample_batch)

    iree_module = build_iree_module_in_dir(
        dataloader=dataloader,
        get_iree_jax_program=get_iree_jax_program,
        artifacts_dir=os.getcwd(),
        use_cache=True,
        iree_backend=testing_args.target_backend,
        iree_runtime=testing_args.driver)
    train_iree_module(module=iree_module, dataloader=dataloader)
    logger.debug("Iree train step done.")
    iree_unet_optimizer_state = iree_module.get_unet_optimizer_state()
    iree_unet_params = iree_module.get_unet_params()

    jax_trainer = JaxTrainer(jax_train_state)
    train_jax_moduel(jax_trainer, dataloader)
    logger.debug("Jax train step done.")
    jax_unet_optimizer_state = jax_train_state.unet_optimizer_state
    jax_unet_params = jax_train_state.unet_params
    del jax_train_state

    assert_array_list_allclose(
        tree_flatten(iree_unet_optimizer_state)[0],
        tree_flatten(jax_unet_optimizer_state)[0])
    assert_array_list_allclose(
        tree_flatten(iree_unet_params)[0],
        tree_flatten(jax_unet_params)[0])


if __name__ == "__main__":
    main()
