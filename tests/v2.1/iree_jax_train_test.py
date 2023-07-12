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
                                                   print_array_list_diff, main
                                                   as testing_main, args as
                                                   testing_args)
from nod_stable_diffusion_training.iree_jax import (
    create_optimizer, load_train_state, JaxTrainer, train_jax_moduel,
    build_iree_module, train_iree_module, create_small_model_train_state,
    create_dataloader, create_iree_jax_program, create_full_model_train_state,
    call_iree_function, export_mlir_jax_trainer, legalize_array_for_iree_input,
    shard)
from copy import deepcopy
import numpy as np
import argparse
from typing import List, Optional
import sys
import iree.compiler
import iree.runtime
from multiprocessing import Pool

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def build_iree_module_in_dir(dataloader: torch.utils.data.DataLoader,
                             iree_backend,
                             iree_runtime,
                             get_iree_jax_program: Callable[[], Program],
                             artifacts_dir: str,
                             use_cache: bool = True,
                             mlir_format: str = "bytecode"):
    module = build_iree_module(
        get_iree_jax_program=get_iree_jax_program,
        mlir_module_path=os.path.join(artifacts_dir,
                                      f"stable_diffusion_training.mlir"),
        iree_module_path=os.path.join(
            artifacts_dir, f"stable_diffusion_training_{iree_backend}.vmfb"),
        use_cache=use_cache,
        iree_backend=iree_backend,
        iree_runtime=iree_runtime,
        mlir_format=mlir_format)
    return module


def test_training_with_iree_jax_pretrained():
    """Test that training with IREE Jax produces the same result as Jax.
    The model tested is a full pretrained version."""
    seed = 12345
    set_seed(seed)

    rng = jax.random.PRNGKey(seed)
    jax_train_state = load_train_state(
        optimizer=create_optimizer(),
        pretrained_model_name_or_path=testing_args.
        pretrained_model_name_or_path,
        rng=rng,
        distribution_count=testing_args.distribution_count,
        weight_dtype=jnp.float32)
    iree_train_state = deepcopy(jax_train_state)

    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    tokenizer = CLIPTokenizer.from_pretrained(
        testing_args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataloader = create_dataloader(dataset,
                                   tokenizer=tokenizer,
                                   seed=seed,
                                   train_batch_size=testing_args.batch_size,
                                   max_train_samples=testing_args.batch_size)

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
        use_cache=testing_args.use_cache,
        iree_backend=testing_args.target_backend,
        iree_runtime=testing_args.driver,
        mlir_format=testing_args.mlir_format)
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
    jax_train_state = create_small_model_train_state(
        optimizer=optimizer,
        seed=seed,
        output_gradient=True,
        distribution_count=testing_args.distribution_count)
    iree_train_state = deepcopy(jax_train_state)
    # jax_train_state_orig = deepcopy(jax_train_state)

    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    tokenizer = CLIPTokenizer.from_pretrained(
        testing_args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataloader = create_dataloader(dataset,
                                   tokenizer=tokenizer,
                                   seed=seed,
                                   resolution=8,
                                   max_train_samples=testing_args.batch_size,
                                   train_batch_size=testing_args.batch_size)

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
        use_cache=testing_args.use_cache,
        iree_backend=testing_args.target_backend,
        iree_runtime=testing_args.driver,
        mlir_format=testing_args.mlir_format)

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
    The model tested is full sized."""
    seed = 12345
    set_seed(seed)
    optimizer = create_optimizer()
    jax_train_state = create_full_model_train_state(
        optimizer=optimizer,
        seed=seed,
        distribution_count=testing_args.distribution_count)
    iree_train_state = deepcopy(jax_train_state)

    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    tokenizer = CLIPTokenizer.from_pretrained(
        testing_args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataloader = create_dataloader(dataset,
                                   tokenizer=tokenizer,
                                   seed=seed,
                                   train_batch_size=testing_args.batch_size,
                                   max_train_samples=testing_args.batch_size)

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
        use_cache=testing_args.use_cache,
        iree_backend=testing_args.target_backend,
        iree_runtime=testing_args.driver,
        mlir_format=testing_args.mlir_format)
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


def get_shard(rank: int, *args):
    return jax.tree_util.tree_map(lambda x: x[rank], args)


def run_shard(shard_index: Optional[int], mlir_filepath: str,
              iree_module_filepath: str, args):
    if testing_args.distribution_count > 1:
        os.environ["OMPI_COMM_WORLD_SIZE"] = str(
            testing_args.distribution_count)
        os.environ["OMPI_COMM_WORLD_RANK"] = str(shard_index)
    iree_module = iree.runtime.system_api.load_vm_flatbuffer_file(
        iree_module_filepath, driver=testing_args.driver)

    with open(f"{mlir_filepath}.shard-{shard_index}.in.npy", "wb") as f:
        for arr in args:
            np.save(f, arr)

    # Run IREE module.
    res = call_iree_function(iree_module.main, *args)
    return res


def test_distributed_iree_module_from_jax_trainer():
    seed = 12345
    set_seed(seed)
    optimizer = create_optimizer()
    jax_train_state = create_small_model_train_state(
        optimizer=optimizer,
        seed=seed,
        output_gradient=True,
        distribution_count=testing_args.distribution_count)

    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    tokenizer = CLIPTokenizer.from_pretrained(
        testing_args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataloader = create_dataloader(dataset,
                                   tokenizer=tokenizer,
                                   seed=seed,
                                   resolution=8,
                                   max_train_samples=testing_args.batch_size,
                                   train_batch_size=testing_args.batch_size)

    sample_batch = None
    for batch in dataloader:
        sample_batch = batch
        break
    assert (sample_batch is not None)

    jax_trainer = JaxTrainer(jax_train_state)

    # Export MLIR.
    mlir_filepath = f"test_compile_iree_module_from_jax_trainer_stable_diffusion_pure_train_step_fn.dist-count-{testing_args.distribution_count}.mlir"
    if not testing_args.use_cache or not os.path.exists(mlir_filepath):
        export_mlir_jax_trainer(jax_trainer, sample_batch, mlir_filepath)

    # Compile IREE module.
    iree_module_filepath = f"{mlir_filepath}.{testing_args.target_backend}.vmfb"
    if not testing_args.use_cache or not os.path.exists(iree_module_filepath):
        iree.compiler.tools.compile_file(
            input_file=mlir_filepath,
            output_file=iree_module_filepath,
            target_backends=[testing_args.target_backend],
            input_type="stablehlo")

    shareded_sample_batch = shard(sample_batch,
                                  testing_args.distribution_count)
    module_args_for_all_shards = (shareded_sample_batch
                                  if testing_args.distribution_count > 1 else
                                  sample_batch,
                                  jax_train_state.unet_optimizer_state,
                                  jax_train_state.unet_params,
                                  jax_train_state.rng)
    module_args_for_all_shards = tree_flatten(module_args_for_all_shards)[0]
    module_args_for_all_shards = [
        legalize_array_for_iree_input(arr)
        for arr in module_args_for_all_shards
    ]

    if testing_args.distribution_count == 1:
        run_shard(None, mlir_filepath, iree_module_filepath,
                  module_args_for_all_shards)
    else:
        with Pool(testing_args.distribution_count) as pool:
            starmap_args = [[
                shard_index, mlir_filepath, iree_module_filepath,
                get_shard(shard_index, *module_args_for_all_shards)
            ] for shard_index in range(testing_args.distribution_count)]
            res = pool.starmap(run_shard, starmap_args)

    # TODO: verify result is correct when compared to JAX.


def parse_args(args: List[str] = sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path",
                        type=str,
                        default="flax/stable-diffusion-2-1")
    return parser.parse_known_args(args=args)


def main():
    new_args, remaining_args = parse_args()
    global testing_args
    testing_args.update(vars(new_args))
    testing_main([sys.argv[0]] + remaining_args)


if __name__ == "__main__":
    main()
