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
    create_optimizer, load_train_state, JaxTrainer, TrainState,
    train_jax_moduel, build_iree_module, train_iree_module,
    create_small_model_train_state, create_dataloader, create_iree_jax_program,
    create_full_model_train_state, call_iree_function, export_mlir_jax_trainer,
    legalize_array_for_iree_input, shard)
from copy import deepcopy
import numpy as np
import argparse
from typing import List, Optional
import sys
import iree.compiler
import iree.runtime
import iree.runtime.distributed
import iree.runtime.distributed.sharding_pass_validation
from multiprocessing import Pool
from numpy.typing import ArrayLike

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


def test_distributed_training_with_iree(train_state: TrainState,
                                        output_file_name_prefix: str,
                                        rng_seed: int):
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(path="lambdalabs/pokemon-blip-captions")
    tokenizer = CLIPTokenizer.from_pretrained(
        testing_args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataloader = create_dataloader(dataset,
                                   tokenizer=tokenizer,
                                   seed=rng_seed,
                                   resolution=8,
                                   max_train_samples=testing_args.batch_size,
                                   train_batch_size=testing_args.batch_size)

    sample_batch = None
    for batch in dataloader:
        sample_batch = batch
        break
    assert (sample_batch is not None)
    shareded_sample_batch = shard(sample_batch,
                                  testing_args.distribution_count)

    distributed_jax_trainer = JaxTrainer(train_state)

    # Export MLIRs.
    distributed_mlir_filepath = f"{output_file_name_prefix}.dist-count-{testing_args.distribution_count}.mlir"
    if not testing_args.use_cache or not os.path.exists(
            distributed_mlir_filepath):
        export_mlir_jax_trainer(distributed_jax_trainer, sample_batch,
                                distributed_mlir_filepath)

    # Compile IREE modules.
    distributed_iree_module_filepath = f"{distributed_mlir_filepath}.{testing_args.target_backend}.vmfb"
    if not testing_args.use_cache or not os.path.exists(
            distributed_iree_module_filepath):
        iree.compiler.tools.compile_file(
            input_file=distributed_mlir_filepath,
            output_file=distributed_iree_module_filepath,
            target_backends=[testing_args.target_backend],
            input_type="stablehlo")

    # Run distributed JAX model.
    distributed_jax_results = distributed_jax_trainer._train_step(
        batch=shareded_sample_batch,
        unet_optimizer_state=train_state.unet_optimizer_state,
        unet_params=train_state.unet_params,
        rng=train_state.rng)
    distributed_jax_results = tree_flatten(distributed_jax_results)[0]
    distributed_jax_results = iree.runtime.distributed.sharding_pass_validation.swap_shard_axis(
        distributed_jax_results)

    # Run distributed IREE module.
    module_args_for_all_shards = (shareded_sample_batch,
                                  train_state.unet_optimizer_state,
                                  train_state.unet_params, train_state.rng)
    module_args_for_all_shards = tree_flatten(module_args_for_all_shards)[0]
    module_args_for_all_shards = [
        legalize_array_for_iree_input(arr)
        for arr in module_args_for_all_shards
    ]
    module_args_for_all_shards = iree.runtime.distributed.sharding_pass_validation.swap_shard_axis(
        module_args_for_all_shards)
    distributed_iree_results = iree.runtime.distributed.run_ranks(
        num_ranks=testing_args.distribution_count,
        module_filepath=distributed_iree_module_filepath,
        function="main",
        inputs=module_args_for_all_shards,
        driver=testing_args.driver,
        measure_execution_time=True,
        warmup=1)
    rank_execution_times_seconds = [
        rank_results[-1] for rank_results in distributed_iree_results
    ]
    distributed_iree_results = [
        rank_results[:-1] for rank_results in distributed_iree_results
    ]
    print(
        f"rank_execution_times_seconds = {np.array(rank_execution_times_seconds).flatten()}"
    )
    print(
        f"max rank_execution_times_seconds = {np.max(rank_execution_times_seconds)}"
    )

    assert (len(distributed_iree_results) == len(distributed_jax_results))
    for rank, rank_results in enumerate(distributed_iree_results):
        print_array_list_diff(rank_results, distributed_jax_results[rank],
                              f"distributed_iree_vs_jax_results_rank_{rank}")
    for rank, rank_results in enumerate(distributed_iree_results):
        assert_array_list_allclose(rank_results, distributed_jax_results[rank])


def test_data_prallel_small_model_training_with_iree():
    seed = 12345
    set_seed(seed)
    optimizer = create_optimizer()
    distributed_jax_train_state = create_small_model_train_state(
        optimizer=optimizer,
        seed=seed,
        distribution_count=testing_args.distribution_count)
    test_distributed_training_with_iree(
        train_state=distributed_jax_train_state,
        output_file_name_prefix=
        "test_data_prallel_small_model_training_with_iree_stable_diffusion_pure_train_step_fn",
        rng_seed=seed)


def test_data_prallel_full_model_training_with_iree():
    seed = 12345
    set_seed(seed)
    optimizer = create_optimizer()
    distributed_jax_train_state = create_full_model_train_state(
        optimizer=optimizer,
        seed=seed,
        distribution_count=testing_args.distribution_count)
    test_distributed_training_with_iree(
        train_state=distributed_jax_train_state,
        output_file_name_prefix=
        "test_data_prallel_full_model_training_with_iree_stable_diffusion_pure_train_step_fn",
        rng_seed=seed)


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
