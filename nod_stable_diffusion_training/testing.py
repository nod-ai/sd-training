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
from jax.tree_util import tree_flatten
from iree import runtime as iree_rt
from tempfile import TemporaryDirectory
import os
import logging


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
