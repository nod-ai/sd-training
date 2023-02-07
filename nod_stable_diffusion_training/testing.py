from typing import Callable, Any
import sys
import argparse
from typing import List, TypeVar
import numpy as np
import pytest

Tensor = TypeVar('Tensor')

args = None


def main(argv: List[str] = sys.argv):
    global args
    args, remaining_args = parse_args(argv[1:])
    pytest.main(args=[argv[0]] + remaining_args)


def parse_args(args: List[str] = sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_backend", type=str, default="llvm-cpu")
    parser.add_argument("--driver", type=str, default="local-task")
    return parser.parse_known_args(args=args)


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


def parse_args(args: List[str] = sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_backend", type=str, default="llvm-cpu")
    parser.add_argument("--driver", type=str, default="local-task")
    return parser.parse_known_args(args=args)


DEFAULT_ABS_TOLERANCE = 1e-6
DEFAULT_REL_TOLERANCE = 1e-3


def allclose(a: Tensor,
             b: Tensor,
             rtol=DEFAULT_REL_TOLERANCE,
             atol=DEFAULT_ABS_TOLERANCE):
    return np.allclose(np.asarray(a), np.asarray(b), rtol, atol)


def array_equal(a: Tensor, b: Tensor):
    return np.array_equal(np.asarray(a), np.asarray(b))


def assert_array_list_compare(array_compare_fn, a: Tensor, b: Tensor):
    assert (len(a) == len(b))
    for x, y in zip(a, b):
        np.testing.assert_array_compare(array_compare_fn, x, y)


def assert_array_list_equal(a: List[Tensor], b: List[Tensor]):
    assert_array_list_compare(array_equal, a, b)


def assert_array_list_allclose(a: List[Tensor],
                               b: List[Tensor],
                               rtol=DEFAULT_REL_TOLERANCE,
                               atol=DEFAULT_ABS_TOLERANCE):
    assert_array_list_compare(lambda x, y: allclose(x, y, rtol, atol), a, b)
