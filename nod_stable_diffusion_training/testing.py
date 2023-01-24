from typing import Callable, Any
import sys
import argparse
from typing import List, TypeVar, Tuple
from numbers import Real, Integral
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


DEFAULT_ABS_TOLERANCE = 1e-6
DEFAULT_REL_TOLERANCE = 1e-5


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


def array_compare_metrics(a: Tensor,
                          b: Tensor) -> Tuple[Real, Real, Integral, Integral]:
    """Return a tuple of max absolute relative difference, absolute difference and their cooresponding indices."""
    abs_diff = np.abs(a - b)
    max_abs_diff = np.max(abs_diff)
    nonzero = np.bool_(b != 0)
    if np.all(~nonzero):
        max_rel_diff = 0
        max_rel_diff_index = 0
    else:
        rel_diff = abs_diff / np.abs(np.asarray(b))
        max_rel_diff = np.max(rel_diff[nonzero])
        if not np.all(nonzero):
            rel_diff[~nonzero] = 0
        max_rel_diff_index = np.argmax(rel_diff)
    return max_rel_diff, max_abs_diff, max_rel_diff_index, np.argmax(abs_diff)


def array_list_compare_metrics(
    a: List[Tensor], b: List[Tensor]
) -> Tuple[List[Real], List[Real], List[Integral], List[Integral]]:
    """Return a tuple of max absolute relative differences and absolute differences
    and their cooresponding indices
    for all corresponding array pairs."""
    assert len(a) == len(b)
    rel_diff_list = []
    abs_diff_list = []
    rel_diff_index_list = []
    abs_diff_index_list = []
    for x, y in zip(a, b):
        rel_diff, abs_diff, rel_diff_index, abs_diff_index = array_compare_metrics(
            x, y)
        rel_diff_list.append(rel_diff)
        abs_diff_list.append(abs_diff)
        rel_diff_index_list.append(rel_diff_index)
        abs_diff_index_list.append(abs_diff_index)
    return (np.asarray(rel_diff_list), np.asarray(abs_diff_list),
            np.asarray(rel_diff_index_list), np.asarray(abs_diff_index_list))


def print_array_list_diff(a: List[Tensor], b: List[Tensor], name: str):
    rel_diff, abs_diff, rel_diff_idx, abs_diff_idx = array_list_compare_metrics(
        a, b)
    #print(f"{name} rel diff = {rel_diff}")
    #print(f"{name} abs diff = {abs_diff}")
    max_rel_diff = np.nanmax(rel_diff)
    max_abs_diff = np.max(abs_diff)
    max_rel_diff_idx = np.nanargmax(rel_diff)
    max_abs_diff_idx = np.argmax(abs_diff)
    print(
        f"{name} max rel diff, argmax = {max_rel_diff}, [{max_rel_diff_idx}][{rel_diff_idx[max_rel_diff_idx]}]"
    )
    print(
        f"{name} max abs diff, argmax = {max_abs_diff}, [{max_abs_diff_idx}][{abs_diff_idx[max_abs_diff_idx]}]"
    )
