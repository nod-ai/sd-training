from typing import Callable, Any
import sys
import argparse
from typing import List
import numpy as np
import pytest

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


def assert_array_almost_equal(a, b):
    np_a = np.asarray(a)
    np_b = np.asarray(b)

    if (np.issubdtype(np_a.dtype, np.integer)
            and np.issubdtype(np_b.dtype, np.integer)):
        np.testing.assert_array_equal(np_a, np_b)
        return

    # Test for absolute error.
    np.testing.assert_array_almost_equal(np_a, np_b, decimal=4)
    # Test for relative error while ignoring errors from
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


def parse_args(args: List[str] = sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_backend", type=str, default="llvm-cpu")
    parser.add_argument("--driver", type=str, default="local-task")
    return parser.parse_known_args(args=args)


DEFAULT_DECIMAL = 5
DEFAULT_EPSILON = 10**-7
DEFAULT_NULP = 10**8


def assert_array_almost_equal(a,
                              b,
                              decimal=DEFAULT_DECIMAL,
                              epsilon=DEFAULT_EPSILON,
                              nulp=DEFAULT_NULP):
    np_a = np.asarray(a)
    np_b = np.asarray(b)

    if (np.issubdtype(np_a.dtype, np.integer)
            and np.issubdtype(np_b.dtype, np.integer)):
        np.testing.assert_array_equal(np_a, np_b)
        return

    # Test for absolute error.
    np.testing.assert_array_almost_equal(np_a, np_b, decimal=decimal)
    # Test for relative error while ignoring false errors from
    # catastrophic cancellation.
    np.testing.assert_array_almost_equal_nulp(np.abs(np_a - np_b) + epsilon,
                                              np.zeros_like(np_a),
                                              nulp=nulp)


def assert_array_list_equal(
    a,
    b,
    array_compare_fn: Callable[[Any, Any],
                               None] = np.testing.assert_array_equal):
    assert (len(a) == len(b))
    for x, y in zip(a, b):
        array_compare_fn(x, y)


def assert_array_list_almost_equal(a,
                                   b,
                                   decimal=DEFAULT_DECIMAL,
                                   epsilon=DEFAULT_EPSILON,
                                   nulp=DEFAULT_NULP):
    assert_array_list_equal(
        a, b,
        lambda x, y: assert_array_almost_equal(x, y, decimal, epsilon, nulp))
