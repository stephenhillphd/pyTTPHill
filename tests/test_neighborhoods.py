import numpy as np
from pyTTP.neighborhoods import (swap_homes, swap_rounds, swap_teams,
                                 partial_swap_rounds, partial_swap_teams)

sol_nl4 = np.array([[-4, -2, -3,  2,  4,  3],
                    [3,  1, -4, -1, -3,  4],
                    [-2, -4,  1,  4,  2, -1],
                    [1,  3,  2, -3, -1, -2]], dtype=np.int32)


def test_swap_homes():
    result = swap_homes(sol_nl4, 1, 2)
    expected_result = np.array([[-4,  2, -3, -2,  4,  3],
                                [3, -1, -4,  1, -3,  4],
                                [-2, -4,  1,  4,  2, -1],
                                [1,  3,  2, -3, -1, -2]], dtype=np.int32)
    np.testing.assert_array_equal(result, expected_result)


def test_swap_rounds():
    result = swap_rounds(sol_nl4, 1, 2)
    expected_result = np.array([[-2, -4, -3,  2,  4,  3],
                                [1,  3, -4, -1, -3,  4],
                                [-4, -2,  1,  4,  2, -1],
                                [3,  1,  2, -3, -1, -2]], dtype=np.int32)
    np.testing.assert_array_equal(result, expected_result)


def test_swap_teams():
    result = swap_teams(sol_nl4, 1, 2)
    expected_result = np.array([[3, -2, -4,  2, -3,  4],
                                [-4,  1, -3, -1,  4,  3],
                                [-1, -4,  2,  4,  1, -2],
                                [2,  3,  1, -3, -2, -1]], dtype=np.int32)
    np.testing.assert_array_equal(result, expected_result)


def test_partial_swap_rounds():
    result = partial_swap_rounds(sol_nl4, 1, 2, 3)
    expected_result = np.array([[-4, -3, -2,  2,  4,  3],
                                [3, -4,  1, -1, -3,  4],
                                [-2,  1, -4,  4,  2, -1],
                                [1,  2,  3, -3, -1, -2]], dtype=np.int32)
    np.testing.assert_array_equal(result, expected_result)


def test_partial_swap_teams():
    result = partial_swap_teams(sol_nl4, 1, 2, 3)
    expected_result = np.array([[-3,  2, -4, -2,  3,  4],
                                [4, -1, -3,  1, -4,  3],
                                [1, -4,  2,  4, -1, -2],
                                [-2,  3,  1, -3,  2, -1]], dtype=np.int32)
    np.testing.assert_array_equal(result, expected_result)
