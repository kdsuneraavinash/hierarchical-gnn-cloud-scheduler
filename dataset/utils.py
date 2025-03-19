from typing import Any
import numpy as np


def random_list(arr_len: int, arr_sum: int, rng: np.random.RandomState, min_value: int = 1) -> list[int]:
    value_arr = rng.rand(arr_len)
    value_arr: np.ndarray[tuple[int, ...], Any] = value_arr / value_arr.sum()
    scaling_factor = arr_sum - (arr_len * min_value)
    value_arr = np.floor(value_arr * scaling_factor) + min_value
    value_arr[-1] += arr_sum - value_arr.sum()
    return value_arr.astype(int).tolist()
