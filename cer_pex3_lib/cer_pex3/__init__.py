from typing import Tuple as Tuple
import numpy as np

from .robot_arm import RobotArm
from .target_specifications import TargetSpecifications, target_specs_pd, target_specs_pid
from .trajectory_generation_tests import traj_gen_tests_1d, traj_gen_tests_2d, traj_trans_tests

from .printing import get_letter_print_path


def test_true(test_name, value: bool):
    if value:
        print("\033[92mTest {}: passed.\033[0m".format(test_name))
    else:
        print("\033[91mTest {}: failed.\033[0m".format(test_name))


def test_almost_equal(test_name, value: float, target: float, precision: float = 1e-4):
    test_true(test_name, abs(value - target) < precision)


def test_almost_zero(test_name, value: float, precision: float = 1e-4):
    test_almost_equal(test_name, value, 0.0, precision=precision)


def test_gt(test_name, upper: float, lower: float, precision: float = 1e-4):
    test_true(test_name, upper > lower - precision)


def test_geq(test_name, upper: float, lower: float, precision: float = 1e-4):
    test_true(test_name, upper >= lower - precision)


def check_shape(arr: np.ndarray, target_shape: Tuple[int, ...], var_name: str):
    num_undefined_lengths = sum(d == -1 for d in target_shape)
    if num_undefined_lengths == 1:
        placeholders = ["N"]
    elif num_undefined_lengths == 2:
        placeholders = ["N", "M"]
    else:
        placeholders = ["N" + str(i + 1) for i in range(num_undefined_lengths)]
    target_shape_elements = []
    placeholder_index = 0
    for e in target_shape:
        if e == -1:
            target_shape_elements.append(placeholders[placeholder_index])
            placeholder_index += 1
        else:
            target_shape_elements.append(str(e))
    target_shape_str = "({})".format(", ".join(target_shape_elements))
    assert all(et == -1 or e == et for e, et, in zip(arr.shape, target_shape)), \
        "Expected {} to have shape {} but got {}.".format(var_name, target_shape_str, arr.shape)
