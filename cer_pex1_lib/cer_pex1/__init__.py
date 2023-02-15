from .robot_arm import RobotArm


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
