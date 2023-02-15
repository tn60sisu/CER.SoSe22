from typing import List, Tuple


def get_letter_print_path(letter: str) -> List[Tuple[Tuple[float, float], bool]]:
    """
    Given a letter, generates the path to print it.
    :param letter: String containing a single letter (either "I", "A", or "S").
    :return: A list of print coordinates. Each entry of the list is a tuple (p, e),
    where p is a 2D coordinate where the robot should move and e is a boolean indicating
    whether the robot should print on the way to the coordinate.

    Example:
        [((0.0, 0.0), False),
         ((0.0, 1.0), True),
         ((1.0, 1.0), False),
         ((1.0, 0.0), True))]
    In this example, the robot would first move to coordinate (0.0, 0.0) without printing.
    Then, from (0.0, 0.0) it would move to (0.0, 1.0) while drawing a line behind itself.
    Afterwards, it would move from (0.0, 1.0) to (1.0, 1.0), not drawing, and finally from
    (1.0, 1.0) to (1.0, 0.0), again drawing a line. If the robot moves straight between
    the points, the result would be two parallel vertical lines.
    """
    if letter == "I":
        return [
            ((-0.2, -1.0), False),
            ((-0.2, 1.0), True),
            ((0.2, 1.0), True),
            ((0.2, -1.0), True),
            ((-0.2, -1.0), True)]
    elif letter == "A":
        return [
            ((-0.8, -1.0), False),
            ((-0.8, 1.0), True),
            ((0.8, 1.0), True),
            ((0.8, -1.0), True),
            ((0.4, -1.0), True),
            ((0.4, -0.2), True),
            ((-0.4, -0.2), True),
            ((-0.4, -1.0), True),
            ((-0.8, -1.0), True),
            ((-0.4, 0.2), False),
            ((-0.4, 0.6), True),
            ((0.4, 0.6), True),
            ((0.4, 0.2), True),
            ((-0.4, 0.2), True)]
    elif letter == "S":
        return [
            ((-0.8, -1.0), False),
            ((-0.8, -0.6), True),
            ((0.4, -0.6), True),
            ((0.4, -0.2), True),
            ((-0.8, -0.2), True),
            ((-0.8, 1.0), True),
            ((0.8, 1.0), True),
            ((0.8, 0.6), True),
            ((-0.4, 0.6), True),
            ((-0.4, 0.2), True),
            ((0.8, 0.2), True),
            ((0.8, -1.0), True),
            ((-0.8, -1.0), True)]
