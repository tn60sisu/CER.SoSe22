import time
from typing import Iterable

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from .single_robot_arm import SingleRobotArm


def isnotebook():
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


class RobotArm:
    # Two joint robot arm in 2D, which supports drawing a shadow arm

    def __init__(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, l: Iterable[float]):
        self._l = np.array(l)
        assert self._l.shape == (2,)

        self._arm = SingleRobotArm(fig, ax, l, base_z_order=7)
        self._shadow_arm = SingleRobotArm(
            fig, ax, l, base_color="#404040A0", arm_color="#606060A0", line_color="#404040A0", base_z_order=5)

        self.target_position_marker = None
        self.render()

    def set_joint_pos(self, q: Iterable[float]):
        self._arm.set_joint_pos(q)

    def set_shadow_joint_pos(self, q: Iterable[float]):
        self._shadow_arm.set_joint_pos(q)

    def render(self):
        self._arm.render()
        if isnotebook():
            self._arm.fig.canvas.draw()
            self._arm.fig.canvas.flush_events()
            time.sleep(0.001)
        else:
            plt.pause(0.001)
            self._arm.fig.canvas.flush_events()

    def reset(self):
        self._arm.reset()
        self._shadow_arm.reset()
        self.render()

    def start_drawing(self):
        self._arm.start_drawing()
        self._shadow_arm.start_drawing()

    def stop_drawing(self):
        self._arm.stop_drawing()
        self._shadow_arm.stop_drawing()

    @property
    def l(self):
        return self._l.copy()

    @property
    def q(self) -> np.ndarray:
        return self._arm.q

    @property
    def axis(self) -> matplotlib.axes.Axes:
        return self._arm.axis

    @property
    def fig(self):
        return self._arm.fig
