from typing import Tuple, Iterable

import matplotlib.axes
import matplotlib.figure
import numpy as np


class SingleRobotArm:
    def __init__(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, l: Iterable[float],
                 base_color="#185f90", arm_color="#1f77b4", line_color="#ff7f0e", base_z_order: int = 10):
        self._l = np.array(l)
        assert self._l.shape == (2,)
        self._ax = ax
        self._fig = fig
        self._q = np.zeros(2)
        self._line_segments = [[]]
        self._last_step = 0.0
        self._base_z_order = base_z_order
        self._plt_arm_base = \
            self._ax.plot([0.0], [0.0], marker="o", color=base_color, zorder=base_z_order + 1, markersize=10)[0]
        self._plt_arm = self._ax.plot([], [], linewidth=5.0, marker="o", color=arm_color, zorder=base_z_order)[0]
        self._plt_line_segments = []
        self._line_color = line_color
        self._drawing = False

    def set_joint_pos(self, q: Iterable[float]):
        q = np.array(q)
        assert q.shape == (2,)
        self._q = q
        if self._drawing:
            _, p2 = self._forward_kinematics(self._q)
            self._line_segments[-1].append(p2)
        self.render()

    def _forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1 = self._l[0] * np.cos(q[0])
        y1 = self._l[0] * np.sin(q[0])
        x2 = x1 + self._l[1] * np.cos(q[0] + q[1])
        y2 = y1 + self._l[1] * np.sin(q[0] + q[1])
        return np.array([x1, y1]), np.array([x2, y2])

    def render(self):
        # Draw the arm
        p1, p2 = self._forward_kinematics(self._q)
        self._plt_arm.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])

        # Ensure that the correct number of line segments is present in the plot
        while len(self._line_segments) > len(self._plt_line_segments):
            self._plt_line_segments.append(
                self._ax.plot([], [], color=self._line_color, zorder=self._base_z_order + 10)[0])
        for segment in self._plt_line_segments[len(self._line_segments):]:
            segment.remove()
        self._plt_line_segments = self._plt_line_segments[:len(self._line_segments)]
        # Draw the trajectories
        for segment, line in zip(self._line_segments, self._plt_line_segments):
            if len(segment) > 0:
                line.set_data(*zip(*segment))
            else:
                line.set_data([], [])

    def reset(self):
        self._q = np.zeros(2)
        self._line_segments = [[]]
        self.render()

    @property
    def l(self):
        return self._l.copy()

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    @property
    def axis(self) -> matplotlib.axes.Axes:
        return self._ax

    @property
    def fig(self) -> matplotlib.figure.Figure:
        return self._fig

    def start_drawing(self):
        self._drawing = True

    def stop_drawing(self):
        self._drawing = False
        if len(self._line_segments[-1]) > 0:
            self._line_segments.append([])
