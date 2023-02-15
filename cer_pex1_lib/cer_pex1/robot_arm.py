import time
from typing import Tuple, Callable, Iterable, Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def isnotebook():
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


class RobotArm:
    # simple two joint robot arm in 2D

    def __init__(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, l: Iterable[float],
                 kinematics_model: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                 joint_vel: float = 1.0, dt: float = 0.005, render_interval: int = 10) -> None:
        self._l = np.array(l)
        assert self._l.shape == (2,)
        self._kinematics_model = kinematics_model
        self._ax = ax
        self._fig = fig
        self.joint_vel = joint_vel
        self._dt = dt
        self._q = np.zeros(2)
        self._line_segments = [[]]
        self._render_interval = render_interval
        self._steps_since_last_render = None
        self._last_step = 0.0
        self._plt_target_marker = None
        self._plt_arm_base = self._ax.plot([0.0], [0.0], marker="o", color="#185f90", zorder=6, markersize=10)[0]
        self._plt_arm = self._ax.plot([], [], linewidth=5.0, marker="o", color="#1f77b4", zorder=5)[0]
        self._plt_shadow_arm = self._ax.plot([], [], linewidth=5.0, marker="o", color=(0, 0, 0, 0.5), zorder=4)[0]
        self._plt_target_marker = self._ax.plot([], [], marker="o", color="#d62728", zorder=3)[0]
        self._plt_line_segments = []
        self.target_position_marker = None
        self.render()

    def set_joint_pos(self, q: Iterable[float]):
        q = np.array(q)
        assert q.shape == (2,)
        self._q = q
        self._finish_line_segment()
        self.render()

    def move_to_joint_pos(self, q: Iterable[float], draw_trace: bool = False):
        q = np.array(q)
        assert q.shape == (2,)
        last_render_time = time.time()
        if draw_trace:
            if len(self._line_segments[-1]) == 0:
                self._add_point_to_line_segment()
        else:
            self._finish_line_segment()
        max_dist = np.max(np.abs(q - self._q))
        step_count = int(np.ceil(max_dist / (self._dt * self.joint_vel)))
        steps = np.linspace(self._q, q, step_count + 1)[1:]
        for i, qs in enumerate(steps):
            self._q = qs
            if draw_trace:
                self._add_point_to_line_segment()
            self.render(force=i == len(steps) - 1)
            next_render_time = last_render_time + self._dt
            time.sleep(max(0.0, next_render_time - time.time()))
            last_render_time = next_render_time

    def step_vel(self, q_vel: Iterable[float], draw_trace: bool = False):
        step_time = time.time() + self._dt
        q_vel = np.array(q_vel)
        assert q_vel.shape == (2,)
        if draw_trace:
            if len(self._line_segments[-1]) == 0:
                self._add_point_to_line_segment()
        else:
            self._finish_line_segment()
        self._q += self._dt * q_vel
        if draw_trace:
            self._add_point_to_line_segment()
        self.render(force=False)
        time.sleep(max(0.0, step_time - time.time()))

    def render(self, force: bool = True):
        if force or self._steps_since_last_render >= self._render_interval:
            self._steps_since_last_render = 0

            # Draw the target position marker
            if self.target_position_marker is not None:
                self._plt_target_marker.set_data([self.target_position_marker[0]], [self.target_position_marker[1]])
            else:
                self._plt_target_marker.set_data([], [])

            # Draw the arm
            p1, p2 = self._kinematics_model(self._q, self._l)
            self._plt_arm.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])

            # Ensure that the correct number of line segments is present in the plot
            while len(self._line_segments) > len(self._plt_line_segments):
                self._plt_line_segments.append(self._ax.plot([], [], color="#ff7f0e", zorder=10)[0])
            for segment in self._plt_line_segments[len(self._line_segments):]:
                segment.remove()
            self._plt_line_segments = self._plt_line_segments[:len(self._line_segments)]
            # Draw the trajectories
            for segment, line in zip(self._line_segments, self._plt_line_segments):
                if len(segment) > 0:
                    line.set_data(*zip(*segment))
                else:
                    line.set_data([], [])

            if isnotebook():
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
                time.sleep(0.001)
            else:
                plt.pause(0.001)
                self._fig.canvas.flush_events()
        else:
            self._steps_since_last_render += 1

    def set_shadow(self, q: Sequence[float]):
        p1, p2 = self._kinematics_model(np.array(q), self._l)
        self._plt_shadow_arm.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])
        self.render()

    def remove_shadow(self):
        self._plt_shadow_arm.set_data([], [])
        self.render()

    def reset(self):
        self._q = np.zeros(2)
        self._line_segments = [[]]
        self.target_position_marker = None
        self.render()

    @property
    def l(self):
        return self._l.copy()

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    @property
    def dt(self):
        return self._dt

    @property
    def axis(self) -> matplotlib.axes.Axes:
        return self._ax

    def _finish_line_segment(self):
        if len(self._line_segments[-1]) > 0:
            self._line_segments.append([])

    def _add_point_to_line_segment(self):
        _, p2 = self._kinematics_model(self._q, self._l)
        self._line_segments[-1].append(p2)
