from collections import deque
from typing import Tuple, Iterable, Optional
import time

from matplotlib.axes import Axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def isnotebook():
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


class RobotArm:
    def __init__(self, l: Iterable[float], m: Iterable[float], fig: Optional[matplotlib.figure.Figure] = None,
                 ax_arm: Optional[Axes] = None, ax_pos1: Optional[Axes] = None, ax_pos2: Optional[Axes] = None,
                 ax_vel1: Optional[Axes] = None, ax_vel2: Optional[Axes] = None, gravity: float = 9.81,
                 dt: float = 0.001, torque_limit: float = 50.0,
                 simulator_substeps: int = 10, base_z_order: int = 10):
        self._l = np.array(l)
        assert self._l.shape == (2,)
        self._m = np.array(m)
        assert self._m.shape == (2,)
        self._base_z_order = base_z_order
        self._fig = fig

        self._ax_arm = ax_arm
        self._ax_pos = [ax_pos1, ax_pos2]
        self._ax_vel = [ax_vel1, ax_vel2]

        if self._ax_arm is not None:
            self._plt_arm_base = \
                self._ax_arm.plot([0.0], [0.0], marker="o", color="#185f90", zorder=base_z_order + 3, markersize=10)[0]
            self._plt_arm = self._ax_arm.plot(
                [], [], linewidth=5.0, marker="o", color="#1f77b4", zorder=base_z_order + 2)[0]
            self._plt_target_dot = self._ax_arm.plot([], [], marker="o", color="#d62728", zorder=base_z_order + 1)[0]
            self._plt_target_traj = self._ax_arm.plot([], [], color="#d62728", zorder=base_z_order)[0]
        else:
            self._plt_arm = self._plt_arm_base = self._plt_target_dot = None

        self._plt_line_segments = []
        self._plt_pos = [
            ax.plot([], [], zorder=base_z_order + 4)[0] if ax is not None else None for ax in self._ax_pos]
        self._plt_vel = [
            ax.plot([], [], zorder=base_z_order + 4)[0] if ax is not None else None for ax in self._ax_vel]

        self._dt = dt
        self._simulator_substeps = simulator_substeps
        self._gravity = gravity
        self._torque_limit = torque_limit

        self._line_segments = [[]]
        self._trajectory_q = deque()
        self._trajectory_dq = deque()
        self._last_step = 0.0
        self._drawing = False
        self._q = np.zeros(2)
        self._dq = np.zeros(2)
        self._q_tar = None
        self._steps_since_reset = 0
        self.reset(np.zeros(2), np.zeros(2))

    def set_target_traj(
            self, q: Iterable[float], dq: Iterable[float], q_error_bounds: Optional[Iterable[float]] = None,
            dq_error_bounds: Optional[Iterable[float]] = None):
        q = np.asarray(q)
        dq = np.asarray(dq)

        assert len(q.shape) == 2 and q.shape[1] == 2
        assert q.shape == dq.shape

        if q_error_bounds is None:
            q_error_bounds = np.zeros((q.shape[0], 2, 2))
        else:
            q_error_bounds = np.asarray(q_error_bounds)
        if dq_error_bounds is None:
            dq_error_bounds = np.zeros((dq.shape[0], 2, 2))
        else:
            dq_error_bounds = np.asarray(dq_error_bounds)

        assert q_error_bounds.shape == (q.shape[0], 2, 2)
        assert dq_error_bounds.shape == q_error_bounds.shape

        ts = np.arange(q.shape[0]) * self._dt
        for i in range(2):
            if self._ax_pos[i] is not None:
                self._ax_pos[i].plot(ts, q[:, i], zorder=self._base_z_order + 2, color="#d62728")
                self._ax_pos[i].fill_between(
                    ts, q[:, i] + q_error_bounds[:, i, 0], q[:, i] + q_error_bounds[:, i, 1], color="#d6272840")
                self._ax_pos[i].set_xlim(0, q.shape[0] * self.dt)
                q_min = np.min(q[:, i] + q_error_bounds[:, i, 0])
                q_max = np.max(q[:, i] + q_error_bounds[:, i, 1])
                diff = q_max - q_min
                self._ax_pos[i].set_ylim(q_min - diff * 0.2 - 0.05, q_max + diff * 0.2 + 0.05)
                self._plt_target_traj.set_data(*self._forward_kinematics(q)[1].T)
            if self._ax_vel[i] is not None:
                self._ax_vel[i].plot(ts, dq[:, i], zorder=self._base_z_order + 2, color="#d62728")
                self._ax_vel[i].fill_between(
                    ts, dq[:, i] + dq_error_bounds[:, i, 0], dq[:, i] + dq_error_bounds[:, i, 1], color="#d6272840")
                self._ax_vel[i].set_xlim(0, q.shape[0] * self.dt)
                dq_min = np.min(dq[:, i] + dq_error_bounds[:, i, 0])
                dq_max = np.max(dq[:, i] + dq_error_bounds[:, i, 1])
                diff = dq_max - dq_min
                self._ax_vel[i].set_ylim(dq_min - diff * 0.2 - 0.05, dq_max + diff * 0.2 + 0.05)

    def step_simulator(self, torques: Iterable[float] = (0.0, 0.0)):
        torques = np.array(torques)
        assert torques.shape == (2,)
        torques_clipped = np.clip(torques, -self._torque_limit, self._torque_limit)
        for s in range(self._simulator_substeps):
            self._single_simulator_step(
                torques_clipped, self._dt / self._simulator_substeps, record_traj=s == self._simulator_substeps - 1)
        self._steps_since_reset += 1

    def reset(self, q: Iterable[float], dq: Iterable[float] = (0.0, 0.0)):
        self._trajectory_q.clear()
        self._trajectory_dq.clear()
        self._line_segments = [[]]
        q = np.array(q)
        dq = np.array(dq)
        self._set_state(q, dq)
        self._drawing = False
        self._steps_since_reset = 0
        self.render()

    def inverse_kinematics(self, target_ee_pos: Iterable[float], q0: Optional[Iterable[float]] = None) -> np.ndarray:
        """
        Computes the inverse kinematics of the robot arm using newton's method.
        :param p2: A 2D numpy array containing the target position of the end-effector in meters.
        :param q0: A 2D numpy array containing the initial solution for the Newton method in radians.
        :return: A 2D numpy array containing the resulting joint angles in radians.
        """
        if q0 is None:
            q0 = self._q
        else:
            q0 = np.asarray(q0)
            assert q0.shape == (2,)
        target_ee_pos = np.array(target_ee_pos)
        assert target_ee_pos.shape == (2,)

        q = q0
        alpha = 0.1
        for itr in range(10000):
            p = self._forward_kinematics(q)[1][:2]
            if np.linalg.norm(target_ee_pos - p) < 1e-5:
                return q
            jac = -self.compute_jacobian(q)
            update = np.linalg.solve(jac, target_ee_pos - p)
            q -= alpha * update
        print("Warning: Newton method did not converge within 10000 iterations.")
        return q

    def compute_jacobian(self, q: Optional[Iterable[float]] = None) -> np.ndarray:
        """
        Computes the Jacobian (first derivative) of the forward kinematics of the robot arm.
        :param q: 2D vector containing the joint angles in radians.
        :return: A 2x2 numpy array containing the Jacobian.
        """
        if q is None:
            q = self._q
        else:
            q = np.asarray(q)
            assert q.shape == (2,)
        l = self._l
        return np.array(
            [[-l[0] * np.sin(q[0]) - l[1] * np.sin(q[0] + q[1]), -l[1] * np.sin(q[0] + q[1])],
             [l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1]), l[1] * np.cos(q[0] + q[1])]])

    def _set_state(self, q: np.ndarray, dq: np.ndarray, record_traj: bool = True):
        assert q.shape == (2,)
        assert dq.shape == (2,)
        self._q = q
        self._dq = dq
        if record_traj:
            if self._drawing:
                _, p2 = self._forward_kinematics(self._q)
                self._line_segments[-1].append(p2)
            self._trajectory_q.append(q)
            self._trajectory_dq.append(dq)

    def _forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x1 = self._l[0] * np.cos(q[..., 0])
        y1 = self._l[0] * np.sin(q[..., 0])
        x2 = x1 + self._l[1] * np.cos(q[..., 0] + q[..., 1])
        y2 = y1 + self._l[1] * np.sin(q[..., 0] + q[..., 1])
        return np.stack([x1, y1], axis=-1), np.stack([x2, y2], axis=-1)

    def _forward_dynamics(self, x: np.ndarray, torques: np.ndarray) -> np.ndarray:
        m1, m2 = self._m
        l1, l2 = self._l
        f1, f2 = torques
        q1, q2, dq1, dq2 = x
        denom = l1 * (m1 + m2 * np.sin(q2) ** 2)
        num = 0.5 * dq1 ** 2 * l1 * m2 * np.sin(2 * q2) + m2 * (dq1 + dq2) ** 2 * l2 * np.sin(q2) \
              - self._gravity * ((m1 + m2) * np.cos(q1) - m2 * np.cos(q1 + q2) * np.cos(q2)) \
              - f2 * (1 + l1 / l2 * np.cos(q2)) + f1 / l1
        ddq1 = num / denom

        # Incorporate angular acceleration caused by motors
        ddq2 = - ddq1 * (1 + l1 / l2 * np.cos(q2)) - dq1 ** 2 * l1 / l2 * np.sin(q2) \
               - self._gravity / l2 * np.cos(q1 + q2) + f2 / (m2 * l2 ** 2)
        return np.array([dq1, dq2, ddq1, ddq2])

    def _single_simulator_step(self, torques: np.ndarray, dt: float, record_traj: bool = True):
        # Use RK4 to compute a single simulator step
        x = np.concatenate([self._q, self._dq])
        s1 = self._forward_dynamics(x, torques)
        s2 = self._forward_dynamics(x + 0.5 * dt * s1, torques)
        s3 = self._forward_dynamics(x + 0.5 * dt * s2, torques)
        s4 = self._forward_dynamics(x + dt * s3, torques)
        x_new = x + dt * (1 / 6. * s1 + 1 / 3. * s2 + 1 / 3. * s3 + 1 / 6. * s4)
        self._set_state(x_new[:2], x_new[2:], record_traj=record_traj)

    def render(self):
        if self._plt_arm is not None:
            # Draw the arm
            p1, p2 = self._forward_kinematics(self._q)
            self._plt_arm.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])

            # Ensure that the correct number of line segments is present in the plot
            while len(self._line_segments) > len(self._plt_line_segments):
                self._plt_line_segments.append(
                    self._ax_arm.plot([], [], color="#ff7f0e", zorder=self._base_z_order + 10)[0])
            for segment in self._plt_line_segments[len(self._line_segments):]:
                segment.remove()
            self._plt_line_segments = self._plt_line_segments[:len(self._line_segments)]
            # Draw the trajectories
            for segment, line in zip(self._line_segments, self._plt_line_segments):
                if len(segment) > 0:
                    line.set_data(*zip(*segment))
                else:
                    line.set_data([], [])

        traj_q = np.array(self._trajectory_q)
        traj_dq = np.array(self._trajectory_dq)
        ts = np.arange(len(traj_q)) * self._dt

        if self._q_tar is not None:
            self._plt_target_dot.set_data(*self._forward_kinematics(
                self._q_tar[min(self._q_tar.shape[0] - 1, self._steps_since_reset)])[1])

        for i in range(2):
            if self._plt_pos[i] is not None:
                self._plt_pos[i].set_data(ts, traj_q[:, i])
            if self._plt_vel[i] is not None:
                self._plt_vel[i].set_data(ts, traj_dq[:, i])

        if self._fig is not None:
            if isnotebook():
                self._fig.canvas.draw()
                self._fig.canvas.flush_events()
                time.sleep(0.001)
            else:
                plt.pause(0.001)
                self._fig.canvas.flush_events()

    @property
    def l(self):
        return self._l.copy()

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    @property
    def dq(self) -> np.ndarray:
        return self._dq.copy()

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def torque_limit(self) -> float:
        return self._torque_limit

    @property
    def ax_arm(self) -> matplotlib.axes.Axes:
        return self._ax_arm

    @property
    def ax_pos(self) -> Tuple[Optional[matplotlib.axes.Axes]]:
        return tuple(self._ax_pos)

    @property
    def ax_vel(self) -> Tuple[Optional[matplotlib.axes.Axes]]:
        return tuple(self._ax_vel)

    @property
    def fig(self) -> matplotlib.figure.Figure:
        return self._fig

    def start_drawing(self):
        if not self._drawing:
            self._drawing = True
            self._line_segments.append([self._forward_kinematics(self._q)[-1]])

    def stop_drawing(self):
        self._drawing = False

    @property
    def is_drawing(self) -> bool:
        return self._drawing
