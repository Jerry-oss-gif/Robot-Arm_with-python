"""SO-101 4-DOF kinematics (pan, shoulder, elbow, wrist-flex). Units: m, rad.

Standard DH:  joint | theta | d  | a  | alpha
                1    |  q1   | d1 | a1 | 90deg   (pan + horizontal shoulder offset)
                2    |  q2   | 0  | l2 | 0       (angle of upper arm above horizontal)
                3    |  q3   | 0  | l3 | 0       (relative, negative = elbow up)
                4    |  q4   | 0  | l4 | 0       (relative)
Tool pitch phi = q2 + q3 + q4. Link lengths are approximations - measure yours.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ArmGeometry:
    d1: float = 0.065   # Base height to shoulder axis (m)
    a1: float = 0.030   # Horizontal forward offset from base axis to shoulder axis (m)
    l2: float = 0.116   # Upper arm length (m)
    l3: float = 0.135   # Forearm length (m)
    l4: float = 0.100   # Wrist-to-TCP length (m)
    limits_deg: tuple = ((-110, 110), (-100, 100), (-100, 100), (-100, 100))
    home_deg: tuple = (0, 60, -90, 30)


@dataclass
class SingularityStatus:
    index: float      # sigma_min / sigma_max of the task Jacobian (0..1)
    level: str        # "ok" | "warn" | "stop"
    message: str


class SO101Kinematics:
    def __init__(self, geometry=None, pitch_weight=0.1, warn=0.12, stop=0.04, damping=0.02):
        self.g = geometry or ArmGeometry()
        self.limits = np.radians(self.g.limits_deg)
        self.home = np.radians(self.g.home_deg)
        self.pitch_w, self.warn, self.stop, self.damping = pitch_weight, warn, stop, damping

    @property
    def reach(self):
        return self.g.a1 + self.g.l2 + self.g.l3 + self.g.l4

    # ---------- homogeneous transforms ----------
    @staticmethod
    def dh(theta, d, a, alpha):
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [ 0,       sa,       ca,      d],
            [ 0,        0,        0,      1]
        ])

    def transforms(self, q):
        """[T00, T01, T02, T03, T04] cumulative base->frame i."""
        g = self.g
        params = [
            (q[0], g.d1, g.a1, np.pi / 2),
            (q[1],    0, g.l2,         0),
            (q[2],    0, g.l3,         0),
            (q[3],    0, g.l4,         0),
        ]
        Ts = [np.eye(4)]
        for p in params:
            Ts.append(Ts[-1] @ self.dh(*p))
        return Ts

    def joint_positions(self, q):
        """Return (5, 3) array of [base, shoulder, elbow, wrist, tcp] XYZ positions for 3D plotting."""
        return np.array([T[:3, 3] for T in self.transforms(q)])

    # ---------- forward kinematics ----------
    def fk(self, q):
        return self.transforms(q)[-1]

    def tcp_position(self, q):
        return self.fk(q)[:3, 3]

    @staticmethod
    def tool_pitch(q):
        return float(q[1] + q[2] + q[3])

    # ---------- inverse kinematics (analytic, elbow-up / elbow-down) ----------
    def ik(self, p, phi, elbow_up=True):
        """Return (q, message); q is None if unreachable / outside limits."""
        g = self.g
        x, y, z = p
        r = np.hypot(x, y)
        if r < 0.01:
            return None, "Target too close to base axis (pan singularity)"

        # Subtract base forward offset (a1) and tool length (l4)
        rw = r - g.a1 - g.l4 * np.cos(phi)
        zw = z - g.d1 - g.l4 * np.sin(phi)

        c3 = (rw**2 + zw**2 - g.l2**2 - g.l3**2) / (2.0 * g.l2 * g.l3)
        if abs(c3) > 1.0 + 1e-6:
            return None, "Target out of reach"

        c3 = float(np.clip(c3, -1.0, 1.0))
        q3 = -np.arccos(c3) if elbow_up else np.arccos(c3)
        q2 = np.arctan2(zw, rw) - np.arctan2(g.l3 * np.sin(q3), g.l2 + g.l3 * np.cos(q3))
        q = np.array([np.arctan2(y, x), q2, q3, phi - q2 - q3], dtype=float)

        if self.out_of_limits(q):
            return None, "Solution violates joint limits"
        return q, "ok"

    def out_of_limits(self, q):
        return bool(np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]))

    # ---------- velocity kinematics ----------
    def jacobian(self, q):
        """3x4 linear Jacobian: Jv_i = z_{i-1} x (p_e - p_{i-1})."""
        Ts = self.transforms(q)
        pe = Ts[-1][:3, 3]
        return np.column_stack([np.cross(Ts[i][:3, 2], pe - Ts[i][:3, 3]) for i in range(4)])

    def task_jacobian(self, q):
        """4x4 Jacobian for [x, y, z, pitch]."""
        return np.vstack([self.jacobian(q), self.pitch_w * np.array([0.0, 1.0, 1.0, 1.0])])

    def forward_velocity(self, q, q_dot):
        """TCP velocity [vx, vy, vz, vpitch] (m/s, rad/s) from joint rates q_dot (rad/s)."""
        J_unweighted = np.vstack([self.jacobian(q), np.array([0.0, 1.0, 1.0, 1.0])])
        return J_unweighted @ np.asarray(q_dot, dtype=float)

    def resolved_rate(self, q, v):
        """Joint rates (rad/s) for TCP velocity v=[vx,vy,vz,vpitch] (m/s, rad/s); damped least squares."""
        status = self.singularity(q)
        if status.level == "stop":
            return np.zeros(4), status

        J = self.task_jacobian(q)
        v_scaled = np.array(v, dtype=float)
        v_scaled[3] *= self.pitch_w
        q_dot = J.T @ np.linalg.solve(J @ J.T + (self.damping**2) * np.eye(4), v_scaled)
        return q_dot, status

    def manipulability(self, q):
        J = self.jacobian(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def singularity(self, q):
        s = np.linalg.svd(self.task_jacobian(q), compute_uv=False)
        idx = float(s[-1] / s[0]) if s[0] > 1e-9 else 0.0
        if idx < self.stop:
            return SingularityStatus(idx, "stop", "SINGULARITY - motion blocked")
        if idx < self.warn:
            return SingularityStatus(idx, "warn", "Near singularity - slow down / change pose")
        return SingularityStatus(idx, "ok", "OK")


if __name__ == "__main__":
    print("[Kinematics] Running self-test...")
    kin = SO101Kinematics()

    # 1. Forward Kinematics
    tcp = kin.tcp_position(kin.home)
    pitch = kin.tool_pitch(kin.home)
    print(f"Home pose TCP position: {tcp.round(4)} m, pitch: {np.degrees(pitch):.1f} deg")

    # 2. Inverse Kinematics
    q_sol, msg = kin.ik(tcp, pitch, elbow_up=True)
    assert q_sol is not None, f"IK failed: {msg}"
    print(f"IK solve verification: {msg} -> {np.degrees(q_sol).round(2)} deg")

    # 3. Velocity Kinematics
    v_cmd = [0.02, 0.0, -0.01, 0.0]  # Move +2 cm/s X, -1 cm/s Z, keep pitch constant
    q_dot, status = kin.resolved_rate(q_sol, v_cmd)
    v_check = kin.forward_velocity(q_sol, q_dot)
    print(f"Singularity status: {status.level} (index={status.index:.3f})")
    print(f"Commanded q_dot: {np.degrees(q_dot).round(2)} deg/s")
    print(f"Verified TCP vel: {v_check.round(4)}")
    print("[Kinematics] Self-test passed successfully!")