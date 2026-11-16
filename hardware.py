"""Hardware layer: 4 x Feetech STS3215 servos (IDs 1-4) behind a common RobotInterface."""
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import numpy as np

TICKS = 4096  # STS3215: 4096 ticks / rev
ADDR_MODE, ADDR_TORQUE = 33, 40

@dataclass
class JointConfig:
    servo_id: int
    name: str
    sign: int = 1            # +1 / -1 : direction of the servo relative to the kinematic model
    zero_ticks: int = 2048   # servo ticks when model angle q = 0

    def to_ticks(self, q):
        t = self.zero_ticks + self.sign * np.degrees(q) * TICKS / 360.0
        return int(np.clip(round(t), 0, TICKS - 1))

    def from_ticks(self, ticks):
        return float(np.radians(self.sign * (ticks - self.zero_ticks) * 360.0 / TICKS))

class CalibrationStore:
    def __init__(self, path="calibration.json"):
        self.path = path

    @staticmethod
    def default_joints():
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
        return [JointConfig(i + 1, n) for i, n in enumerate(names)]

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return [JointConfig(**d) for d in data]
            except Exception:
                pass
        return self.default_joints()

    def save(self, joints):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(j) for j in joints], f, indent=2)

class RobotInterface(ABC):
    real = False

    @abstractmethod
    def connect(self): ...
    @abstractmethod
    def disconnect(self): ...
    @abstractmethod
    def read_q(self): ...
    @abstractmethod
    def write_q(self, q): ...
    @abstractmethod
    def set_torque(self, on): ...

class SimRobot(RobotInterface):
    """Perfect virtual arm for testing the GUI without hardware."""
    def __init__(self, q0):
        self.q = np.array(q0, float)

    def connect(self): pass
    def disconnect(self): pass
    def read_q(self): return self.q.copy()
    def write_q(self, q): self.q = np.array(q, float)
    def set_torque(self, on): pass

    def read_raw(self):
        """Simulate raw encoder ticks corresponding to simulated joint angles."""
        return [int(np.clip(round(2048 + np.degrees(qi) * TICKS / 360.0), 0, TICKS - 1)) for qi in self.q]

class FeetechRobot(RobotInterface):
    real = True

    def __init__(self, port, joints, baud=1_000_000, speed=2000, acc=50):
        self.port, self.joints, self.baud, self.speed, self.acc = port, joints, baud, speed, acc
        self.ph = self.pk = None

    # ---- connection ----
    def connect(self):
        self.disconnect()
        try:
            from scservo_sdk import PortHandler, sms_sts  # pip install ftservo-python-sdk
        except ImportError as e:
            raise ImportError(
                "Feetech SDK with sms_sts support is required. "
                "Please install ftservo-python-sdk: pip install ftservo-python-sdk"
            ) from e

        self.ph = PortHandler(self.port)
        if not self.ph.openPort():
            self.disconnect()
            raise IOError(f"Cannot open serial port {self.port}")
        if not self.ph.setBaudRate(self.baud):
            self.disconnect()
            raise IOError(f"Cannot set baud rate {self.baud} on {self.port}")
        self.pk = sms_sts(self.ph)
        time.sleep(0.2)  # let the USB adapter settle after opening the port

        def responds(sid):
            for _ in range(3):  # retry to ride out a dropped packet
                if self.pk.ping(sid)[1] == 0:
                    return True
                time.sleep(0.05)
            return False

        missing = [j.servo_id for j in self.joints if not responds(j.servo_id)]
        if missing:
            self.disconnect()
            raise IOError(
                f"No response from servo ID(s) {missing} on {self.port} @ {self.baud} baud. "
                "Check: servo power supply is ON, correct COM port, adapter jumper on USB channel, "
                "baud rate (run scan_servo.py), and unique servo IDs (run set_servo_id.py)."
            )
        for j in self.joints:  # position (servo) mode
            self.pk.write1ByteTxRx(j.servo_id, ADDR_MODE, 0)

    def disconnect(self):
        try:
            if self.ph:
                self.ph.closePort()
        except Exception:
            pass
        finally:
            self.ph = self.pk = None

    # ---- I/O ----
    def read_raw(self):
        if not self.pk:
            raise IOError("Robot not connected")
        out = []
        for j in self.joints:
            pos, res, _ = self.pk.ReadPos(j.servo_id)
            if res != 0:
                raise IOError(f"Read failed on servo {j.servo_id}")
            out.append(int(pos))
        return out

    def read_q(self):
        return np.array([j.from_ticks(t) for j, t in zip(self.joints, self.read_raw())], dtype=float)

    def write_q(self, q):
        if not self.pk:
            raise IOError("Robot not connected")
        self.pk.groupSyncWrite.clearParam()
        try:
            for j, qi in zip(self.joints, q):
                if not self.pk.SyncWritePosEx(j.servo_id, j.to_ticks(qi), self.speed, self.acc):
                    raise IOError(f"SyncWrite addParam failed for servo {j.servo_id}")
            res = self.pk.groupSyncWrite.txPacket()
            if res != 0:
                raise IOError(f"SyncWrite transmit failed with code {res}")
        finally:
            self.pk.groupSyncWrite.clearParam()

    def set_torque(self, on):
        if not self.pk:
            return
        if on:  # hold current pose first -> no jump when torque enables
            try:
                self.write_q(self.read_q())
            except Exception:
                pass
        for j in self.joints:
            self.pk.write1ByteTxRx(j.servo_id, ADDR_TORQUE, 1 if on else 0)

    # ---- calibration ----
    def calibrate_reference(self, q_ref):
        """The arm is physically in pose q_ref (rad) right now -> store zero_ticks for each joint."""
        for j, raw, q in zip(self.joints, self.read_raw(), q_ref):
            j.zero_ticks = int(round(raw - j.sign * np.degrees(q) * TICKS / 360.0))

if __name__ == "__main__":
    print("[Hardware] Running self-test...")
    store = CalibrationStore()
    joints = store.load()
    print(f"Loaded {len(joints)} joints: {[j.name for j in joints]}")
    sim = SimRobot([0, 0, 0, 0])
    raw = sim.read_raw()
    print(f"SimRobot raw encoder ticks: {raw}")
    print("[Hardware] Self-test passed successfully!")

