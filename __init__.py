"""SO-101 Robot Arm Package."""
try:
    from .kinematics import SO101Kinematics, ArmGeometry, SingularityStatus
    from .hardware import JointConfig, CalibrationStore, RobotInterface, SimRobot, FeetechRobot
    from .controller import MotionController
    from .gui import RobotStudioApp
except (ImportError, ValueError):
    from kinematics import SO101Kinematics, ArmGeometry, SingularityStatus
    from hardware import JointConfig, CalibrationStore, RobotInterface, SimRobot, FeetechRobot
    from controller import MotionController
    from gui import RobotStudioApp

__all__ = [
    "SO101Kinematics",
    "ArmGeometry",
    "SingularityStatus",
    "JointConfig",
    "CalibrationStore",
    "RobotInterface",
    "SimRobot",
    "FeetechRobot",
    "MotionController",
    "RobotStudioApp",
]
