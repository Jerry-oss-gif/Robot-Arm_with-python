"""SO-101 Robot Studio entry point.  Run:  python main.py"""
import os
import sys

# Make imports and relative files (logo, calibration.json) work from any working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gui import RobotStudioApp

if __name__ == "__main__":
    RobotStudioApp().mainloop()
