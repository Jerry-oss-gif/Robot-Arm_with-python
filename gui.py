"""Dark Tkinter GUI built from small classes: one per panel + header + calibration dialog + app."""
import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

try:
    from .kinematics import SO101Kinematics
    from .hardware import SimRobot, FeetechRobot, CalibrationStore
    from .controller import MotionController
except (ImportError, ValueError):
    from kinematics import SO101Kinematics
    from hardware import SimRobot, FeetechRobot, CalibrationStore
    from controller import MotionController

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # logo + calibration live next to the code
BAUDS = ["1000000", "500000", "250000", "115200", "57600"]

BG, PANEL, CYAN, GREEN, ORANGE, RED = "#1e1e1e", "#252526", "#00bcd4", "#4caf50", "#ffa726", "#c62828"
LEVEL_COLOR = {"ok": "#2e7d32", "warn": "#ef6c00", "stop": RED}
JOINT_NAMES = ["J1: Base (Yaw)", "J2: Shoulder (Pitch)", "J3: Elbow (Pitch)", "J4: Wrist (Pitch)"]

class Panel(tk.LabelFrame):
    """Base class for every panel: styled frame + refresh(ctrl) hook."""
    def __init__(self, parent, title):
        super().__init__(parent, text=title, bg=PANEL, fg=CYAN, bd=1, relief="groove", font=("Segoe UI", 10, "bold"))

    def label(self, parent, text="", fg="white", font=("Segoe UI", 9), **kw):
        return tk.Label(parent, text=text, bg=PANEL, fg=fg, font=font, **kw)

    def refresh(self, ctrl): pass

# ------------------------------------------------------------------ panels
class JointPanel(Panel):
    def __init__(self, parent, ctrl):
        super().__init__(parent, "Joint Control (FK)")
        self.ctrl = ctrl
        self.sliders, self.vals, self.spds = [], [], []
        self._dragging = [False] * 4
        for i, name in enumerate(JOINT_NAMES):
            row = tk.Frame(self, bg=PANEL)
            row.pack(fill="x", padx=6, pady=2)
            row.columnconfigure(0, weight=1)
            self.label(row, name).grid(row=0, column=0, sticky="w")
            sp = self.label(row, "0.0°/s", ORANGE, ("Consolas", 9))
            sp.grid(row=0, column=1, sticky="e")
            lo, hi = np.degrees(ctrl.kin.limits[i])
            s = tk.Scale(row, from_=lo, to=hi, resolution=0.5, orient="horizontal", showvalue=0, length=190, bg=PANEL,
                         troughcolor="#cfcfc8", highlightthickness=0,
                         command=lambda v, idx=i: self._on_slide(idx, float(v)))
            s.bind("<Button-1>", lambda e, idx=i: self._set_drag(idx, True))
            s.bind("<ButtonRelease-1>", lambda e, idx=i: self._set_drag(idx, False))
            s.grid(row=1, column=0, sticky="w")
            vl = self.label(row, "0.0°", font=("Consolas", 10), width=7)
            vl.grid(row=1, column=1)
            self.sliders.append(s)
            self.vals.append(vl)
            self.spds.append(sp)

    def _set_drag(self, i, state):
        self._dragging[i] = state

    def _on_slide(self, i, deg):
        self.ctrl.set_joint_target(i, np.radians(deg))

    def _set_slider_silent(self, i, deg):
        s = self.sliders[i]
        cmd = s.cget("command")
        s.configure(command="")
        s.set(deg)
        s.configure(command=cmd)

    def refresh(self, ctrl):
        for i in range(4):
            des_deg = round(float(np.degrees(ctrl.q_des[i])), 1)
            if not self._dragging[i] and self.focus_get() is not self.sliders[i]:
                if abs(self.sliders[i].get() - des_deg) >= 0.25:
                    self._set_slider_silent(i, des_deg)
            self.vals[i].config(text=f"{np.degrees(ctrl.q_fb[i]):.1f}°")  # actual angle
            self.spds[i].config(text=f"{np.degrees(ctrl.qd[i]):.1f}°/s")

class TCPPanel(Panel):
    def __init__(self, parent):
        super().__init__(parent, "Actual Tool Center Point")
        self.vars = [tk.StringVar() for _ in range(4)]
        for v in self.vars:
            tk.Label(self, textvariable=v, bg=PANEL, fg=GREEN, font=("Consolas", 13, "bold"), anchor="w").pack(fill="x", padx=10)

    def refresh(self, ctrl):
        for v, name, val in zip(self.vars, "XYZ", ctrl.tcp_mm):
            v.set(f"{name}: {val:8.2f} mm")
        self.vars[3].set(f"Pitch: {ctrl.pitch_deg:6.1f} °")

class TargetPanel(Panel):
    def __init__(self, parent, ctrl):
        super().__init__(parent, "Target Coordinates (IK)")
        self.ctrl, self.entries = ctrl, []
        for r, (name, d) in enumerate([("X (mm)", "260"), ("Y (mm)", "0"), ("Z (mm)", "150"), ("Pitch (°)", "0")]):
            self.label(self, "Target " + name).grid(row=r, column=0, sticky="w", padx=6, pady=3)
            e = tk.Entry(self, width=9)
            e.insert(0, d)
            e.grid(row=r, column=1, padx=6)
            self.entries.append(e)
        tk.Button(self, text="Move to Target (IK Solve)", bg="#0078d7", fg="white", font=("Segoe UI", 10, "bold"),
                  command=self._send).grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=8)

    def _send(self):
        try:
            x, y, z, pitch = [float(e.get()) for e in self.entries]
        except ValueError:
            return messagebox.showerror("Input", "Enter numeric X, Y, Z, Pitch")
        ok, msg = self.ctrl.move_to([x, y, z], pitch)
        if not ok:
            messagebox.showerror("Move to Target", msg)

class SpeedPanel(Panel):
    def __init__(self, parent, ctrl):
        super().__init__(parent, "TCP Speed Setting")
        self.ctrl = ctrl
        self.lbl = self.label(self, "", ORANGE, ("Consolas", 10))
        self.scale = tk.Scale(self, from_=5, to=150, resolution=5, orient="horizontal", showvalue=0, length=150, bg=PANEL,
                              troughcolor="#cfcfc8", highlightthickness=0, command=self._set)
        self.scale.pack(side="left", padx=6, pady=6)
        self.lbl.pack(side="left")
        self.scale.set(ctrl.speed_mm_s)

    def _set(self, v):
        self.ctrl.speed_mm_s = float(v)
        self.lbl.config(text=f"{float(v):.0f} mm/s")

class JogPanel(Panel):
    def __init__(self, parent, ctrl):
        super().__init__(parent, "Jog X, Y, Z (Press & Hold)")
        self.label(self, "* hold to move / release to stop *", "#888").grid(row=0, column=0, columnspan=2, pady=4)
        for ax, name in enumerate("XYZ"):
            for c, sgn in enumerate([1, -1]):
                b = tk.Button(self, text=f"{'+' if sgn > 0 else '-'}{name}", width=7, height=2, bg="#2b2b3a", fg="white",
                              font=("Segoe UI", 10, "bold"))
                b.grid(row=ax + 1, column=c, padx=6, pady=4)
                b.bind("<ButtonPress-1>", lambda e, a=ax, s=sgn: ctrl.start_jog(a, s))
                b.bind("<ButtonRelease-1>", lambda e: ctrl.stop_jog())

class SingularityPanel(Panel):
    def __init__(self, parent):
        super().__init__(parent, "Singularity Index")
        self.txt = tk.StringVar()
        tk.Label(self, textvariable=self.txt, bg=PANEL, fg=GREEN, font=("Consolas", 9), justify="left").pack(padx=6, pady=4)
        self.bar = tk.Canvas(self, width=240, height=16, bg="#cfcfc8", highlightthickness=0)
        self.bar.pack(padx=6)
        self.alert = tk.Label(self, text="OK", bg=LEVEL_COLOR["ok"], fg="white", width=30, wraplength=230, font=("Segoe UI", 9, "bold"))
        self.alert.pack(padx=6, pady=8)

    def refresh(self, ctrl):
        s = ctrl.singularity
        self.txt.set(f"Index (smin/smax): {s.index:.3f}\nManipulability w: {ctrl.manipulability_mm3:,.0f}")
        color = LEVEL_COLOR.get(s.level, "#2e7d32")
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, 240 * min(s.index / 0.5, 1.0), 16, fill=color, width=0)
        self.alert.config(text=ctrl.message or s.message, bg=RED if ctrl.message else color)

class Visualizer3D(Panel):
    def __init__(self, parent, ctrl):
        super().__init__(parent, "[3] 3D Robot Motion Visualizer")
        fig = Figure(figsize=(6, 5), facecolor=PANEL)
        ax = fig.add_subplot(111, projection="3d", facecolor=PANEL)
        ax.set_xlim(-350, 350)
        ax.set_ylim(-350, 350)
        ax.set_zlim(0, 350)
        ax.set_box_aspect((2, 2, 1))
        ax.view_init(26, 40)
        for a, n in zip("xyz", "XYZ"):
            ax.tick_params(axis=a, colors="#bbb", labelsize=7)
            getattr(ax, f"set_{a}label")(f"{n} (mm)", color="#bbb")
        t, r_reach = np.linspace(0, 2 * np.pi, 100), ctrl.kin.reach * 1000
        ax.plot(r_reach * np.cos(t), r_reach * np.sin(t), 0 * t, "--", color="#ddd", lw=0.8)
        self.links, = ax.plot([], [], [], color=CYAN, lw=5, solid_capstyle="round")
        self.tool, = ax.plot([], [], [], color=ORANGE, lw=6)
        self.joints, = ax.plot([], [], [], "o", color="white", ms=7)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.kin, self._last = ctrl.kin, None

    def refresh(self, ctrl):
        if self._last is not None and np.allclose(self._last, ctrl.q_fb):
            return
        p_transforms = np.array([t[:3, 3] for t in self.kin.transforms(ctrl.q_fb)]) * 1000
        self.links.set_data_3d(*p_transforms.T)
        self.joints.set_data_3d(*p_transforms.T)
        self.tool.set_data_3d(*p_transforms[-2:].T)
        self.canvas.draw_idle()
        self._last = ctrl.q_fb.copy()

# ------------------------------------------------------------------ header & dialog
class HeaderBar(tk.Frame):
    def __init__(self, app):
        
        super().__init__(app, bg=BG)
        self.app = app
        tk.Label(self, text="(SO-101) (4-DOF ROBOT STUDIO)", bg=BG, fg="white", font=("Segoe UI", 13, "bold")).pack(side="left")
        self.estop = tk.Button(self, text="EMERGENCY STOP", bg=RED, fg="white", font=("Segoe UI", 10, "bold"),
                               command=app.toggle_estop)
        self.estop.pack(side="right")
        tk.Button(self, text="Calibrate", command=app.open_calibration).pack(side="right", padx=4)
        tk.Button(self, text="Home", command=app.ctrl.go_home).pack(side="right", padx=4)
        self.torque = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="Torque ON", variable=self.torque, bg=BG, fg="white", selectcolor=BG,
                       activebackground=BG,
                       command=lambda: app.ctrl.set_torque(self.torque.get())).pack(side="right", padx=6)
        tk.Button(self, text="Connect", command=app.connect).pack(side="right", padx=4)
        ports = self._ports()
        self.port = tk.StringVar(value=ports[0])
        self.port_combo = ttk.Combobox(self, textvariable=self.port, values=ports, width=10,
                                       postcommand=self._update_ports)
        self.port_combo.pack(side="right")
        self.baud = tk.StringVar(value=BAUDS[0])
        ttk.Combobox(self, textvariable=self.baud, values=BAUDS, width=9).pack(side="right", padx=4)
        self.mode = tk.StringVar(value="Simulation")
        ttk.Combobox(self, textvariable=self.mode, values=["Simulation", "Feetech STS3215"], width=15,
                     state="readonly").pack(side="right", padx=4)

    def _update_ports(self):
        self.port_combo.config(values=self._ports())

    @staticmethod
    def _ports():
        try:
            from serial.tools import list_ports
            return [p.device for p in list_ports.comports()] or ["COM3"]
        except Exception:
            return ["COM3", "/dev/ttyACM0"]

    def refresh(self, ctrl):
        self.torque.set(ctrl.torque)
        self.estop.config(text="RESET E-STOP" if ctrl.estopped else "EMERGENCY STOP",
                          bg="#555" if ctrl.estopped else RED)

class CalibrationDialog(tk.Toplevel):
    """Place the arm in a known pose by hand (torque OFF), enter that pose, press Apply."""
    REF_DEG = ["0", "90", "-90", "0"]

    def __init__(self, app):
        super().__init__(app, bg=PANEL)
        self.title("Calibration")
        self.app = app
        self._timer = None
        txt = ("1. Torque OFF.  2. Put the arm in the reference pose: base straight ahead, upper arm VERTICAL,\n"
               "forearm HORIZONTAL forward, wrist/tool horizontal (or edit the angles below to match your pose).\n"
               "3. Move each joint by hand: if the 3D model moves the wrong way tick Invert.  4. Apply & Save.")
        tk.Label(self, text=txt, bg=PANEL, fg="white", justify="left").grid(row=0, column=0, columnspan=5, padx=8, pady=8)
        self.inv, self.ref, self.raw = [], [], []
        for i, j in enumerate(app.joints):
            tk.Label(self, text=f"ID {j.servo_id} {j.name}", bg=PANEL, fg=CYAN).grid(row=i + 1, column=0, padx=6, sticky="w")
            v = tk.BooleanVar(value=j.sign < 0)
            self.inv.append(v)
            tk.Checkbutton(self, text="Invert", variable=v, bg=PANEL, fg="white", selectcolor=BG, activebackground=PANEL).grid(row=i + 1, column=1)
            e = tk.Entry(self, width=7)
            e.insert(0, self.REF_DEG[i])
            e.grid(row=i + 1, column=2)
            self.ref.append(e)
            tk.Label(self, text="deg", bg=PANEL, fg="#aaa").grid(row=i + 1, column=3)
            r = tk.Label(self, text="raw: --", bg=PANEL, fg=ORANGE, font=("Consolas", 10), width=12)
            r.grid(row=i + 1, column=4)
            self.raw.append(r)
        tk.Button(self, text="Apply & Save", bg="#0078d7", fg="white", command=self.apply).grid(row=6, column=0, columnspan=5, pady=10)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()

    def _poll(self):
        if not self.winfo_exists():
            return
        try:
            if hasattr(self.app.ctrl.robot, "read_raw"):
                for r, t in zip(self.raw, self.app.ctrl.robot.read_raw()):
                    r.config(text=f"raw: {t}")
        except Exception:
            pass
        self._timer = self.after(300, self._poll)

    def on_close(self):
        if self._timer:
            try:
                self.after_cancel(self._timer)
            except Exception:
                pass
        self.destroy()

    def apply(self):
        c = self.app.ctrl
        if not c.robot.real:
            return messagebox.showinfo("Calibration", "Connect the real robot first.", parent=self)
        if c.torque:
            return messagebox.showwarning("Calibration", "Turn Torque OFF first.", parent=self)
        try:
            for j, v in zip(self.app.joints, self.inv):
                j.sign = -1 if v.get() else 1
            c.robot.calibrate_reference(np.radians([float(e.get()) for e in self.ref]))
            self.app.store.save(self.app.joints)
            try:
                q = c.robot.read_q()
                c.q_fb, c.q_cmd, c.q_des = q, q.copy(), q.copy()
            except Exception:
                pass
            messagebox.showinfo("Calibration", "Saved to calibration.json", parent=self)
        except Exception as e:
            messagebox.showerror("Calibration", str(e), parent=self)


# ------------------------------------------------------------------ application
class RobotStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SO-101: 4-DOF Controller & 3D Visualizer")
        self.configure(bg=BG)
        icon_path = os.path.join(BASE_DIR, "robot_logo.png")
        if os.path.exists(icon_path):
            try:
                img = Image.open(icon_path)
                self._icon_img = ImageTk.PhotoImage(img)  # Store reference
                self.wm_iconphoto(True, self._icon_img)
            except Exception as e:
                print(f"Failed to load title bar icon: {e}")
        else:
            print(f"Warning: Icon file '{icon_path}' not found.")
            
        self.kin = SO101Kinematics()
        self.store = CalibrationStore(os.path.join(BASE_DIR, "calibration.json"))
        self.joints = self.store.load()
        self.ctrl = MotionController(self.kin, SimRobot(self.kin.home))
        self.ctrl.set_torque(True)
        self._running = True
        self._tick_id = None
        self._cal_dialog = None

        self.header = HeaderBar(self)
        self.header.pack(fill="x", padx=10, pady=6)
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=4)
        left = Panel(main, "[1] State & Kinematics")
        left.pack(side="left", fill="y", padx=4)
        mid = Panel(main, "[2] Motion & Jacobian")
        mid.pack(side="left", fill="y", padx=4)
        self.viz = Visualizer3D(main, self.ctrl)
        self.viz.pack(side="left", fill="both", expand=True, padx=4)

        self.panels = [JointPanel(left, self.ctrl), TCPPanel(left), TargetPanel(left, self.ctrl),
                       SpeedPanel(mid, self.ctrl), JogPanel(mid, self.ctrl), SingularityPanel(mid), self.viz, self.header]
        for p in self.panels[:6]:
            p.pack(fill="x", padx=8, pady=6)
        self._n = 0
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.tick()

    # ---- actions ----
    def connect(self):
        self.ctrl.shutdown()
        mode_val = self.header.mode.get()
        try:
            if mode_val == "Simulation":
                robot = SimRobot(self.kin.home)
            else:
                port = self.header.port.get().strip()
                baud = int(self.header.baud.get())
                robot = FeetechRobot(port, self.joints, baud=baud)
            self.config(cursor="watch")
            self.update_idletasks()
            robot.connect()
            self.config(cursor="")
            self.ctrl.attach(robot)
            if not robot.real:
                self.ctrl.set_torque(True)
            else:
                messagebox.showinfo("Connected", "Servos found. Calibrate once, then tick 'Torque ON'.")
        except Exception as e:
            self.config(cursor="")
            messagebox.showerror("Connection error", str(e))
            self.header.mode.set("Simulation")
            self.ctrl.attach(SimRobot(self.kin.home))
            self.ctrl.set_torque(True)

    def toggle_estop(self):
        self.ctrl.reset_estop() if self.ctrl.estopped else self.ctrl.emergency()

    def open_calibration(self):
        if self._cal_dialog is not None and self._cal_dialog.winfo_exists():
            self._cal_dialog.lift()
            self._cal_dialog.focus_force()
            return
        self._cal_dialog = CalibrationDialog(self)

    def on_close(self):
        if self.ctrl.robot.real and self.ctrl.torque and not messagebox.askokcancel(
                "Quit", "Torque will be disabled and the arm may fall.\nSupport the arm, then press OK."):
            return
        self._running = False
        if self._tick_id:
            try:
                self.after_cancel(self._tick_id)
            except Exception:
                pass
        self.ctrl.shutdown()
        self.destroy()

    # ---- main loop ----
    def tick(self):
        if not self._running:
            return
        self.ctrl.update()
        self._n += 1
        if self._n % 2 == 0:
            for p in self.panels:
                p.refresh(self.ctrl)
        if self._running:
            self._tick_id = self.after(int(self.ctrl.DT * 1000), self.tick)
