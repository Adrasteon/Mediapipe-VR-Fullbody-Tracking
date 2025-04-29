# inference_gui.py Creates the runtime parameter control GUI using Tkinter.

import tkinter as tk
from tkinter import ttk # Use themed widgets for a better look
import numpy as np
import logging # Use logging
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    logging.error("Scipy not found. Please install it: pip install scipy")
    import sys
    sys.exit(1)

# Local imports (ensure helpers.py is accessible)
try:
    from helpers import shutdown, sendToSteamVR
except ImportError:
    logging.error("Failed to import helpers module. Ensure helpers.py is in the same directory or Python path.")
    import sys
    sys.exit(1)


class InferenceWindow(tk.Frame):
    def __init__(self, root, params, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)

        self.params = params
        # Assigning self to params.gui allows the main script/webui to call methods here (like autocalibrate)
        # It's a common pattern, but be mindful of potential circular dependencies if overused.
        params.gui = self
        self.root = root
        root.title("Runtime Controls") # Set a title for the window

        # --- Calibration Section ---
        calib_frame = ttk.LabelFrame(self.root, text="Calibration Settings", padding=(10, 5))
        calib_frame.pack(padx=10, pady=5, fill="x")

        # Rotation Calibration
        self.calib_rot_var = tk.BooleanVar(value=self.params.calib_rot)
        self.calib_flip_var = tk.BooleanVar(value=self.params.flip)
        self.rot_y_var = tk.DoubleVar(value=self.params.euler_rot_y)
        self.calibrate_rotation_frame(calib_frame)

        # Tilt Calibration
        self.calib_tilt_var = tk.BooleanVar(value=self.params.calib_tilt)
        self.rot_x_var = tk.DoubleVar(value=self.params.euler_rot_x)
        self.rot_z_var = tk.DoubleVar(value=self.params.euler_rot_z)
        self.calibrate_tilt_frame(calib_frame)

        # Scale Calibration
        self.calib_scale_var = tk.BooleanVar(value=self.params.calib_scale)
        self.scale_var = tk.DoubleVar(value=self.params.posescale)
        self.calibrate_scale_frame(calib_frame)

        # Recalibrate Button
        recalibrate_button = ttk.Button(self.root, text='Recalibrate Now (uses checked settings above)',
                                        command=self.autocalibrate)
        recalibrate_button.pack(pady=5)

        self.put_separator()

        # --- Tracking Control Section ---
        tracking_control_frame = ttk.LabelFrame(self.root, text="Tracking Control", padding=(10, 5))
        tracking_control_frame.pack(padx=10, pady=5, fill="x")

        # Pause Button
        pause_button = ttk.Button(tracking_control_frame, text='Pause / Unpause Tracking',
                                  command=self.pause_tracking)
        pause_button.pack(pady=5)

        # Mirror Image Checkbox
        img_mirror_var = tk.BooleanVar(value=self.params.mirror)
        img_mirror_check = ttk.Checkbutton(tracking_control_frame, text="Mirror Camera Image", variable=img_mirror_var,
                                           command=lambda: self.params.change_mirror(img_mirror_var.get()))
        img_mirror_check.pack(pady=5)

        # Image Rotation Radio Buttons
        self.change_image_rotation_frame(tracking_control_frame)

        self.put_separator()

        # --- Smoothing & Latency Section ---
        smooth_frame = ttk.LabelFrame(self.root, text="Smoothing & Latency", padding=(10, 5))
        smooth_frame.pack(padx=10, pady=5, fill="x")

        # Profile Labels (if advanced)
        if params.advanced:
            frame_profile = tk.Frame(smooth_frame)
            frame_profile.pack(fill="x")
            tk.Label(frame_profile, text=" ", width=20).pack(side='left') # Spacer
            tk.Label(frame_profile, text="Profile 1", width=10).pack(side='left')
            tk.Label(frame_profile, text=" ", width=5).pack(side='left') # Spacer
            tk.Label(frame_profile, text="Profile 2", width=10).pack(side='left')

        # Smoothing Window
        if params.advanced:
            self.change_smoothing_frame(smooth_frame)

        # Additional Smoothing
        self.change_add_smoothing_frame(smooth_frame)

        # Camera Latency (if advanced)
        if params.advanced:
            self.change_cam_lat_frame(smooth_frame)

        self.put_separator()

        # --- Advanced Options Section ---
        if params.advanced:
            adv_options_frame = ttk.LabelFrame(self.root, text="Advanced Options", padding=(10, 5))
            adv_options_frame.pack(padx=10, pady=5, fill="x")

            # Neck Offset
            self.change_neck_offset_frame(adv_options_frame)

            # Frametime Logging Checkbox
            self.log_frametime_var = tk.BooleanVar(value=self.params.log_frametime)
            log_frametime_check = ttk.Checkbutton(adv_options_frame, text="Log Frametimes to Console",
                                                  variable=self.log_frametime_var, command=self.change_log_frametime)
            log_frametime_check.pack(pady=5)

        # --- NEW: Keyboard Head Control Section ---
        keyboard_frame = ttk.LabelFrame(self.root, text="Keyboard Head Control (SteamVR Only)", padding=(10, 5))
        keyboard_frame.pack(padx=10, pady=5, fill="x")
        self.add_keyboard_control_frame(keyboard_frame)
        # --- End Keyboard Head Control ---

        self.put_separator()

        # --- Exit Button ---
        exit_button = ttk.Button(self.root, text='Save Settings & Exit', command=self.params.ready2exit)
        exit_button.pack(pady=10)

        # Handle window close button (X)
        root.protocol("WM_DELETE_WINDOW", self.params.ready2exit)

        # Initialize variable displays (optional, can sometimes cause issues if params update elsewhere)
        # self.set_rot_y_var()
        # self.set_rot_x_var()
        # self.set_rot_z_var()
        # self.set_scale_var()

    def put_separator(self):
        """Adds a horizontal separator."""
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill='x', pady=5, padx=10)

    # --- Frame Creation Methods ---

    def calibrate_rotation_frame(self, parent_frame):
        """Creates the rotation calibration controls."""
        frame = tk.Frame(parent_frame)
        frame.pack(fill="x", pady=2)

        rot_check = ttk.Checkbutton(frame, text="Auto Calibrate Rotation (Y)", variable=self.calib_rot_var, command=self.change_rot_auto)
        rot_check.pack(anchor=tk.W)

        flip_check = ttk.Checkbutton(frame, text="Flip Calibration (180°)", variable=self.calib_flip_var, command=self.change_rot_flip)
        flip_check.pack(anchor=tk.W)

        rot_y_frame = tk.Frame(frame)
        rot_y_frame.pack(fill="x")

        rot_y_scale = ttk.Scale(rot_y_frame, from_=-40, to=400,
                                orient=tk.HORIZONTAL, length=350, variable=self.rot_y_var)
        rot_y_scale.pack(side='left', padx=5)
        # Use trace_add for continuous updates from Scale/Entry
        self.rot_y_var.trace_add('write', lambda var, index, mode: self.params.rot_change_y(self.rot_y_var.get()))

        # Optional: Add Entry box for precise input
        rot_y_entry = ttk.Entry(rot_y_frame, textvariable=self.rot_y_var, width=5)
        rot_y_entry.pack(side='left', padx=5)

    def calibrate_tilt_frame(self, parent_frame):
        """Creates the tilt calibration controls."""
        frame = tk.Frame(parent_frame)
        frame.pack(fill="x", pady=2)

        tilt_check = ttk.Checkbutton(frame, text="Auto Calibrate Tilt (X/Z)", variable=self.calib_tilt_var, command=self.change_tilt_auto)
        tilt_check.pack(anchor=tk.W)

        rot_x_frame = tk.Frame(frame)
        rot_x_frame.pack(fill="x")
        rot_z_frame = tk.Frame(frame)
        rot_z_frame.pack(fill="x")

        # Rotation X
        ttk.Label(rot_x_frame, text="Tilt X:", width=6).pack(side='left')
        rot_x_scale = ttk.Scale(rot_x_frame, from_=0, to=180,
                                orient=tk.HORIZONTAL, length=350, variable=self.rot_x_var)
        rot_x_scale.pack(side='left', padx=5)
        self.rot_x_var.trace_add('write', lambda var, index, mode: self.params.rot_change_x(self.rot_x_var.get()))
        rot_x_entry = ttk.Entry(rot_x_frame, textvariable=self.rot_x_var, width=5)
        rot_x_entry.pack(side='left', padx=5)

        # Rotation Z
        ttk.Label(rot_z_frame, text="Tilt Z:", width=6).pack(side='left')
        rot_z_scale = ttk.Scale(rot_z_frame, from_=90, to=270,
                                orient=tk.HORIZONTAL, length=350, variable=self.rot_z_var)
        rot_z_scale.pack(side='left', padx=5)
        self.rot_z_var.trace_add('write', lambda var, index, mode: self.params.rot_change_z(self.rot_z_var.get()))
        rot_z_entry = ttk.Entry(rot_z_frame, textvariable=self.rot_z_var, width=5)
        rot_z_entry.pack(side='left', padx=5)

    def calibrate_scale_frame(self, parent_frame):
        """Creates the scale calibration controls."""
        frame = tk.Frame(parent_frame)
        frame.pack(fill="x", pady=2)

        scale_check = ttk.Checkbutton(frame, text="Auto Calibrate Scale", variable=self.calib_scale_var, command=self.change_scale_auto)
        scale_check.pack(anchor=tk.W)

        scale_frame = tk.Frame(frame)
        scale_frame.pack(fill="x")

        ttk.Label(scale_frame, text="Scale:", width=6).pack(side='left')
        scale_scale = ttk.Scale(scale_frame, from_=0.5, to=2.0,
                                orient=tk.HORIZONTAL, length=350, variable=self.scale_var)
        scale_scale.pack(side='left', padx=5)
        self.scale_var.trace_add('write', lambda var, index, mode: self.params.change_scale(self.scale_var.get()))
        scale_entry = ttk.Entry(scale_frame, textvariable=self.scale_var, width=5)
        scale_entry.pack(side='left', padx=5)

    def change_smoothing_frame(self, parent_frame):
        """Creates the smoothing window controls (Advanced)."""
        frame = tk.Frame(parent_frame)
        frame.pack(fill="x", pady=2)

        ttk.Label(frame, text="Smoothing Window:", width=20).pack(side='left')
        smoothing_entry1 = ttk.Entry(frame, width=10)
        smoothing_entry1.pack(side='left', padx=5)
        smoothing_entry1.insert(0, f"{self.params.smoothing_1:.2f}")
        ttk.Button(frame, text='Update P1', width=10,
                   command=lambda: self.params.change_smoothing(float(smoothing_entry1.get()), 1)).pack(side='left')

        if self.params.advanced:
            smoothing_entry2 = ttk.Entry(frame, width=10)
            smoothing_entry2.pack(side='left', padx=5)
            smoothing_entry2.insert(0, f"{self.params.smoothing_2:.2f}")
            ttk.Button(frame, text='Update P2', width=10,
                       command=lambda: self.params.change_smoothing(float(smoothing_entry2.get()), 2)).pack(side='left')

        ttk.Button(frame, text='Disable', width=8,
                   command=lambda: self.params.change_smoothing(0.0)).pack(side='left', padx=5)

    def change_add_smoothing_frame(self, parent_frame):
        """Creates the additional smoothing controls."""
        frame = tk.Frame(parent_frame)
        frame.pack(fill="x", pady=2)

        ttk.Label(frame, text="Additional Smoothing:", width=20).pack(side='left')
        add_smoothing_entry1 = ttk.Entry(frame, width=10)
        add_smoothing_entry1.pack(side='left', padx=5)
        add_smoothing_entry1.insert(0, f"{self.params.additional_smoothing_1:.2f}")
        ttk.Button(frame, text='Update P1', width=10,
                   command=lambda: self.params.change_additional_smoothing(float(add_smoothing_entry1.get()), 1)).pack(side='left')

        if self.params.advanced:
            add_smoothing_entry2 = ttk.Entry(frame, width=10)
            add_smoothing_entry2.pack(side='left', padx=5)
            add_smoothing_entry2.insert(0, f"{self.params.additional_smoothing_2:.2f}")
            ttk.Button(frame, text='Update P2', width=10,
                       command=lambda: self.params.change_additional_smoothing(float(add_smoothing_entry2.get()), 2)).pack(side='left')

        ttk.Button(frame, text='Disable', width=8,
                   command=lambda: self.params.change_additional_smoothing(0.0)).pack(side='left', padx=5)

    def change_cam_lat_frame(self, parent_frame):
        """Creates the camera latency control (Advanced)."""
        frame = tk.Frame(parent_frame)
        frame.pack(fill="x", pady=2)

        ttk.Label(frame, text="Camera Latency (s):", width=20).pack(side='left')
        latency_entry = ttk.Entry(frame, width=10)
        latency_entry.pack(side='left', padx=5)
        latency_entry.insert(0, f"{self.params.camera_latency:.3f}")
        ttk.Button(frame, text='Update', width=10,
                   command=lambda: self.params.change_camera_latency(float(latency_entry.get()))).pack(side='left')

    def change_image_rotation_frame(self, parent_frame):
        """Creates the image rotation radio buttons."""
        frame = tk.Frame(parent_frame)
        frame.pack(fill="x", pady=2)

        rot_img_var = tk.IntVar(value=self.params.img_rot_dict_rev.get(self.params.rotate_image, 0))
        ttk.Label(frame, text="Image Rotation:", width=15).pack(side='left')
        ttk.Radiobutton(frame, text="0°", variable=rot_img_var, value=0).pack(side='left')
        ttk.Radiobutton(frame, text="90°", variable=rot_img_var, value=1).pack(side='left')
        ttk.Radiobutton(frame, text="180°", variable=rot_img_var, value=2).pack(side='left')
        ttk.Radiobutton(frame, text="270°", variable=rot_img_var, value=3).pack(side='left')
        # Use trace_add for immediate update when radio button changes
        rot_img_var.trace_add('write', lambda var, index, mode: self.params.change_img_rot(rot_img_var.get()))

    def change_neck_offset_frame(self, parent_frame):
        """Creates the HMD neck offset controls (Advanced)."""
        frame = tk.Frame(parent_frame)
        frame.pack(fill="x", pady=2)

        ttk.Label(frame, text="HMD Neck Offset (X Y Z):", width=22).pack(side='left')
        offset_x_entry = ttk.Entry(frame, width=6)
        offset_x_entry.pack(side='left', padx=2)
        offset_x_entry.insert(0, f"{self.params.hmd_to_neck_offset[0]:.2f}")
        offset_y_entry = ttk.Entry(frame, width=6)
        offset_y_entry.pack(side='left', padx=2)
        offset_y_entry.insert(0, f"{self.params.hmd_to_neck_offset[1]:.2f}")
        offset_z_entry = ttk.Entry(frame, width=6)
        offset_z_entry.pack(side='left', padx=2)
        offset_z_entry.insert(0, f"{self.params.hmd_to_neck_offset[2]:.2f}")

        ttk.Button(frame, text='Update', width=8,
                   command=lambda: self.params.change_neck_offset(
                       float(offset_x_entry.get()),
                       float(offset_y_entry.get()),
                       float(offset_z_entry.get()))
                   ).pack(side='left', padx=5)

    def add_keyboard_control_frame(self, parent_frame):
        """Creates the keyboard head control elements."""
        frame = tk.Frame(parent_frame)
        frame.pack(fill="x", pady=2)

        # Enable Checkbox
        keyboard_enabled_var = tk.BooleanVar(value=self.params.keyboard_head_control_enabled)
        keyboard_enabled_check = ttk.Checkbutton(
            frame,
            text="Enable Keyboard Control",
            variable=keyboard_enabled_var,
            # Directly update the parameter attribute. The main loop checks this attribute.
            command=lambda: setattr(self.params, 'keyboard_head_control_enabled', keyboard_enabled_var.get())
        )
        keyboard_enabled_check.pack(side="left", padx=5)

        # Speed Slider
        speed_frame = tk.Frame(frame)
        speed_frame.pack(side="left", fill="x", expand=True, padx=10)

        ttk.Label(speed_frame, text="Speed:").pack(side="left")
        keyboard_speed_var = tk.DoubleVar(value=self.params.keyboard_head_control_speed)
        keyboard_speed_scale = ttk.Scale(
            speed_frame,
            from_=0.5, to=5.0, # Adjust range as needed
            orient=tk.HORIZONTAL,
            length=150,
            variable=keyboard_speed_var,
            # Directly update the parameter attribute. The main loop checks this attribute.
            command=lambda val: setattr(self.params, 'keyboard_head_control_speed', float(val))
        )
        keyboard_speed_scale.pack(side="left", padx=5)

        # Optional: Display current speed value
        speed_label_val = ttk.Label(speed_frame, textvariable=keyboard_speed_var, width=4)
        speed_label_val.pack(side="left")
        # Format the label initially and on change
        def format_speed_label(*args):
            try:
                speed_label_val.config(text=f"{keyboard_speed_var.get():.1f}")
            except tk.TclError: # Handle window closing error
                pass
        keyboard_speed_var.trace_add('write', format_speed_label)
        format_speed_label() # Initial formatting


    # --- Action Methods ---

    def pause_tracking(self):
        """Toggles the pause state."""
        self.params.paused = not self.params.paused
        logging.info(f"Pose estimation {'paused' if self.params.paused else 'unpaused'}")

    def change_log_frametime(self):
        """Toggles frametime logging."""
        self.params.log_frametime = self.log_frametime_var.get()
        logging.info(f"{'Enabled' if self.params.log_frametime else 'Disabled'} frametime logging")

    def change_rot_auto(self):
        """Updates the auto rotation calibration flag."""
        self.params.calib_rot = self.calib_rot_var.get()
        logging.info(f"Automatic rotation calibration {'enabled' if self.params.calib_rot else 'disabled'}")

    def change_rot_flip(self):
        """Updates the rotation flip flag."""
        self.params.flip = self.calib_flip_var.get()
        logging.info(f"Rotation calibration flip {'enabled' if self.params.flip else 'disabled'}")

    def change_tilt_auto(self):
        """Updates the auto tilt calibration flag."""
        self.params.calib_tilt = self.calib_tilt_var.get()
        logging.info(f"Automatic tilt calibration {'enabled' if self.params.calib_tilt else 'disabled'}")

    def change_scale_auto(self):
        """Updates the auto scale calibration flag."""
        self.params.calib_scale = self.calib_scale_var.get()
        logging.info(f"Automatic scale calibration {'enabled' if self.params.calib_scale else 'disabled'}")

    # --- Variable Update Methods (Called by autocalibrate) ---

    def set_rot_y_var(self):
        """Updates the Y rotation slider/variable."""
        angle = self.params.euler_rot_y
        # Note: Flip logic might be better handled purely within calibration calculation
        # if self.params.flip: angle += 180
        angle = angle % 360 # Normalize to 0-360
        self.rot_y_var.set(round(angle, 1))

    def set_rot_x_var(self):
        """Updates the X rotation slider/variable."""
        self.rot_x_var.set(round(self.params.euler_rot_x, 1))

    def set_rot_z_var(self):
        """Updates the Z rotation slider/variable."""
        self.rot_z_var.set(round(self.params.euler_rot_z, 1))

    def set_scale_var(self):
        """Updates the scale slider/variable."""
        self.scale_var.set(round(self.params.posescale, 2))

    # --- Autocalibration Logic ---

    def autocalibrate(self):
        """Performs automatic calibration based on current pose and HMD data."""
        logging.info("Attempting autocalibration...")

        # Check if pose data is available
        if not hasattr(self.params, 'pose3d_og') or self.params.pose3d_og is None:
            logging.warning("Cannot autocalibrate: No pose data available (params.pose3d_og is None).")
            return

        # --- Get HMD Data (if applicable) ---
        headsetpos = None
        headsetrot = None
        use_steamvr = self.params.backend == 1

        if use_steamvr:
            try:
                array = sendToSteamVR("getdevicepose 0", num_tries=3, wait_time=0.05)
                if array is None or len(array) < 10 or array[0] == "error":
                    logging.error(f"Autocalibrate failed: Could not get valid HMD pose from SteamVR. Response: {array}")
                    return
                # Safely parse HMD data
                headsetpos = np.array([float(array[3]), float(array[4]), float(array[5])])
                # Assuming WXYZ order from driver: [w, x, y, z] -> Scipy wants [x, y, z, w]
                qx, qy, qz, qw = float(array[7]), float(array[8]), float(array[9]), float(array[6])
                if np.isclose(np.linalg.norm([qx, qy, qz, qw]), 0.0):
                     logging.error("Autocalibrate failed: HMD quaternion from driver is zero.")
                     return
                headsetrot = R.from_quat([qx, qy, qz, qw])
                logging.info(f"HMD Data | Pos: {headsetpos}, Rot Quat (xyzw): {headsetrot.as_quat()}")
            except (IndexError, ValueError, TypeError) as e:
                logging.error(f"Autocalibrate failed: Error parsing HMD pose data: {e}")
                return
            except Exception as e:
                 logging.error(f"Autocalibrate failed: Unexpected error getting HMD pose: {e}", exc_info=True)
                 return

        # --- Perform Calibrations ---
        temp_pose = self.params.pose3d_og.copy() # Work on a copy

        # 1. Tilt Calibration (Roll Z, Tilt X) - Requires only pose data
        if self.params.calib_tilt:
            try:
                feet_middle = (temp_pose[0] + temp_pose[5]) / 2 # Midpoint between ankles
                logging.debug(f"Tilt Calib | Feet Middle (Before): {feet_middle}")

                # Roll (Z) Calculation: Angle in XY plane relative to negative Y axis
                roll_angle_rad = np.arctan2(feet_middle[0], -feet_middle[1])
                logging.info(f"Tilt Calib | Calculated Roll (Z) Correction: {-np.degrees(roll_angle_rad):.1f}°")
                self.params.rot_change_z(-np.degrees(roll_angle_rad) + 180.0) # Apply correction
                self.set_rot_z_var() # Update GUI
                temp_pose = self.params.global_rot_z.apply(temp_pose) # Apply to temp pose for next step

                # Tilt (X) Calculation: Angle in YZ plane relative to negative Y axis (after Z correction)
                feet_middle_after_z = (temp_pose[0] + temp_pose[5]) / 2
                tilt_angle_rad = np.arctan2(feet_middle_after_z[2], -feet_middle_after_z[1])
                logging.info(f"Tilt Calib | Calculated Tilt (X) Correction: {np.degrees(tilt_angle_rad):.1f}°")
                self.params.rot_change_x(np.degrees(tilt_angle_rad) + 90.0) # Apply correction
                self.set_rot_x_var() # Update GUI
                temp_pose = self.params.global_rot_x.apply(temp_pose) # Apply to temp pose

                feet_middle_final = (temp_pose[0] + temp_pose[5]) / 2
                logging.debug(f"Tilt Calib | Feet Middle (After): {feet_middle_final}")

            except Exception as e:
                logging.error(f"Tilt calibration failed: {e}", exc_info=True)

        # 2. Rotation Calibration (Y) - Requires HMD data
        if use_steamvr and self.params.calib_rot:
            if headsetrot is None:
                 logging.warning("Skipping rotation calibration: HMD data not available.")
            else:
                try:
                    # Vector between feet in XZ plane (after tilt correction)
                    feet_vec = temp_pose[0] - temp_pose[5]
                    pose_yaw_rad = np.arctan2(feet_vec[0], feet_vec[2]) # Angle relative to Z axis

                    # HMD Yaw (Rotation around Y axis)
                    # Extract Y rotation from HMD quaternion or matrix
                    hmd_yaw_rad = headsetrot.as_euler('yxz', degrees=False)[0] # Yaw is the first element

                    logging.debug(f"Rot Calib | Pose Yaw: {np.degrees(pose_yaw_rad):.1f}°, HMD Yaw: {np.degrees(hmd_yaw_rad):.1f}°")

                    # Calculate correction angle
                    correction_rad = pose_yaw_rad - hmd_yaw_rad
                    correction_deg = -np.degrees(correction_rad) # Apply opposite rotation

                    if self.params.flip:
                        correction_deg += 180.0
                        logging.info("Rot Calib | Applying 180° flip.")

                    logging.info(f"Rot Calib | Calculated Rotation (Y) Correction: {correction_deg:.1f}°")
                    self.params.rot_change_y(correction_deg) # Apply correction
                    self.set_rot_y_var() # Update GUI
                    temp_pose = self.params.global_rot_y.apply(temp_pose) # Apply to temp pose

                    feet_vec_after = temp_pose[0] - temp_pose[5]
                    pose_yaw_after_rad = np.arctan2(feet_vec_after[0], feet_vec_after[2])
                    logging.debug(f"Rot Calib | Pose Yaw (After): {np.degrees(pose_yaw_after_rad):.1f}°")

                except Exception as e:
                    logging.error(f"Rotation calibration failed: {e}", exc_info=True)

        # 3. Scale Calibration - Requires HMD data
        if use_steamvr and self.params.calib_scale:
            if headsetpos is None:
                 logging.warning("Skipping scale calibration: HMD data not available.")
            else:
                try:
                    # Use HMD height as target height
                    hmd_height = headsetpos[1]
                    if hmd_height <= 0.1: # Avoid division by zero or unrealistic heights
                        logging.warning(f"Skipping scale calibration: Invalid HMD height ({hmd_height:.2f}m). Ensure HMD is tracking.")
                    else:
                        # Calculate skeleton height (vertical span after other calibrations)
                        skel_min_y = np.min(temp_pose[:, 1])
                        skel_max_y = np.max(temp_pose[:, 1])
                        skel_height = skel_max_y - skel_min_y

                        if skel_height <= 0.1:
                             logging.warning(f"Skipping scale calibration: Invalid skeleton height ({skel_height:.2f}m).")
                        else:
                            new_scale = hmd_height / skel_height
                            logging.info(f"Scale Calib | HMD Height: {hmd_height:.2f}m, Skel Height: {skel_height:.2f}m => New Scale: {new_scale:.3f}")
                            self.params.change_scale(new_scale) # Apply correction
                            self.set_scale_var() # Update GUI
                except Exception as e:
                    logging.error(f"Scale calibration failed: {e}", exc_info=True)

        # Signal that calibration attempt is done (even if parts failed)
        self.params.recalibrate = False
        logging.info("Autocalibration attempt finished.")


# --- Main Function to Create GUI ---

def make_inference_gui(_params):
    """Creates and runs the Tkinter inference GUI window."""
    try:
        root = tk.Tk()
        # Apply theme if available (e.g., 'clam', 'alt', 'default', 'classic')
        try:
            style = ttk.Style(root)
            # print(style.theme_names()) # See available themes
            style.theme_use('clam') # Or another theme like 'vista' on Windows
        except tk.TclError:
            logging.warning("Failed to set ttk theme. Using default.")

        app = InferenceWindow(root, _params)
        app.pack(side="top", fill="both", expand=True)
        root.mainloop()
    except Exception as e:
        logging.exception(f"Error creating inference GUI: {e}")
        # Ensure shutdown is called if GUI fails critically
        if _params and not _params.exit_ready:
             _params.exit_ready = True
             # shutdown(_params, 1) # Avoid calling shutdown directly here, let main loop handle it


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Create a dummy params object for testing
    class DummyParams:
        def __init__(self):
            self.calib_rot = True
            self.flip = False
            self.euler_rot_y = 180.0
            self.calib_tilt = True
            self.euler_rot_x = 90.0
            self.euler_rot_z = 180.0
            self.calib_scale = True
            self.posescale = 1.0
            self.paused = False
            self.advanced = True # Enable advanced options for testing
            self.smoothing_1 = 0.0
            self.smoothing_2 = 0.5
            self.additional_smoothing_1 = 0.7
            self.additional_smoothing_2 = 0.9
            self.camera_latency = 0.050
            self.rotate_image = None
            self.img_rot_dict_rev = {None: 0, cv2.ROTATE_90_CLOCKWISE: 1, cv2.ROTATE_180: 2, cv2.ROTATE_90_COUNTERCLOCKWISE: 3}
            self.hmd_to_neck_offset = [0.0, -0.2, 0.1]
            self.log_frametime = False
            self.mirror = False
            self.exit_ready = False
            self.backend = 1 # Simulate SteamVR backend for testing calibration
            self.pose3d_og = None # Needs to be set for calibration test
            self.keyboard_head_control_enabled = False
            self.keyboard_head_control_speed = 1.5
            self.gui = None

        # Dummy methods matching those called by the GUI
        def rot_change_y(self, val): print(f"Params: Set rot_y to {val}")
        def rot_change_x(self, val): print(f"Params: Set rot_x to {val}")
        def rot_change_z(self, val): print(f"Params: Set rot_z to {val}")
        def change_scale(self, val): print(f"Params: Set scale to {val}")
        def change_smoothing(self, val, pid): print(f"Params: Set smoothing {pid} to {val}")
        def change_additional_smoothing(self, val, pid): print(f"Params: Set add_smoothing {pid} to {val}")
        def change_camera_latency(self, val): print(f"Params: Set cam_latency to {val}")
        def change_img_rot(self, val): print(f"Params: Set img_rot to {val}")
        def change_neck_offset(self, x, y, z): print(f"Params: Set neck_offset to {[x, y, z]}")
        def change_mirror(self, val): print(f"Params: Set mirror to {val}")
        def ready2exit(self): print("Params: Exit requested!"); self.exit_ready = True; root.quit() # Quit root for testing

        # Dummy methods for keyboard control (actual logic is in keyboard_helper)
        def change_keyboard_control_enabled(self, enabled): print(f"Params: Set keyboard_enabled to {enabled}"); self.keyboard_head_control_enabled = enabled
        def change_keyboard_control_speed(self, speed): print(f"Params: Set keyboard_speed to {speed}"); self.keyboard_head_control_speed = speed

    logging.basicConfig(level=logging.INFO)
    print("Running inference_gui.py directly for testing...")
    test_params = DummyParams()
    # Simulate some pose data for calibration testing
    test_params.pose3d_og = np.random.rand(29, 3) * 2 - 1 # Random data around origin

    make_inference_gui(test_params)
    print("GUI closed.")
