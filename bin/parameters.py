# parameters.py Defines and manages application parameters, including loading/saving.
from init_gui import getparams
from scipy.spatial.transform import Rotation as R
import cv2
import json
import logging # Use logging

class Parameters():
    def __init__(self) -> None:
        param = None
        while param == None:
            # getparams() handles the initial GUI popup
            param = getparams()

        # --- Parameters from Initial GUI ---
        self.advanced = param["advanced"]
        self.model = param["model_complexity"] # Still relevant for Holistic
        self.smooth_landmarks = param["smooth_landmarks"] # Still relevant for Holistic
        self.min_tracking_confidence = param["min_tracking_confidence"] # Still relevant for Holistic
        # self.static_image = param["static_image"] # Removed, not used by Holistic stream

        self.maximgsize = param["imgsize"]
        self.cameraid = param["camid"]
        self.preview_skeleton = param["prevskel"]
        # self.dont_wait_hmd = param["waithmd"] # This param seems unused, consider removing later
        self.feet_rotation = param["feetrot"]
        self.use_hands = param["use_hands"]
        self.use_face = param["use_face"] # Added face tracking flag
        self.ignore_hip = param["ignore_hip"]

        self.camera_settings = param["camera_settings"]
        self.camera_width = param["camera_width"]
        self.camera_height = param["camera_height"]

        self.backend = param["backend"]
        self.backend_ip = param["backend_ip"]
        self.backend_port = param["backend_port"]

        self.webui = param["webui"]

        # --- Internal State & Default Parameters ---
        self.hmd_to_neck_offset = [0.0, -0.2, 0.1] # Default offset
        self.rotate_image = None # Default, loaded from save file later
        self.camera_latency = 0.0 # Default, loaded/set later
        self.smoothing_1 = 0.0 # Default for profile 1
        self.additional_smoothing_1 = 0.7 # Default for profile 1
        self.smoothing_2 = 0.5 # Default for profile 2
        self.additional_smoothing_2 = 0.9 # Default for profile 2

        self.calib_rot = True
        self.calib_tilt = True
        self.calib_scale = True
        self.recalibrate = False # Flag to trigger recalibration

        # Default rotations in degrees! Loaded from save file later.
        self.euler_rot_y = 180.0
        self.euler_rot_x = 90.0
        self.euler_rot_z = 180.0
        self.posescale = 1.0 # Default scale

        # --- Keyboard Control Defaults ---
        self.keyboard_head_control_enabled = False # Default disabled
        self.keyboard_head_control_speed = 1.5     # Default speed (degrees per tick)

        self.exit_ready = False # Flag to signal exit request
        self.paused = False # Flag for pausing tracking
        self.flip = False # Flag for flipping calibration rotation
        self.log_frametime = False # Flag for logging FPS
        self.mirror = False # Flag for mirroring camera image

        # Dictionaries for image rotation mapping
        self.img_rot_dict = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
        self.img_rot_dict_rev = {v: k for k, v in self.img_rot_dict.items()} # Create reverse mapping

        # --- Load Saved Parameters ---
        self.load_params() # Overwrites defaults with saved values if available

        # --- Initialize Runtime Variables ---
        # Calculate initial rotation objects from Euler angles
        self.update_rotation_matrices()

        # Set initial active smoothing profile (default to profile 1)
        self.smoothing = self.smoothing_1
        self.additional_smoothing = self.additional_smoothing_1

        # If advanced mode is disabled, force basic smoothing/latency
        if not self.advanced:
            self.smoothing = 0.0
            self.smoothing_1 = 0.0 # Ensure profile 1 is also reset
            self.camera_latency = 0.0

        # --- Link to Inference GUI ---
        self.gui = None # Placeholder for the inference GUI instance


    def update_rotation_matrices(self):
        """Updates Scipy Rotation objects based on current Euler angles."""
        try:
            # Apply standard adjustments for coordinate system alignment
            self.global_rot_y = R.from_euler('y', self.euler_rot_y, degrees=True)
            self.global_rot_x = R.from_euler('x', self.euler_rot_x - 90, degrees=True)
            self.global_rot_z = R.from_euler('z', self.euler_rot_z - 180, degrees=True)
        except Exception as e:
            logging.error(f"Error updating rotation matrices: {e}")
            # Fallback to identity rotations?
            self.global_rot_y = R.identity()
            self.global_rot_x = R.identity()
            self.global_rot_z = R.identity()

    def change_recalibrate(self):
        """Sets the flag to trigger recalibration."""
        self.recalibrate = True

    # --- Callback functions for GUI/WebUI changes ---
    def rot_change_y(self, value):
        logging.info(f"Changed y rotation value to {value}")
        self.euler_rot_y = float(value)
        self.update_rotation_matrices()

    def rot_change_x(self, value):
        logging.info(f"Changed x rotation value to {value}")
        self.euler_rot_x = float(value)
        self.update_rotation_matrices()

    def rot_change_z(self, value):
        logging.info(f"Changed z rotation value to {value}")
        self.euler_rot_z = float(value)
        self.update_rotation_matrices()

    def change_scale(self, value):
        logging.info(f"Changed scale value to {value}")
        self.posescale = float(value)

    def change_img_rot(self, val):
        rotation_degrees = val * 90
        logging.info(f"Changed image rotation to {rotation_degrees} clockwise")
        self.rotate_image = self.img_rot_dict.get(int(val), None)

    def change_smoothing(self, val, paramid=0):
        value = float(val)
        logging.info(f"Changed smoothing value to {value} (Profile {paramid})")
        self.smoothing = value # Update active smoothing

        if paramid == 1:
            self.smoothing_1 = value
        if paramid == 2:
            self.smoothing_2 = value

    def change_additional_smoothing(self, val, paramid=0):
        value = float(val)
        logging.info(f"Changed additional smoothing value to {value} (Profile {paramid})")
        self.additional_smoothing = value # Update active smoothing

        if paramid == 1:
            self.additional_smoothing_1 = value
        if paramid == 2:
            self.additional_smoothing_2 = value

    def change_camera_latency(self, val):
        value = float(val)
        logging.info(f"Changed camera latency to {value}")
        self.camera_latency = value

    def change_neck_offset(self, x, y, z):
        offset = [float(x), float(y), float(z)]
        logging.info(f"HMD to neck offset changed to: {offset}")
        self.hmd_to_neck_offset = offset

    def change_mirror(self, mirror):
        is_mirrored = bool(mirror)
        logging.info(f"Image mirror set to {is_mirrored}")
        self.mirror = is_mirrored

    # --- Keyboard Control Callbacks (Add if needed for GUI) ---
    # def change_keyboard_control_enabled(self, enabled):
    #     is_enabled = bool(enabled)
    #     logging.info(f"Keyboard head control set to {is_enabled}")
    #     self.keyboard_head_control_enabled = is_enabled
    #     # Note: Actual enabling/disabling happens via keyboard_helper module

    # def change_keyboard_control_speed(self, speed):
    #     new_speed = float(speed)
    #     if new_speed > 0:
    #         logging.info(f"Keyboard head control speed set to {new_speed}")
    #         self.keyboard_head_control_speed = new_speed
    #     else:
    #         logging.warning("Keyboard head control speed must be positive.")
    #     # Note: Actual speed setting happens via keyboard_helper module

    def ready2exit(self):
        """Signals the main loop to exit."""
        self.exit_ready = True

    def save_params(self):
        """Saves current parameters to a JSON file."""
        param = {
            "rotate": self.img_rot_dict_rev.get(self.rotate_image, 0),
            "smooth1": self.smoothing_1,
            "smooth2": self.smoothing_2,
            "camlatency": self.camera_latency,
            "addsmooth1": self.additional_smoothing_1,
            "addsmooth2": self.additional_smoothing_2,
            "roty": self.euler_rot_y,
            "rotx": self.euler_rot_x,
            "rotz": self.euler_rot_z,
            "scale": self.posescale,
            "calibrot": self.calib_rot,
            "calibtilt": self.calib_tilt,
            "calibscale": self.calib_scale,
            "flip": self.flip,
            "hmd_to_neck_offset": self.hmd_to_neck_offset,
            "mirror": self.mirror,
            "use_face": self.use_face, # Save face tracking flag
            # --- Save Keyboard Control Params ---
            "keyboard_head_control_enabled": self.keyboard_head_control_enabled,
            "keyboard_head_control_speed": self.keyboard_head_control_speed,
            # Add other parameters loaded initially if they should be saved
        }

        try:
            with open("saved_params.json", "w") as f:
                json.dump(param, f, indent=4) # Add indent for readability
            logging.info("Parameters saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save parameters: {e}")

    def load_params(self):
        """Loads parameters from the JSON file if it exists."""
        try:
            with open("saved_params.json", "r") as f:
                param = json.load(f)

            self.rotate_image = self.img_rot_dict.get(param.get("rotate", 0), None)
            self.smoothing_1 = param.get("smooth1", self.smoothing_1)
            self.smoothing_2 = param.get("smooth2", self.smoothing_2)
            self.camera_latency = param.get("camlatency", self.camera_latency)
            self.additional_smoothing_1 = param.get("addsmooth1", self.additional_smoothing_1)
            self.additional_smoothing_2 = param.get("addsmooth2", self.additional_smoothing_2)

            self.euler_rot_y = param.get("roty", self.euler_rot_y)
            self.euler_rot_x = param.get("rotx", self.euler_rot_x)
            self.euler_rot_z = param.get("rotz", self.euler_rot_z)
            self.posescale = param.get("scale", self.posescale)

            self.calib_rot = param.get("calibrot", self.calib_rot)
            self.calib_tilt = param.get("calibtilt", self.calib_tilt)
            self.calib_scale = param.get("calibscale", self.calib_scale)

            self.mirror = param.get("mirror", self.mirror)

            # Only load neck offset if advanced mode is enabled (or maybe always load?)
            # Let's always load it, but GUI might hide it.
            self.hmd_to_neck_offset = param.get("hmd_to_neck_offset", self.hmd_to_neck_offset)

            self.flip = param.get("flip", self.flip)
            self.use_face = param.get("use_face", self.use_face) # Load face tracking flag

            # --- Load Keyboard Control Params ---
            self.keyboard_head_control_enabled = param.get("keyboard_head_control_enabled", self.keyboard_head_control_enabled)
            self.keyboard_head_control_speed = param.get("keyboard_head_control_speed", self.keyboard_head_control_speed)

            logging.info("Parameters loaded successfully.")

        except FileNotFoundError:
            logging.info("Save file (saved_params.json) not found, using defaults. Will be created on exit.")
        except Exception as e:
            logging.error(f"Failed to load parameters: {e}. Using defaults.")

# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # This part won't run the GUI, just tests class instantiation
    try:
        # Need a dummy getparams for direct execution
        def dummy_getparams():
            # Include defaults for keyboard params here if needed for standalone test
            return {
                "advanced": False, "model_complexity": 1, "smooth_landmarks": True,
                "min_tracking_confidence": 0.5, "imgsize": 640, "camid": "0",
                "prevskel": False, "feetrot": False, "use_hands": False, "use_face": False,
                "ignore_hip": False, "camera_settings": False, "camera_width": 640,
                "camera_height": 480, "backend": 1, "backend_ip": "127.0.0.1",
                "backend_port": 9000, "webui": False
            }
        getparams = dummy_getparams # Override for testing
        params_instance = Parameters()
        print("Parameters instance created with defaults/loaded values.")
        print(f"Keyboard Control Enabled: {params_instance.keyboard_head_control_enabled}")
        print(f"Keyboard Control Speed: {params_instance.keyboard_head_control_speed}")
        # params_instance.save_params() # Test saving
    except Exception as e:
        print(f"Error during direct execution test: {e}")
