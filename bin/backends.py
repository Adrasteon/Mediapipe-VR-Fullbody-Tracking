# backends.py Defines backend interfaces and implementations (SteamVR, VRChat OSC, Dummy).
import time
import logging  # Use logging module
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple  # For type hinting

# Third-party imports (ensure these are installed)
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    logging.error("Scipy not found. Please install it: pip install scipy")
    import sys
    sys.exit(1)
try:
    from pythonosc import osc_bundle_builder, osc_message_builder, udp_client
    # Specific OSC parameter names for VRChat Avatars 3.0
    # https://docs.vrchat.com/docs/osc-avatar-parameters
    OSC_HAND_PARAMS_LEFT = {
        "ThumbCurl": "/avatar/parameters/LeftThumb",
        "IndexCurl": "/avatar/parameters/LeftIndex",
        "MiddleCurl": "/avatar/parameters/LeftMiddle",
        "RingCurl": "/avatar/parameters/LeftRing",
        "PinkyCurl": "/avatar/parameters/LeftLittle", # VRC uses "Little"
        "ThumbSplay": "/avatar/parameters/LeftThumbSplay", # Example, might vary
        "IndexSplay": "/avatar/parameters/LeftIndexSplay",
        "MiddleSplay": "/avatar/parameters/LeftMiddleSplay",
        "RingSplay": "/avatar/parameters/LeftRingSplay",
        # Pinky splay often not a standard parameter
    }
    OSC_HAND_PARAMS_RIGHT = {
        "ThumbCurl": "/avatar/parameters/RightThumb",
        "IndexCurl": "/avatar/parameters/RightIndex",
        "MiddleCurl": "/avatar/parameters/RightMiddle",
        "RingCurl": "/avatar/parameters/RightRing",
        "PinkyCurl": "/avatar/parameters/RightLittle",
        "ThumbSplay": "/avatar/parameters/RightThumbSplay",
        "IndexSplay": "/avatar/parameters/RightIndexSplay",
        "MiddleSplay": "/avatar/parameters/RightMiddleSplay",
        "RingSplay": "/avatar/parameters/RightRingSplay",
    }
    OSC_AVAILABLE = True
except ImportError:
    logging.warning(
        "python-osc not found. VRChatOSC backend will not be available. Install with: pip install python-osc"
    )
    # Define dummy classes/vars if OSC is not available
    udp_client = None
    osc_bundle_builder = None
    osc_message_builder = None
    OSC_HAND_PARAMS_LEFT = {}
    OSC_HAND_PARAMS_RIGHT = {}
    OSC_AVAILABLE = False


# Local imports
try:
    # Import sendToSteamVR, shutdown, and potentially a new hand rotation helper
    from helpers import sendToSteamVR, shutdown, _get_landmark_coords, HAND_LANDMARK, WRIST_INDEX # Import helpers needed
    import numpy as np
except ImportError as e:
    logging.error(f"Failed to import local modules in backends.py: {e}")
    import sys
    sys.exit(1)


# --- Abstract Base Class ---

class Backend(ABC):
    """Abstract base class for different tracking backends."""

    @abstractmethod
    def onparamchanged(self, params) -> None:
        """Called when parameters (like smoothing) change."""
        ...

    @abstractmethod
    def connect(self, params) -> None:
        """Establishes connection to the backend."""
        ...

    @abstractmethod
    def updatepose(
        self,
        params,
        pose3d: np.ndarray,
        pose_rots: Optional[tuple], # Renamed from rots for clarity
        processed_hand_data: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]],
        face_landmarks: Optional[Any] # Accept face landmarks (currently unused)
    ) -> bool:
        """
        Sends the updated pose data to the backend.

        Args:
            params: Application parameters.
            pose3d: Numpy array (N, 3) of the processed 3D pose skeleton.
            pose_rots: Tuple containing rotations for hip, left leg, right leg (or None).
            processed_hand_data: Tuple containing (left_hand_dict, right_hand_dict) with finger data (or None).
            face_landmarks: MediaPipe face landmarks object (or None).

        Returns:
            bool: True if the update was successful or handled, False otherwise.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Closes the connection to the backend."""
        ...


# --- Dummy Backend ---

class DummyBackend(Backend):
    """A backend that does nothing, useful for testing without SteamVR or OSC."""

    def __init__(self, **kwargs):
        logging.info("Using Dummy Backend.")

    def onparamchanged(self, params) -> None:
        pass  # Nothing to do

    def connect(self, params) -> None:
        pass  # Nothing to do

    def updatepose(
        self,
        params,
        pose3d: np.ndarray,
        pose_rots: Optional[tuple],
        processed_hand_data: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]],
        face_landmarks: Optional[Any]
    ) -> bool:
        # Log pose data if needed for debugging
        # logging.debug(f"DummyBackend received pose data. Hip: {pose3d[6]}")
        # if params.use_hands and processed_hand_data and processed_hand_data[0]:
        #     logging.debug(f"DummyBackend received Left Hand data: {processed_hand_data[0]}")
        return True  # Always succeeds

    def disconnect(self) -> None:
        pass  # Nothing to do


# --- SteamVR Backend ---

class SteamVRBackend(Backend):
    """Backend implementation for sending tracker data to SteamVR via named pipe/socket."""

    def __init__(self, **kwargs):
        logging.info("Initializing SteamVR Backend.")
        # Initialize attributes for storing last known good HMD pose
        self.last_headsetpos: Optional[np.ndarray] = None
        self.last_headsetrot: Optional[R] = None
        self.stale_data_warning_logged: bool = False

    def _calculate_hand_rotation_from_landmarks(self, hand_landmarks) -> Optional[np.ndarray]:
        """
        Calculates a simple hand rotation quaternion using wrist, index MCP, and pinky MCP.
        More accurate than using pose landmarks.
        """
        if not hand_landmarks:
            return None

        wrist = _get_landmark_coords(hand_landmarks, WRIST_INDEX)
        index_mcp = _get_landmark_coords(hand_landmarks, HAND_LANDMARK.INDEX_FINGER_MCP)
        pinky_mcp = _get_landmark_coords(hand_landmarks, HAND_LANDMARK.PINKY_MCP)

        if wrist is None or index_mcp is None or pinky_mcp is None:
            # logging.debug("Missing landmarks for SteamVR hand rotation calculation.")
            return None

        try:
            # Vector from wrist to pinky (roughly forward/X axis)
            x_vec = pinky_mcp - wrist
            # Vector from wrist to index (defines plane with x_vec, roughly up/Y axis direction)
            w_vec = index_mcp - wrist

            x_norm = np.linalg.norm(x_vec)
            w_norm = np.linalg.norm(w_vec)
            if x_norm < 1e-6 or w_norm < 1e-6: return None
            x_vec /= x_norm
            w_vec /= w_norm

            # Calculate orthogonal axes
            z_vec = np.cross(x_vec, w_vec) # Roughly side/Z axis
            z_norm = np.linalg.norm(z_vec)
            if z_norm < 1e-6: return None
            z_vec /= z_norm

            y_vec = np.cross(z_vec, x_vec) # Recalculate Y axis to be orthogonal
            y_norm = np.linalg.norm(y_vec)
            if y_norm < 1e-6: return None
            y_vec /= y_norm

            # Construct rotation matrix (columns are X, Y, Z axes)
            rot_matrix = np.vstack((x_vec, y_vec, z_vec)).T
            quat = R.from_matrix(rot_matrix).as_quat() # [x, y, z, w]
            return quat

        except (ValueError, ZeroDivisionError) as e:
            logging.warning(f"Error calculating hand rotation from landmarks: {e}")
            return None


    def onparamchanged(self, params) -> None:
        """Sends updated smoothing settings to the SteamVR driver."""
        logging.info(
            f"SteamVRBackend: Updating settings - Smoothing={params.smoothing}, AdditionalSmoothing={params.additional_smoothing}"
        )
        # Note: The meaning of '50' in the settings command is unclear. Could be interpolation time ms?
        resp = sendToSteamVR(f"settings 50 {params.smoothing} {params.additional_smoothing}")
        if resp is None:
            logging.error("SteamVRBackend: Could not send 'settings' command to driver during onparamchanged.")

    def connect(self, params) -> None:
        """Connects to the SteamVR driver, checks/adds trackers, and sends initial settings."""
        logging.info("Connecting to SteamVR driver...")

        numtrackers_response = sendToSteamVR("numtrackers")
        if numtrackers_response is None:
            logging.error("Could not connect to SteamVR driver after multiple tries! Launch SteamVR and check driver status.")
            shutdown(params, exit_code=1)
            return

        logging.debug(f"Received 'numtrackers' response list: {numtrackers_response}")

        numtrackers = 0
        parsed_count = False
        try:
            if "numtrackers" in numtrackers_response:
                keyword_index = numtrackers_response.index("numtrackers")
                for i in range(keyword_index + 1, len(numtrackers_response)):
                    element = numtrackers_response[i]
                    if element.isdigit():
                        numtrackers = int(element)
                        logging.info(f"Driver reported {numtrackers} existing trackers (parsed).")
                        parsed_count = True
                        break
            if not parsed_count:
                logging.error(f"Could not parse tracker count from driver response: {numtrackers_response}")
                if any("." in item for item in numtrackers_response):
                    logging.warning("Response might contain a version number or other non-numeric data.")
                logging.warning("Assuming 0 existing trackers due to parsing failure.")
                numtrackers = 0
        except Exception as e:
            logging.exception(f"Unexpected error parsing 'numtrackers' response: {e}")
            logging.error(f"Raw response list was: {numtrackers_response}")
            shutdown(params, exit_code=1)
            return

        # Determine total trackers needed
        if params.preview_skeleton:
            totaltrackers = 23
        else:
            totaltrackers = 3
            if params.ignore_hip: totaltrackers -= 1
            if params.use_hands: totaltrackers += 2

        # Define roles
        roles = []
        if not params.preview_skeleton:
            if not params.ignore_hip: roles.append("TrackerRole_Waist")
            roles.append("TrackerRole_RightFoot")
            roles.append("TrackerRole_LeftFoot")
            if params.use_hands:
                roles.append("TrackerRole_LeftHand")
                roles.append("TrackerRole_RightHand")
        for i in range(len(roles), totaltrackers):
            roles.append("None")

        # Add missing trackers
        if numtrackers < totaltrackers:
            logging.info(f"Driver has {numtrackers} trackers, ensuring {totaltrackers} are present...")
            for i in range(numtrackers, totaltrackers):
                tracker_role = roles[i] if i < len(roles) else "None"
                tracker_name = f"MPTracker{i}"
                logging.info(f"Adding tracker {tracker_name} with role {tracker_role}")
                resp = sendToSteamVR(f"addtracker {tracker_name} {tracker_role}")
                if resp is None:
                    logging.error(f"Could not send 'addtracker {tracker_name}' command to SteamVR driver.")
                    shutdown(params, exit_code=1)
                    return
        elif numtrackers > totaltrackers:
            logging.warning(f"Driver reports {numtrackers} trackers, but only {totaltrackers} needed. Excess ignored.")
        else:
            logging.info(f"Driver reports {numtrackers} trackers, matching required count ({totaltrackers}).")

        self.onparamchanged(params)
        logging.info("Successfully connected to SteamVR driver and configured trackers.")

    def updatepose(
        self,
        params,
        pose3d: np.ndarray,
        pose_rots: Optional[tuple],
        processed_hand_data: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]],
        face_landmarks: Optional[Any] # Accept face landmarks (currently unused)
    ) -> bool:
        """Gets HMD pose, calculates offsets, and sends tracker updates to the SteamVR driver."""
        # Initialize last known good pose if not already done
        if self.last_headsetpos is None: self.last_headsetpos = np.array([0.0, 0.0, 0.0])
        if self.last_headsetrot is None: self.last_headsetrot = R.identity()

        headsetpos = self.last_headsetpos
        headsetrot = self.last_headsetrot
        new_data_received = False

        try:
            array = sendToSteamVR("getdevicepose 0")

            if array is None:
                if not self.stale_data_warning_logged:
                    logging.warning("Could not get HMD pose from SteamVR driver (connection failed?). Reusing last known HMD pose.")
                    self.stale_data_warning_logged = True
            else:
                if len(array) < 10 or array[0] == "error":
                    if not self.stale_data_warning_logged:
                        logging.warning(f"Unexpected HMD pose response format or error: {array}. Reusing last known HMD pose.")
                        self.stale_data_warning_logged = True
                else:
                    try:
                        def safe_float(value_str, default=0.0):
                            if not value_str: return default
                            try: return float(value_str)
                            except ValueError:
                                logging.warning(f"Could not convert '{value_str}' to float, using default {default}.")
                                return default

                        parsed_pos = np.array([safe_float(array[3]), safe_float(array[4]), safe_float(array[5])])
                        qx, qy, qz = safe_float(array[7]), safe_float(array[8]), safe_float(array[9])
                        qw = safe_float(array[6], default=1.0)
                        parsed_rot = R.from_quat([qx, qy, qz, qw])
                        quat_norm = np.linalg.norm([qx, qy, qz, qw])

                        if np.isclose(quat_norm, 0.0):
                            logging.warning(f"Parsed HMD quaternion is near zero {([qx, qy, qz, qw])}. Using last known rotation.")
                        else:
                            headsetpos = parsed_pos
                            headsetrot = parsed_rot
                            new_data_received = True
                    except IndexError:
                        logging.error(f"Index error parsing HMD pose response: {array}. Reusing last known HMD pose.")
                        if not self.stale_data_warning_logged: self.stale_data_warning_logged = True
                    except Exception as e:
                         logging.exception(f"Unexpected error parsing HMD pose data: {e}. Reusing last known HMD pose.")
                         if not self.stale_data_warning_logged: self.stale_data_warning_logged = True

            if new_data_received:
                self.last_headsetpos = headsetpos
                self.last_headsetrot = headsetrot
                self.stale_data_warning_logged = False

            # --- Continue with calculations ---
            neckoffset = headsetrot.apply(params.hmd_to_neck_offset)
            if params.recalibrate:
                logging.debug("Recalibration pending, skipping pose update.")
                return False

            pose3d_scaled = pose3d * params.posescale
            offset = pose3d_scaled[7] - (headsetpos + neckoffset) # Offset relative to mid-shoulder

            # --- Prepare Tracker Updates ---
            trackers_to_update = []
            if not params.preview_skeleton:
                steamvr_idx = 0
                tracker_updates: Dict[int, Dict[str, Any]] = {} # {steamvr_idx: {pose_idx: ..., rot: ...}}

                # Body Trackers (Hip, Feet)
                if not params.ignore_hip:
                    tracker_updates[steamvr_idx] = {"pose_idx": 6, "rot": pose_rots[0] if pose_rots else None} # MidHip
                    steamvr_idx += 1
                tracker_updates[steamvr_idx] = {"pose_idx": 5, "rot": pose_rots[2] if pose_rots else None} # R Ankle
                steamvr_idx += 1
                tracker_updates[steamvr_idx] = {"pose_idx": 0, "rot": pose_rots[1] if pose_rots else None} # L Ankle
                steamvr_idx += 1

                # Hand Trackers (if enabled)
                if params.use_hands:
                    left_hand_rot = None
                    right_hand_rot = None
                    # processed_hand_data is Tuple[Optional[Dict], Optional[Dict]]
                    # We need the raw landmarks passed from mediapipepose.py for rotation calculation
                    # Let's assume processed_hand_data now contains the raw landmarks:
                    # processed_hand_data = (results.left_hand_landmarks, results.right_hand_landmarks)
                    # **NOTE:** This requires changing the call in mediapipepose.py if it's sending curl dicts
                    if processed_hand_data and processed_hand_data[0]: # Left hand landmarks
                        left_hand_rot = self._calculate_hand_rotation_from_landmarks(processed_hand_data[0])
                    if processed_hand_data and processed_hand_data[1]: # Right hand landmarks
                        right_hand_rot = self._calculate_hand_rotation_from_landmarks(processed_hand_data[1])

                    tracker_updates[steamvr_idx] = {"pose_idx": 10, "rot": left_hand_rot} # L Wrist
                    steamvr_idx += 1
                    tracker_updates[steamvr_idx] = {"pose_idx": 15, "rot": right_hand_rot} # R Wrist
                    steamvr_idx += 1

                # Build update commands
                for idx, update_data in tracker_updates.items():
                    pose_idx = update_data["pose_idx"]
                    joint_pos = pose3d_scaled[pose_idx] - offset
                    quat = update_data["rot"] if update_data["rot"] is not None else np.array([0.0, 0.0, 0.0, 1.0])

                    if not isinstance(quat, np.ndarray): quat = np.array(quat)
                    norm = np.linalg.norm(quat)
                    if np.isclose(norm, 0.0): quat = np.array([0.0, 0.0, 0.0, 1.0])
                    elif not np.isclose(norm, 1.0): quat /= norm

                    cmd = (f"updatepose {idx} {joint_pos[0]:.4f} {joint_pos[1]:.4f} {joint_pos[2]:.4f} "
                           f"{quat[3]:.4f} {quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} "
                           f"{params.camera_latency:.4f} 0.8") # 0.8 = smoothing factor?
                    trackers_to_update.append(cmd)

            else: # Preview skeleton mode
                num_points_to_send = min(23, len(pose3d_scaled))
                for i in range(num_points_to_send):
                    joint_pos = pose3d_scaled[i] - offset
                    cmd = (f"updatepose {i} {joint_pos[0]:.4f} {joint_pos[1]:.4f} {joint_pos[2] - 2.0:.4f} " # Z offset for viz
                           f"1.0 0 0 0 {params.camera_latency:.4f} 0.8")
                    trackers_to_update.append(cmd)

            # Send all update commands
            for cmd in trackers_to_update:
                resp = sendToSteamVR(cmd, num_tries=1)
                if resp is None or resp[0] == "error":
                     logging.warning(f"Failed to send command '{cmd.split(' ')[0]} {cmd.split(' ')[1]}...' to driver. Response: {resp}")

        except (IndexError, ValueError, TypeError, ZeroDivisionError) as e:
             logging.exception(f"Error during SteamVR updatepose processing: {e}")
             return False # Indicate update failed

        return True # Indicate update succeeded


    def disconnect(self) -> None:
        """Placeholder for disconnection logic if needed."""
        logging.info("Disconnecting from SteamVR driver (if applicable).")
        pass

# --- VRChat OSC Backend ---

# Helper functions for OSC message building (moved outside class)
def _build_osc_message(address: str, value: float) -> Optional[Any]:
    """Builds a single OSC message with one float argument."""
    if osc_message_builder is None: return None
    try:
        builder = osc_message_builder.OscMessageBuilder(address=address)
        builder.add_arg(float(value)) # VRChat OSC expects floats
        return builder.build()
    except Exception as e:
        logging.error(f"Failed to build OSC message for {address}: {e}")
        return None

def _build_osc_tracking_bundle(trackers: List[Dict[str, Any]]) -> Optional[Any]:
    """Builds an OSC bundle for VRChat tracker positions/rotations."""
    if osc_bundle_builder is None or not trackers: return None
    try:
        builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
        # Head message (position only)
        head_msg = _osc_build_msg(trackers[0]['name'], "position", trackers[0]['position'])
        if head_msg: builder.add_content(head_msg)

        # Other trackers (position and rotation)
        for tracker in trackers[1:]:
            pos_msg = _osc_build_msg(tracker['name'], "position", tracker['position'])
            rot_msg = _osc_build_msg(tracker['name'], "rotation", tracker['rotation'])
            if pos_msg: builder.add_content(pos_msg)
            if rot_msg: builder.add_content(rot_msg)
        return builder.build()
    except Exception as e:
        logging.error(f"Failed to build OSC tracking bundle: {e}")
        return None

def _build_osc_parameter_bundle(hand_data: Dict[str, float], param_map: Dict[str, str]) -> Optional[Any]:
    """Builds an OSC bundle for VRChat avatar parameters (e.g., finger curls)."""
    if osc_bundle_builder is None or not hand_data: return None
    try:
        builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
        for key, value in hand_data.items():
            if value is not None and key in param_map:
                address = param_map[key]
                msg = _build_osc_message(address, value)
                if msg:
                    builder.add_content(msg)
        return builder.build()
    except Exception as e:
        logging.error(f"Failed to build OSC parameter bundle: {e}")
        return None


class VRChatOSCBackend(Backend):
    """Backend implementation for sending tracker data to VRChat via OSC."""

    def __init__(self, **kwargs):
        logging.info("Initializing VRChat OSC Backend.")
        self.client: Optional[udp_client.UDPClient] = None
        self.prev_pose3d: np.ndarray = np.zeros((29, 3), dtype=np.float32) # For smoothing

    def onparamchanged(self, params) -> None:
        # Smoothing is handled internally in updatepose for OSC
        pass

    def connect(self, params) -> None:
        """Initializes the UDP client for sending OSC messages."""
        if not OSC_AVAILABLE:
             logging.error("VRChatOSCBackend cannot connect: python-osc library not found.")
             shutdown(params, exit_code=1)
             return

        ip = params.backend_ip if hasattr(params, "backend_ip") else "127.0.0.1"
        port = params.backend_port if hasattr(params, "backend_port") else 9000
        logging.info(f"Connecting VRChat OSC client to {ip}:{port}")
        try:
            self.client = udp_client.UDPClient(ip, port)
        except Exception as e:
            logging.error(f"Failed to create UDP client for VRChat OSC: {e}")
            shutdown(params, exit_code=1)

    def updatepose(
        self,
        params,
        pose3d: np.ndarray,
        pose_rots: Optional[tuple],
        processed_hand_data: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]],
        face_landmarks: Optional[Any] # Accept face landmarks (currently unused)
    ) -> bool:
        """Applies smoothing, calculates offsets, formats, and sends tracker/parameter data via OSC."""
        if self.client is None:
            logging.warning("VRChat OSC client not initialized. Skipping update.")
            return False

        try:
            # --- Coordinate System Adjustments for VRChat OSC ---
            pose3d_osc = pose3d.copy()
            pose3d_osc[:, 0] *= -1.0 # Flip X (Left -> Right)
            pose3d_osc[:, 2] *= -1.0 # Flip Z (Towards -> Forward)

            # Apply smoothing
            alpha = np.clip(params.additional_smoothing, 0.0, 1.0)
            pose3d_smoothed = self.prev_pose3d * alpha + pose3d_osc * (1.0 - alpha)
            self.prev_pose3d = pose3d_smoothed

            # --- Offset Calculation (Relative to playspace origin) ---
            headsetpos = np.array([0.0, 0.0, 0.0]) # Assume HMD is origin for offset calc
            headsetrot = R.identity()
            neckoffset = headsetrot.apply(params.hmd_to_neck_offset)

            if params.recalibrate:
                logging.debug("Recalibration pending, skipping OSC pose update.")
                return False

            pose3d_final = pose3d_smoothed * params.posescale
            offset = pose3d_final[7] - (headsetpos + neckoffset) # Offset relative to mid-shoulder

            # --- Prepare and Send OSC Tracking Bundle ---
            if not params.preview_skeleton:
                trackers_osc: List[Dict[str, Any]] = []
                trackers_osc.append({"name": "head", "position": [0.0, 0.0, 0.0]}) # Head required

                osc_idx = 1
                tracker_map: Dict[int, Dict[str, Any]] = {} # {pose_idx: {osc_name: ..., rot: ...}}

                # Body Trackers
                if not params.ignore_hip:
                    tracker_map[6] = {"osc_name": str(osc_idx), "rot": pose_rots[0] if pose_rots else None} # MidHip
                    osc_idx += 1
                tracker_map[5] = {"osc_name": str(osc_idx), "rot": pose_rots[2] if pose_rots else None} # R Ankle
                osc_idx += 1
                tracker_map[0] = {"osc_name": str(osc_idx), "rot": pose_rots[1] if pose_rots else None} # L Ankle
                osc_idx += 1
                # Note: OSC Hands are usually handled by parameters, not trackers 4 & 5

                # Build tracker list for bundle
                for pose_idx, osc_data in tracker_map.items():
                    position = pose3d_final[pose_idx] - offset
                    quat = osc_data["rot"] if osc_data["rot"] is not None else np.array([0.0, 0.0, 0.0, 1.0])
                    if not isinstance(quat, np.ndarray): quat = np.array(quat)

                    # Convert rotation to Euler angles (degrees) for OSC (ZXY order)
                    try:
                        norm = np.linalg.norm(quat)
                        if np.isclose(norm, 0.0): final_rotation = [0.0, 0.0, 0.0]
                        else:
                            if not np.isclose(norm, 1.0): quat /= norm
                            rotation_euler = R.from_quat(quat).as_euler("zxy", degrees=True)
                            final_rotation = [rotation_euler[0], rotation_euler[1], rotation_euler[2]] # Yaw, Pitch, Roll
                    except ValueError: final_rotation = [0.0, 0.0, 0.0]

                    trackers_osc.append({
                        "name": osc_data['osc_name'],
                        "position": position.tolist(),
                        "rotation": final_rotation
                    })

                # Build and send the tracking bundle
                tracking_bundle = _build_osc_tracking_bundle(trackers_osc)
                if tracking_bundle:
                    self.client.send(tracking_bundle)

            else: # Preview skeleton mode not implemented for OSC
                logging.log_every_n(logging.WARNING, "OSC Preview skeleton mode is not implemented.", 100) # Log less frequently


            # --- Prepare and Send OSC Parameter Bundles (Hands) ---
            if params.use_hands and processed_hand_data:
                left_hand_dict, right_hand_dict = processed_hand_data

                # Send Left Hand Parameters
                if left_hand_dict:
                    left_param_bundle = _build_osc_parameter_bundle(left_hand_dict, OSC_HAND_PARAMS_LEFT)
                    if left_param_bundle:
                        self.client.send(left_param_bundle)

                # Send Right Hand Parameters
                if right_hand_dict:
                    right_param_bundle = _build_osc_parameter_bundle(right_hand_dict, OSC_HAND_PARAMS_RIGHT)
                    if right_param_bundle:
                        self.client.send(right_param_bundle)

            # --- Face Landmark Processing (Placeholder) ---
            if params.use_face and face_landmarks:
                # Future: Process face_landmarks and send relevant OSC messages
                # (e.g., for eye tracking, blend shapes if supported)
                pass


        except (IndexError, ValueError, TypeError, ZeroDivisionError, AttributeError) as e:
            logging.exception(f"Error during VRChat OSC updatepose processing: {e}")
            return False # Indicate update failed

        return True # Indicate update succeeded

    def disconnect(self) -> None:
        """Closes the UDP client."""
        logging.info("Disconnecting VRChat OSC client.")
        self.client = None
