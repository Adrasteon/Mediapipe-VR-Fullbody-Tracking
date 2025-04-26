# helpers.py Utility functions for pose processing, math, communication, and camera handling.
import time
import socket
import cv2
import numpy as np
import sys
import threading
import logging # Use logging module
from sys import platform
from typing import Optional, List, Tuple, Dict, Any # For type hinting

# Third-party imports (ensure these are installed)
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    logging.error("Scipy not found. Please install it: pip install scipy")
    sys.exit(1)
try:
    import mediapipe as mp # Import mediapipe for landmark definitions
except ImportError:
    logging.error("Mediapipe not found. Please install it: pip install mediapipe")
    sys.exit(1)

# --- Constants ---
# Define EDGES if it's used by draw_pose (assuming it's a list of pairs)
# Example: EDGES = [[0, 1], [1, 2], ...] # Replace with actual skeleton edges if needed
EDGES = [] # Placeholder - Define your skeleton edges here if draw_pose is used

# MediaPipe Hand Landmark Indices (from https://google.github.io/mediapipe/solutions/hands.html#hand-landmark-model)
# Used for curl/splay calculations
HAND_LANDMARK = mp.solutions.hands.HandLandmark
FINGER_INDICES = {
    'thumb': [HAND_LANDMARK.THUMB_CMC, HAND_LANDMARK.THUMB_MCP, HAND_LANDMARK.THUMB_IP, HAND_LANDMARK.THUMB_TIP],
    'index': [HAND_LANDMARK.INDEX_FINGER_MCP, HAND_LANDMARK.INDEX_FINGER_PIP, HAND_LANDMARK.INDEX_FINGER_DIP, HAND_LANDMARK.INDEX_FINGER_TIP],
    'middle': [HAND_LANDMARK.MIDDLE_FINGER_MCP, HAND_LANDMARK.MIDDLE_FINGER_PIP, HAND_LANDMARK.MIDDLE_FINGER_DIP, HAND_LANDMARK.MIDDLE_FINGER_TIP],
    'ring': [HAND_LANDMARK.RING_FINGER_MCP, HAND_LANDMARK.RING_FINGER_PIP, HAND_LANDMARK.RING_FINGER_DIP, HAND_LANDMARK.RING_FINGER_TIP],
    'pinky': [HAND_LANDMARK.PINKY_MCP, HAND_LANDMARK.PINKY_PIP, HAND_LANDMARK.PINKY_DIP, HAND_LANDMARK.PINKY_TIP]
}
WRIST_INDEX = HAND_LANDMARK.WRIST

# --- Pose Processing ---

def draw_pose(frame: np.ndarray, pose: np.ndarray, size: int) -> None:
    """Draws the 2D pose skeleton on the frame."""
    # Check if EDGES is defined and pose has enough points
    if not EDGES or pose.shape[0] < max(max(edge) for edge in EDGES) + 1:
        logging.warning("EDGES not defined or pose array too small for drawing.")
        return

    pose_scaled = (pose * size).astype(int)
    for sk in EDGES:
        pt1 = (pose_scaled[sk[0], 1], pose_scaled[sk[0], 0]) # Assuming (y, x) order in pose
        pt2 = (pose_scaled[sk[1], 1], pose_scaled[sk[1], 0])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

def mediapipeTo3dpose(lms) -> np.ndarray:
    """
    Converts MediaPipe 3D world POSE landmarks to a custom 29-point skeleton format.

    Args:
        lms: MediaPipe POSE landmarks object (e.g., results.pose_world_landmarks.landmark).

    Returns:
        A numpy array (29, 3) representing the 3D pose.
    """
    # 33 pose landmarks as in https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
    pose = np.zeros((29, 3), dtype=np.float32) # Use float32 for consistency

    # --- Body ---
    pose[0] = [lms[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x, lms[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y, lms[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].z]
    pose[1] = [lms[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x, lms[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y, lms[mp.solutions.pose.PoseLandmark.LEFT_KNEE].z]
    pose[2] = [lms[mp.solutions.pose.PoseLandmark.LEFT_HIP].x, lms[mp.solutions.pose.PoseLandmark.LEFT_HIP].y, lms[mp.solutions.pose.PoseLandmark.LEFT_HIP].z]
    pose[3] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_HIP].z]
    pose[4] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].z]
    pose[5] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].z]

    # Calculate mid-hip (Point 6)
    pose[6] = (pose[2] + pose[3]) / 2

    # Calculate mid-shoulder (Point 7)
    pose[7] = [(lms[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x + lms[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
               (lms[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y + lms[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y) / 2,
               (lms[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].z + lms[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].z) / 2]

    # Calculate neck base (Point 16) - Approximation using mid-shoulder
    pose[16] = pose[7]

    # --- Head ---
    pose[8] = [(lms[mp.solutions.pose.PoseLandmark.MOUTH_LEFT].x + lms[mp.solutions.pose.PoseLandmark.MOUTH_RIGHT].x) / 2,
               (lms[mp.solutions.pose.PoseLandmark.MOUTH_LEFT].y + lms[mp.solutions.pose.PoseLandmark.MOUTH_RIGHT].y) / 2,
               (lms[mp.solutions.pose.PoseLandmark.MOUTH_LEFT].z + lms[mp.solutions.pose.PoseLandmark.MOUTH_RIGHT].z) / 2]
    pose[9] = [lms[mp.solutions.pose.PoseLandmark.NOSE].x, lms[mp.solutions.pose.PoseLandmark.NOSE].y, lms[mp.solutions.pose.PoseLandmark.NOSE].z]

    # --- Arms ---
    pose[10] = [lms[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x, lms[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y, lms[mp.solutions.pose.PoseLandmark.LEFT_WRIST].z]
    pose[11] = [lms[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x, lms[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y, lms[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].z]
    pose[12] = [lms[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x, lms[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y, lms[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].z]
    pose[13] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].z]
    pose[14] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].z]
    pose[15] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].z]

    # --- Feet Rotation Points (using heel, foot index, ankle from POSE landmarks) ---
    pose[17] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX].z]
    pose[18] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_HEEL].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_HEEL].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_HEEL].z]
    pose[19] = pose[5] # RIGHT_ANKLE
    pose[20] = [lms[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].x, lms[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].y, lms[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].z]
    pose[21] = [lms[mp.solutions.pose.PoseLandmark.LEFT_HEEL].x, lms[mp.solutions.pose.PoseLandmark.LEFT_HEEL].y, lms[mp.solutions.pose.PoseLandmark.LEFT_HEEL].z]
    pose[22] = pose[0] # LEFT_ANKLE

    # --- Hand Rotation Points (using wrist, index, pinky from POSE landmarks) ---
    # These are less accurate than using Holistic hand landmarks but kept for compatibility
    pose[23] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_PINKY].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_PINKY].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_PINKY].z]
    pose[24] = pose[15] # RIGHT_WRIST
    pose[25] = [lms[mp.solutions.pose.PoseLandmark.RIGHT_INDEX].x, lms[mp.solutions.pose.PoseLandmark.RIGHT_INDEX].y, lms[mp.solutions.pose.PoseLandmark.RIGHT_INDEX].z]
    pose[26] = [lms[mp.solutions.pose.PoseLandmark.LEFT_PINKY].x, lms[mp.solutions.pose.PoseLandmark.LEFT_PINKY].y, lms[mp.solutions.pose.PoseLandmark.LEFT_PINKY].z]
    pose[27] = pose[10] # LEFT_WRIST
    pose[28] = [lms[mp.solutions.pose.PoseLandmark.LEFT_INDEX].x, lms[mp.solutions.pose.PoseLandmark.LEFT_INDEX].y, lms[mp.solutions.pose.PoseLandmark.LEFT_INDEX].z]

    return pose

# --- Coordinate Transformations ---

def keypoints_to_original(scale: float, center: Tuple[float, float], points: np.ndarray) -> np.ndarray:
    """Converts normalized keypoints back to original image coordinates."""
    scores = points[:, 2].copy() # Preserve scores
    points_orig = points[:, :2].copy() # Work on xy coordinates
    points_orig -= 0.5
    points_orig *= scale
    points_orig[:, 0] += center[0]
    points_orig[:, 1] += center[1]

    # Combine coordinates and scores
    result = np.zeros_like(points)
    result[:, :2] = points_orig
    result[:, 2] = scores
    return result

def normalize_screen_coordinates(X: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalizes screen coordinates to [-1, 1] range while preserving aspect ratio."""
    assert X.shape[-1] == 2, "Input must have 2 columns (x, y)"
    if w == 0 or h == 0:
        logging.warning("Screen width or height is zero, cannot normalize.")
        return X # Return original if dimensions are invalid

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    # Map to [0, 2] range first based on width
    normalized_X = X / w * 2
    # Center around 0 -> [-1, 1] for x
    normalized_X[:, 0] -= 1
    # Adjust y based on aspect ratio and center -> [-h/w, h/w]
    normalized_X[:, 1] -= h / w
    return normalized_X

# --- Rotation Calculation ---

def _calculate_rotation_matrix(p_forward: np.ndarray, p_back: np.ndarray, p_up: np.ndarray, axes_order: str = 'xyz') -> Optional[np.ndarray]:
    """Helper to calculate rotation matrix from 3 points defining axes."""
    x_vec = p_forward - p_back
    w_vec = p_up - p_back # Vector roughly defining the plane

    # Normalize vectors
    x_norm = np.linalg.norm(x_vec)
    w_norm = np.linalg.norm(w_vec)
    if x_norm < 1e-6 or w_norm < 1e-6:
        logging.debug("Skipping rotation calculation: Input vectors too small.")
        return None

    x_vec /= x_norm
    w_vec /= w_norm # Not strictly necessary to normalize w_vec if only used for cross product direction

    # Calculate orthogonal axes using cross products
    z_vec = np.cross(x_vec, w_vec)
    z_norm = np.linalg.norm(z_vec)
    if z_norm < 1e-6:
        logging.debug("Skipping rotation calculation: Cross product resulted in zero vector (collinear points?).")
        return None
    z_vec /= z_norm

    y_vec = np.cross(z_vec, x_vec)
    y_norm = np.linalg.norm(y_vec) # Should be close to 1 if z and x are orthogonal and normalized
    if y_norm < 1e-6:
         logging.debug("Skipping rotation calculation: Final cross product failed.")
         return None
    y_vec /= y_norm

    # Construct rotation matrix based on desired axis order
    if axes_order == 'xyz':
        rot_matrix = np.vstack((x_vec, y_vec, z_vec)).T
    elif axes_order == 'zxy': # As used in get_rot_hands
        rot_matrix = np.vstack((z_vec, y_vec, -x_vec)).T # Note the -x
    elif axes_order == 'xzy': # As used in get_rot_mediapipe/get_rot
        rot_matrix = np.vstack((x_vec, y_vec, z_vec)).T
    else:
        logging.error(f"Unsupported axes_order: {axes_order}")
        return None

    return rot_matrix


def get_rot_hands(pose3d: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculates basic hand rotations based on POSE landmarks (wrist, pinky, index).
    NOTE: This is less accurate than using Holistic hand landmarks.
    """
    # Left Hand (indices 26, 27, 28 -> pinky, wrist, index from POSE)
    l_hand_rot_mat = _calculate_rotation_matrix(pose3d[26], pose3d[27], pose3d[28], axes_order='zxy')
    # Right Hand (indices 23, 24, 25 -> pinky, wrist, index from POSE)
    r_hand_rot_mat = _calculate_rotation_matrix(pose3d[23], pose3d[24], pose3d[25], axes_order='zxy')

    if l_hand_rot_mat is None or r_hand_rot_mat is None:
        return None

    try:
        l_hand_rot = R.from_matrix(l_hand_rot_mat).as_quat()
        r_hand_rot = R.from_matrix(r_hand_rot_mat).as_quat()
        # Scipy quat format: [x, y, z, w]
        return l_hand_rot, r_hand_rot
    except ValueError as e:
        logging.warning(f"Scipy rotation conversion failed for hands (using pose landmarks): {e}")
        return None


def get_rot_mediapipe(pose3d: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates hip and foot rotations using MediaPipe POSE foot landmarks."""
    # Hip (indices 2, 3, 16 -> left_hip, right_hip, neck_base/mid_shoulder)
    hip_rot_mat = _calculate_rotation_matrix(pose3d[3], pose3d[2], pose3d[16], axes_order='xzy')
    # Left Foot (indices 20, 21, 22 -> foot_index, heel, ankle from POSE)
    l_foot_rot_mat = _calculate_rotation_matrix(pose3d[20], pose3d[21], pose3d[22], axes_order='xzy')
    # Right Foot (indices 17, 18, 19 -> foot_index, heel, ankle from POSE)
    r_foot_rot_mat = _calculate_rotation_matrix(pose3d[17], pose3d[18], pose3d[19], axes_order='xzy')

    if hip_rot_mat is None or l_foot_rot_mat is None or r_foot_rot_mat is None:
        return None

    try:
        hip_rot = R.from_matrix(hip_rot_mat).as_quat()
        l_foot_rot = R.from_matrix(l_foot_rot_mat).as_quat()
        r_foot_rot = R.from_matrix(r_foot_rot_mat).as_quat()
        return hip_rot, l_foot_rot, r_foot_rot
    except ValueError as e:
        logging.warning(f"Scipy rotation conversion failed for mediapipe feet: {e}")
        return None


def get_rot(pose3d: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates hip and leg rotations using standard skeleton joints from POSE landmarks."""
    # Indices for standard skeleton
    hip_left = 2
    hip_right = 3
    hip_up = 16 # Using neck_base/mid_shoulder as 'up' reference for hip
    knee_left = 1
    knee_right = 4
    ankle_left = 0
    ankle_right = 5

    # Hip Rotation
    hip_rot_mat = _calculate_rotation_matrix(pose3d[hip_right], pose3d[hip_left], pose3d[hip_up], axes_order='xzy')

    # Leg Rotations (using knee-ankle as primary axis 'y')
    # Right Leg
    y_r = pose3d[knee_right] - pose3d[ankle_right]
    w_r = pose3d[hip_right] - pose3d[ankle_right] # Vector within the plane of the leg
    z_r = np.cross(w_r, y_r)
    z_r_norm = np.linalg.norm(z_r)
    if z_r_norm < 1e-6: # Handle case where points might be collinear
        w_r_alt = pose3d[hip_left] - pose3d[ankle_right]
        z_r = np.cross(w_r_alt, y_r)
        z_r_norm = np.linalg.norm(z_r)

    leg_r_rot_mat = None
    if z_r_norm >= 1e-6:
        z_r /= z_r_norm
        y_r_norm = np.linalg.norm(y_r)
        if y_r_norm >= 1e-6:
            y_r /= y_r_norm
            x_r = np.cross(y_r, z_r)
            x_r_norm = np.linalg.norm(x_r)
            if x_r_norm >= 1e-6:
                x_r /= x_r_norm
                leg_r_rot_mat = np.vstack((x_r, y_r, z_r)).T
            else:
                logging.debug("Skipping right leg rotation: x_r vector too small.")
        else:
             logging.debug("Skipping right leg rotation: y_r vector too small.")
    else:
        logging.debug("Skipping right leg rotation: z_r vector too small (collinear points?).")


    # Left Leg
    y_l = pose3d[knee_left] - pose3d[ankle_left]
    w_l = pose3d[hip_left] - pose3d[ankle_left]
    z_l = np.cross(w_l, y_l)
    z_l_norm = np.linalg.norm(z_l)
    if z_l_norm < 1e-6:
        w_l_alt = pose3d[hip_right] - pose3d[ankle_left]
        z_l = np.cross(w_l_alt, y_l)
        z_l_norm = np.linalg.norm(z_l)

    leg_l_rot_mat = None
    if z_l_norm >= 1e-6:
        z_l /= z_l_norm
        y_l_norm = np.linalg.norm(y_l)
        if y_l_norm >= 1e-6:
            y_l /= y_l_norm
            x_l = np.cross(y_l, z_l)
            x_l_norm = np.linalg.norm(x_l)
            if x_l_norm >= 1e-6:
                x_l /= x_l_norm
                leg_l_rot_mat = np.vstack((x_l, y_l, z_l)).T
            else:
                logging.debug("Skipping left leg rotation: x_l vector too small.")
        else:
             logging.debug("Skipping left leg rotation: y_l vector too small.")
    else:
        logging.debug("Skipping left leg rotation: z_l vector too small (collinear points?).")


    if hip_rot_mat is None or leg_l_rot_mat is None or leg_r_rot_mat is None:
        logging.warning("Failed to calculate one or more rotation matrices in get_rot.")
        return None

    try:
        rot_hip = R.from_matrix(hip_rot_mat).as_quat()
        rot_leg_l = R.from_matrix(leg_l_rot_mat).as_quat()
        rot_leg_r = R.from_matrix(leg_r_rot_mat).as_quat()
        return rot_hip, rot_leg_l, rot_leg_r
    except ValueError as e:
        logging.warning(f"Scipy rotation conversion failed in get_rot: {e}")
        return None

# --- Hand Landmark Processing (Holistic) ---

def _get_landmark_coords(landmarks, index: int) -> Optional[np.ndarray]:
    """Safely gets the x, y, z coordinates of a landmark."""
    if landmarks and len(landmarks.landmark) > index:
        lm = landmarks.landmark[index]
        return np.array([lm.x, lm.y, lm.z])
    return None

def calculate_finger_curl(landmarks, joint_indices: List[int]) -> Optional[float]:
    """
    Calculates the curl of a finger based on the angle between segments.
    Uses PIP, DIP, and TIP joints relative to MCP.

    Args:
        landmarks: MediaPipe hand landmarks (NormalizedLandmarkList).
        joint_indices: List of [MCP, PIP, DIP, TIP] indices for the finger.

    Returns:
        Curl value (0.0 = straight, 1.0 = fully curled), or None on error.
    """
    mcp_coord = _get_landmark_coords(landmarks, joint_indices[0])
    pip_coord = _get_landmark_coords(landmarks, joint_indices[1])
    dip_coord = _get_landmark_coords(landmarks, joint_indices[2])
    tip_coord = _get_landmark_coords(landmarks, joint_indices[3])

    if any(coord is None for coord in [mcp_coord, pip_coord, dip_coord, tip_coord]):
        # logging.debug("Missing landmarks for finger curl calculation.")
        return None

    try:
        # Vectors for the two main segments (MCP->PIP and PIP->DIP or DIP->TIP)
        vec_mcp_pip = pip_coord - mcp_coord
        vec_pip_dip = dip_coord - pip_coord
        vec_dip_tip = tip_coord - dip_coord

        # Normalize vectors
        norm_mcp_pip = np.linalg.norm(vec_mcp_pip)
        norm_pip_dip = np.linalg.norm(vec_pip_dip)
        norm_dip_tip = np.linalg.norm(vec_dip_tip)

        if norm_mcp_pip < 1e-6 or norm_pip_dip < 1e-6 or norm_dip_tip < 1e-6:
            # logging.debug("Segment length too small for curl calculation.")
            return None

        vec_mcp_pip /= norm_mcp_pip
        vec_pip_dip /= norm_pip_dip
        vec_dip_tip /= norm_dip_tip

        # Angle at PIP joint (between MCP->PIP and PIP->DIP)
        dot_pip = np.dot(vec_mcp_pip, vec_pip_dip)
        angle_pip = np.arccos(np.clip(dot_pip, -1.0, 1.0))

        # Angle at DIP joint (between PIP->DIP and DIP->TIP)
        dot_dip = np.dot(vec_pip_dip, vec_dip_tip)
        angle_dip = np.arccos(np.clip(dot_dip, -1.0, 1.0))

        # Combine angles and normalize (max angle is ~pi for each joint, total ~2*pi)
        # Simple average normalization, might need tuning per finger/avatar
        total_angle = angle_pip + angle_dip
        curl = np.clip(total_angle / (np.pi * 1.5), 0.0, 1.0) # Normalize based on ~270 degrees max curl?

        return curl

    except Exception as e:
        logging.exception(f"Error calculating finger curl: {e}")
        return None

def calculate_finger_splay(landmarks, finger_mcp_idx: int, adjacent_mcp_idx: int) -> Optional[float]:
    """
    Calculates finger splay based on the angle between finger MCPs relative to the wrist.
    More robust than the previous projection method.

    Args:
        landmarks: MediaPipe hand landmarks.
        finger_mcp_idx: Index of the finger's MCP joint.
        adjacent_mcp_idx: Index of the adjacent finger's MCP joint (towards the center of the hand).

    Returns:
        Splay value (approx 0.0 = together, 1.0 = spread), or None on error.
    """
    wrist_coord = _get_landmark_coords(landmarks, WRIST_INDEX)
    mcp1_coord = _get_landmark_coords(landmarks, finger_mcp_idx)
    mcp2_coord = _get_landmark_coords(landmarks, adjacent_mcp_idx)

    if any(coord is None for coord in [wrist_coord, mcp1_coord, mcp2_coord]):
        # logging.debug("Missing landmarks for finger splay calculation.")
        return None

    try:
        # Vectors from wrist to MCPs
        vec_wrist_mcp1 = mcp1_coord - wrist_coord
        vec_wrist_mcp2 = mcp2_coord - wrist_coord

        norm1 = np.linalg.norm(vec_wrist_mcp1)
        norm2 = np.linalg.norm(vec_wrist_mcp2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            # logging.debug("Segment length too small for splay calculation.")
            return None

        vec_wrist_mcp1 /= norm1
        vec_wrist_mcp2 /= norm2

        # Angle between the vectors
        dot_product = np.dot(vec_wrist_mcp1, vec_wrist_mcp2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        # Normalize angle (needs tuning based on max expected splay angle)
        # Assuming max splay angle is around 45 degrees (pi/4 radians)
        max_splay_angle = np.pi / 4.0
        splay = np.clip(angle / max_splay_angle, 0.0, 1.0)

        return splay

    except Exception as e:
        logging.exception(f"Error calculating finger splay: {e}")
        return None


def process_holistic_hands(left_hand_landmarks, right_hand_landmarks) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """
    Processes Holistic hand landmarks to calculate finger curls and splays for OSC.

    Args:
        left_hand_landmarks: MediaPipe hand landmarks for the left hand.
        right_hand_landmarks: MediaPipe hand landmarks for the right hand.

    Returns:
        A tuple containing dictionaries for left and right hand data, suitable for OSC parameters.
        Returns (None, None) if no hands are tracked or errors occur.
    """
    left_data: Dict[str, Optional[float]] = {}
    right_data: Dict[str, Optional[float]] = {}

    # --- Left Hand ---
    if left_hand_landmarks:
        # Curls (Thumb uses PIP, IP, TIP relative to MCP)
        left_data['ThumbCurl'] = calculate_finger_curl(left_hand_landmarks, FINGER_INDICES['thumb'][1:]) # Use MCP, IP, TIP for thumb angle? Needs testing.
        left_data['IndexCurl'] = calculate_finger_curl(left_hand_landmarks, FINGER_INDICES['index'])
        left_data['MiddleCurl'] = calculate_finger_curl(left_hand_landmarks, FINGER_INDICES['middle'])
        left_data['RingCurl'] = calculate_finger_curl(left_hand_landmarks, FINGER_INDICES['ring'])
        left_data['PinkyCurl'] = calculate_finger_curl(left_hand_landmarks, FINGER_INDICES['pinky'])

        # Splays (relative to adjacent finger towards the middle)
        # Thumb splay is often handled differently (angle relative to index MCP?) - simplified here
        left_data['ThumbSplay'] = calculate_finger_splay(left_hand_landmarks, FINGER_INDICES['thumb'][1], FINGER_INDICES['index'][0]) # Thumb MCP vs Index MCP
        left_data['IndexSplay'] = calculate_finger_splay(left_hand_landmarks, FINGER_INDICES['index'][0], FINGER_INDICES['middle'][0])
        left_data['MiddleSplay'] = calculate_finger_splay(left_hand_landmarks, FINGER_INDICES['middle'][0], FINGER_INDICES['ring'][0])
        left_data['RingSplay'] = calculate_finger_splay(left_hand_landmarks, FINGER_INDICES['ring'][0], FINGER_INDICES['pinky'][0])
        # Pinky splay could be relative to ring, or maybe a fixed axis if needed

    # --- Right Hand ---
    if right_hand_landmarks:
        # Curls
        right_data['ThumbCurl'] = calculate_finger_curl(right_hand_landmarks, FINGER_INDICES['thumb'][1:])
        right_data['IndexCurl'] = calculate_finger_curl(right_hand_landmarks, FINGER_INDICES['index'])
        right_data['MiddleCurl'] = calculate_finger_curl(right_hand_landmarks, FINGER_INDICES['middle'])
        right_data['RingCurl'] = calculate_finger_curl(right_hand_landmarks, FINGER_INDICES['ring'])
        right_data['PinkyCurl'] = calculate_finger_curl(right_hand_landmarks, FINGER_INDICES['pinky'])

        # Splays
        right_data['ThumbSplay'] = calculate_finger_splay(right_hand_landmarks, FINGER_INDICES['thumb'][1], FINGER_INDICES['index'][0])
        right_data['IndexSplay'] = calculate_finger_splay(right_hand_landmarks, FINGER_INDICES['index'][0], FINGER_INDICES['middle'][0])
        right_data['MiddleSplay'] = calculate_finger_splay(right_hand_landmarks, FINGER_INDICES['middle'][0], FINGER_INDICES['ring'][0])
        right_data['RingSplay'] = calculate_finger_splay(right_hand_landmarks, FINGER_INDICES['ring'][0], FINGER_INDICES['pinky'][0])

    # Filter out None values before returning (optional, depends on backend needs)
    # left_data_filtered = {k: v for k, v in left_data.items() if v is not None}
    # right_data_filtered = {k: v for k, v in right_data.items() if v is not None}

    return left_data if left_data else None, right_data if right_data else None


# --- SteamVR Communication (Pipe/Socket) ---

def sendToPipe(text: str) -> bytes:
    """Sends text to the named pipe (Windows) or Unix socket (Linux)."""
    data_to_send = text.encode('utf-8') + b'\0' # Ensure null termination

    if platform.startswith('win32'):
        pipe_path = r'\.\pipe\ApriltagPipeIn'
        try:
            # Use 'with' statement for automatic closing
            with open(pipe_path, 'rb+', buffering=0) as pipe:
                pipe.write(data_to_send)
                # Consider adding a timeout for reading
                resp = pipe.read(1024) # Read response
            return resp
        except FileNotFoundError:
            logging.error(f"Pipe not found: {pipe_path}. Is the SteamVR driver running?")
            raise # Re-raise the exception to be caught by caller
        except Exception as e:
            logging.error(f"Error communicating with pipe {pipe_path}: {e}")
            raise # Re-raise

    elif platform.startswith('linux'):
        socket_path = "/tmp/ApriltagPipeIn"
        client = None # Initialize client to None
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            client.settimeout(1.0) # Add a timeout for connection and operations
            client.connect(socket_path)
            client.send(data_to_send)
            resp = client.recv(1024)
            return resp
        except socket.timeout:
            logging.error(f"Timeout connecting or communicating with socket {socket_path}.")
            raise
        except FileNotFoundError:
            logging.error(f"Socket not found: {socket_path}. Is the SteamVR driver running?")
            raise
        except ConnectionRefusedError:
             logging.error(f"Connection refused for socket {socket_path}. Driver running but not accepting connections?")
             raise
        except Exception as e:
            logging.error(f"Error communicating with socket {socket_path}: {e}")
            raise
        finally:
            if client:
                client.close() # Ensure socket is closed
    else:
        logging.error(f"Unsupported platform: {platform}")
        raise OSError(f"Platform {platform} not supported for pipe/socket communication.")


def sendToSteamVR_(text: str) -> List[str]:
    """Internal function to send command and parse response."""
    try:
        resp_bytes = sendToPipe(text)
        # Decode and split, handling potential empty responses
        resp_str = resp_bytes.decode("utf-8").strip()
        if not resp_str:
            logging.warning(f"Received empty response from SteamVR driver for command: {text}")
            return ["error", "empty_response"]
        return resp_str.split(" ")
    except Exception as e:
        # Error is already logged in sendToPipe, just return error indicator
        return ["error", str(e)]


def sendToSteamVR(text: str, num_tries: int = 10, wait_time: float = 0.1) -> Optional[List[str]]:
    """
    Wrapped function to send command to SteamVR driver with retries.

    Args:
        text: The command string to send.
        num_tries: Maximum number of attempts.
        wait_time: Seconds to wait between retries.

    Returns:
        A list of strings from the driver response, or None if connection failed after retries.
    """
    for i in range(num_tries):
        ret = sendToSteamVR_(text)
        if ret and ret[0] != "error":
            return ret # Success
        # Log specific error if available
        error_detail = ret[1] if len(ret) > 1 else "Unknown"
        logging.warning(f"Error connecting to SteamVR driver (Attempt {i+1}/{num_tries}): {error_detail}. Retrying in {wait_time}s...")
        time.sleep(wait_time)

    logging.error(f"Could not connect to SteamVR driver after {num_tries} tries.")
    return None # Failed after all retries

# --- Camera Handling ---

class CameraStream():
    """
    Manages camera capture in a separate thread.

    Args:
        params: The application parameters object.
        shutdown_callback: Optional function to call on critical camera error.
    """
    def __init__(self, params, shutdown_callback=None):
        self.params = params
        self.image_ready = threading.Event() # Use Event for signaling
        self.latest_frame: Optional[np.ndarray] = None
        self.stopped = False
        self.shutdown_callback = shutdown_callback
        self.lock = threading.Lock() # Lock for thread-safe access to frame/flags

        # --- Camera ID Parsing ---
        camera_id_str = params.cameraid
        camera_id = None
        if len(camera_id_str) <= 2: # Assume it's an integer ID
            try:
                camera_id = int(camera_id_str)
            except ValueError:
                logging.error(f"Invalid camera ID format: {camera_id_str}. Should be integer or URL.")
                self._trigger_shutdown(1)
                return
        else: # Assume it's a URL/path
            camera_id = camera_id_str

        # --- OpenCV Capture Initialization ---
        capture_backend = cv2.CAP_DSHOW if platform.startswith('win32') and params.camera_settings else cv2.CAP_ANY
        logging.info(f"Attempting to open camera: {camera_id} with backend: {capture_backend}")
        self.cap = cv2.VideoCapture(camera_id, capture_backend)

        if not self.cap.isOpened():
            logging.error(f"ERROR: Could not open camera: {camera_id}. Try another id/IP or backend.")
            self._trigger_shutdown(1)
            return

        # --- Set Camera Properties ---
        props_set = []
        if params.camera_width != 0:
            if self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(params.camera_width)):
                props_set.append(f"Width={int(params.camera_width)}")
            else:
                 logging.warning(f"Failed to set camera width to {params.camera_width}")
        if params.camera_height != 0:
            if self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(params.camera_height)):
                 props_set.append(f"Height={int(params.camera_height)}")
            else:
                 logging.warning(f"Failed to set camera height to {params.camera_height}")

        # CAP_PROP_SETTINGS often opens a dialog, which is problematic. Avoid if possible.
        if platform.startswith('win32') and params.camera_settings:
            logging.info("Attempting to open camera settings dialog (CAP_PROP_SETTINGS)...")
            # This might block or fail silently depending on the camera driver
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
            # Give some time for potential dialog interaction? Risky.
            # time.sleep(2)

        # Log actual resolution
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logging.info(f"Camera {camera_id} opened. Set props: [{', '.join(props_set)}]. Actual resolution: {int(actual_width)}x{int(actual_height)}")

        # --- Start Thread ---
        logging.info("Starting camera capture thread...")
        self.thread = threading.Thread(target=self._update_loop, args=(), daemon=True)
        self.thread.start()


    def _update_loop(self):
        """Continuously grabs frames from the camera."""
        while not self.stopped:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("ERROR: Camera capture failed! No frame returned or camera disconnected.")
                    self.params.exit_ready = True # Signal main thread
                    self._trigger_shutdown(1) # Use callback if available
                    self.stopped = True # Ensure loop terminates
                    break

                with self.lock:
                    self.latest_frame = frame
                    self.image_ready.set() # Signal that a new frame is ready

                # Optional small sleep to prevent high CPU if camera is very fast
                # time.sleep(0.001)

            except Exception as e:
                 logging.exception(f"Exception in camera update loop: {e}")
                 self.params.exit_ready = True
                 self._trigger_shutdown(1)
                 self.stopped = True
                 break

        # --- Cleanup when loop exits ---
        if self.cap.isOpened():
            logging.info("Releasing camera resource...")
            self.cap.release()
        logging.info("Camera thread stopped.")


    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Gets the latest frame if available, waiting briefly.

        Args:
            timeout: Maximum time to wait for a new frame (seconds).

        Returns:
            A copy of the latest frame (np.ndarray) or None if no new frame or stopped.
        """
        # Wait for the event to be set, with a timeout
        event_is_set = self.image_ready.wait(timeout=timeout)

        if event_is_set and not self.stopped:
            with self.lock:
                self.image_ready.clear() # Reset event after getting frame
                if self.latest_frame is not None:
                    # Return a copy to prevent race conditions if main thread modifies it
                    return self.latest_frame.copy()
        return None


    def stop(self):
        """Signals the camera thread to stop and waits for it to join."""
        if not self.stopped:
            logging.info("Stopping camera thread...")
            self.stopped = True
            self.image_ready.set() # Wake up thread if it's waiting
            if self.thread.is_alive():
                 self.thread.join(timeout=2.0) # Wait for thread to finish
                 if self.thread.is_alive():
                     logging.warning("Camera thread did not stop cleanly after 2 seconds.")


    def _trigger_shutdown(self, exit_code: int):
        """Calls the shutdown callback if provided."""
        if self.shutdown_callback:
            logging.info("Triggering shutdown via callback...")
            # Run callback in a separate thread to avoid blocking camera thread?
            # Or assume callback is safe to call directly? For now, call directly.
            try:
                self.shutdown_callback(self.params, exit_code=exit_code)
            except Exception as e:
                logging.error(f"Error executing shutdown callback: {e}")
        else:
            # Fallback if no callback provided (though main script should handle exit_ready)
            logging.warning("No shutdown callback provided to CameraStream.")
            # sys.exit(exit_code) # Avoid calling sys.exit directly from thread


# --- Application Shutdown ---

def shutdown(params, exit_code: int = 0):
    """Saves parameters and exits the application."""
    # First save parameters
    if params and hasattr(params, 'save_params'):
        try:
            logging.info("Saving parameters...")
            params.save_params()
        except Exception as e:
            logging.error(f"Failed to save parameters: {e}")

    # Close OpenCV windows (safely)
    # cv2.destroyAllWindows() # Might cause issues if called from non-main thread.
    # Let main thread handle window destruction.

    logging.info(f"Exiting application with code {exit_code}...")
    # Add a small delay to allow logs to flush?
    time.sleep(0.1)
    sys.exit(exit_code) # This raises SystemExit

# --- Example Usage (for testing helpers directly) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(asctime)s: %(message)s')
    logging.info("Running helpers.py directly for testing...")

    # --- Test Hand Landmark Processing ---
    logging.info("Testing Hand Landmark Processing...")
    # Create dummy landmarks for testing (replace with actual data if available)
    class Landmark: # Mock class
        def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
    class LandmarkList: # Mock class
        def __init__(self, landmarks): self.landmark = landmarks

    # Simulate a slightly curled index finger on the left hand
    left_lms = [Landmark(0,0,0)] * 21 # Initialize with zeros
    left_lms[WRIST_INDEX] = Landmark(0, 0, 0)
    left_lms[HAND_LANDMARK.INDEX_FINGER_MCP] = Landmark(0.1, -0.05, 0)
    left_lms[HAND_LANDMARK.INDEX_FINGER_PIP] = Landmark(0.2, -0.08, 0.02) # Slightly curled in Z
    left_lms[HAND_LANDMARK.INDEX_FINGER_DIP] = Landmark(0.28, -0.09, 0.05) # More curled
    left_lms[HAND_LANDMARK.INDEX_FINGER_TIP] = Landmark(0.35, -0.10, 0.08)
    # Simulate middle finger close to index
    left_lms[HAND_LANDMARK.MIDDLE_FINGER_MCP] = Landmark(0.1, 0.05, 0)
    # Simulate thumb slightly splayed
    left_lms[HAND_LANDMARK.THUMB_MCP] = Landmark(0.05, 0.1, 0)

    dummy_left_hand = LandmarkList(left_lms)
    dummy_right_hand = None # Simulate no right hand detected

    left_hand_data, right_hand_data = process_holistic_hands(dummy_left_hand, dummy_right_hand)

    if left_hand_data:
        print("Left Hand Data:")
        for key, value in left_hand_data.items():
            print(f"  {key}: {value:.3f}" if value is not None else f"  {key}: None")
    else:
        print("No left hand data.")

    if right_hand_data:
        print("Right Hand Data:")
        for key, value in right_hand_data.items():
             print(f"  {key}: {value:.3f}" if value is not None else f"  {key}: None")
    else:
        print("No right hand data.")
    # --- End Hand Landmark Test ---


    # Example: Test SteamVR connection
    logging.info("Testing SteamVR connection...")
    response = sendToSteamVR("numtrackers", num_tries=3)
    if response:
        logging.info(f"SteamVR driver responded: {' '.join(response)}")
    else:
        logging.error("SteamVR connection test failed.")

    # Example: Test CameraStream (requires a dummy params object)
    class DummyParams:
        cameraid = "0" # Or your camera ID/URL
        camera_settings = False
        camera_width = 640
        camera_height = 480
        exit_ready = False
        def save_params(self): pass

    logging.info("Testing CameraStream...")
    test_params = DummyParams()
    cam = None
    try:
        cam = CameraStream(test_params, shutdown_callback=shutdown)
        if not test_params.exit_ready:
            logging.info("CameraStream initialized. Waiting for first frame...")
            frame = cam.get_frame(timeout=5.0) # Wait up to 5 seconds
            if frame is not None:
                logging.info(f"Received frame with shape: {frame.shape}")
                # cv2.imshow("Camera Test", frame) # Requires main thread for GUI
                # cv2.waitKey(1000)
            else:
                logging.error("Did not receive frame from camera.")
            time.sleep(1) # Keep running briefly
    except Exception as e:
        logging.exception(f"Error during CameraStream test: {e}")
    finally:
        if cam:
            cam.stop()
        # cv2.destroyAllWindows()
        logging.info("CameraStream test finished.")

    logging.info("Helpers test complete.")
