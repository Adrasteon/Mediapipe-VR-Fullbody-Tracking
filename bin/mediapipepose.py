# mediapipepose.py Main application script for MediaPipe-based full-body tracking using Holistic model.
"""
Main application script for MediaPipe-based full-body tracking.

Initializes camera, pose detection (using Holistic model), GUI, web UI,
and backend communication (SteamVR or VRChat OSC) to translate webcam pose
estimation into VR tracking data. Includes flags to enable/disable hand and face tracking processing.
Includes optional keyboard head rotation control for SteamVR.
"""

import os
import sys
import time
import threading
import logging # Use logging instead of print for info/errors
from typing import Optional, Tuple, Dict, Any # For type hinting

# Third-party imports
import cv2
import mediapipe as mp
import numpy as np
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("ERROR: Scipy not found. Please install it: pip install scipy")
    sys.exit(1)


# Ensure the current directory is in the path for embedded Python
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Local application imports (Grouped)
try:
    # Import pose processing and hand processing functions from helpers
    from helpers import (CameraStream, shutdown, mediapipeTo3dpose,
                         get_rot_mediapipe, get_rot, process_holistic_hands,
                         sendToSteamVR) # Added sendToSteamVR for keyboard control
    from backends import DummyBackend, SteamVRBackend, VRChatOSCBackend, Backend
    import webui
    import inference_gui
    import parameters
    import keyboard_helper # Import the keyboard helper module
except ImportError as e:
    print(f"ERROR: Failed to import local modules: {e}")
    print("Ensure all required .py files (helpers.py, backends.py, keyboard_helper.py, etc.) are in the same directory or in the Python path.")
    sys.exit(1)


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Constants ---
ESC_KEY = 27
OUTPUT_WINDOW_NAME = "MediaPipe Holistic Tracking" # Updated window name


def main() -> None:
    """Initializes and runs the main pose tracking application loop."""

    # --- Initialization ---
    mp_drawing = mp.solutions.drawing_utils
    # Use mp.solutions.holistic instead of mp.solutions.pose
    mp_holistic = mp.solutions.holistic

    logging.info("Reading parameters...")
    params = None
    try:
        params = parameters.Parameters()
        # Ensure 'use_face' parameter exists, default to False if not added yet
        if not hasattr(params, 'use_face'):
            logging.warning("Parameter 'use_face' not found, defaulting to False.")
            params.use_face = False
        # Add placeholder attributes for keyboard control if not in parameters.py yet
        if not hasattr(params, 'keyboard_head_control_enabled'):
            logging.warning("Parameter 'keyboard_head_control_enabled' not found, defaulting to False.")
            params.keyboard_head_control_enabled = False
        if not hasattr(params, 'keyboard_head_control_speed'):
             logging.warning("Parameter 'keyboard_head_control_speed' not found, defaulting to 1.5.")
             params.keyboard_head_control_speed = 1.5

    except Exception as e:
        logging.error(f"Failed during parameter initialization: {e}")
        # Attempt graceful shutdown even if params partially failed
        if params:
            shutdown(params, exit_code=1)
        else:
            # If params object itself failed, provide a dummy for shutdown
            class DummyParams:
                def save_params(self): pass
            shutdown(DummyParams(), exit_code=1)
        return # Exit main function

    if params.exit_ready: # Check if init GUI requested exit
        logging.info("Exiting after parameter setup.")
        shutdown(params, exit_code=0)
        return

    # --- WebUI Setup ---
    webui_thread = None
    if params.webui:
        try:
            webui_thread = threading.Thread(target=webui.start_webui, args=(params,), daemon=True)
            webui_thread.start()
        except Exception as e:
            logging.error(f"Failed to start WebUI: {e}")
    else:
        logging.info("WebUI disabled in parameters.")

    # --- Backend Setup ---
    backend: Optional[Backend] = None
    try:
        backends = {0: DummyBackend, 1: SteamVRBackend, 2: VRChatOSCBackend}
        selected_backend_class = backends.get(params.backend)

        if selected_backend_class is None:
            logging.error(f"Invalid backend selected: {params.backend}. Exiting.")
            shutdown(params, exit_code=1)
            return

        backend = selected_backend_class()
        # Ensure backend.connect signature matches if needed (currently doesn't need hand/face flags)
        backend.connect(params)
        logging.info(f"Using backend: {selected_backend_class.__name__}")

    except Exception as e:
        logging.error(f"Failed to connect to backend: {e}")
        if backend:
            backend.disconnect()
        shutdown(params, exit_code=1)
        return

    # --- Camera Setup ---
    camera_thread: Optional[CameraStream] = None
    try:
        logging.info("Opening camera...")
        camera_thread = CameraStream(params, shutdown_callback=shutdown)
        if params.exit_ready:
             logging.error("Camera initialization failed (detected via exit_ready flag).")
             if backend: backend.disconnect()
             return

    except TypeError as e:
        logging.error(f"TypeError during CameraStream init: {e}")
        logging.error("Ensure CameraStream.__init__ in helpers.py accepts 'shutdown_callback'.")
        if backend: backend.disconnect()
        shutdown(params, exit_code=1)
        return
    except Exception as e:
        logging.error(f"Failed to initialize camera: {e}")
        if backend: backend.disconnect()
        if not params.exit_ready:
            shutdown(params, exit_code=1)
        return

    # --- GUI Setup ---
    gui_thread: Optional[threading.Thread] = None
    try:
        gui_thread = threading.Thread(target=inference_gui.make_inference_gui, args=(params,), daemon=True)
        gui_thread.start()
    except Exception as e:
        logging.error(f"Failed to start GUI thread: {e}")

    # --- Keyboard Helper Setup ---
    try:
        keyboard_helper.start_keyboard_listener()
        # Set initial state based on parameters
        keyboard_helper.set_keyboard_control_enabled(params.keyboard_head_control_enabled)
        keyboard_helper.set_keyboard_rotation_speed(params.keyboard_head_control_speed)
    except Exception as e:
        logging.error(f"Failed to start keyboard listener: {e}")
        # Continue without keyboard control if it fails

    # --- MediaPipe Holistic Setup ---
    holistic: Optional[mp_holistic.Holistic] = None # type: ignore # Use holistic variable
    try:
        logging.info("Starting Holistic detector...")
        holistic = mp_holistic.Holistic(
            model_complexity=params.model,
            min_detection_confidence=0.5, # Adjust confidence as needed
            min_tracking_confidence=params.min_tracking_confidence,
            smooth_landmarks=params.smooth_landmarks,
            # static_image_mode=params.static_image, # Not applicable to Holistic stream
            refine_face_landmarks=params.use_face, # Use flag to enable detailed face mesh
            enable_segmentation=False # Disable segmentation unless needed
        )
    except Exception as e:
        logging.error(f"Failed to initialize MediaPipe Holistic: {e}")
        if camera_thread: camera_thread.stop()
        if backend: backend.disconnect()
        keyboard_helper.stop_keyboard_listener() # Stop keyboard listener on error
        shutdown(params, exit_code=1)
        return

    try:
        cv2.namedWindow(OUTPUT_WINDOW_NAME)
    except Exception as e:
        logging.error(f"Failed to create OpenCV window: {e}")
        if holistic: holistic.close()
        if camera_thread: camera_thread.stop()
        if backend: backend.disconnect()
        keyboard_helper.stop_keyboard_listener() # Stop keyboard listener on error
        shutdown(params, exit_code=1)
        return


    # --- Main Loop ---
    prev_smoothing = params.smoothing
    prev_add_smoothing = params.additional_smoothing
    # Track previous keyboard parameter states
    prev_keyboard_enabled = params.keyboard_head_control_enabled
    prev_keyboard_speed = params.keyboard_head_control_speed

    last_log_time = time.time()

    logging.info("Starting main loop...")
    while not params.exit_ready:
        try:
            loop_start_time = time.perf_counter()

            # --- Parameter Change Check ---
            # Smoothing
            if prev_smoothing != params.smoothing or prev_add_smoothing != params.additional_smoothing:
                logging.info(f"Smoothing changed: Window={params.smoothing:.2f}, Additional={params.additional_smoothing:.2f}")
                prev_smoothing = params.smoothing
                prev_add_smoothing = params.additional_smoothing
                try:
                    backend.onparamchanged(params)
                except Exception as e:
                    logging.error(f"Error calling backend.onparamchanged: {e}")

            # Keyboard Control Parameters
            if hasattr(params, 'keyboard_head_control_enabled') and prev_keyboard_enabled != params.keyboard_head_control_enabled:
                prev_keyboard_enabled = params.keyboard_head_control_enabled
                keyboard_helper.set_keyboard_control_enabled(prev_keyboard_enabled)

            if hasattr(params, 'keyboard_head_control_speed') and prev_keyboard_speed != params.keyboard_head_control_speed:
                 prev_keyboard_speed = params.keyboard_head_control_speed
                 keyboard_helper.set_keyboard_rotation_speed(prev_keyboard_speed)


            # --- Get Frame ---
            img: Optional[np.ndarray] = camera_thread.get_frame()

            if img is None:
                if params.exit_ready:
                    logging.warning("Camera thread signaled exit. Breaking loop.")
                    break
                time.sleep(0.005)
                continue

            # --- Image Preprocessing ---
            if params.rotate_image is not None:
                try:
                    img = cv2.rotate(img, params.rotate_image)
                except cv2.error as e:
                    logging.warning(f"cv2.rotate error: {e}. Skipping rotation.")
                    params.rotate_image = None

            if params.mirror:
                img = cv2.flip(img, 1)

            img_h, img_w = img.shape[:2]
            if max(img_h, img_w) > params.maximgsize:
                scale_ratio = params.maximgsize / max(img_h, img_w)
                new_w = int(img_w * scale_ratio)
                new_h = int(img_h * scale_ratio)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # --- Pause Check ---
            if params.paused:
                try:
                    cv2.putText(img, "PAUSED", (img_w // 2 - 50, img_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow(OUTPUT_WINDOW_NAME, img)
                    key = cv2.waitKey(30)
                    if key == ESC_KEY:
                        params.exit_ready = True
                except cv2.error as e:
                    logging.warning(f"cv2.imshow/waitKey error while paused: {e}")
                    params.exit_ready = True
                continue

            # --- Apply Keyboard Head Rotation (SteamVR Only) ---
            if keyboard_helper.is_keyboard_control_enabled() and params.backend == 1:
                # Get deltas accumulated since last loop iteration
                yaw_deg, pitch_deg, roll_deg = keyboard_helper.get_and_reset_keyboard_deltas()

                if yaw_deg != 0.0 or pitch_deg != 0.0 or roll_deg != 0.0:
                    try:
                        # 1. Get current HMD pose from SteamVR
                        hmd_pose_str = sendToSteamVR("getdevicepose 0", num_tries=1, wait_time=0.01)
                        if hmd_pose_str and not hmd_pose_str[0].startswith("error"):
                            parts = hmd_pose_str[0].split()
                            # Adapt parsing based on actual driver output format
                            try:
                                current_hmd_pos = [float(p) for p in parts[3:6]]
                                # Assuming driver outputs W, X, Y, Z quaternion
                                current_hmd_rot_quat_wxyz = [float(q) for q in parts[6:10]]
                            except (IndexError, ValueError) as parse_err:
                                 logging.warning(f"Could not parse HMD pose string: '{hmd_pose_str[0]}'. Error: {parse_err}")
                                 continue # Skip this rotation update

                            # 2. Create rotation objects
                            # Convert SteamVR WXYZ to Scipy XYZW: [x, y, z, w]
                            current_rot_quat_xyzw = [
                                current_hmd_rot_quat_wxyz[1],
                                current_hmd_rot_quat_wxyz[2],
                                current_hmd_rot_quat_wxyz[3],
                                current_hmd_rot_quat_wxyz[0]
                            ]
                            # Check for invalid quaternion before creating Rotation object
                            if np.isclose(np.linalg.norm(current_rot_quat_xyzw), 0.0):
                                logging.warning("Current HMD quaternion from driver is zero. Skipping keyboard rotation.")
                                continue
                            current_rot = R.from_quat(current_rot_quat_xyzw)

                            # 3. Create delta rotation (Yaw around Y, Pitch around X, Roll around Z)
                            # Euler sequence 'yxz' is common for head rotation (Yaw first, then Pitch, then Roll)
                            delta_rot = R.from_euler('yxz', [yaw_deg, pitch_deg, roll_deg], degrees=True)

                            # 4. Combine rotations: Apply keyboard delta to the current HMD rotation
                            new_rot = delta_rot * current_rot # Apply delta in the HMD's local frame

                            # 5. Get new quaternion in SteamVR format (W, X, Y, Z)
                            new_rot_quat_xyzw = new_rot.as_quat() # Scipy gives [x, y, z, w]
                            new_rot_quat_wxyz = [
                                new_rot_quat_xyzw[3], # w
                                new_rot_quat_xyzw[0], # x
                                new_rot_quat_xyzw[1], # y
                                new_rot_quat_xyzw[2]  # z
                            ]

                            # 6. Send update command (only rotation changes)
                            update_cmd = (f"updatepose 0 {current_hmd_pos[0]:.4f} {current_hmd_pos[1]:.4f} {current_hmd_pos[2]:.4f} "
                                          f"{new_rot_quat_wxyz[0]:.4f} {new_rot_quat_wxyz[1]:.4f} {new_rot_quat_wxyz[2]:.4f} {new_rot_quat_wxyz[3]:.4f} "
                                          f"0.0 0.0") # Latency, smoothing
                            sendToSteamVR(update_cmd, num_tries=1, wait_time=0.01)
                            # logging.debug(f"Applied keyboard rotation: Yaw={yaw_deg}, Pitch={pitch_deg}, Roll={roll_deg}")

                        else:
                            # Log only occasionally if HMD pose cannot be retrieved
                            # logging.warning(f"Could not get HMD pose for keyboard rotation: {hmd_pose_str}")
                            pass
                    except Exception as e:
                        logging.error(f"Error applying keyboard rotation: {e}", exc_info=True)
            # --- End Keyboard Head Rotation ---


            # --- Holistic Estimation ---
            t_start = time.perf_counter()

            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb.flags.writeable = False
                results = holistic.process(img_rgb) # Use holistic model
            except cv2.error as e:
                logging.error(f"cv2.cvtColor error: {e}. Skipping frame.")
                continue
            except Exception as e:
                logging.error(f"MediaPipe holistic.process error: {e}. Skipping frame.")
                continue

            inference_time = time.perf_counter() - t_start

            # --- Process Results ---
            pose3d = None
            rots = None
            # Variables to hold processed hand/face data for the backend
            processed_hand_data: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]] = None
            face_landmarks_for_backend: Optional[Any] = None # Keep as Any for now

            if results.pose_world_landmarks:
                try:
                    # --- Process POSE landmarks ---
                    landmarks = results.pose_world_landmarks.landmark
                    pose3d = mediapipeTo3dpose(landmarks) # Assuming this helper still works

                    # Coordinate System Adjustment for POSE
                    # NOTE: Consider moving backend-specific adjustments into the backend classes.
                    pose3d[:, 0] *= -1
                    pose3d[:, 1] *= -1
                    params.pose3d_og = pose3d.copy() # For calibration GUI

                    # Apply Global Rotations
                    pose3d = params.global_rot_z.apply(pose3d)
                    pose3d = params.global_rot_x.apply(pose3d)
                    pose3d = params.global_rot_y.apply(pose3d)

                    # Calculate POSE Joint Rotations (Hip/Feet)
                    if not params.feet_rotation:
                        rots = get_rot(pose3d)
                    else:
                        rots = get_rot_mediapipe(pose3d)

                    # --- Process HAND landmarks (if enabled) ---
                    if params.use_hands and (results.left_hand_landmarks or results.right_hand_landmarks):
                        # Call the helper function to calculate curls/splays etc.
                        processed_hand_data = process_holistic_hands(results.left_hand_landmarks, results.right_hand_landmarks)
                        # The backend will receive this tuple: (left_hand_dict, right_hand_dict)

                    # --- Process FACE landmarks (if enabled, primarily for potential future backend use) ---
                    if params.use_face and results.face_landmarks:
                        # Pass raw face landmarks if a backend needs them
                        face_landmarks_for_backend = results.face_landmarks
                        # Currently no backend uses this, but structure is ready.

                    # --- Send to Backend ---
                    # Pass pose, pose rotations, and optional processed hand/face landmark data
                    # ** IMPORTANT: Ensure backend.updatepose signatures are updated **
                    if not backend.updatepose(params, pose3d, rots, processed_hand_data, face_landmarks_for_backend):
                         pass # Backend indicated not ready (e.g., waiting for calibration)

                except Exception as e:
                    logging.exception(f"Error processing landmarks or updating backend: {e}")

            # --- Logging & Display ---
            current_time = time.time()
            if params.log_frametime and (current_time - last_log_time >= 1.0):
                 fps = 1.0 / inference_time if inference_time > 0 else 0
                 total_loop_time = time.perf_counter() - loop_start_time
                 logging.info(f"Inference: {inference_time*1000:.1f} ms ({fps:.1f} FPS) | Total Loop: {total_loop_time*1000:.1f} ms")
                 last_log_time = current_time

            # --- Draw Landmarks ---
            try:
                # Draw POSE landmarks
                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks, # Use screen landmarks for drawing
                    mp_holistic.POSE_CONNECTIONS, # Use Holistic connections
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                # Draw HAND landmarks (if enabled)
                if params.use_hands:
                    mp_drawing.draw_landmarks(
                        img,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2), # Left Hand Color
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                    )
                    mp_drawing.draw_landmarks(
                        img,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # Right Hand Color
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

                # Draw FACE landmarks (if enabled)
                if params.use_face:
                    # Choose either Tesselation or Contours for less clutter, or both
                    mp_drawing.draw_landmarks(
                        img,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_TESSELATION, # Full mesh
                        landmark_drawing_spec=None, # Do not draw individual points
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1) # Tesselation color
                    )
                    mp_drawing.draw_landmarks(
                        img,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS, # Outline contours
                        landmark_drawing_spec=None, # Do not draw individual points
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1) # Contours color
                    )

                # Add FPS text
                fps_display = 1 / inference_time if inference_time > 0 else 0
                cv2.putText(img, f"FPS: {fps_display:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow(OUTPUT_WINDOW_NAME, img)

            except cv2.error as e:
                 logging.warning(f"cv2 drawing or imshow error: {e}")
                 params.exit_ready = True
            except Exception as e:
                 logging.exception(f"Error during drawing/display: {e}")
                 params.exit_ready = True


            # --- Exit Check ---
            key = cv2.waitKey(1)
            if key == ESC_KEY:
                logging.info("ESC key pressed. Exiting...")
                params.exit_ready = True

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Exiting...")
            params.exit_ready = True
        except Exception as e:
            logging.exception(f"Unhandled error in main loop: {e}")
            params.exit_ready = True

    # --- Cleanup ---
    logging.info("Exiting main loop.")

    try:
        if holistic: # Use holistic variable
            logging.info("Closing MediaPipe Holistic...") # Update log message
            holistic.close() # Close holistic model
        if camera_thread:
            logging.info("Stopping camera thread...")
            camera_thread.stop()
        if backend:
            logging.info("Disconnecting backend...")
            backend.disconnect()
        # Stop keyboard listener during cleanup
        logging.info("Stopping keyboard listener...")
        keyboard_helper.stop_keyboard_listener()

        logging.info("Destroying OpenCV windows...")
        cv2.destroyAllWindows()

    except Exception as e:
        logging.exception(f"Error during cleanup: {e}")
    finally:
        logging.info("Calling final shutdown...")
        if 'params' in locals() and params is not None:
             shutdown(params, exit_code=0)
        else:
             class DummyParams:
                 def save_params(self): pass
             shutdown(DummyParams(), exit_code=1)

        logging.info("Shutdown process complete.")


if __name__ == "__main__":
    main()
