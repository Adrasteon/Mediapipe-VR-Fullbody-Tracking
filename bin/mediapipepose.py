# mediapipepose.py Main application script for MediaPipe-based full-body tracking using Holistic model.
"""
Main application script for MediaPipe-based full-body tracking.

Initializes camera, pose detection (using Holistic model), GUI, web UI,
and backend communication (SteamVR or VRChat OSC) to translate webcam pose
estimation into VR tracking data. Includes flags to enable/disable hand and face tracking processing.
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
from scipy.spatial.transform import Rotation as R

# Ensure the current directory is in the path for embedded Python
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Local application imports (Grouped)
try:
    # Import pose processing and hand processing functions from helpers
    from helpers import (CameraStream, shutdown, mediapipeTo3dpose,
                         get_rot_mediapipe, get_rot, process_holistic_hands) # Added process_holistic_hands
    from backends import DummyBackend, SteamVRBackend, VRChatOSCBackend, Backend
    import webui
    import inference_gui
    import parameters
except ImportError as e:
    print(f"ERROR: Failed to import local modules: {e}")
    print("Ensure all required .py files (helpers.py, backends.py, etc.) are in the same directory or in the Python path.")
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
        shutdown(params, exit_code=1)
        return

    try:
        cv2.namedWindow(OUTPUT_WINDOW_NAME)
    except Exception as e:
        logging.error(f"Failed to create OpenCV window: {e}")
        if holistic: holistic.close()
        if camera_thread: camera_thread.stop()
        if backend: backend.disconnect()
        shutdown(params, exit_code=1)
        return


    # --- Main Loop ---
    prev_smoothing = params.smoothing
    prev_add_smoothing = params.additional_smoothing
    last_log_time = time.time()

    logging.info("Starting main loop...")
    while not params.exit_ready:
        try:
            loop_start_time = time.perf_counter()

            # --- Parameter Change Check ---
            if prev_smoothing != params.smoothing or prev_add_smoothing != params.additional_smoothing:
                logging.info(f"Smoothing changed: Window={params.smoothing:.2f}, Additional={params.additional_smoothing:.2f}")
                prev_smoothing = params.smoothing
                prev_add_smoothing = params.additional_smoothing
                try:
                    backend.onparamchanged(params)
                except Exception as e:
                    logging.error(f"Error calling backend.onparamchanged: {e}")

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
