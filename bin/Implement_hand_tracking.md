# Project Plan: Detailed Hand Tracking Implementation

**Version:** 1.0
**Date:** 2023-10-27

## 1. Introduction & Goal

This document outlines the plan to upgrade the existing hand tracking capabilities within the MediaPipe VR Fullbody Tracking project. The primary goal is to replace the current basic hand orientation tracking with detailed, articulated finger tracking using MediaPipe's dedicated hand models and integrate this data with the VRChat OSC backend. This will significantly enhance user immersion and expressiveness within VRChat by enabling natural hand gestures and movements mirrored on the user's avatar.

## 2. Current State Analysis

*   **Pose Estimation Model:** The project currently uses `mediapipe.solutions.pose`.
*   **Hand Landmarks:** The Pose model provides limited hand landmarks (wrist, thumb, index, pinky - indices 15-22).
*   **Hand Rotation Calculation:** The `helpers.get_rot_hands` function calculates a single overall rotation quaternion for each hand based on these limited landmarks.
*   **Backend Integration:**
    *   `SteamVRBackend`: Sends the single hand rotation quaternion if `params.use_hands` is enabled.
    *   `VRChatOSCBackend`: Currently **does not** send any hand tracking data. Finger articulation is not supported.
*   **Limitations:** The current system only provides basic hand orientation, lacking any finger movement, which limits expressiveness and interaction possibilities in VR.

## 3. Proposed Solution Overview

The proposed solution involves leveraging MediaPipe's more advanced models and integrating their output with the VRChat OSC protocol for finger tracking.

1.  **Switch to MediaPipe Holistic:** Replace `mp.solutions.pose` with `mp.solutions.holistic`. The Holistic model efficiently combines pose, face, and **detailed hand tracking (21 landmarks per hand)** into a single pipeline.
2.  **Calculate Finger Joint Data:** Implement logic to process the 21 landmarks per hand provided by Holistic to calculate normalized curl and splay values for each finger joint.
3.  **Integrate with VRChat OSC:** Update the `VRChatOSCBackend` to send the calculated finger curl/splay data using the specific OSC parameters expected by VRChat avatars configured for finger tracking.

## 4. Detailed Implementation Steps

### Step 1: Refactor `mediapipepose.py` to Use `mp.solutions.holistic`

*   **Objective:** Replace the Pose model with the Holistic model to gain access to detailed hand landmarks alongside pose landmarks.
*   **Tasks:**
    *   Modify imports: Change `import mediapipe as mp` related lines to import `mp.solutions.holistic`.
        ```python
        # In mediapipepose.py
        import mediapipe as mp
        # ... other imports ...
        mp_drawing = mp.solutions.drawing_utils
        # mp_pose = mp.solutions.pose # Remove this
        mp_holistic = mp.solutions.holistic # Add this
        ```
    *   Update MediaPipe Initialization: Change the `mp_pose.Pose(...)` call to `mp_holistic.Holistic(...)`. Configure it similarly regarding model complexity, confidence, etc.
        ```python
        # In mediapipepose.py - main()
        logging.info("Starting Holistic detector...")
        try:
            # pose = mp_pose.Pose(...) # Remove this
            holistic = mp_holistic.Holistic(
                model_complexity=params.model,
                min_detection_confidence=0.5, # Or use param
                min_tracking_confidence=params.min_tracking_confidence,
                smooth_landmarks=params.smooth_landmarks,
                # static_image_mode=params.static_image # Holistic might not have this exact param
            )
        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe Holistic: {e}")
            # ... error handling ...
        ```
    *   Modify Main Loop Processing: Change `pose.process(img_rgb)` to `holistic.process(img_rgb)`.
        ```python
        # In mediapipepose.py - main() loop
        # results = pose.process(img_rgb) # Remove this
        results = holistic.process(img_rgb) # Add this
        ```
    *   Access Results: Update code to access landmarks from the `results` object provided by Holistic.
        ```python
        # In mediapipepose.py - main() loop, after process()
        pose3d = None
        left_hand_landmarks = results.left_hand_landmarks # Access left hand
        right_hand_landmarks = results.right_hand_landmarks # Access right hand

        if results.pose_world_landmarks: # Keep using pose_world_landmarks for body
            landmarks = results.pose_world_landmarks.landmark
            pose3d = mediapipeTo3dpose(landmarks)
            # ... existing pose processing logic ...

            # Pass hand landmarks to backend updatepose
            try:
                # Modify backend call to include hand landmarks
                if not backend.updatepose(params, pose3d, rots, left_hand_landmarks, right_hand_landmarks):
                     pass
            except Exception as e:
                logging.error(f"Error updating backend pose: {e}")

        # Update drawing logic if needed (Holistic provides connections for hands too)
        # Example: Draw pose landmarks
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks, # Use pose_landmarks for 2D drawing
            mp_holistic.POSE_CONNECTIONS, # Use Holistic connections
            # ... drawing specs ...
        )
        # Example: Draw hand landmarks
        mp_drawing.draw_landmarks(
            img,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            # ... drawing specs ...
        )
        mp_drawing.draw_landmarks(
            img,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            # ... drawing specs ...
        )

        ```
    *   Cleanup: Remove the now unused `get_rot_hands` function call if its single rotation is no longer needed.

### Step 2: Implement Finger Joint Calculation (`helpers.py`)

*   **Objective:** Create a function to convert the 21 MediaPipe hand landmarks into normalized finger curl and splay values suitable for VRChat OSC.
*   **Tasks:**
    *   Create a new function `calculate_finger_data(hand_landmarks, is_left_hand)` in `helpers.py`.
    *   **Input:** `hand_landmarks` (MediaPipe landmark list or `None`), `is_left_hand` (boolean, potentially needed for axis orientation).
    *   **Output:** A dictionary containing normalized finger data (e.g., `{ "Index1Curl": 0.8, "Index2Curl": 0.7, ..., "IndexSplay": 0.1, ... }`) or `None` if no hand landmarks are provided.
    *   **Internal Logic:**
        *   Check if `hand_landmarks` is valid. If not, return `None`.
        *   Extract necessary landmark coordinates (WRIST, MCP, PIP, DIP, TIP for each finger). Convert landmark objects to NumPy arrays.
        *   **Curl Calculation:**
            *   For each finger (Thumb, Index, Middle, Ring, Pinky):
                *   Define vectors for each segment (e.g., `vec_mcp_pip = landmarks[PIP] - landmarks[MCP]`).
                *   Calculate the angle between consecutive segments using the dot product formula: `angle = acos(dot(vec1, vec2) / (norm(vec1) * norm(vec2)))`. Handle potential division by zero.
                *   Sum or average angles for multi-joint curls (e.g., MCP-PIP angle + PIP-DIP angle for primary curl).
                *   Normalize the calculated angle(s) to a 0-1 range (0 = straight, 1 = fully curled). This requires defining reasonable min/max angles based on observation or experimentation.
        *   **Splay Calculation (Abduction/Adduction):**
            *   Define vectors representing the direction of the metacarpal bones (e.g., `vec_index_mcp = landmarks[INDEX_MCP] - landmarks[WRIST]`, `vec_middle_mcp = landmarks[MIDDLE_MCP] - landmarks[WRIST]`).
            *   Calculate the angle between adjacent metacarpal vectors (e.g., angle between index and middle).
            *   Normalize this angle to a suitable range (e.g., -1 to 1 or 0-1, where 0.5 is neutral). Define min/max splay angles.
        *   Store normalized values in the output dictionary using keys that match (or can be easily mapped to) VRChat OSC parameter names.
    *   **Example Snippet (Conceptual Curl):**
        ```python
        # Inside calculate_finger_data (Conceptual Example)
        import numpy as np
        import math

        def get_angle(vec1, vec2):
            unit_vec1 = vec1 / np.linalg.norm(vec1)
            unit_vec2 = vec2 / np.linalg.norm(vec2)
            dot_product = np.dot(unit_vec1, unit_vec2)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) # Clip for stability
            return math.degrees(angle)

        # --- For Index Finger Curl ---
        # Assuming landmarks are NumPy arrays
        wrist = landmarks[mp_holistic.HandLandmark.WRIST]
        index_mcp = landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP]
        index_pip = landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP]
        index_dip = landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP]
        index_tip = landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP]

        vec_mcp_pip = index_pip - index_mcp
        vec_pip_dip = index_dip - index_pip
        vec_dip_tip = index_tip - index_dip

        # Angle at PIP joint (example)
        angle_pip = get_angle(vec_mcp_pip, vec_pip_dip)
        # Angle at DIP joint (example)
        angle_dip = get_angle(vec_pip_dip, vec_dip_tip)

        # Normalize (Example: assuming max curl angle is ~180 degrees total)
        # These normalization ranges need careful tuning!
        max_curl_angle = 180.0
        normalized_curl = np.clip((angle_pip + angle_dip) / max_curl_angle, 0.0, 1.0)

        finger_data["IndexCurl"] = normalized_curl # Store with appropriate key
        # Repeat for other joints/fingers/splay...
        ```

### Step 3: Update `VRChatOSCBackend` (`backends.py`)

*   **Objective:** Modify the OSC backend to process detailed hand landmarks and send finger curl/splay data to VRChat.
*   **Tasks:**
    *   Modify `updatepose` signature to accept `left_hand_landmarks` and `right_hand_landmarks`.
        ```python
        # In backends.py - VRChatOSCBackend
        # def updatepose(self, params, pose3d, rots, hand_rots): # Old
        def updatepose(self, params, pose3d, rots, left_hand_landmarks, right_hand_landmarks): # New
            # ... existing smoothing ...
        ```
    *   Call the new helper function:
        ```python
        # Inside updatepose
        left_finger_data = calculate_finger_data(left_hand_landmarks, is_left_hand=True)
        right_finger_data = calculate_finger_data(right_hand_landmarks, is_left_hand=False)
        ```
    *   Modify OSC message building:
        *   Remove the "Sending hand trackers unsupported" comment.
        *   **Crucially:** Identify the correct VRChat OSC parameter names for finger tracking. These are typically avatar parameters like `/avatar/parameters/LeftIndexCurl`, `/avatar/parameters/RightThumbSplay`, etc. (Consult VRChat OSC documentation or common avatar setups).
        *   Create new helper functions or extend existing ones (`osc_build_msg`, `osc_build_bundle`) to build messages for these float avatar parameters.
        *   If `left_finger_data` is not `None`, iterate through its keys/values and add corresponding OSC messages to the bundle.
        *   If `right_finger_data` is not `None`, do the same for the right hand.
    *   **Example OSC Message Building (Conceptual):**
        ```python
        # Inside VRChatOSCBackend - updatepose

        # Assuming osc_build_bundle is modified or a new function handles bundle creation
        bundle_builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)

        # ... (Add existing head/hip/foot messages to bundle_builder) ...

        def add_finger_messages(builder, finger_data, hand_prefix):
            if finger_data is None:
                return
            # Map internal keys to VRChat OSC parameter names
            # This mapping needs to be accurate based on VRChat OSC docs!
            param_map = {
                "IndexCurl": f"{hand_prefix}IndexCurl", # Example VRChat param name
                "IndexSplay": f"{hand_prefix}IndexSplay",
                # ... add mappings for all fingers/curls/splays ...
            }
            for key, osc_param_name in param_map.items():
                if key in finger_data:
                    msg_builder = osc_message_builder.OscMessageBuilder(address=f"/avatar/parameters/{osc_param_name}")
                    msg_builder.add_arg(float(finger_data[key])) # Send as float
                    builder.add_content(msg_builder.build())

        add_finger_messages(bundle_builder, left_finger_data, "Left") # Or "L" depending on convention
        add_finger_messages(bundle_builder, right_finger_data, "Right") # Or "R"

        # Send the complete bundle
        final_bundle = bundle_builder.build()
        self.client.send(final_bundle)
        ```

### Step 4: Update `Parameters` (`parameters.py`)

*   **Objective:** Adjust configuration parameters related to hand tracking.
*   **Tasks:**
    *   Review the `use_hands` parameter. Decide if it's still needed. If detailed tracking is always on when using OSC, it might become redundant or could be repurposed (e.g., to enable/disable sending hand data entirely).
    *   Consider adding new parameters for finger data smoothing or filtering coefficients if implementing filters in Step 5.
    *   Ensure parameter saving/loading (`save_params`, `load_params`) is updated if parameters are added/removed.

### Step 5: Testing and Refinement

*   **Objective:** Ensure the detailed hand tracking works correctly and smoothly in VRChat.
*   **Tasks:**
    *   Use a VRChat avatar that is correctly configured to receive OSC finger tracking data via avatar parameters.
    *   Run the application and observe avatar hand movements in VRChat.
    *   Verify that finger curls and splays correspond reasonably to the user's actual hand movements.
    *   **Tune Normalization:** Adjust the min/max angle ranges used in `calculate_finger_data` to ensure the full 0-1 range is utilized effectively and movements feel natural.
    *   **Implement Filtering (Optional but Recommended):** Apply a filter (e.g., OneEuroFilter, EMA) to the calculated curl/splay values *before* sending them via OSC. This will significantly reduce jitter and create smoother animations. The filter could be applied within `calculate_finger_data` or in the `VRChatOSCBackend`.
    *   Test performance impact. Holistic is more computationally expensive than Pose. Monitor FPS.

## 5. VRChat OSC Finger Tracking Specification (Reference)

VRChat receives finger tracking data primarily through **Avatar Parameters**. The exact names of these parameters depend on how the specific avatar is set up, but common conventions exist.

*   **Curl:** Typically parameters like `/avatar/parameters/LeftIndexCurl`, `/avatar/parameters/LeftMiddleCurl`, ..., `/avatar/parameters/RightPinkyCurl`. These usually expect a `float` value from `0.0` (straight) to `1.0` (fully curled). Some avatars might use separate parameters for different joints (e.g., `LeftIndex1Curl`, `LeftIndex2Curl`, `LeftIndex3Curl`).
*   **Splay:** Typically parameters like `/avatar/parameters/LeftIndexSplay`, `/avatar/parameters/LeftMiddleSplay`, etc. These often expect a `float` value representing abduction/adduction, potentially in a `-1.0` to `1.0` or `0.0` to `1.0` range (where `0.5` is neutral).

**Action:** Confirm the exact parameter names required by target VRChat avatars or standard OSC configurations (e.g., those used by VRCFaceTracking setups often define standard parameter names). Update the mapping in `VRChatOSCBackend` accordingly.

## 6. Potential Challenges

*   **Tracking Accuracy/Jitter:** MediaPipe hand tracking can be sensitive to lighting, occlusions, and fast movements, leading to jitter. Filtering (Step 5) is crucial.
*   **Performance:** `mp.solutions.holistic` requires more computational resources than `mp.solutions.pose`. Performance testing on the target hardware is necessary.
*   **OSC Message Limits:** VRChat has limits on OSC message frequency and bundle size. Ensure the number of messages sent per frame is within reasonable bounds. Sending only changed values could be an optimization if needed.
*   **Coordinate Systems/Normalization:** Correctly calculating and normalizing angles for curl/splay requires careful definition of vectors and tuning of normalization ranges to match avatar expectations.
*   **Avatar Setup:** The system will only work correctly with VRChat avatars specifically configured to accept OSC input for finger tracking parameters.

## 7. Future Enhancements

*   **Gesture Recognition:** Implement logic to detect specific static or dynamic hand gestures based on the calculated finger states (e.g., fist, pointing, peace sign).
*   **Adaptive Filtering:** Tune filter parameters dynamically based on movement speed.
*   **UI Controls:** Add GUI elements to fine-tune finger tracking sensitivity or filter parameters.

