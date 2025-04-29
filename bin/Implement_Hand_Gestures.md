# Project Breakdown: Implement Hand Gestures as VRChat OSC Inputs

This checklist outlines the steps required to modify the existing project to detect hand gestures (pinch for trigger, fist for grab) using MediaPipe landmarks and send corresponding OSC messages to VRChat.

## Phase 1: Gesture Detection Logic (`helpers.py`)

-   [ ] **Modify `_get_landmark_coords`:** Ensure this helper function exists and correctly returns `np.ndarray` or `None`. (Already exists, verify functionality).
-   [ ] **Add Pinch Detection Logic:**
    -   Create a new helper function `calculate_pinch_strength(landmarks, thumb_tip_idx, index_tip_idx) -> Optional[float]`.
    -   Inside this function:
        -   Get coordinates for `thumb_tip_idx` and `index_tip_idx` using `_get_landmark_coords`.
        -   If both landmarks exist, calculate the Euclidean distance between them.
        -   Define a `max_pinch_distance` (e.g., `0.05` - needs tuning) and a `min_pinch_distance` (e.g., `0.01` - needs tuning).
        -   Normalize the distance: `strength = 1.0 - np.clip((distance - min_pinch_distance) / (max_pinch_distance - min_pinch_distance), 0.0, 1.0)`. This gives `1.0` for fully pinched, `0.0` for open.
        -   Return the calculated `strength`. Return `None` if landmarks are missing.
-   [ ] **Add Grab Detection Logic:**
    -   Create a new helper function `calculate_grab_strength(finger_curls: Dict[str, Optional[float]]) -> Optional[float]`.
    -   Inside this function:
        -   Check if curls for 'IndexCurl', 'MiddleCurl', 'RingCurl', 'PinkyCurl' exist in the input dictionary.
        -   Calculate the average curl of these four fingers (ignoring `None` values).
        -   Define a `min_grab_curl` (e.g., `0.6` - needs tuning) and a `max_grab_curl` (e.g., `0.9` - needs tuning).
        -   Normalize the average curl: `strength = np.clip((avg_curl - min_grab_curl) / (max_grab_curl - min_grab_curl), 0.0, 1.0)`. This gives `0.0` for open, `1.0` for fully grabbed.
        -   Return the calculated `strength`. Return `None` if insufficient curl data exists.
-   [ ] **Modify `process_holistic_hands` Return Type:**
    -   Change the return type annotation to `Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]`. The dictionary will now hold curls, splays, *and* gesture strengths.
-   [ ] **Integrate Gesture Calculations into `process_holistic_hands`:**
    -   Inside the `if left_hand_landmarks:` block:
        -   After calculating curls and splays, call `calculate_pinch_strength` using `left_hand_landmarks` and the correct indices (`HAND_LANDMARK.THUMB_TIP`, `HAND_LANDMARK.INDEX_FINGER_TIP`). Store the result in `left_data['PinchStrength']`.
        -   Call `calculate_grab_strength` using the calculated `left_data` curls. Store the result in `left_data['GrabStrength']`.
    -   Repeat the above steps for the `if right_hand_landmarks:` block, storing results in `right_data['PinchStrength']` and `right_data['GrabStrength']`.
    -   Ensure the function returns the `left_data` and `right_data` dictionaries, which now contain the new strength values (or `None` if not calculated).

## Phase 2: OSC Message Sending (`backends.py`)

-   [ ] **Update `VRChatOSCBackend.updatepose` Signature:**
    -   Ensure the `processed_hand_data` parameter annotation matches the new return type from `process_holistic_hands`: `Optional[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]`.
-   [ ] **Add OSC Input Addresses:**
    -   Define constants for VRChat input addresses near the top of the file (if not already present for other inputs):
        ```python
        OSC_INPUT_USE_LEFT = "/input/UseLeft"
        OSC_INPUT_GRAB_LEFT = "/input/GrabLeft"
        OSC_INPUT_USE_RIGHT = "/input/UseRight"
        OSC_INPUT_GRAB_RIGHT = "/input/GrabRight"
        # Add others like Jump, MoveForward etc. if mapping other gestures/buttons later
        ```
-   [ ] **Modify `VRChatOSCBackend.updatepose` Logic:**
    -   Inside the `if params.use_hands and processed_hand_data:` block:
        -   Unpack `left_hand_dict, right_hand_dict = processed_hand_data`.
        -   **After** sending the parameter bundles (curls/splays), create a *new* OSC bundle specifically for inputs.
        -   Initialize `input_bundle_builder = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)` and `input_added = False`.
        -   **Left Hand Inputs:**
            -   If `left_hand_dict` is not `None`:
                -   Get pinch strength: `pinch_val = left_hand_dict.get('PinchStrength', 0.0)` (default to 0.0 if missing).
                -   Get grab strength: `grab_val = left_hand_dict.get('GrabStrength', 0.0)` (default to 0.0 if missing).
                -   Build messages using `_build_osc_message`:
                    -   `use_msg = _build_osc_message(OSC_INPUT_USE_LEFT, pinch_val)`
                    -   `grab_msg = _build_osc_message(OSC_INPUT_GRAB_LEFT, grab_val)`
                -   Add valid messages to the bundle:
                    -   `if use_msg: input_bundle_builder.add_content(use_msg); input_added = True`
                    -   `if grab_msg: input_bundle_builder.add_content(grab_msg); input_added = True`
        -   **Right Hand Inputs:**
            -   If `right_hand_dict` is not `None`:
                -   Get pinch strength: `pinch_val = right_hand_dict.get('PinchStrength', 0.0)`.
                -   Get grab strength: `grab_val = right_hand_dict.get('GrabStrength', 0.0)`.
                -   Build messages:
                    -   `use_msg = _build_osc_message(OSC_INPUT_USE_RIGHT, pinch_val)`
                    -   `grab_msg = _build_osc_message(OSC_INPUT_GRAB_RIGHT, grab_val)`
                -   Add valid messages to the bundle:
                    -   `if use_msg: input_bundle_builder.add_content(use_msg); input_added = True`
                    -   `if grab_msg: input_bundle_builder.add_content(grab_msg); input_added = True`
        -   **Send Input Bundle:**
            -   If `input_added`:
                -   `input_bundle = input_bundle_builder.build()`
                -   `self.client.send(input_bundle)`

## Phase 3: Parameter Passing (`mediapipepose.py`)

-   [ ] **Verify `process_holistic_hands` Call:**
    -   Ensure the call `process_holistic_hands(results.left_hand_landmarks, results.right_hand_landmarks)` is correctly placed within the `if results.pose_world_landmarks:` block (specifically, after pose processing but before calling `backend.updatepose`).
-   [ ] **Verify `backend.updatepose` Call:**
    -   Ensure the `processed_hand_data` variable (containing the output from `process_holistic_hands`) is correctly passed to `backend.updatepose`. The signature should look like: `backend.updatepose(params, pose3d, rots, processed_hand_data, face_landmarks_for_backend)`.

## Phase 4: Tuning and Testing

-   [ ] **Test Basic Functionality:** Run the application with the VRChatOSC backend enabled and `use_hands=True`. Observe if VRChat receives OSC messages on port 9000 (using OSC debug tools or VRChat's built-in OSC debug menu).
-   [ ] **Tune Pinch Thresholds (`helpers.py`):**
    -   Adjust `max_pinch_distance` and `min_pinch_distance` in `calculate_pinch_strength` until the pinch gesture feels responsive and covers the desired range of motion for triggering `/input/Use`. Log the calculated distance and strength during testing.
-   [ ] **Tune Grab Thresholds (`helpers.py`):**
    -   Adjust `min_grab_curl` and `max_grab_curl` in `calculate_grab_strength` until the fist/grab gesture feels responsive for triggering `/input/Grab`. Log the average curl and calculated strength during testing.
-   [ ] **(Optional) Add GUI/WebUI Controls for Thresholds:**
    -   Add new parameters in `parameters.py` for the min/max pinch distances and grab curls.
    -   Load/Save these parameters in `parameters.py`.
    -   Add sliders or entry boxes in `inference_gui.py` and potentially `webui.py` to allow runtime adjustment of these thresholds without restarting.
    -   Update the `calculate_pinch_strength` and `calculate_grab_strength` functions to use these parameters instead of hardcoded values.

