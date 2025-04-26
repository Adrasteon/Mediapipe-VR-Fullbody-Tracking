# Speed and Accuracy Improvements for MediaPipe VR Tracking

This document outlines potential areas for improving the speed and accuracy of the MediaPipe-based VR full-body tracking project.

## 1. CUDA Acceleration

*   **MediaPipe Python & CUDA:** The standard `mediapipe` Python package primarily uses CPU inference. Direct CUDA acceleration is not typically exposed or easily configured via the high-level Python API. GPU acceleration, if used, often relies on delegates like OpenGL ES (via ANGLE on Windows) or Metal (on macOS/iOS).
*   **"Nvidia Compute Components Loaded":** Messages indicating NVIDIA components are loaded usually stem from other libraries like **OpenCV** or **PyTorch/TensorFlow**, which might be installed with CUDA support. This doesn't automatically mean MediaPipe's pose estimation is running on CUDA.
*   **Achieving CUDA Speedup:**
    *   **Different Library/Model:** Replace MediaPipe's `mp_pose` with a different pose estimation model and inference engine that has robust Python CUDA bindings. Options include:
        *   **ONNX Runtime + CUDA Execution Provider:** Run models (e.g., from MMPose, converted to ONNX) using ONNX Runtime configured for CUDA.
        *   **NVIDIA TensorRT:** Convert models to TensorRT format for optimized inference on NVIDIA GPUs.
        *   **PyTorch/TensorFlow:** Use models directly within these frameworks configured for GPU execution.
        *   *Note:* This requires significant refactoring (replacing `pose.process`, `mediapipeTo3dpose`, etc.).
    *   **MediaPipe C++:** Use the MediaPipe C++ API for more control over GPU delegates. Requires C++ bindings or a separate process.

## 2. Accuracy Improvements

*   **MediaPipe Model Complexity:**
    *   Use `params.model = 2` (Heavy) for MediaPipe's highest accuracy, accepting the speed trade-off. Found in `mediapipepose.py`.
*   **Confidence Thresholds:**
    *   Experiment with `min_detection_confidence` (currently hardcoded at 0.5 in `mediapipepose.py`) and `params.min_tracking_confidence`.
    *   Higher values might improve stability but risk losing tracking more easily.
*   **Input Resolution:**
    *   Higher camera resolution (`params.camera_width`, `params.camera_height`) or `params.maximgsize` can improve detail detection but impacts speed.
*   **Post-Processing / Filtering:**
    *   **Current:** `VRChatOSCBackend` uses a simple exponential moving average (`additional_smoothing`).
    *   **Recommendation:** Implement a more sophisticated filter like the **OneEuroFilter** on `pose3d` points or calculated `rots` *before* sending to the backend (in `mediapipepose.py`). This reduces jitter while minimizing lag. Python implementations are available.
*   **Calibration (`autocalibrate` in `inference_gui.py`):**
    *   Ensure the user performs the calibration pose correctly (e.g., T-pose facing camera).
    *   Review the `arctan2` math and coordinate system assumptions if calibration results seem inaccurate.
    *   Verify the scale calibration logic (`headsetpos[1]/skelSize[1]`) assumption holds (user's feet near Y=0 in VR space).

## 3. Speed Improvements (Beyond CUDA)

*   **MediaPipe Model Complexity:**
    *   Use `params.model = 0` (Lite) or `1` (Full) for faster inference if accuracy is sufficient.
*   **Input Resolution:**
    *   Decrease `params.maximgsize`, `params.camera_width`, or `params.camera_height`. Smaller images process faster.
*   **Threading:**
    *   The current threading model (Camera, GUI, WebUI, Main) is generally good.
    *   **Crucial:** Implement thread safety (using `threading.Lock`) in `CameraStream` (`helpers.py`) when accessing shared image data (`image_from_thread`, `image_ready`) to prevent race conditions, as discussed previously.
*   **Code Profiling:**
    *   Use Python's `cProfile` on the `main()` function in `mediapipepose.py` to identify bottlenecks outside the core `pose.process()` call (e.g., in preprocessing, rotation calculations).
*   **Avoid Unnecessary Work:**
    *   Ensure calculations are conditional (e.g., `get_rot_hands` only if `params.use_hands`). The code generally follows this.
*   **Preprocessing Optimization:**
    *   The current preprocessing (rotate, flip, resize) in `mediapipepose.py` uses OpenCV and NumPy, which are generally efficient. Profiling can confirm if optimization is needed here.

## Recommendations & Next Steps

1.  **Confirm CUDA Source:** Verify which library is actually loading NVIDIA components.
2.  **Implement OneEuroFilter:** High potential impact on perceived accuracy/smoothness.
3.  **Benchmark Model Complexity:** Test models 0, 1, 2 for speed/accuracy trade-off.
4.  **Implement Thread Safety:** Add `threading.Lock` to `CameraStream`.
5.  **Profile:** Use `cProfile` to find non-MediaPipe bottlenecks.
6.  **Consider ONNX Runtime:** If CUDA is essential and current performance is insufficient, investigate ONNX Runtime + CUDA.
