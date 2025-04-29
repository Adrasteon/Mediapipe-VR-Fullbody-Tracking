# keyboard_helper.py

import threading
import logging
from pynput import keyboard
from typing import Tuple

# --- Configuration ---
DEFAULT_ROTATION_SPEED_DEG_PER_TICK = 1.5 # Degrees applied per update cycle when key is held

# --- Shared State ---
# Use a lock for thread safety when modifying/reading state from different threads
_state_lock = threading.Lock()
_yaw_delta_deg = 0.0    # Accumulated yaw change in degrees
_pitch_delta_deg = 0.0  # Accumulated pitch change in degrees
_roll_delta_deg = 0.0   # Accumulated roll change (optional, e.g., with Q/E keys)
_rotation_speed_deg = DEFAULT_ROTATION_SPEED_DEG_PER_TICK
_control_enabled = False
_listener_thread = None
_stop_event = threading.Event()

# --- Keyboard Listener Callbacks ---
def _on_press(key):
    """Handles key press events."""
    global _yaw_delta_deg, _pitch_delta_deg, _roll_delta_deg
    if not _control_enabled:
        return

    try:
        # Use lock to safely modify shared state
        with _state_lock:
            if key == keyboard.Key.left:
                _yaw_delta_deg -= _rotation_speed_deg
            elif key == keyboard.Key.right:
                _yaw_delta_deg += _rotation_speed_deg
            elif key == keyboard.Key.up:
                _pitch_delta_deg += _rotation_speed_deg
            elif key == keyboard.Key.down:
                _pitch_delta_deg -= _rotation_speed_deg
            # Optional: Add Roll control (e.g., Q/E keys)
            # elif key == keyboard.KeyCode.from_char('q'):
            #     _roll_delta_deg += _rotation_speed_deg
            # elif key == keyboard.KeyCode.from_char('e'):
            #     _roll_delta_deg -= _rotation_speed_deg

    except AttributeError:
        # Ignore non-special keys or keys we don't handle
        pass
    except Exception as e:
        logging.error(f"[KeyboardHelper] Error in on_press: {e}", exc_info=True)


def _on_release(key):
    """Handles key release events."""
    # Currently, we apply rotation based on accumulated delta per frame loop,
    # so release doesn't need to reset the delta immediately.
    # If you wanted rotation to *stop* on release, you'd add logic here.
    pass

# --- Listener Control ---
def _listener_loop():
    """Target function for the listener thread."""
    logging.info("[KeyboardHelper] Listener thread started.")
    # Setup the listener (blocking call within this thread)
    with keyboard.Listener(on_press=_on_press, on_release=_on_release) as listener:
        # Keep the thread alive until stop_event is set
        _stop_event.wait() # Wait until stop() is called
        listener.stop()
    logging.info("[KeyboardHelper] Listener thread stopped.")

def start_keyboard_listener():
    """Starts the keyboard listener in a separate thread."""
    global _listener_thread
    if _listener_thread is None or not _listener_thread.is_alive():
        _stop_event.clear() # Ensure stop event is reset
        _listener_thread = threading.Thread(target=_listener_loop, daemon=True, name="KeyboardListenerThread")
        _listener_thread.start()
        logging.info("[KeyboardHelper] Keyboard listener initiated.")
    else:
        logging.warning("[KeyboardHelper] Listener already running.")

def stop_keyboard_listener():
    """Stops the keyboard listener thread."""
    global _listener_thread
    if _listener_thread and _listener_thread.is_alive():
        logging.info("[KeyboardHelper] Stopping keyboard listener...")
        _stop_event.set() # Signal the listener thread to exit
        _listener_thread.join(timeout=1.0) # Wait briefly for the thread to finish
        if _listener_thread.is_alive():
             logging.warning("[KeyboardHelper] Listener thread did not stop gracefully.")
        _listener_thread = None
        logging.info("[KeyboardHelper] Keyboard listener stopped.")
    else:
        logging.info("[KeyboardHelper] Listener not running or already stopped.")

# --- Public Interface ---
def get_and_reset_keyboard_deltas() -> Tuple[float, float, float]:
    """
    Safely retrieves the current accumulated rotation deltas (yaw, pitch, roll)
    in degrees and resets them to zero.

    Returns:
        Tuple[float, float, float]: (yaw_delta, pitch_delta, roll_delta) in degrees.
    """
    global _yaw_delta_deg, _pitch_delta_deg, _roll_delta_deg
    with _state_lock:
        yaw = _yaw_delta_deg
        pitch = _pitch_delta_deg
        roll = _roll_delta_deg
        # Reset after reading
        _yaw_delta_deg = 0.0
        _pitch_delta_deg = 0.0
        _roll_delta_deg = 0.0
        return yaw, pitch, roll

def set_keyboard_control_enabled(enabled: bool):
    """Enable or disable keyboard head rotation control."""
    global _control_enabled
    with _state_lock:
        if _control_enabled != enabled:
            _control_enabled = enabled
            logging.info(f"[KeyboardHelper] Keyboard control {'enabled' if enabled else 'disabled'}.")
            # Reset deltas when disabling to prevent leftover rotation
            if not enabled:
                global _yaw_delta_deg, _pitch_delta_deg, _roll_delta_deg
                _yaw_delta_deg = 0.0
                _pitch_delta_deg = 0.0
                _roll_delta_deg = 0.0

def is_keyboard_control_enabled() -> bool:
    """Check if keyboard control is currently enabled."""
    with _state_lock:
        return _control_enabled

def set_keyboard_rotation_speed(speed_deg_per_tick: float):
    """Set the rotation speed in degrees per update cycle."""
    global _rotation_speed_deg
    if speed_deg_per_tick > 0:
        with _state_lock:
            _rotation_speed_deg = speed_deg_per_tick
            logging.info(f"[KeyboardHelper] Rotation speed set to {speed_deg_per_tick:.2f} deg/tick.")
    else:
        logging.warning("[KeyboardHelper] Rotation speed must be positive.")

def get_keyboard_rotation_speed() -> float:
    """Get the current rotation speed."""
    with _state_lock:
        return _rotation_speed_deg

# --- Example Usage (for testing this file directly) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("Starting keyboard helper test.")
    print("Use Arrow Keys (Left/Right/Up/Down) to control rotation.")
    print("Press Ctrl+C to exit.")

    start_keyboard_listener()
    set_keyboard_control_enabled(True) # Enable control for testing

    try:
        while True:
            # In a real application, this would be inside the main loop
            yaw, pitch, roll = get_and_reset_keyboard_deltas()
            if yaw != 0.0 or pitch != 0.0 or roll != 0.0:
                print(f"Tick: Yaw Delta={yaw:.2f}, Pitch Delta={pitch:.2f}, Roll Delta={roll:.2f}")

            # Simulate main loop work
            time.sleep(0.05) # Simulate ~20 FPS update rate

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping...")
    finally:
        set_keyboard_control_enabled(False) # Disable before stopping
        stop_keyboard_listener()
        print("Keyboard helper test finished.")

