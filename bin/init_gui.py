# init_gui.py Creates the initial parameter setup GUI using Tkinter.
import tkinter as tk
import sys
import pickle
import logging
from sys import platform
from tkinter import ttk # Use themed widgets

# Default parameters dictionary
DEFAULT_PARAMS = {
    "camid": '0', # Default to 0 for USB webcam
    "imgsize": 640,
    "prevskel": False,
    "feetrot": False,
    "use_hands": False,
    "use_face": False,
    "ignore_hip": False,
    "camera_settings": False,
    "camera_width": 640,
    "camera_height": 480,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "min_tracking_confidence": 0.5,
    "backend": 1, # Default to SteamVR
    "backend_ip": "127.0.0.1",
    "backend_port": 9000,
    "advanced": False,
    "webui": False,
    # --- Keyboard Control Defaults ---
    "keyboard_head_control_enabled": False, # Default disabled
    "keyboard_head_control_speed": 1.5,     # Default speed (degrees per tick)
    # Internal flag, not loaded/saved directly
    "switch_advanced": False
}

def set_advanced(window, param):
    """Sets flag to switch advanced mode and closes window."""
    param["switch_advanced"] = True
    window.quit()

def safe_float(value, default=0.0):
    """Safely convert a value to float, returning default on error."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert a value to int, returning default on error."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def getparams():
    """Creates and runs the initial parameter GUI, returning selected parameters."""
    param = {}
    try:
        # Load previously saved parameters to pre-fill the GUI
        # Note: Using pickle can be fragile. Consider switching to JSON like saved_params.json
        loaded_params = pickle.load(open("params.p", "rb"))
        # Update defaults with loaded values, keeping defaults for missing keys
        param = DEFAULT_PARAMS.copy()
        param.update(loaded_params)
        logging.info("Loaded previous settings from params.p")
    except FileNotFoundError:
        logging.info("params.p not found, using default values for initial GUI.")
        param = DEFAULT_PARAMS.copy()
    except Exception as e:
        logging.error(f"Error loading params.p: {e}. Using default values.")
        param = DEFAULT_PARAMS.copy()

    window = tk.Tk()
    window.title("Initial Setup")
    # Apply theme if available
    try:
        style = ttk.Style(window)
        style.theme_use('clam') # Or 'vista', 'xpnative', etc.
    except tk.TclError:
        logging.warning("Failed to set ttk theme. Using default.")

    def on_close():
        """Handles window close event."""
        logging.info("Initial setup window closed by user. Exiting.")
        window.destroy()
        sys.exit(0) # Exit cleanly if initial window is closed

    window.protocol("WM_DELETE_WINDOW", on_close)

    # --- Camera Section ---
    camera_frame = ttk.LabelFrame(window, text="Camera Setup", padding=(10, 5))
    camera_frame.pack(padx=10, pady=5, fill="x")

    if not param["advanced"]:
        ttk.Label(camera_frame, text="Use '0', '1', etc. for USB webcams.\nUse 'http://<ip>:<port>/video' for IP Webcam app.", justify=tk.LEFT).pack(anchor=tk.W)

    ttk.Label(camera_frame, text="Camera ID or IP URL:").pack(anchor=tk.W)
    camid_entry = ttk.Entry(camera_frame, width=50)
    camid_entry.pack(fill="x")
    camid_entry.insert(0, param["camid"])

    res_frame = tk.Frame(camera_frame)
    res_frame.pack(fill="x", pady=(5,0))

    if not param["advanced"]:
        ttk.Label(res_frame, text="NOTE: Higher resolution impacts performance.", justify=tk.LEFT).pack(anchor=tk.W)

    ttk.Label(res_frame, text="Width (0=auto):").pack(side=tk.LEFT)
    camwidth_entry = ttk.Entry(res_frame, width=7)
    camwidth_entry.pack(side=tk.LEFT, padx=(0,10))
    camwidth_entry.insert(0, param["camera_width"])

    ttk.Label(res_frame, text="Height (0=auto):").pack(side=tk.LEFT)
    camheight_entry = ttk.Entry(res_frame, width=7)
    camheight_entry.pack(side=tk.LEFT)
    camheight_entry.insert(0, param["camera_height"])

    if platform == "win32":
        if not param["advanced"]:
            ttk.Label(camera_frame, text="NOTE: 'Attempt camera settings' might change camera ID or behavior.", justify=tk.LEFT).pack(anchor=tk.W, pady=(5,0))
        varcamsettings = tk.BooleanVar(value=param.get("camera_settings", False)) # Use .get for safety
        cam_settings_check = ttk.Checkbutton(camera_frame, text="Attempt to open camera settings (Windows DShow)", variable=varcamsettings)
        cam_settings_check.pack(anchor=tk.W)
    else:
        varcamsettings = tk.BooleanVar(value=False) # Ensure variable exists even if not packed

    # --- Tracking Options Section ---
    tracking_frame = ttk.LabelFrame(window, text="Tracking Options", padding=(10, 5))
    tracking_frame.pack(padx=10, pady=5, fill="x")

    varfeet = tk.BooleanVar(value=param.get("feetrot", False))
    rot_feet_check = ttk.Checkbutton(tracking_frame, text="Enable experimental foot rotation", variable=varfeet)
    rot_feet_check.pack(anchor=tk.W)

    if not param["advanced"]:
        ttk.Label(tracking_frame, text="NOTE: VRChat works better with hip tracking.\nOnly disable if using other software (e.g., owoTrack).", justify=tk.LEFT).pack(anchor=tk.W)

    varhip = tk.BooleanVar(value=param.get("ignore_hip", False))
    hip_check = ttk.Checkbutton(tracking_frame, text="Disable hip tracker", variable=varhip)
    hip_check.pack(anchor=tk.W)

    # Hand and Face Tracking Options
    varhand = tk.BooleanVar(value=param.get("use_hands", False))
    hand_check = ttk.Checkbutton(tracking_frame, text="Enable hand tracking features (requires capable backend)", variable=varhand)
    hand_check.pack(anchor=tk.W)

    varface = tk.BooleanVar(value=param.get("use_face", False))
    face_check = ttk.Checkbutton(tracking_frame, text="Enable face tracking features (visual only unless backend supports)", variable=varface)
    face_check.pack(anchor=tk.W)

    # --- Backend Section ---
    backend_outer_frame = ttk.LabelFrame(window, text="Backend Communication", padding=(10, 5))
    backend_outer_frame.pack(padx=10, pady=5, fill="x")

    backend_selection_frame = tk.Frame(backend_outer_frame)
    backend_options_frame = tk.Frame(backend_outer_frame)

    varbackend = tk.IntVar(value=param.get("backend", 1))

    def show_hide_backend_options():
        """Shows/hides IP/Port options based on backend selection."""
        if varbackend.get() == 2: # VRChatOSC
            backend_options_frame.pack(side=tk.BOTTOM, fill="x", pady=(5,0))
        else:
            backend_options_frame.pack_forget()

    ttk.Label(backend_selection_frame, text="Backend: ").pack(side=tk.LEFT)
    ttk.Radiobutton(backend_selection_frame, text="SteamVR", variable=varbackend, value=1, command=show_hide_backend_options).pack(side=tk.LEFT)
    ttk.Radiobutton(backend_selection_frame, text="VRChatOSC", variable=varbackend, value=2, command=show_hide_backend_options).pack(side=tk.LEFT)
    backend_selection_frame.pack(side=tk.TOP, fill="x")

    # Options specific to VRChatOSC
    ttk.Label(backend_options_frame, text="OSC IP:").pack(side=tk.LEFT)
    backend_ip_entry = ttk.Entry(backend_options_frame, width=15)
    backend_ip_entry.insert(0, param.get("backend_ip", "127.0.0.1"))
    backend_ip_entry.pack(side=tk.LEFT, padx=(0, 5))
    ttk.Label(backend_options_frame, text="Port:").pack(side=tk.LEFT)
    backend_port_entry = ttk.Entry(backend_options_frame, width=5)
    backend_port_entry.insert(0, param.get("backend_port", 9000))
    backend_port_entry.pack(side=tk.LEFT)

    show_hide_backend_options() # Initial check

    # --- WebUI Section ---
    webui_frame = ttk.LabelFrame(window, text="Web Interface", padding=(10, 5))
    webui_frame.pack(padx=10, pady=5, fill="x")
    varwebui = tk.BooleanVar(value=param.get("webui", False))
    webui_check = ttk.Checkbutton(webui_frame, text="Enable WebUI (control parameters from another device)", variable=varwebui)
    webui_check.pack(anchor=tk.W)

    # --- Keyboard Control Section ---
    keyboard_frame = ttk.LabelFrame(window, text="Keyboard Head Control (SteamVR Only)", padding=(10, 5))
    keyboard_frame.pack(padx=10, pady=5, fill="x")

    varkeyboard_enabled = tk.BooleanVar(value=param.get("keyboard_head_control_enabled", False))
    keyboard_enabled_check = ttk.Checkbutton(keyboard_frame, text="Enable Keyboard Control", variable=varkeyboard_enabled)
    keyboard_enabled_check.pack(anchor=tk.W)

    keyboard_speed_frame = tk.Frame(keyboard_frame)
    keyboard_speed_frame.pack(fill="x", pady=(5,0))
    ttk.Label(keyboard_speed_frame, text="Rotation Speed (deg/tick):").pack(side=tk.LEFT)
    keyboard_speed_entry = ttk.Entry(keyboard_speed_frame, width=7)
    keyboard_speed_entry.pack(side=tk.LEFT, padx=(5,0))
    keyboard_speed_entry.insert(0, param.get("keyboard_head_control_speed", 1.5))

    # --- Advanced Mode Section ---
    if param["advanced"]:
        advanced_frame = ttk.LabelFrame(window, text="Advanced MediaPipe Settings", padding=(10, 5))
        advanced_frame.pack(padx=10, pady=5, fill="x")

        imgsize_frame = tk.Frame(advanced_frame)
        imgsize_frame.pack(fill="x")
        ttk.Label(imgsize_frame, text="Max Image Size (pixels):").pack(side=tk.LEFT)
        maximgsize_entry = ttk.Entry(imgsize_frame, width=7)
        maximgsize_entry.pack(side=tk.LEFT, padx=(5,0))
        maximgsize_entry.insert(0, param.get("imgsize", 640))

        modelc_frame = tk.Frame(advanced_frame)
        modelc_frame.pack(fill="x", pady=(5,0))
        ttk.Label(modelc_frame, text="Model Complexity:").pack(side=tk.LEFT)
        modelc_combo = ttk.Combobox(modelc_frame, values=[0, 1, 2], width=5, state="readonly")
        modelc_combo.set(param.get("model_complexity", 1))
        modelc_combo.pack(side=tk.LEFT, padx=(5,0))
        ttk.Label(modelc_frame, text="(0=Fast, 1=Default, 2=Accurate)").pack(side=tk.LEFT, padx=(5,0))


        varmsmooth = tk.BooleanVar(value=param.get("smooth_landmarks", True))
        msmooth_check = ttk.Checkbutton(advanced_frame, text="Smooth Landmarks (MediaPipe internal)", variable=varmsmooth)
        msmooth_check.pack(anchor=tk.W, pady=(5,0))

        trackc_frame = tk.Frame(advanced_frame)
        trackc_frame.pack(fill="x", pady=(5,0))
        ttk.Label(trackc_frame, text="Min Tracking Confidence:").pack(side=tk.LEFT)
        trackc_entry = ttk.Entry(trackc_frame, width=5)
        trackc_entry.pack(side=tk.LEFT, padx=(5,0))
        trackc_entry.insert(0, param.get("min_tracking_confidence", 0.5))

        varskel = tk.BooleanVar(value=param.get("prevskel", False))
        skeleton_check = ttk.Checkbutton(advanced_frame, text="DEV: Preview skeleton in VR (SteamVR only)", variable=varskel)
        skeleton_check.pack(anchor=tk.W, pady=(5,0))
    else:
        # Need dummy variables if not in advanced mode so retrieval doesn't fail
        maximgsize_entry = None
        modelc_combo = None
        varmsmooth = tk.BooleanVar(value=DEFAULT_PARAMS["smooth_landmarks"])
        trackc_entry = None
        varskel = tk.BooleanVar(value=DEFAULT_PARAMS["prevskel"])


    # --- Control Buttons ---
    button_frame = tk.Frame(window)
    button_frame.pack(padx=10, pady=10, fill="x")

    param["switch_advanced"] = False # Reset flag before showing button
    advanced_button_text = 'Disable Advanced Mode' if param["advanced"] else 'Enable Advanced Mode'
    ttk.Button(button_frame, text=advanced_button_text, command=lambda: set_advanced(window, param)).pack(side=tk.LEFT, expand=True, padx=(0,5))

    ttk.Button(button_frame, text='Save and Continue', command=window.quit).pack(side=tk.RIGHT, expand=True, padx=(5,0))

    window.mainloop()

    # --- Retrieve values after mainloop finishes ---
    final_params = {}
    final_params["camid"] = camid_entry.get()
    final_params["camera_width"] = safe_int(camwidth_entry.get(), DEFAULT_PARAMS["camera_width"])
    final_params["camera_height"] = safe_int(camheight_entry.get(), DEFAULT_PARAMS["camera_height"])
    final_params["camera_settings"] = varcamsettings.get()

    final_params["feetrot"] = varfeet.get()
    final_params["ignore_hip"] = varhip.get()
    final_params["use_hands"] = varhand.get()
    final_params["use_face"] = varface.get()

    final_params["backend"] = varbackend.get()
    final_params["backend_ip"] = backend_ip_entry.get()
    final_params["backend_port"] = safe_int(backend_port_entry.get(), DEFAULT_PARAMS["backend_port"])

    final_params["webui"] = varwebui.get()

    # --- Retrieve Keyboard Control Params ---
    final_params["keyboard_head_control_enabled"] = varkeyboard_enabled.get()
    final_params["keyboard_head_control_speed"] = safe_float(keyboard_speed_entry.get(), DEFAULT_PARAMS["keyboard_head_control_speed"])

    # Retrieve advanced params only if they were displayed
    if param["advanced"] and maximgsize_entry:
        final_params["imgsize"] = safe_int(maximgsize_entry.get(), DEFAULT_PARAMS["imgsize"])
        final_params["model_complexity"] = safe_int(modelc_combo.get(), DEFAULT_PARAMS["model_complexity"])
        final_params["smooth_landmarks"] = varmsmooth.get()
        final_params["min_tracking_confidence"] = safe_float(trackc_entry.get(), DEFAULT_PARAMS["min_tracking_confidence"])
        final_params["prevskel"] = varskel.get()
    else:
        # Use defaults if not in advanced mode
        final_params["imgsize"] = DEFAULT_PARAMS["imgsize"]
        final_params["model_complexity"] = DEFAULT_PARAMS["model_complexity"]
        final_params["smooth_landmarks"] = DEFAULT_PARAMS["smooth_landmarks"]
        final_params["min_tracking_confidence"] = DEFAULT_PARAMS["min_tracking_confidence"]
        final_params["prevskel"] = DEFAULT_PARAMS["prevskel"]

    # Carry over the advanced mode switch flag
    final_params["advanced"] = param["advanced"]
    if param["switch_advanced"]:
        final_params["advanced"] = not param["advanced"] # Toggle if button was pressed

    # Save the final selected parameters for next launch
    try:
        # Consider changing this to JSON to match saved_params.json
        pickle.dump(final_params, open("params.p", "wb"))
        logging.info("Initial parameters saved to params.p")
    except Exception as e:
        logging.error(f"Error saving initial parameters to params.p: {e}")

    window.destroy()

    # If advanced mode was switched, return None to signal main script to restart loop
    if param["switch_advanced"]:
        logging.info("Advanced mode switched. Restarting parameter loading.")
        return None
    else:
        return final_params

# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    returned_params = getparams()
    if returned_params:
        print("\nParameters selected:")
        import json
        print(json.dumps(returned_params, indent=2))
    else:
        print("\nAdvanced mode switched, loop should restart.")
