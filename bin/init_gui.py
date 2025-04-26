# init_gui.py Creates the initial parameter setup GUI using Tkinter.
import tkinter as tk
import sys
import pickle
import logging
from sys import platform

# Default parameters dictionary
DEFAULT_PARAMS = {
    "camid": '0', # Default to 0 for USB webcam
    "imgsize": 640,
    # "neckoffset": [0.0, -0.2, 0.1], # Neck offset is handled in parameters.py/saved_params.json
    "prevskel": False,
    # "waithmd": False, # Seems unused
    "feetrot": False,
    "use_hands": False,
    "use_face": False, # Added face tracking default
    "ignore_hip": False,
    "camera_settings": False,
    "camera_width": 640,
    "camera_height": 480,
    "model_complexity": 1,
    "smooth_landmarks": True,
    "min_tracking_confidence": 0.5,
    # "static_image": False, # Removed, not used by Holistic
    "backend": 1, # Default to SteamVR
    "backend_ip": "127.0.0.1",
    "backend_port": 9000,
    "advanced": False,
    "webui": False,
    # Internal flag, not loaded/saved directly
    "switch_advanced": False
}

def set_advanced(window, param):
    """Sets flag to switch advanced mode and closes window."""
    param["switch_advanced"] = True
    window.quit()

def getparams():
    """Creates and runs the initial parameter GUI, returning selected parameters."""
    param = {}
    try:
        # Load previously saved parameters to pre-fill the GUI
        loaded_params = pickle.load(open("params.p", "rb"))
        # Update defaults with loaded values, keeping defaults for missing keys
        param = DEFAULT_PARAMS.copy()
        param.update(loaded_params)
    except FileNotFoundError:
        logging.info("params.p not found, using default values for initial GUI.")
        param = DEFAULT_PARAMS.copy()
    except Exception as e:
        logging.error(f"Error loading params.p: {e}. Using default values.")
        param = DEFAULT_PARAMS.copy()

    window = tk.Tk()
    window.title("Initial Setup")

    def on_close():
        """Handles window close event."""
        logging.info("Initial setup window closed by user. Exiting.")
        window.destroy()
        sys.exit(0) # Exit cleanly if initial window is closed

    window.protocol("WM_DELETE_WINDOW", on_close)

    # --- Camera Section ---
    camera_frame = tk.LabelFrame(window, text="Camera Setup", padx=5, pady=5)
    camera_frame.pack(padx=10, pady=5, fill="x")

    if not param["advanced"]:
        tk.Label(camera_frame, text="Use '0', '1', etc. for USB webcams.\nUse 'http://<ip>:<port>/video' for IP Webcam app.", justify=tk.LEFT).pack()

    tk.Label(camera_frame, text="Camera ID or IP URL:").pack()
    camid_entry = tk.Entry(camera_frame, width=50)
    camid_entry.pack()
    camid_entry.insert(0, param["camid"])

    if not param["advanced"]:
        tk.Label(camera_frame, text="NOTE: Higher resolution impacts performance.", justify=tk.LEFT).pack()

    tk.Label(camera_frame, text="Camera Width (0 for auto):").pack()
    camwidth_entry = tk.Entry(camera_frame, width=20)
    camwidth_entry.pack()
    camwidth_entry.insert(0, param["camera_width"])

    tk.Label(camera_frame, text="Camera Height (0 for auto):").pack()
    camheight_entry = tk.Entry(camera_frame, width=20)
    camheight_entry.pack()
    camheight_entry.insert(0, param["camera_height"])

    if platform == "win32":
        if not param["advanced"]:
            tk.Label(camera_frame, text="NOTE: 'Attempt camera settings' might change camera ID or behavior.", justify=tk.LEFT).pack()
        varcamsettings = tk.BooleanVar(value=param["camera_settings"])
        cam_settings_check = tk.Checkbutton(camera_frame, text="Attempt to open camera settings (Windows DShow)", variable=varcamsettings)
        cam_settings_check.pack()
    else:
        varcamsettings = tk.BooleanVar(value=False) # Ensure variable exists even if not packed

    # --- Tracking Options Section ---
    tracking_frame = tk.LabelFrame(window, text="Tracking Options", padx=5, pady=5)
    tracking_frame.pack(padx=10, pady=5, fill="x")

    varfeet = tk.BooleanVar(value=param["feetrot"])
    rot_feet_check = tk.Checkbutton(tracking_frame, text="Enable experimental foot rotation", variable=varfeet)
    rot_feet_check.pack(anchor=tk.W)

    if not param["advanced"]:
        tk.Label(tracking_frame, text="NOTE: VRChat works better with hip tracking.\nOnly disable if using other software (e.g., owoTrack).", justify=tk.LEFT).pack()

    varhip = tk.BooleanVar(value=param["ignore_hip"])
    hip_check = tk.Checkbutton(tracking_frame, text="Disable hip tracker", variable=varhip)
    hip_check.pack(anchor=tk.W)

    # Hand and Face Tracking Options
    varhand = tk.BooleanVar(value=param["use_hands"])
    hand_check = tk.Checkbutton(tracking_frame, text="Enable hand tracking features (requires capable backend)", variable=varhand)
    hand_check.pack(anchor=tk.W)

    varface = tk.BooleanVar(value=param["use_face"]) # Added face checkbox
    face_check = tk.Checkbutton(tracking_frame, text="Enable face tracking features (visual only unless backend supports)", variable=varface)
    face_check.pack(anchor=tk.W)

    # --- Backend Section ---
    backend_outer_frame = tk.LabelFrame(window, text="Backend Communication", padx=5, pady=5)
    backend_outer_frame.pack(padx=10, pady=5, fill="x")

    backend_selection_frame = tk.Frame(backend_outer_frame)
    backend_options_frame = tk.Frame(backend_outer_frame)

    varbackend = tk.IntVar(value=param["backend"])

    def show_hide_backend_options():
        """Shows/hides IP/Port options based on backend selection."""
        if varbackend.get() == 2: # VRChatOSC
            backend_options_frame.pack(side=tk.BOTTOM, fill="x", pady=(5,0))
        else:
            backend_options_frame.pack_forget()

    tk.Label(backend_selection_frame, text="Backend: ").pack(side=tk.LEFT)
    tk.Radiobutton(backend_selection_frame, text="SteamVR", variable=varbackend, value=1, command=show_hide_backend_options).pack(side=tk.LEFT)
    tk.Radiobutton(backend_selection_frame, text="VRChatOSC", variable=varbackend, value=2, command=show_hide_backend_options).pack(side=tk.LEFT)
    backend_selection_frame.pack(side=tk.TOP)

    # Options specific to VRChatOSC
    tk.Label(backend_options_frame, text="OSC IP:").pack(side=tk.LEFT)
    backend_ip_entry = tk.Entry(backend_options_frame, width=15)
    backend_ip_entry.insert(0, param["backend_ip"])
    backend_ip_entry.pack(side=tk.LEFT, padx=(0, 5))
    tk.Label(backend_options_frame, text="Port:").pack(side=tk.LEFT)
    backend_port_entry = tk.Entry(backend_options_frame, width=5)
    backend_port_entry.insert(0, param["backend_port"])
    backend_port_entry.pack(side=tk.LEFT)

    show_hide_backend_options() # Initial check

    # --- WebUI Section ---
    webui_frame = tk.LabelFrame(window, text="Web Interface", padx=5, pady=5)
    webui_frame.pack(padx=10, pady=5, fill="x")
    varwebui = tk.BooleanVar(value=param["webui"])
    webui_check = tk.Checkbutton(webui_frame, text="Enable WebUI (control parameters from another device)", variable=varwebui)
    webui_check.pack(anchor=tk.W)

    # --- Advanced Mode Section ---
    if param["advanced"]:
        advanced_frame = tk.LabelFrame(window, text="Advanced MediaPipe Settings", padx=5, pady=5)
        advanced_frame.pack(padx=10, pady=5, fill="x")

        tk.Label(advanced_frame, text="Max Image Size (pixels):").pack()
        maximgsize_entry = tk.Entry(advanced_frame, width=20)
        maximgsize_entry.pack()
        maximgsize_entry.insert(0, param["imgsize"])

        tk.Label(advanced_frame, text="Model Complexity (0=Fast, 1=Default, 2=Accurate):").pack()
        modelc_entry = tk.Entry(advanced_frame, width=20)
        modelc_entry.pack()
        modelc_entry.insert(0, param["model_complexity"])

        varmsmooth = tk.BooleanVar(value=param["smooth_landmarks"])
        msmooth_check = tk.Checkbutton(advanced_frame, text="Smooth Landmarks (MediaPipe internal)", variable=varmsmooth)
        msmooth_check.pack(anchor=tk.W)

        tk.Label(advanced_frame, text="Min Tracking Confidence (0.0 - 1.0):").pack()
        trackc_entry = tk.Entry(advanced_frame, width=20)
        trackc_entry.pack()
        trackc_entry.insert(0, param["min_tracking_confidence"])

        # Removed static image mode checkbox
        # varstatic = tk.BooleanVar(value=param["static_image"])
        # static_check = tk.Checkbutton(advanced_frame, text="Static image mode", variable=varstatic)
        # static_check.pack(anchor=tk.W)

        varskel = tk.BooleanVar(value=param["prevskel"])
        skeleton_check = tk.Checkbutton(advanced_frame, text="DEV: Preview skeleton in VR (SteamVR only)", variable=varskel)
        skeleton_check.pack(anchor=tk.W)
    else:
        # Need dummy variables if not in advanced mode so retrieval doesn't fail
        maximgsize_entry = None
        modelc_entry = None
        varmsmooth = tk.BooleanVar(value=DEFAULT_PARAMS["smooth_landmarks"])
        trackc_entry = None
        # varstatic = tk.BooleanVar(value=DEFAULT_PARAMS["static_image"])
        varskel = tk.BooleanVar(value=DEFAULT_PARAMS["prevskel"])


    # --- Control Buttons ---
    button_frame = tk.Frame(window)
    button_frame.pack(padx=10, pady=10, fill="x")

    param["switch_advanced"] = False # Reset flag before showing button
    advanced_button_text = 'Disable Advanced Mode' if param["advanced"] else 'Enable Advanced Mode'
    tk.Button(button_frame, text=advanced_button_text, command=lambda: set_advanced(window, param)).pack(side=tk.LEFT, expand=True)

    tk.Button(button_frame, text='Save and Continue', command=window.quit).pack(side=tk.RIGHT, expand=True)

    window.mainloop()

    # --- Retrieve values after mainloop finishes ---
    final_params = {}
    final_params["camid"] = camid_entry.get()
    final_params["camera_width"] = int(camwidth_entry.get()) if camwidth_entry.get().isdigit() else DEFAULT_PARAMS["camera_width"]
    final_params["camera_height"] = int(camheight_entry.get()) if camheight_entry.get().isdigit() else DEFAULT_PARAMS["camera_height"]
    final_params["camera_settings"] = varcamsettings.get()

    final_params["feetrot"] = varfeet.get()
    final_params["ignore_hip"] = varhip.get()
    final_params["use_hands"] = varhand.get()
    final_params["use_face"] = varface.get() # Get face tracking value

    final_params["backend"] = varbackend.get()
    final_params["backend_ip"] = backend_ip_entry.get()
    final_params["backend_port"] = int(backend_port_entry.get()) if backend_port_entry.get().isdigit() else DEFAULT_PARAMS["backend_port"]

    final_params["webui"] = varwebui.get()

    # Retrieve advanced params only if they were displayed
    if param["advanced"] and maximgsize_entry:
        final_params["imgsize"] = int(maximgsize_entry.get()) if maximgsize_entry.get().isdigit() else DEFAULT_PARAMS["imgsize"]
        final_params["model_complexity"] = int(modelc_entry.get()) if modelc_entry.get().isdigit() else DEFAULT_PARAMS["model_complexity"]
        final_params["smooth_landmarks"] = varmsmooth.get()
        try:
            final_params["min_tracking_confidence"] = float(trackc_entry.get())
        except ValueError:
            final_params["min_tracking_confidence"] = DEFAULT_PARAMS["min_tracking_confidence"]
        # final_params["static_image"] = varstatic.get() # Removed
        final_params["prevskel"] = varskel.get()
    else:
        # Use defaults if not in advanced mode
        final_params["imgsize"] = DEFAULT_PARAMS["imgsize"]
        final_params["model_complexity"] = DEFAULT_PARAMS["model_complexity"]
        final_params["smooth_landmarks"] = DEFAULT_PARAMS["smooth_landmarks"]
        final_params["min_tracking_confidence"] = DEFAULT_PARAMS["min_tracking_confidence"]
        # final_params["static_image"] = DEFAULT_PARAMS["static_image"] # Removed
        final_params["prevskel"] = DEFAULT_PARAMS["prevskel"]

    # Carry over the advanced mode switch flag
    final_params["advanced"] = param["advanced"]
    if param["switch_advanced"]:
        final_params["advanced"] = not param["advanced"] # Toggle if button was pressed

    # Save the final selected parameters for next launch
    try:
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
