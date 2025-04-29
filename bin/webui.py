# webui.py Flask application for remote parameter control.

import logging
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify # Added jsonify

# Global variable to hold the parameters object
params = None

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# --- Constants for Button Actions ---
options = {
    "<<": -10,
    "<": -1,
    ">": 1,
    ">>": 10
}

# --- Helper Function ---
def safe_float(value, default=0.0):
    """Safely convert a value to float, returning default on error."""
    try:
        return float(value)
    except (ValueError, TypeError):
        logging.warning(f"Could not convert '{value}' to float, using default {default}.")
        return default

# --- Main Route ---
@app.route('/')
def index():
    """Renders the main control page."""
    # Pass current parameter values to the template
    template_data = {
        "name": "there", # Placeholder name
        "smooth": np.round(params.additional_smoothing, 2),
        "rot_y": np.round(params.euler_rot_y, 1),
        "rot_x": np.round(params.euler_rot_x, 1),
        "rot_z": np.round(params.euler_rot_z, 1),
        "scale": np.round(params.posescale, 2),
        "keyboard_enabled": params.keyboard_head_control_enabled,
        "keyboard_speed": np.round(params.keyboard_head_control_speed, 2)
    }
    return render_template('index.html', **template_data)

# --- Parameter Update Routes ---

@app.route('/smoothing', methods=["POST"])
def smoothing():
    """Handles smoothing parameter updates."""
    try:
        if "action" in request.form and request.form["action"] in options:
            newvalue = params.additional_smoothing + 0.1 * options[request.form["action"]]
        elif "value" in request.form:
            newvalue = safe_float(request.form["value"], params.additional_smoothing)
        else:
            return redirect(url_for("index")) # No valid input

        newvalue = np.clip(newvalue, 0, 1)
        params.change_additional_smoothing(newvalue, paramid=1) # Assuming profile 1 for webui
    except Exception as e:
        logging.error(f"Error in /smoothing route: {e}", exc_info=True)
    return redirect(url_for("index"))

@app.route('/roty', methods=["POST"])
def roty():
    """Handles Y rotation updates."""
    try:
        if "action" in request.form and request.form["action"] in options:
            cur = params.euler_rot_y
            change = options[request.form["action"]]
            params.rot_change_y(cur + change)
    except Exception as e:
        logging.error(f"Error in /roty route: {e}", exc_info=True)
    return redirect(url_for("index"))

@app.route('/rotx', methods=["POST"])
def rotx():
    """Handles X rotation updates."""
    try:
        if "action" in request.form and request.form["action"] in options:
            cur = params.euler_rot_x
            change = options[request.form["action"]]
            params.rot_change_x(cur + change)
    except Exception as e:
        logging.error(f"Error in /rotx route: {e}", exc_info=True)
    return redirect(url_for("index"))

@app.route('/rotz', methods=["POST"])
def rotz():
    """Handles Z rotation updates."""
    try:
        if "action" in request.form and request.form["action"] in options:
            cur = params.euler_rot_z
            change = options[request.form["action"]]
            params.rot_change_z(cur + change)
    except Exception as e:
        logging.error(f"Error in /rotz route: {e}", exc_info=True)
    return redirect(url_for("index"))

@app.route('/scale', methods=["POST"])
def scale():
    """Handles scale updates."""
    try:
        if "action" in request.form and request.form["action"] in options:
            cur = params.posescale
            # Adjust change magnitude for scale
            change = options[request.form["action"]] / 100.0
            params.change_scale(cur + change)
    except Exception as e:
        logging.error(f"Error in /scale route: {e}", exc_info=True)
    return redirect(url_for("index"))

@app.route('/autocalib', methods=["POST"])
def autocalib():
    """Triggers autocalibration via the inference GUI instance."""
    try:
        if params.gui and hasattr(params.gui, 'autocalibrate'):
            # Trigger autocalibration in the main GUI thread
            # Note: This relies on params.gui being set and might have thread-safety implications
            # depending on what autocalibrate does. Consider using a thread-safe queue if issues arise.
            params.gui.autocalibrate()
            logging.info("Autocalibration triggered via WebUI.")
        else:
            logging.warning("Cannot trigger autocalibration: params.gui not set or lacks autocalibrate method.")
    except Exception as e:
        logging.error(f"Error triggering autocalibration via WebUI: {e}", exc_info=True)
    return redirect(url_for("index"))

# --- NEW: Keyboard Control Route ---
@app.route('/keyboard_control_update', methods=['POST'])
def keyboard_control_update():
    """Handles updates for keyboard head control enable/speed via JSON."""
    global params # Ensure we're using the global params object
    if not params:
        return jsonify({"status": "error", "message": "Parameters not initialized"}), 500

    try:
        data = request.get_json()
        if not data:
             return jsonify({"status": "error", "message": "No JSON data received"}), 400

        enabled = data.get('enabled')
        speed = data.get('speed')

        # Update parameters if values are provided in the JSON data
        if enabled is not None:
            # Ensure the parameter object has the method before calling
            if hasattr(params, 'change_keyboard_control_enabled'):
                 params.change_keyboard_control_enabled(bool(enabled))
            else:
                 # Fallback: Directly set attribute if method missing (less ideal)
                 params.keyboard_head_control_enabled = bool(enabled)
                 logging.warning("Called keyboard_control_update but params object missing change_keyboard_control_enabled method.")

        if speed is not None:
             # Ensure the parameter object has the method before calling
             if hasattr(params, 'change_keyboard_control_speed'):
                 params.change_keyboard_control_speed(safe_float(speed, params.keyboard_head_control_speed))
             else:
                 # Fallback: Directly set attribute if method missing (less ideal)
                 params.keyboard_head_control_speed = safe_float(speed, params.keyboard_head_control_speed)
                 logging.warning("Called keyboard_control_update but params object missing change_keyboard_control_speed method.")

        return jsonify({"status": "success"})

    except Exception as e:
        logging.error(f"Failed to update keyboard control via WebUI: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# --- Main Driver Function ---
def start_webui(param_obj):
    """Starts the Flask web server."""
    global params
    params = param_obj # Store the passed parameters object globally

    # Configure Flask logging to match application level if desired
    # log = logging.getLogger('werkzeug')
    # log.setLevel(logging.INFO) # Or ERROR to reduce console spam

    logging.info("Starting WebUI...")
    logging.info("Access the WebUI via the displayed 'Running on' IP addresses in your browser.")
    try:
        # run() method of Flask class runs the application
        # on the local development server.
        # host='0.0.0.0' makes it accessible on the network.
        app.run(host="0.0.0.0", port=5000) # Default Flask port is 5000
    except OSError as e:
        logging.error(f"Failed to start WebUI (port 5000 might be in use?): {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while running the WebUI: {e}", exc_info=True)

