from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Paths for the ONNX models
scaler_path = 'standard_scaler.onnx'
model_path = 'random_forest_regressor.onnx'

# Load the ONNX Scaler model
try:
    scaler_session = ort.InferenceSession(scaler_path)
    print("Scaler model loaded successfully.")
    for input_meta in scaler_session.get_inputs():
        print(f"Scaler Input Name: {input_meta.name}")
        print(f"Scaler Input Shape: {input_meta.shape}")
except Exception as e:
    print("Error loading scaler model:", e)

# Load the ONNX Random Forest model
try:
    model_session = ort.InferenceSession(model_path)
    print("Prediction model loaded successfully.")
    for input_meta in model_session.get_inputs():
        print(f"Model Input Name: {input_meta.name}")
        print(f"Model Input Shape: {input_meta.shape}")
except Exception as e:
    print("Error loading prediction model:", e)

# Buffers to store continuous data and predictions
data_log = []          # Continuous storage for all received data
prediction_log = []    # Rolling storage for predictions

# Define the input feature order for the model
input_features = [
    "Position_X", "Position_Y", "Position_Z", "Rotation_X", "Rotation_Y",
    "Rotation_Z", "Rotation_W", "Framerate", "TimeDiff",
    "VelocityX", "VelocityY", "VelocityZ", "AccelerationX", "AccelerationY",
    "AccelerationZ", "LinearAcceleration", "LinearVelocity", "LinearJerk",
    "AngularVelocityX", "AngularVelocityY", "AngularVelocityZ",
    "AngularAccelerationX", "AngularAccelerationY", "AngularAccelerationZ",
    "AngularJerkX", "AngularJerkY", "AngularJerkZ",
    "AngularVelocityMagnitude", "AngularAccelerationMagnitude",
    "AngularJerkMagnitude"
]

@app.route('/headtracking', methods=['POST'])
def receive_headtracking_data():
    global data_log, prediction_log

    # Get JSON data from request
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    # Ensure the data is a list of lists
    if not isinstance(data, list) or not all(isinstance(entry, list) for entry in data):
        return jsonify({"status": "error", "message": "Expected a batch of lists"}), 400

    # Convert incoming data to a NumPy array
    try:
        data_array = np.array(data, dtype=np.float32)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Data conversion error: {e}"}), 400

    # Validate feature dimensions
    if data_array.ndim != 2 or data_array.shape[1] != len(input_features):
        return jsonify({"status": "error", 
                        "message": "Each entry must have the correct number of features"}), 400

    try:
        # Scale the entire batch of rows
        scaler_inputs = {scaler_session.get_inputs()[0].name: data_array}
        scaled_data = scaler_session.run(None, scaler_inputs)[0]

        # Run predictions on the scaled batch
        model_inputs = {model_session.get_inputs()[0].name: scaled_data}
        predictions = model_session.run(None, model_inputs)[0]

        # Compute the average prediction across all rows
        prediction_value = float(np.mean(predictions))

        # Log each individual row with a timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for row in data_array:
            data_log.append({"timestamp": timestamp, "data": row.tolist()})
        
        # Log the aggregated prediction with timestamp
        prediction_log.append({"timestamp": timestamp, "prediction": prediction_value})

        # Return the aggregated prediction
        return jsonify({"status": "success", "prediction": prediction_value}), 200

    except Exception as e:
        return jsonify({"status": "error", 
                        "message": f"Error during batch processing: {e}"}), 500


@app.route('/display')
def display_data():
    return render_template('display.html', data=data_log)

@app.route('/prediction')
def display_prediction():
    # Compute the mean of the last 10 predictions
    recent_predictions = [entry["prediction"] for entry in prediction_log[-10:]]
    if recent_predictions:
        aggregate_prediction = sum(recent_predictions) / len(recent_predictions)
    else:
        aggregate_prediction = "N/A"

    return render_template(
        'prediction.html',
        predictions=prediction_log,
        aggregate=aggregate_prediction
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
