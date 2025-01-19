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
    for input in scaler_session.get_inputs():
        print(f"Scaler Input Name: {input.name}")
        print(f"Scaler Input Shape: {input.shape}")
except Exception as e:
    print("Error loading scaler model:", e)

# Load the ONNX Random Forest model
try:
    model_session = ort.InferenceSession(model_path)
    print("Prediction model loaded successfully.")
    for input in model_session.get_inputs():
        print(f"Model Input Name: {input.name}")
        print(f"Model Input Shape: {input.shape}")
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

    # Check if all entries have the correct number of features
    if any(len(entry) != len(input_features) for entry in data):
        return jsonify({"status": "error", "message": "One or more entries have an incorrect number of features"}), 400

    try:
        # Combine the data into a single sample by averaging the rows
        combined_data = np.mean(np.array(data, dtype=np.float32), axis=0)

        # Check the combined data shape matches the expected input features
        if combined_data.shape[0] != len(input_features):
            return jsonify({"status": "error", "message": "Combined data has incorrect feature dimensions"}), 400

        # Prepare data for the scaler
        scaler_inputs = {scaler_session.get_inputs()[0].name: combined_data.reshape(1, -1)}
        scaled_data = scaler_session.run(None, scaler_inputs)[0]

        # Run the prediction on the scaled data
        model_inputs = {model_session.get_inputs()[0].name: scaled_data}
        prediction = model_session.run(None, model_inputs)[0]

        # Assuming the model outputs a single value
        prediction_value = float(prediction[0]) if isinstance(prediction, (list, np.ndarray)) and np.size(prediction) == 1 else float(prediction)

        # Log the combined data and prediction
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data_log.append({"timestamp": timestamp, "data": combined_data.tolist()})
        prediction_log.append({"timestamp": timestamp, "prediction": prediction_value})

        # Return the single prediction
        return jsonify({"status": "success", "prediction": prediction_value}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during batch processing: {e}"}), 500


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
