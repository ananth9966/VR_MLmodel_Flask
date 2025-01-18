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

    if not isinstance(data, list):
        return jsonify({"status": "error", "message": "Expected a batch of JSON objects"}), 400

    feature_batch = []
    indices = []  # To keep track of original indices for predictions

    # Extract feature values for each entry and log data
    for idx, entry in enumerate(data):
        try:
            feature_values = [entry[feature] for feature in input_features]
            feature_batch.append(feature_values)
            data_entry = {"timestamp": datetime.now(), "data": feature_values}
            data_log.append(data_entry)
            indices.append(idx)
        except KeyError as e:
            return jsonify({"status": "error", "message": f"Missing feature {e} in entry {idx}"}), 400

    # Convert the batch of features into a NumPy array
    raw_input_data = np.array(feature_batch, dtype=np.float32)

    try:
        # Scale the entire batch using the Scaler ONNX model
        scaler_inputs = {scaler_session.get_inputs()[0].name: raw_input_data}
        scaled_data = scaler_session.run(None, scaler_inputs)[0]

        # Run the prediction on the entire scaled batch
        model_inputs = {model_session.get_inputs()[0].name: scaled_data}
        output = model_session.run(None, model_inputs)

        # Assuming the model outputs an array of shape (batch_size, 1) or (batch_size,)
        predictions_array = output[0]
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during batch prediction: {e}"}), 500

    batch_predictions = []
    # Process each prediction and log it
    for i, pred in enumerate(predictions_array):
        # If prediction comes as an array with one element, extract that element
        prediction_value = float(pred[0]) if isinstance(pred, (list, np.ndarray)) and np.size(pred) == 1 else float(pred)
        prediction_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "prediction": prediction_value
        }
        prediction_log.append(prediction_entry)
        batch_predictions.append({"index": indices[i], "prediction": prediction_value})

    return jsonify({"status": "success", "predictions": batch_predictions}), 200

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
