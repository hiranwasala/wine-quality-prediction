from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # Import CORS

# Load the trained ML model
model = joblib.load('model.pkl')

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        quality = "High" if prediction[0] == 1 else "Low"

        return jsonify({"quality": quality})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
