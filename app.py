from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # <-- ADD THIS to handle CORS

# Load the trained model from the pickle file
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route to serve the index.html (optional if needed)
@app.route('/', methods=['GET'])
def index():
    return "Server is running!"  # (Simple text or you can serve HTML if you want)

# Prediction route
@app.route('/check_rank', methods=['POST'])
def predict():
    try:
        # Get the form data (for JSON input)
        data = request.get_json()
        name = data['name']
        score = float(data['score'])  # Ensure score is float

        # Predict the rank
        predicted_rank = model.predict(np.array([[score]]))  # Make sure it's 2D array
        predicted_rank = predicted_rank[0]  # Get the single value

        # Prepare the message
        if predicted_rank > 3500:
            message = "You are not having sufficient skills! Please practice more problems."
        else:
            message = "You have sufficient skills! You can apply for any problem-solving interviews."

        # Return the result as a JSON response
        return jsonify({'status': 'success', 'message': message, 'rank': predicted_rank})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'status': 'error', 'message': 'Invalid input or error in prediction: ' + str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(port=3000, debug=True)
