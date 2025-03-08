from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load ML models
models = {
    "gradient": pickle.load(open("models/gradient_model.pkl", "rb")),
    "logreg": pickle.load(open("models/logreg_model.pkl", "rb")),
    "rf": pickle.load(open("models/rf_model.pkl", "rb")),
    "lstm_pickle": pickle.load(open("models/lstm_model.pkl", "rb")),
    #"cnn_pickle": pickle.load("models/cnn_model.pkl","rb"),
   # "lstm_pickle": pickle.load("models/lstm_model.pkl","rb"),
    #"rnn_pickle": pickle.load("models/rnn_model.pkl","rb"),
}
deep_models = {
    "cnn": tf.keras.models.load_model("models/cnn_model.h5"),
    "lstm": tf.keras.models.load_model("models/lstm_model.h5"),
    "rnn": tf.keras.models.load_model("models/rnn_model.h5"),
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_name = data.get("model")
        features = np.array(data.get("features")).reshape(1, -1)

        if model_name in models:
            prediction = models[model_name].predict(features)
        elif model_name in deep_models:
            prediction = deep_models[model_name].predict(features)
        else:
            return jsonify({"error": "Model not found"}), 400

        return jsonify({"model": model_name, "prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)