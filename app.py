from flask import Flask, render_template, request, jsonify, send_file
from gtts import gTTS
from training import (predict_alzheimer, brain_predict, predict_disease)
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json
from werkzeug.utils import secure_filename
from PIL import Image
import base64

app = Flask(__name__)

# Load symptoms
with open("symptoms.json", "r") as f:
    symptoms_list = json.load(f)

# diabetes
with open("models/diabetes-model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# alzheimer
model1 = load_model("models/alzheimer-model.h5")
class2label = {0: 'Mild Demented', 1: 'Moderate Demented', 2: 'Non Demented', 3: 'Very Mild Demented'}
# Configure upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# heart disease
with open('models/heart-model.pkl', 'rb') as model_file:
    heart_model = pickle.load(model_file) 

# kidney disease
with open('models/kidney-model.pkl', 'rb') as model_file:
    kidney_model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            # Get user input from form
            features = [float(request.form[field]) for field in [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]]
            # Preprocess input
            features = np.array(features).reshape(1, -1)
            features = scaler.transform(features)
            # Make prediction
            prediction = model.predict(features)[0]
            result = "Diabetic" if prediction == 1 else "Not Diabetic"
            return render_template('diabetes.html', result=result)
        except Exception as e:
            return render_template('diabetes.html', error="Invalid input! Please enter valid numbers.")
    return render_template('diabetes.html')


@app.route("/alzheimer", methods=["GET", "POST"])
def alzheimer():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({"error": "No file part"})
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})
        # Save uploaded image
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        # Get prediction
        label = predict_alzheimer(file_path)
        # Convert image to base64 for display
        with open(file_path, "rb") as img2str:
            converted_string = base64.b64encode(img2str.read()).decode()
        return jsonify({
            "label": label,
            "photo": f"data:image/png;base64,{converted_string}"
        })
    return render_template("alzheimer.html")


@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        try:
            # Extract features from form data
            features = [float(request.form.get(feature)) for feature in [
                'age', 'gender', 'chest_pain', 'resting_bp', 'cholesterol',
                'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina',
                'oldpeak', 'st_slope', 'ca', 'thal'
            ]]
            features_array = np.array(features).reshape(1, -1)
            prediction = heart_model.predict(features_array)
            result = 'You have heart disease.' if prediction[0] == 1 else 'You do not have heart disease.'
            return render_template('heart.html', prediction_result=result)
        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"
            return render_template('heart.html', error=error_message)
    return render_template('heart.html')


@app.route("/brain", methods=["GET", "POST"])
def brain():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded image
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Call the trained model for prediction
        prediction = brain_predict(file.filename)  # Ensure brain_predict handles filenames correctly

        return jsonify({
            "image_name": filename,
            "prediction": prediction
        })
    
    return render_template('brain.html')

@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        try:
            # Extract features from form data
            features = [float(request.form.get(feature)) for feature in [
                'sg', 'htn', 'hemo', 'dm', 'al',
                'appet', 'rc', 'pc'
            ]]
            features_array = np.array(features).reshape(1, -1)
            prediction = kidney_model.predict(features_array)
            result = 'You have kidney disease.' if prediction[0] == 1 else 'You do not have kidney disease.'
            return render_template('kidney.html', prediction_result=result)
        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"
            return render_template('kidney.html', error=error_message)
    return render_template('kidney.html')

@app.route('/general', methods=['GET', 'POST'])
def general():
    if request.method == "POST":
        user_symptoms = request.form.getlist("symptoms[]")  # Get symptoms as a list
        days = int(request.form.get("days", 5))
        if not user_symptoms:
            return jsonify({"error": "No symptoms entered"}), 400
        advice, predictions = predict_disease(user_symptoms, days)
        response = {
            "advice": advice,
            "predictions": []
        }
        for disease, details in predictions.items():
            response["predictions"].append({
                "disease": disease,
                "description": details["desc"],
                "precautions": details["prec"],
                "medications": details["drugs"]["Medications"],
                "diet": details["drugs"]["Diet"]
            })

        #**Generate Speech**
        speech_text = "The predicted disease is: "
        speech_text += f" {disease}. Description: {details['desc']}. "
        speech_text += f"Precautions to take: {', '.join(details['prec'])}. "
        speech_text += f"Recommended medications: {', '.join(details['drugs']['Medications'])}. "
        speech_text += f"Suggested diet: {', '.join(details['drugs']['Diet'])}. "
        tts = gTTS(text=speech_text, lang='en')
        audio_file = "static/ML/speech.mp3"
        tts.save(audio_file)

        return jsonify(response)

    return render_template('general.html', symptoms=symptoms_list)

@app.route('/speak')
def speak():
    return send_file("static/ML/speech.mp3")

if __name__ == '__main__':
    app.run(debug=True)
