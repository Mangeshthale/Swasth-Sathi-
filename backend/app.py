from flask import Flask, request, jsonify, send_file, render_template
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import os
import pickle
import numpy as np
import pandas as pd

# 🔹 Initialize Flask
app = Flask(__name__, template_folder="templates")

# ==============================🔹 GEMINI CHATBOT SETUP ================================
# 🔹 Configure Gemini API Key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")
chat = model.start_chat()

# 🔹 Initialize Speech Recognizer
recognizer = sr.Recognizer()

@app.route("/")
def home():
    return render_template("chatbot.html")  # chatbot.html must be in templates/

def get_gemini_response(user_text):
    if not user_text:
        return "I didn't catch that. Please try again."
    try:
        response = chat.send_message(user_text)
        return response.text if hasattr(response, 'text') else "🤖 AI did not return a response."
    except Exception as e:
        return f"⚠️ Error getting AI response: {str(e)}"

def text_to_speech(response_text):
    try:
        tts = gTTS(response_text, lang="en")
        tts.save("response.mp3")
        return "response.mp3"
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

@app.route("/chat", methods=["POST"])
def chat_route():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"response": "Please say something.", "audio": ""})

    response = get_gemini_response(user_message)
    audio_file = text_to_speech(response)

    if audio_file:
        return jsonify({"response": response, "audio": f"/audio/{audio_file}"})
    else:
        return jsonify({"response": response, "audio": ""})

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_file(filename, mimetype="audio/mpeg")

# ==============================🔹 ML MODEL SETUP ================================
# 🔹 Load data and model
sym_des = pd.read_csv("data/symtoms_df.csv")
precautions = pd.read_csv("data/precautions_df.csv")
workout = pd.read_csv("data/workout_df.csv")
description = pd.read_csv("data/description.csv")
medications = pd.read_csv("data/medications.csv")
diets = pd.read_csv("data/diets.csv")

svc = pickle.load(open("model/mnb.pkl", "rb"))

# 🔹 Load symptom and disease dicts
# (Keep as-is or read from external JSON if needed)
symptoms_dict = {...}  # truncate for brevity; use your full dictionary here
diseases_list = {...}  # same here

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]
    wrkout = workout[workout['disease'] == dis]['workout']
    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

@app.route("/diagnose", methods=["GET", "POST"])
def diagnose():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms" or not symptoms:
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('sym.html', message=message)
        else:
            user_symptoms = [s.strip("[]' ") for s in symptoms.split(',')]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            my_precautions = [i for i in precautions[0]]

            return render_template('sym.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications,
                                   my_diet=rec_diet, workout=workout)

    return render_template('sym.html')

# ==============================🔹 APP RUN ================================
if __name__ == "__main__":
    print("🚀 Unified Chatbot + ML Diagnosis Server is Running...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
