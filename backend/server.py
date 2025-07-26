from flask import Flask, request, jsonify, send_file, render_template
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import os
import pygame

app = Flask(__name__, template_folder="templates")

# 🔹 Configure Gemini API Key
genai.configure(api_key="GOOGLE_API_KEY")  # Replace with your real API key
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

if __name__ == "__main__":
    print("🚀 Chatbot Server is Running...")
    app.run(debug=True)
