import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)  # allow frontend (Netlify / localhost) to call this API

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"  # or "gemini-1.5-flash" if you want

SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant called Gemini Chat. "
    "Answer clearly and concisely. Use simple language and avoid very long paragraphs."
)

@app.route("/")
def home():
    # Just to test backend is up
    return jsonify({"status": "ok", "message": "Gemini backend running"})

@app.route("/api/chat", methods=["POST"])
def chat():
    if not API_KEY:
        return jsonify({"error": "Server configuration error: API Key missing"}), 500

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        conversation_text = SYSTEM_PROMPT + "\n\n"

        for turn in history:
            role = turn.get("role", "user")
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            if role == "user":
                conversation_text += f"User: {text}\n"
            else:
                conversation_text += f"Assistant: {text}\n"

        conversation_text += f"User: {user_message}\nAssistant:"

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=conversation_text,
        )

        bot_reply = (getattr(response, "text", "") or "").strip()
        if not bot_reply:
            bot_reply = "Sorry, I couldn't generate a reply."

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print("Error calling Gemini API:", repr(e))
        return jsonify({"error": "Error contacting Gemini API"}), 500


if __name__ == "__main__":
    app.run(debug=True)
