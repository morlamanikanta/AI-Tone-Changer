import os
import requests
import gradio as gr
import time
from dotenv import load_dotenv
from collections import deque
from threading import Lock

# Load environment variables from .env file
load_dotenv()

# OpenRouter API configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "HTTP-Referer": "https://github.com/your-username/your-repo",
    "X-Title": "OpenRouter Tone Generator",
    "Content-Type": "application/json"
}
MODEL = "arcee-ai/trinity-large-preview:free"

# Rate limiter settings
MAX_REQUESTS_PER_MINUTE = 10
MAX_REQUESTS_PER_DAY = 100
MINUTE_WINDOW = 60
DAY_WINDOW = 24 * 60 * 60
minute_timestamps = deque()
day_timestamps = deque()
rate_limit_lock = Lock()

# Page title and description
title = "Emotion Based Text Style Transfer"
description = f"""
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
  <p>Transform your text into any tone using the OpenRouter API.</p>
  <p><small>Rate limit: {MAX_REQUESTS_PER_MINUTE}/min, {MAX_REQUESTS_PER_DAY}/day</small></p>
</div>
"""

# ================== CUSTOM CSS (FIXED & WORKING) ==================
custom_css = """
/* ===== IMPORTANT: APPLY BACKGROUND TO GRADIO ROOT ===== */
.gradio-container {
    min-height: 100vh;
    background: linear-gradient(135deg, #e6f2ff, #f2f9ff) !important;
    font-family: "Segoe UI", Roboto, Arial, sans-serif;
    padding: 30px;
}

/* ===== Card Layout ===== */
.gr-box {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 25px rgba(0, 123, 255, 0.2);
}

/* ===== Headings ===== */
h1, h2, h3 {
    text-align: center;
    color: #003366;
}

/* ===== Labels ===== */
label {
    font-weight: 600;
    color: #003366;
}

/* ===== Input Fields ===== */
input, textarea {
    border-radius: 12px;
    border: 1px solid #90c2ff;
    padding: 10px;
    font-size: 14px;
}

input:focus, textarea:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.2);
}

/* ===== Buttons (BLUE) ===== */
button {
    background: linear-gradient(135deg, #4da3ff, #007bff);
    color: white;
    border-radius: 12px;
    font-weight: 600;
    font-size: 15px;
    padding: 10px 16px;
    border: none;
    transition: all 0.2s ease-in-out;
}

button:hover {
    transform: translateY(-1px);
    background: linear-gradient(135deg, #007bff, #0056b3);
}

/* ===== Examples Section ===== */
.gr-examples {
    border-radius: 14px;
    background: #f2f9ff;
    padding: 12px;
}
"""

# ================== RATE LIMIT ==================
def is_rate_limited():
    with rate_limit_lock:
        now = time.time()

        while minute_timestamps and now - minute_timestamps[0] > MINUTE_WINDOW:
            minute_timestamps.popleft()
        if len(minute_timestamps) >= MAX_REQUESTS_PER_MINUTE:
            return "Rate limit exceeded. Please wait a minute."

        while day_timestamps and now - day_timestamps[0] > DAY_WINDOW:
            day_timestamps.popleft()
        if len(day_timestamps) >= MAX_REQUESTS_PER_DAY:
            return "Daily limit reached. Try again tomorrow."

        minute_timestamps.append(now)
        day_timestamps.append(now)
        return False

def get_tone_description(tone):
    predefined = {
        "playful": "fun and lighthearted",
        "serious": "formal and grave",
        "formal": "professional and proper",
        "casual": "relaxed and informal",
        "professional": "business-appropriate",
        "friendly": "warm and approachable",
        "enthusiastic": "energetic and excited",
        "sarcastic": "humorous with irony",
        "poetic": "descriptive and metaphorical",
        "technical": "precise and accurate"
    }
    return predefined.get(tone.lower(), f"{tone} emotional style")

def generate_tone_variation(text, tone):
    try:
        rate_limit_status = is_rate_limited()
        if rate_limit_status:
            return rate_limit_status

        tone_style = get_tone_description(tone)

        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": f"Rewrite text in a {tone} tone ({tone_style})."},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=60
        ).json()

        if "error" in response:
            return f"API Error: {response['error']['message']}"

        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Error: {e}"

# ================== GRADIO UI ==================
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your text",
                placeholder="Type or paste your text here...",
                lines=5
            )
            tone_input = gr.Textbox(
                label="Enter emotion / tone",
                placeholder="e.g., happy, sad, romantic, angry, formal, relaxed"
            )
            generate_btn = gr.Button("Generate Tone Variation")

        with gr.Column():
            output = gr.Textbox(
                label="Modified text",
                lines=5,
                interactive=False
            )

    gr.Examples(
        examples=[
            ["I missed the bus.", "sad"],
            ["The assignment is due tomorrow.", "anxious"],
            ["I love this new cafe!", "happy"],
            ["This is so boring.", "sarcastic"],
            ["Let’s celebrate your success!", "joyful"],
        ],
        inputs=[text_input, tone_input]
    )

    generate_btn.click(
        fn=generate_tone_variation,
        inputs=[text_input, tone_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)
