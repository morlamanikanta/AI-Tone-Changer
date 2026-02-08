import os
import requests
import re
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
MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

# Rate limiter settings
MAX_REQUESTS_PER_MINUTE = 10
MAX_REQUESTS_PER_DAY = 100
MINUTE_WINDOW = 60  # seconds
DAY_WINDOW = 24 * 60 * 60  # seconds (24 hours)
minute_timestamps = deque()
day_timestamps = deque()
rate_limit_lock = Lock()

# Set page title and description
title = "AI Tone Generator"
description = """
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
  <div>
    <p>Transform your text into different tones using the Deepseek V3 model via OpenRouter API. Select a tone and paste your text to get started!</p>
    <p><small>Note: This service is rate-limited to {MAX_REQUESTS_PER_MINUTE} requests per minute and {MAX_REQUESTS_PER_DAY} requests per day.</small></p>
  </div>
</div>
""".format(
    MAX_REQUESTS_PER_MINUTE=MAX_REQUESTS_PER_MINUTE,
    MAX_REQUESTS_PER_DAY=MAX_REQUESTS_PER_DAY
)

def is_rate_limited():
    """Check if the current request would exceed the rate limit"""
    with rate_limit_lock:
        now = time.time()
        
        # Check minute limit
        while minute_timestamps and now - minute_timestamps[0] > MINUTE_WINDOW:
            minute_timestamps.popleft()
        if len(minute_timestamps) >= MAX_REQUESTS_PER_MINUTE:
            return "Rate limit exceeded. Please wait a minute before trying again."
        
        # Check daily limit
        while day_timestamps and now - day_timestamps[0] > DAY_WINDOW:
            day_timestamps.popleft()
        if len(day_timestamps) >= MAX_REQUESTS_PER_DAY:
            return "Daily limit reached. Please try again tomorrow."
        
        # Add current timestamp to both queues
        minute_timestamps.append(now)
        day_timestamps.append(now)
        return False

def get_tone_description(tone):
    """Get the description and example for each tone"""
    tone_descriptions = {
        "playful": "fun and lighthearted, using casual language and maybe even some wordplay",
        "serious": "formal and grave, emphasizing importance and gravity",
        "formal": "professional and proper, using business etiquette and formal vocabulary",
        "casual": "relaxed and informal, like talking to a friend",
        "professional": "business-appropriate, maintaining clarity and professionalism",
        "friendly": "warm and approachable, like chatting with a close friend",
        "enthusiastic": "energetic and excited, using upbeat language and positive expressions",
        "sarcastic": "subtly humorous with a touch of irony and wit",
        "poetic": "flowery and descriptive, using metaphors and vivid language",
        "technical": "precise and technical, focusing on accuracy and specificity"
    }
    return tone_descriptions.get(tone, "neutral")

def generate_tone_variation(text, tone):
    """Generate a tone variation using the OpenRouter API"""
    try:
        # Check rate limit
        rate_limit_status = is_rate_limited()
        if rate_limit_status:
            return rate_limit_status
        
        # Get tone description
        tone_style = get_tone_description(tone)
        
        # Create the system message and user prompt
        system_message = f"""You are an expert at rewriting text in different tones.
Your task is to rewrite the given text in a {tone} tone ({tone_style})."""

        user_prompt = f"""Please rewrite this text in a {tone} tone:
{text}

Make it {tone_style}"""

        # Make API request
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=60
        ).json()

        # Check for errors
        if "error" in response:
            return f"API Error: {response['error']['message']}"

        # Extract and clean the response
        generated_text = response['choices'][0]['message']['content']
        return generated_text.strip()

    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
with gr.Blocks(theme="soft") as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your text",
                placeholder="Type or paste your text here...",
                lines=5
            )
            tone_dropdown = gr.Dropdown(
                choices=[
                    "playful",
                    "serious",
                    "formal",
                    "casual",
                    "professional",
                    "friendly",
                    "enthusiastic",
                    "sarcastic",
                    "poetic",
                    "technical"
                ],
                label="Select tone",
                value="formal"
            )
            generate_btn = gr.Button("Generate Tone Variation")
        
        with gr.Column():
            output = gr.Textbox(
                label="Modified text",
                lines=5,
                interactive=False
            )
    
    # Example inputs
    gr.Examples(
        examples=[
            ["The meeting is scheduled for tomorrow at 2 PM.", "casual"],
            ["I love this new restaurant!", "formal"],
            ["The project deadline is approaching.", "playful"],
            ["The weather is beautiful today.", "poetic"],
            ["Your appointment is on June 1st at 4:30 PM.", "friendly"],
            ["The system requires 16GB of RAM.", "technical"]
        ],
        inputs=[text_input, tone_dropdown]
    )
    
    generate_btn.click(
        fn=generate_tone_variation,
        inputs=[text_input, tone_dropdown],
        outputs=output
    )

# Launch the app
if __name__ == "__main__":
    print("Starting OpenRouter Tone Generator...")
    demo.launch(share=True)  # Set share=False if you don't want to create a public link 
