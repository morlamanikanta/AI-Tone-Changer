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
    "X-Title": "OpenRouter Web Chatbot",
    "Content-Type": "application/json"
}
MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

# Rate limiter settings
MAX_REQUESTS_PER_MINUTE = 10  # Adjust based on your API tier limits
REQUEST_WINDOW = 60  # seconds
request_timestamps = deque()
rate_limit_lock = Lock()

# Set page title and description
title = "OpenRouter AI Chatbot"
description = """
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
  <div>
    <p>This chatbot uses the OpenRouter API with the Deepseek model. Ask me anything!</p>
    <p><small>Note: This chatbot is rate-limited to {MAX_REQUESTS_PER_MINUTE} requests per minute.</small></p>
  </div>
</div>
""".format(MAX_REQUESTS_PER_MINUTE=MAX_REQUESTS_PER_MINUTE)

def is_rate_limited():
    """Check if the current request would exceed the rate limit"""
    with rate_limit_lock:
        now = time.time()
        
        # Remove timestamps older than the window
        while request_timestamps and now - request_timestamps[0] > REQUEST_WINDOW:
            request_timestamps.popleft()
        
        # Check if we've hit the limit
        if len(request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
            return True
        
        # Add current timestamp and allow the request
        request_timestamps.append(now)
        return False

def clean_response(response: str, prompt: str) -> str:
    """Clean the response using same logic as original chatbot"""
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    # Remove any subsequent QA pairs
    for marker in ["\nQ:", "\nA:", "\nHuman:", "\nAssistant:"]:
        if marker in response:
            response = response.split(marker, 1)[0].strip()
    
    # Ensure complete sentences
    if not re.search(r'[.!?;]\s*$', response):
        last_punct = max([response.rfind(c) for c in '.!?;'])
        if last_punct > 0:
            response = response[:last_punct+1]
        else:
            response += "."
    
    return response

def generate_response(message, history):
    """Generate a response using the OpenRouter API"""
    try:
        # Check rate limit
        if is_rate_limited():
            return "Rate limit exceeded. Please try again in a moment."
        
        # Make API request
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "model": MODEL,
                "messages": [
                    # Include chat history for context
                    *[{"role": "user" if i % 2 == 0 else "assistant", 
                       "content": msg} 
                      for i, (user_msg, ai_msg) in enumerate(history) for msg in (user_msg, ai_msg)],
                    # Add the current message
                    {"role": "user", "content": message}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=60  # Add timeout to prevent hanging
        ).json()

        # Check for errors
        if "error" in response:
            return f"API Error: {response['error']['message']}"

        # Extract and clean the response
        full_response = response['choices'][0]['message']['content']
        cleaned_response = clean_response(full_response, message)
        
        return cleaned_response

    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=generate_response,
    title=title,
    description=description,
    examples=[
        "Tell me a short story",
        "How do I create a Python function?",
        "What is the capital of France?",
        "Write code to sort a list in Python"
    ],
    theme="soft"
)

# Launch the app
if __name__ == "__main__":
    print("Starting OpenRouter Web Chatbot...")
    demo.launch(share=True)  # Set share=False if you don't want to create a public link 
