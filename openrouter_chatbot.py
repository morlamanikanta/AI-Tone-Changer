import os
import requests
import re
import time
from dotenv import load_dotenv
from collections import deque

# Load environment variables from .env file
load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "HTTP-Referer": "https://github.com/your-username/your-repo",
    "X-Title": "TinyLM Chatbot",
    "Content-Type": "application/json"
}
MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

# Rate limiter settings
MAX_REQUESTS_PER_MINUTE = 10  # Adjust based on your API tier limits
REQUEST_WINDOW = 60  # seconds
request_timestamps = deque()

def is_rate_limited():
    """Check if the current request would exceed the rate limit"""
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

def main():
    print("=== OpenRouter Chat ===")
    print(f"Note: Rate limited to {MAX_REQUESTS_PER_MINUTE} requests per minute")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            # Check rate limit
            if is_rate_limited():
                print("\nRate limit exceeded. Please wait a moment before trying again.\n")
                # Sleep for a short time to avoid rapid retries
                time.sleep(2) 
                continue

            response = requests.post(
                API_URL,
                headers=HEADERS,
                json={
                    "model": MODEL,
                    "messages": [{
                        "role": "user",
                        "content": user_input
                    }],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            ).json()

            if "error" in response:
                print(f"API Error: {response['error']['message']}")
                continue

            full_response = response['choices'][0]['message']['content']
            cleaned = clean_response(full_response, user_input)
            
            print(f"\nAssistant: {cleaned}\n")

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
