from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process
import time
import json
import random

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

print_highlight("Starting server with cache telemetry enabled...")

server_process, port = launch_server_cmd(
    """
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
 --host 0.0.0.0 --enable-cache-report --mem-fraction-static 0.345
"""
)

wait_for_server(f"http://localhost:{port}")
print_highlight("Server started successfully!")

import requests

def send_request(prompt, request_id=None):
    """Send a request to the server and return the response"""
    url = f"http://localhost:{port}/v1/chat/completions"
    
    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
    }
    
    if request_id:
        data["request_id"] = request_id
    
    response = requests.post(url, json=data)
    return response.json()

def check_telemetry():
    """Get the current telemetry data from the server"""
    try:
        with open("cache_telemetry.json", "r") as f:
            telemetry_data = json.load(f)
        return telemetry_data
    except FileNotFoundError:
        print("Telemetry file not found yet")
        return None
    except json.JSONDecodeError:
        print("Error decoding telemetry file")
        return None

# List of prompts to use for testing - significantly expanded
prompts = [
    # Geography prompts
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?",
    "What is the capital of United Kingdom?",
    "What is the capital of United States?",
    "What is the capital of Canada?",
    "What is the capital of Mexico?",
    "What is the capital of Brazil?",
    "What is the capital of Japan?",
    "What is the capital of China?",
    "What is the capital of India?",
    "What is the capital of Australia?",
    "What is the capital of Russia?",
    
    "Explain the theory of relativity",
    "What is quantum mechanics?",
    "How does photosynthesis work?",
    "Explain the water cycle",
    "What is the structure of DNA?",
    "How do vaccines work?",
    "What is the difference between mitosis and meiosis?",
    "Explain the greenhouse effect",
    
    "Who was Napoleon Bonaparte?",
    "What caused World War I?",
    "What was the Renaissance?",
    "Who was Julius Caesar?",
    "What was the Industrial Revolution?",
    "Who was Genghis Khan?",
    
    "What is machine learning?",
    "How do computers work?",
    "What is cloud computing?",
    "Explain how the internet works",
    "What is blockchain technology?",
    "How do smartphones work?",
    
    "Write a short story about a robot who discovers emotions",
    "Explain the significance of climate change and potential solutions",
    "Compare and contrast democracy and authoritarianism",
    "Discuss the ethical implications of artificial intelligence",
    "Explain the importance of biodiversity in ecosystems",
]

try:
    # First round: Send a batch of initial requests
    print_highlight("\n=== init request sending ===")
    for i in range(30):
        prompt_idx = random.randint(0, len(prompts)-1)
        # prompt_idx = i
        send_request(prompts[prompt_idx], f"init-{i}")

    print_highlight("\n=== Waiting for telemetry thread to record stats ===")
    time.sleep(6)

finally:
    print_highlight("\nShutting down server...")
    terminate_process(server_process)