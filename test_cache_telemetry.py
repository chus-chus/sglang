from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process
import time
import random

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

if __name__ == "__main__":
    print_highlight("Starting server with cache telemetry enabled...")

    server_process, port = launch_server_cmd(
        """
    python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --mem-fraction-static 0.345 --enable-cache-telemetry
    """
    )

    wait_for_server(f"http://localhost:{port}")
    print_highlight("Server started successfully!")

    import requests

    def send_request(prompt, request_id=None):
        url = f"http://localhost:{port}/v1/chat/completions"
        
        data = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if request_id:
            data["request_id"] = request_id
        
        response = requests.post(url, json=data)
        return response.json()

    prompts = [
        "No",
        "Yes"
    ]

    try:
        print_highlight("\n=== init request sending ===")
        for i in range(2):
            prompt_idx = random.randint(0, len(prompts)-1)
            # prompt_idx = i
            send_request(prompts[prompt_idx], f"init-{i}")

        print_highlight("\n=== Waiting for telemetry thread to record stats ===")
        time.sleep(6)

    finally:
        print_highlight("\nShutting down server...")
        terminate_process(server_process)