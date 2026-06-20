import json
import os
import urllib.request


def test_text():
    api_base = "http://localhost:8888/v1"
    api_key = os.getenv("UNSLOTH_API_KEY")
    model = "llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF"

    payload = {"model": model, "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"Testing text prompt at: {api_base}/chat/completions")
    try:
        req = urllib.request.Request(
            f"{api_base}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            status = response.getcode()
            content = response.read().decode("utf-8")
            print(f"Status: {status}")
            print(f"Content: {content}")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"General Error: {e}")


if __name__ == "__main__":
    test_text()
