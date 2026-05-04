import json
import os
import urllib.error
import urllib.request


def check_models():
    api_base = "http://localhost:8888/v1"
    api_key = os.getenv("UNSLOTH_API_KEY", "sk-no-key-required")

    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Checking models at: {api_base}/models")
    try:
        req = urllib.request.Request(f"{api_base}/models", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            print(f"Status: {response.getcode()}")
            models = json.loads(response.read().decode("utf-8"))
            print(json.dumps(models, indent=2))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode()}")
    except Exception as e:
        print(f"General Error: {e}")


if __name__ == "__main__":
    check_models()
