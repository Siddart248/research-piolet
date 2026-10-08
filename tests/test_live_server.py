"""Script to test live HTTP requests against FastAPI running under Uvicorn."""

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from app.main import app


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="warning")


if __name__ == "__main__":
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(1.5)  # Allow server to start

    base_url = "http://127.0.0.1:8008"

    print("Testing GET /health ...")
    req = urllib.request.Request(f"{base_url}/health")
    with urllib.request.urlopen(req) as resp:
        print(f"Status: {resp.status}, Body: {resp.read().decode('utf-8')}")

    print("\nTesting POST /research ...")
    data = json.dumps({"query": "What are the limitations of RAG?"}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/research", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        print(f"Status: {resp.status}")
        print(f"Confidence: {res_json.get('confidence')}")
        print(f"Sources: {res_json.get('sources')}")
        print(f"Answer Snippet: {res_json.get('answer')[:120]}...")

    print("\nTesting POST /chat ...")
    data = json.dumps({"message": "What is 17% of 8500?", "session_id": "live_test_session"}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        print(f"Status: {resp.status}")
        print(f"Answer: {res_json.get('answer')}")

    print("\nTesting GET /evaluation ...")
    req = urllib.request.Request(f"{base_url}/evaluation")
    with urllib.request.urlopen(req) as resp:
        eval_json = json.loads(resp.read().decode("utf-8"))
        print(f"Status: {resp.status}")
        print(f"Pass Rate: {eval_json.get('metrics', {}).get('overall_pass_rate')}")

    print("\nALL LIVE HTTP ENDPOINT TESTS PASSED!")
