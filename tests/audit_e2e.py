"""Full end-to-end audit script for ResearchPilot.

Starts Uvicorn in a background thread, runs all 7 HTTP tests, then shuts down.
"""

import json
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from app.main import app

PORT = 8009
BASE_URL = f"http://127.0.0.1:{PORT}"
PASSED = []
FAILED = []


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def get_json(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def report(test_name, status, data, assertions):
    ok = all(assertions.values())
    symbol = "[PASS]" if ok else "[FAIL]"
    print(f"\n{'='*60}")
    print(f"{symbol} {test_name}")
    print(f"HTTP Status      : {status}")
    if isinstance(data, dict) and "answer" in data:
        print(f"Answer Snippet   : {data.get('answer', '')[:150]}...")
        print(f"Sources          : {data.get('sources')}")
        print(f"Tool Calls       : {data.get('tool_calls')}")
        print(f"Confidence       : {data.get('confidence')}")
        print(f"Insuff. Context  : {data.get('insufficient_context')}")
    for k, v in assertions.items():
        mark = "OK" if v else "FAIL"
        print(f"  [{mark}] {k}")
    if ok:
        PASSED.append(test_name)
    else:
        FAILED.append(test_name)


if __name__ == "__main__":
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2.0)  # Allow Uvicorn startup

    print("\n" + "=" * 60)
    print("RESEARCHPILOT END-TO-END AUDIT")
    print("=" * 60)

    # TEST 1: GET /health
    try:
        status, data = get_json("/health")
        report("GET /health", status, data, {
            "HTTP 200": status == 200,
            "status == ok": data.get("status") == "ok",
        })
    except Exception as e:
        FAILED.append("GET /health")
        print(f"[FAIL] GET /health ERROR: {e}")

    # TEST 2: RAG question
    try:
        status, data = post_json("/research", {"query": "What are the main limitations of RAG systems?"})
        report("POST /research — RAG limitations", status, data, {
            "HTTP 200": status == 200,
            "retrieval used": "retrieve_knowledge" in data.get("tool_calls", []),
            "calculator NOT used": "calculate" not in data.get("tool_calls", []),
            "rag.md cited": "rag.md" in data.get("sources", []),
            "not insufficient": not data.get("insufficient_context", True),
            "confidence > 0": data.get("confidence", 0) > 0,
        })
    except Exception as e:
        FAILED.append("POST /research — RAG limitations")
        print(f"[FAIL] RAG question ERROR: {e}")

    # TEST 3: Calculator question
    try:
        status, data = post_json("/research", {"query": "What is 17% of 8500?"})
        report("POST /research — Calculator 17% of 8500", status, data, {
            "HTTP 200": status == 200,
            "calculator used": "calculate" in data.get("tool_calls", []),
            "answer contains 1445": "1445" in data.get("answer", ""),
            "not insufficient": not data.get("insufficient_context", True),
            "confidence == 1.0": data.get("confidence") == 1.0,
        })
    except Exception as e:
        FAILED.append("POST /research — Calculator 17% of 8500")
        print(f"[FAIL] Calculator question ERROR: {e}")

    # TEST 4: Multi-step
    try:
        status, data = post_json("/research", {"query": "Explain RAG and calculate 15% of 2000."})
        answer_and_kp = data.get("answer", "") + " ".join(data.get("key_points", []))
        report("POST /research — Multi-step RAG + Calculator", status, data, {
            "HTTP 200": status == 200,
            "retrieval used": "retrieve_knowledge" in data.get("tool_calls", []),
            "calculator used": "calculate" in data.get("tool_calls", []),
            "300 in answer or key points": "300" in answer_and_kp,
            "rag.md cited": "rag.md" in data.get("sources", []),
            "not insufficient": not data.get("insufficient_context", True),
        })
    except Exception as e:
        FAILED.append("POST /research — Multi-step RAG + Calculator")
        print(f"[FAIL] Multi-step question ERROR: {e}")

    # TEST 5: Out-of-domain
    try:
        status, data = post_json("/research", {"query": "What year was the Eiffel Tower repainted blue and gold?"})
        report("POST /research — Out-of-domain (anti-hallucination)", status, data, {
            "HTTP 200": status == 200,
            "insufficient_context True": data.get("insufficient_context") is True,
            "confidence == 0.0": data.get("confidence") == 0.0,
            "no sources cited": data.get("sources") == [],
            "refusal message present": "does not contain sufficient information" in data.get("answer", "").lower(),
        })
    except Exception as e:
        FAILED.append("POST /research — Out-of-domain")
        print(f"[FAIL] Out-of-domain question ERROR: {e}")

    # TEST 6 & 7: POST /chat — multi-turn memory
    SID = "audit_session_01"
    try:
        # Turn 1
        status1, data1 = post_json("/chat", {"message": "What is 17% of 8500?", "session_id": SID})
        turn1_ok = "1445" in data1.get("answer", "") and status1 == 200
        print(f"\n[CHAT Turn 1] Status:{status1} Answer: {data1.get('answer', '')[:80]}...")

        # Turn 2 — references turn 1
        status2, data2 = post_json("/chat", {"message": "Can you summarize what you just calculated?", "session_id": SID})
        turn2_ok = status2 == 200
        answer2 = data2.get("answer", "")
        has_memory = "1445" in answer2 or "8500" in answer2
        print(f"[CHAT Turn 2] Status:{status2} Answer: {answer2[:120]}...")

        report("POST /chat — Multi-turn memory (2 turns)", status2, data2, {
            "Turn 1 HTTP 200": status1 == 200,
            "Turn 1 answer contains 1445": turn1_ok,
            "Turn 2 HTTP 200": status2 == 200,
            "Turn 2 references prior context (1445 or 8500)": has_memory,
        })
    except Exception as e:
        FAILED.append("POST /chat — Multi-turn memory")
        print(f"[FAIL] Chat memory ERROR: {e}")

    # TEST 8: GET /evaluation
    try:
        status, data = get_json("/evaluation")
        pass_rate = data.get("metrics", {}).get("overall_pass_rate", 0)
        report("GET /evaluation", status, data if not data.get("detailed_results") else {"metrics": data["metrics"]}, {
            "HTTP 200": status == 200,
            "metrics present": "metrics" in data,
            "pass rate > 0.6": pass_rate > 0.6,
        })
        print(f"  Overall Pass Rate : {pass_rate * 100:.1f}%")
    except Exception as e:
        FAILED.append("GET /evaluation")
        print(f"[FAIL] GET /evaluation ERROR: {e}")

    # Final Summary
    total = len(PASSED) + len(FAILED)
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Tests Passed : {len(PASSED)}/{total}")
    print(f"Tests Failed : {len(FAILED)}/{total}")
    if FAILED:
        print(f"Failed Tests : {FAILED}")
    sys.exit(0 if not FAILED else 1)
