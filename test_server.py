import urllib.request
import json
import subprocess
import time
import sys

p = subprocess.Popen([sys.executable, "server.py"])
time.sleep(1.5)
try:
    with urllib.request.urlopen("http://localhost:8000/api/status") as response:
        status_data = json.loads(response.read().decode())
        print("API /api/status SUCCESS:", status_data)

    req = urllib.request.Request(
        "http://localhost:8000/api/run-next",
        data=json.dumps({}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        run_data = json.loads(response.read().decode())
        print("API /api/run-next SUCCESS: Run #", run_data.get("run_id"), run_data.get("model"))

    # Test LLM consult
    req_llm = urllib.request.Request(
        "http://localhost:8000/api/llm-consult",
        data=json.dumps({"query": "Explain overfit gap"}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req_llm) as response:
        llm_data = json.loads(response.read().decode())
        print("API /api/llm-consult SUCCESS:", llm_data["response"][:80], "...")

    # Test index.html serves
    with urllib.request.urlopen("http://localhost:8000/") as response:
        html = response.read().decode()
        print("GET / SUCCESS: HTML length =", len(html), "bytes")

finally:
    p.terminate()
    p.wait()
    print("Server cleanly stopped.")

