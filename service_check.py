"""
Stage 4 — Inference smoke test.

Sends test requests to the ClearML Serving endpoint and prints results.
Run after the inference service is up.

Usage:
    python serve_check.py
"""
import time

import requests

ENDPOINT = "http://localhost:8890/serve/sentiment"

TESTS = [
    ("This movie was absolutely fantastic, I loved every minute!", "positive"),
    ("Terrible film, complete waste of time and money.", "negative"),
]


def predict(review: str) -> tuple[str, float]:
    """Send review text to the serving endpoint and return (label, latency_ms)."""
    start = time.time()
    resp = requests.post(ENDPOINT, json={"text": review}, timeout=10)
    resp.raise_for_status()
    latency_ms = (time.time() - start) * 1000
    return resp.json()["label"], latency_ms


print(f"Endpoint: {ENDPOINT}\n")

all_ok = True
for text, expected in TESTS:
    label, latency = predict(review=text)
    match = "OK" if expected is None or label == expected else "ERROR"
    if match == "ERROR":
        all_ok = False
    print(f"{match} [{label:8s}]  {latency:6.1f}ms  {text[:60]}")

print()
print("All tests passed." if all_ok else "Some tests FAILED — check endpoint logs.")
