"""Stage 5 — Streamlit sentiment analysis UI."""
import time

import requests
import streamlit as st

ENDPOINT = "http://localhost:8890/serve/sentiment"

st.title("Sentiment Analysis")
st.caption("Powered by ClearML Serving · TF-IDF + LogisticRegression")

review = st.text_area("Enter a movie review:", height=150)

if st.button("Predict") and review.strip():
    try:
        start = time.time()
        resp = requests.post(ENDPOINT, json={"text": review}, timeout=10)
        resp.raise_for_status()
        latency_ms = (time.time() - start) * 1000
        label = resp.json()["label"]

        COLOUR = "green" if label == "positive" else "red"
        st.markdown(f"**Sentiment:** :{COLOUR}[{label.upper()}]")
        st.caption(f"Latency: {latency_ms:.1f} ms")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the inference endpoint. Is ClearML Serving running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The endpoint may be overloaded.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Endpoint returned an error: {e}")
    except KeyError:
        st.error("Unexpected response format from the endpoint.")
