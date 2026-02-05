import os
import requests

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")


def _post_image(endpoint: str, uploaded_file):
    uploaded_file.seek(0)
    files = {"file": uploaded_file}
    r = requests.post(f"{FASTAPI_URL}{endpoint}", files=files, timeout=60)
    r.raise_for_status()
    return r.json()


def predict(uploaded_file):
    return _post_image("/predict", uploaded_file)


def caption(uploaded_file):
    return _post_image("/caption", uploaded_file)


def gradcam(uploaded_file):
    return _post_image("/gradcam", uploaded_file)


def submit_feedback(uploaded_file, payload: dict):
    uploaded_file.seek(0)
    files = {"file": uploaded_file}
    data = {"metadata": json.dumps(payload)}
    r = requests.post(f\"{FASTAPI_URL}/feedback\", files=files, data=data, timeout=30)
    r.raise_for_status()
    return r.json()
