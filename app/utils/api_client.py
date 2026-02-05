import os
import requests

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")


def predict_image(file_obj):
    files = {"file": file_obj}
    r = requests.post(
        f"{FASTAPI_URL}/predict",
        files=files,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def gradcam_image(file_obj, class_idx=None):
    files = {"file": file_obj}
    data = {}
    if class_idx is not None:
        data["class_idx"] = str(class_idx)

    r = requests.post(
        f"{FASTAPI_URL}/gradcam",
        files=files,
        data=data,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def caption_image(file_obj):
    files = {"file": file_obj}
    r = requests.post(
        f"{FASTAPI_URL}/caption",
        files=files,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def submit_feedback(file_obj, entry: dict):
    files = {"file": file_obj}
    data = {"entry": entry}

    r = requests.post(
        f"{FASTAPI_URL}/feedback",
        files=files,
        data=data,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()
