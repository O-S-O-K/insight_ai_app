def predict(_):
    return {
        "label": "mock_cat",
        "confidence": 0.93
    }

def caption(_):
    return "A mock image of a cat sitting on a chair."

def gradcam(_):
    return {
        "heatmap": "mock_base64_string"
    }

def submit_feedback(_, __):
    return {"status": "ok"}
