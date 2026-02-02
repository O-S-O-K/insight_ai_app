import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.cm as cm
from pathlib import Path

# ------------------------------
# Paths
# ------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models/cnn_baseline_functional.h5"

# ------------------------------
# Load canonical CNN model
# ------------------------------
def load_cnn_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find CNN model at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    return model

# ------------------------------
# Predict image
# ------------------------------
def predict_image(model, image, top=3):
    """
    Run an image through your CNN and return top predictions.
    Assumes your CNN outputs one-hot class vectors.
    """
    img_resized = image.resize((224, 224))
    x = img_to_array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]  # (num_classes,)
    # Get top indices
    top_idx = preds.argsort()[-top:][::-1]
    top_scores = preds[top_idx]

    # Class labels: use model's output names if available
    if hasattr(model, "classes_"):  # optional attribute
        labels = model.classes_
    else:
        labels = [f"class_{i}" for i in range(len(preds))]

    top_labels = [labels[i] for i in top_idx]

    return list(zip(top_labels, top_scores))

# ------------------------------
# Grad-CAM helpers
# ------------------------------
def find_last_conv_layer(model):
    """Find last convolutional layer in model"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in model.")

def get_gradcam_heatmap(model, last_conv_layer, img_tensor, pred_index=None):
    """
    Compute Grad-CAM heatmap for a given image tensor.
    Args:
        model: Keras model
        last_conv_layer: last convolutional layer in the model
        img_tensor: preprocessed image tensor (1, H, W, C)
        pred_index: index of the class to generate heatmap for
    Returns:
        heatmap: numpy array of Grad-CAM
    """
    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4, colormap="jet"):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size)

    cmap = cm.get_cmap(colormap)
    colored = cmap(np.array(heatmap)/255.0)
    colored = np.uint8(colored[:, :, :3] * 255)

    overlay = Image.blend(image.convert("RGB"), Image.fromarray(colored), alpha)
    return overlay
