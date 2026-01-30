# utils/gradcam.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from PIL import Image
import matplotlib.cm as cm

# ------------------------------
# Load MobileNetV2
# ------------------------------
def load_cnn_model_pretrained(num_classes=1000):
    model = MobileNetV2(weights="imagenet", include_top=True)
    return model

# ------------------------------
# Grad-CAM helpers
# ------------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None

def get_gradcam_heatmap(model, last_conv_layer, img_tensor, pred_index=None):
    """
    Compute Grad-CAM heatmap for a given image tensor.
    
    Args:
        model: Keras model
        last_conv_layer: last convolutional layer in the model
        img_tensor: preprocessed image tensor
        pred_index: index of the class to generate heatmap for
    Returns:
        heatmap: numpy array of Grad-CAM
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer.name).output, model.output]
    )

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

# ------------------------------
# Overlay
# ------------------------------
def overlay_heatmap(heatmap, image, alpha=0.4, colormap="jet"):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size)

    cmap = cm.get_cmap(colormap)
    colored = cmap(np.array(heatmap) / 255.0)
    colored = np.uint8(colored[:, :, :3] * 255)

    overlay = Image.blend(image.convert("RGB"), Image.fromarray(colored), alpha)
    return overlay
