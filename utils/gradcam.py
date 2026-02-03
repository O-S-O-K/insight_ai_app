import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import matplotlib.cm as cm

IMG_SIZE = (224, 224)

def predict_image(model, image, top=3, label_map=None):
    img = image.resize(IMG_SIZE)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)[0]
    top_idx = preds.argsort()[-top:][::-1]

    if label_map is None:
        labels = [str(i) for i in top_idx]
    else:
        labels = [label_map[i] for i in top_idx]

    scores = preds[top_idx]
    return list(zip(labels, scores))


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found")


def get_gradcam_heatmap(model, last_conv_layer, img_tensor, class_idx=None):
    """
    Generates a Grad-CAM heatmap for the specified class index.
    """
    grad_model = Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size)

    cmap = cm.get_cmap("jet")
    colored = cmap(np.array(heatmap) / 255.0)
    colored = np.uint8(colored[:, :, :3] * 255)

    return Image.blend(image.convert("RGB"), Image.fromarray(colored), alpha)
