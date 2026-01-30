from tensorflow.keras.models import load_model
import os

# Load your existing H5 model
model = load_model("models/cnn_baseline.h5")

# Make sure target folder exists
save_path = "models/cnn_baseline_savedmodel"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Export as TensorFlow SavedModel
model.export(save_path)   # <-- This is the key change
print("SavedModel conversion complete!")
