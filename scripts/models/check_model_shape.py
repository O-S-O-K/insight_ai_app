import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

print("Checking cnn_baseline_functional.h5...")
try:
    model = tf.keras.models.load_model('models/cnn_baseline_functional.h5', compile=False)
    print(f"✓ Loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Model type: {type(model)}")
    print(f"  Layers: {len(model.layers)}")
except Exception as e:
    print(f"✗ Failed to load: {e}")
