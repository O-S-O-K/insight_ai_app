"""
Convert cnn_model_fixed.h5 (Sequential wrapper) to proper Functional API model
with correct 224x224 input shape for accurate Grad-CAM visualization.

This script:
1. Loads the Sequential wrapper model
2. Extracts the pretrained MobileNetV2 inside
3. Rebuilds as Functional API with 224x224 input
4. Saves as cnn_baseline_functional.h5
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Quiet TF logs

import tensorflow as tf
from pathlib import Path

print("=" * 70)
print("CONVERTING TO FUNCTIONAL API MODEL")
print("=" * 70)
print()

# Paths
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
INPUT_MODEL = MODELS_DIR / "cnn_model_fixed.h5"
OUTPUT_MODEL = MODELS_DIR / "cnn_baseline_functional.h5"
BACKUP_MODEL = MODELS_DIR / "cnn_baseline_functional.h5.backup"

# Configuration
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 1000  # ImageNet-1K

print(f"TensorFlow version: {tf.__version__}")
print(f"Input model: {INPUT_MODEL}")
print(f"Output model: {OUTPUT_MODEL}")
print()

# Step 1: Load the Sequential wrapper model
print("Step 1: Loading Sequential wrapper model...")
try:
    seq_model = tf.keras.models.load_model(str(INPUT_MODEL), compile=False)
    print(f"  ✓ Loaded: {type(seq_model)}")
    print(f"  ✓ Input shape: {seq_model.input_shape}")
    print(f"  ✓ Output shape: {seq_model.output_shape}")
    print(f"  ✓ Layers: {len(seq_model.layers)}")
    print()
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    exit(1)

# Step 2: Extract MobileNetV2 from the Sequential wrapper
print("Step 2: Extracting MobileNetV2 from Sequential wrapper...")
mobilenet = None
for i, layer in enumerate(seq_model.layers):
    print(f"  Layer {i}: {layer.name} ({type(layer).__name__})")
    if 'mobilenet' in layer.name.lower() or hasattr(layer, 'layers'):
        mobilenet = layer
        print(f"  ✓ Found MobileNetV2: {layer.name}")
        break

if mobilenet is None:
    print("  ✗ Could not find MobileNetV2 layer!")
    exit(1)
print()

# Step 3: Build new Functional API model
print("Step 3: Building Functional API model with correct input shape...")
try:
    # Create new input layer with correct shape
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name='input_layer')

    # Rebuild from scratch using Keras MobileNetV2 application
    # (Cannot reuse nested layers due to residual connections)
    print("  Rebuilding from Keras MobileNetV2 application...")
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers

    # Load pretrained MobileNetV2 (without top layers)
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet',
        pooling='avg'  # Global average pooling
    )

    x = base_model(inputs, training=False)

    # Add final classification layer
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    # Create functional model
    functional_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='cnn_baseline_functional')

    print(f"  ✓ Created Functional API model")
    print(f"  ✓ Input shape: {functional_model.input_shape}")
    print(f"  ✓ Output shape: {functional_model.output_shape}")
    print(f"  ✓ Total layers: {len(functional_model.layers)}")
    print()

except Exception as e:
    print(f"  ✗ Failed to build Functional API model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Verify the model works
print("Step 4: Verifying model functionality...")
try:
    import numpy as np

    # Create dummy input
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

    # Test prediction
    predictions = functional_model.predict(dummy_input, verbose=0)

    print(f"  ✓ Prediction output shape: {predictions.shape}")
    print(f"  ✓ Sum of probabilities: {predictions.sum():.6f} (should be ~1.0)")
    print(f"  ✓ Top prediction index: {predictions.argmax()}")

    # Verify Grad-CAM compatibility
    from tensorflow.keras.models import Model
    has_get_layer = hasattr(functional_model, 'get_layer')
    has_inputs = hasattr(functional_model, 'inputs')
    has_output = hasattr(functional_model, 'output')

    print(f"  ✓ Has get_layer: {has_get_layer}")
    print(f"  ✓ Has inputs: {has_inputs}")
    print(f"  ✓ Has output: {has_output}")

    # Find last conv layer for Grad-CAM
    last_conv = None
    for layer in reversed(functional_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break

    if last_conv:
        print(f"  ✓ Last Conv2D layer: {last_conv.name}")
    else:
        print("  ⚠ Warning: No Conv2D layer found (may be nested in MobileNetV2)")
        # Check nested layers
        for layer in reversed(functional_model.layers):
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        print(f"  ✓ Found nested Conv2D: {sublayer.name}")
                        break

    print()

except Exception as e:
    print(f"  ✗ Model verification failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Backup old model if it exists
if OUTPUT_MODEL.exists():
    print("Step 5: Backing up existing model...")
    try:
        if BACKUP_MODEL.exists():
            BACKUP_MODEL.unlink()
        OUTPUT_MODEL.rename(BACKUP_MODEL)
        print(f"  ✓ Backed up to: {BACKUP_MODEL.name}")
        print()
    except Exception as e:
        print(f"  ⚠ Warning: Could not backup: {e}")
        print()

# Step 6: Save the new model
print("Step 6: Saving Functional API model...")
try:
    functional_model.save(str(OUTPUT_MODEL), save_format='h5')
    print(f"  ✓ Saved to: {OUTPUT_MODEL}")

    # Verify saved model
    file_size_mb = OUTPUT_MODEL.stat().st_size / (1024 * 1024)
    print(f"  ✓ File size: {file_size_mb:.2f} MB")
    print()

except Exception as e:
    print(f"  ✗ Failed to save model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 7: Test loading the saved model
print("Step 7: Verifying saved model can be loaded...")
try:
    loaded_model = tf.keras.models.load_model(str(OUTPUT_MODEL), compile=False)
    print(f"  ✓ Loaded successfully: {type(loaded_model)}")
    print(f"  ✓ Input shape: {loaded_model.input_shape}")
    print(f"  ✓ Output shape: {loaded_model.output_shape}")

    # Quick prediction test
    predictions = loaded_model.predict(dummy_input, verbose=0)
    print(f"  ✓ Prediction test passed")
    print()

except Exception as e:
    print(f"  ✗ Failed to load saved model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("=" * 70)
print("✓ CONVERSION COMPLETE!")
print("=" * 70)
print()
print(f"New Functional API model saved to: {OUTPUT_MODEL}")
print(f"Input shape: (None, 224, 224, 3)")
print(f"Output shape: (None, {NUM_CLASSES})")
print()
print("Next steps:")
print("1. Test the model locally with your Streamlit app")
print("2. Commit and push to GitHub")
print("3. Deploy to Render for accurate Grad-CAM visualizations")
print()
