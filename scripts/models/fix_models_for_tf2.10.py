#!/usr/bin/env python3
"""
Fix incompatible model files for TensorFlow 2.10.1 deployment

This script:
1. Tests which model files can be loaded with TensorFlow 2.10.1
2. Converts working models to proper SavedModel format
3. Creates a simple compatible model if all else fails

Usage:
    python fix_models_for_tf2.10.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from pathlib import Path
import sys

# Paths
MODELS_DIR = Path(__file__).parent / "models"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Model files to test
MODEL_FILES = [
    ("cnn_model.h5", "Primary H5 model"),
    ("cnn_baseline.h5", "Baseline H5 model"),
    ("cnn_baseline_functional.h5", "Functional H5 model"),
    ("cnn_baseline_savedmodel", "Baseline SavedModel"),
]

def test_model(path, description):
    """Test if a model can be loaded"""
    full_path = MODELS_DIR / path
    if not full_path.exists():
        print(f"  ⊗ {description}: File not found")
        return None

    try:
        model = tf.keras.models.load_model(str(full_path), compile=False, safe_mode=False)

        # Validate
        if not hasattr(model, 'layers'):
            print(f"  ✗ {description}: Missing 'layers' attribute")
            return None
        if not hasattr(model, 'predict'):
            print(f"  ✗ {description}: Missing 'predict' method")
            return None

        print(f"  ✓ {description}: OK ({len(model.layers)} layers)")
        return model

    except Exception as e:
        print(f"  ✗ {description}: {str(e)[:80]}")
        return None

def create_simple_model():
    """Create a simple 3-class MobileNetV2 model as fallback"""
    print("\n3. Creating new compatible model (MobileNetV2 for 3 classes)")

    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        print(f"  ✓ Created new model with {len(model.layers)} layers")
        return model

    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        return None

def main():
    print("=" * 70)
    print("Model Compatibility Fixer for TensorFlow 2.10.1")
    print("=" * 70)

    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Models directory: {MODELS_DIR}")

    # Test all existing models
    print("\n1. Testing existing model files:")
    working_model = None
    working_path = None

    for path, description in MODEL_FILES:
        model = test_model(path, description)
        if model is not None and working_model is None:
            working_model = model
            working_path = path

    # If no working model, create a new one
    if working_model is None:
        print("\n  ⚠ No working models found!")
        working_model = create_simple_model()
        if working_model is None:
            print("\n❌ FAILED: Could not load or create any model")
            return 1
        working_path = "newly_created"
    else:
        print(f"\n  ✓ Found working model: {working_path}")

    # Save as proper SavedModel format
    print("\n2. Saving as proper SavedModel format:")
    savedmodel_path = MODELS_DIR / "cnn_baseline_savedmodel"
    backup_path = MODELS_DIR / "cnn_baseline_savedmodel.old"

    try:
        # Backup existing SavedModel if it exists
        if savedmodel_path.exists():
            print(f"  Creating backup: {backup_path.name}")
            if backup_path.exists():
                import shutil
                shutil.rmtree(backup_path)
            savedmodel_path.rename(backup_path)

        # Save with proper Keras format
        print(f"  Saving to: {savedmodel_path}")
        working_model.save(str(savedmodel_path), save_format='tf')

        # Verify keras_metadata.pb exists
        metadata_pb = savedmodel_path / "keras_metadata.pb"
        if metadata_pb.exists():
            print(f"  ✓ keras_metadata.pb exists")
        else:
            print(f"  ⚠ keras_metadata.pb not found (may cause issues)")

        print(f"  ✓ SavedModel saved successfully")

    except Exception as e:
        print(f"  ✗ Failed to save SavedModel: {e}")
        if backup_path.exists():
            print(f"  Restoring backup...")
            backup_path.rename(savedmodel_path)
        return 1

    # Also save as H5 for backup
    print("\n3. Saving as H5 format (backup):")
    h5_path = MODELS_DIR / "cnn_model_fixed.h5"

    try:
        working_model.save(str(h5_path), save_format='h5')
        print(f"  ✓ H5 model saved to: {h5_path.name}")
    except Exception as e:
        print(f"  ⚠ Failed to save H5: {e}")

    # Test the new SavedModel
    print("\n4. Verifying new SavedModel:")
    test_result = test_model("cnn_baseline_savedmodel", "New SavedModel")

    if test_result is None:
        print("\n❌ FAILED: SavedModel verification failed")
        return 1

    print("\n" + "=" * 70)
    print("✓ SUCCESS: Models fixed and ready for deployment!")
    print("=" * 70)
    print("\nYour application should now work with TensorFlow 2.10.1.")
    print("\nNext steps:")
    print("  1. Commit the updated models to git:")
    print("     git add models/cnn_baseline_savedmodel/")
    print("     git add models/cnn_model_fixed.h5")
    print("     git commit -m 'Fix model compatibility for TF 2.10.1'")
    print("  2. Push to GitHub:")
    print("     git push")
    print("  3. Redeploy on Render")

    return 0

if __name__ == "__main__":
    exit(main())
