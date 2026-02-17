#!/usr/bin/env python3
"""
Regenerate SavedModel format from H5 model file

This script fixes SavedModel files that were created with tf.saved_model.save()
instead of model.save(). The incorrect format loads as _UserObject without Keras
attributes like .layers, .predict, etc.

Usage:
    python regenerate_savedmodel.py

This will:
1. Load the H5 model (cnn_model.h5)
2. Save it properly as SavedModel (cnn_baseline_savedmodel/)
3. Verify the SavedModel loads correctly with all Keras attributes
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import tensorflow as tf
from pathlib import Path

# Paths
MODELS_DIR = Path(__file__).parent / "models"
H5_PATH = MODELS_DIR / "cnn_model.h5"
SAVEDMODEL_DIR = MODELS_DIR / "cnn_baseline_savedmodel"
BACKUP_DIR = MODELS_DIR / "cnn_baseline_savedmodel.backup"

def main():
    print("=" * 70)
    print("SavedModel Regeneration Script")
    print("=" * 70)

    # Check H5 file exists
    if not H5_PATH.exists():
        print(f"❌ ERROR: H5 model not found at {H5_PATH}")
        print("\nAvailable models:")
        for model_file in MODELS_DIR.glob("*.h5"):
            print(f"  - {model_file.name}")
        return 1

    print(f"\n1. Loading H5 model from {H5_PATH}")
    try:
        model = tf.keras.models.load_model(str(H5_PATH), compile=False)
        print(f"   ✓ Model loaded: {type(model)}")
        print(f"   ✓ Layers: {len(model.layers)}")
        print(f"   ✓ Input shape: {model.input_shape}")
        print(f"   ✓ Output shape: {model.output_shape}")
    except Exception as e:
        print(f"   ❌ Failed to load H5 model: {e}")
        return 1

    # Backup existing SavedModel if it exists
    if SAVEDMODEL_DIR.exists():
        print(f"\n2. Backing up existing SavedModel to {BACKUP_DIR.name}")
        if BACKUP_DIR.exists():
            import shutil
            shutil.rmtree(BACKUP_DIR)
        SAVEDMODEL_DIR.rename(BACKUP_DIR)
        print("   ✓ Backup created")
    else:
        print("\n2. No existing SavedModel to backup")

    # Save as SavedModel using correct method
    print(f"\n3. Saving model as SavedModel to {SAVEDMODEL_DIR}")
    try:
        # This is the CORRECT way to save a Keras model as SavedModel
        model.save(str(SAVEDMODEL_DIR), save_format='tf')
        print("   ✓ Model saved successfully")
    except Exception as e:
        print(f"   ❌ Failed to save model: {e}")
        if BACKUP_DIR.exists():
            print("   Restoring backup...")
            BACKUP_DIR.rename(SAVEDMODEL_DIR)
        return 1

    # Verify the saved model loads correctly
    print(f"\n4. Verifying SavedModel loads correctly")
    try:
        loaded_model = tf.keras.models.load_model(str(SAVEDMODEL_DIR), compile=False)
        print(f"   ✓ Model loaded: {type(loaded_model)}")

        # Check for required attributes
        checks = {
            'layers': hasattr(loaded_model, 'layers'),
            'predict': hasattr(loaded_model, 'predict'),
            'get_layer': hasattr(loaded_model, 'get_layer'),
            'inputs': hasattr(loaded_model, 'inputs'),
            'output': hasattr(loaded_model, 'output'),
        }

        all_passed = True
        for attr, has_attr in checks.items():
            status = "✓" if has_attr else "❌"
            print(f"   {status} Has '{attr}' attribute: {has_attr}")
            if not has_attr:
                all_passed = False

        if not all_passed:
            print("\n   ❌ SavedModel validation FAILED")
            print("   The model is missing required Keras attributes")
            if BACKUP_DIR.exists():
                print("   Restoring backup...")
                import shutil
                shutil.rmtree(SAVEDMODEL_DIR)
                BACKUP_DIR.rename(SAVEDMODEL_DIR)
            return 1

        print(f"\n   ✓ Validation passed! Model has {len(loaded_model.layers)} layers")

        # Check for keras_metadata.pb
        metadata_file = SAVEDMODEL_DIR / "keras_metadata.pb"
        if metadata_file.exists():
            print(f"   ✓ keras_metadata.pb exists")
        else:
            print(f"   ⚠ WARNING: keras_metadata.pb not found (may cause issues)")

    except Exception as e:
        print(f"   ❌ Failed to load SavedModel: {e}")
        if BACKUP_DIR.exists():
            print("   Restoring backup...")
            import shutil
            shutil.rmtree(SAVEDMODEL_DIR)
            BACKUP_DIR.rename(SAVEDMODEL_DIR)
        return 1

    print("\n" + "=" * 70)
    print("✓ SUCCESS: SavedModel regenerated successfully!")
    print("=" * 70)
    print("\nYour model is now ready for deployment on Render.")
    print("The SavedModel will load properly with all Keras attributes.")

    if BACKUP_DIR.exists():
        print(f"\nNote: Old SavedModel backed up to {BACKUP_DIR.name}")
        print("You can delete this backup if the new model works correctly.")

    return 0

if __name__ == "__main__":
    exit(main())
