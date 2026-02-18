# Utility Scripts

This directory contains utility scripts for model management, validation, and troubleshooting.

## Directory Structure

```
scripts/
└── models/          # Model management and validation scripts
    ├── check_model_shape.py
    ├── check_tf_keras.py
    ├── check_tf_version.py
    ├── convert_to_functional_api.py
    ├── download_isic_data.py
    ├── fix_models_for_tf2.10.py
    ├── regenerate_savedmodel.py
    └── train_medical.py
```

## Model Management Scripts

### check_model_shape.py

**Purpose**: Verify model loads correctly and inspect input shape and layer count.

**Usage**:
```bash
python scripts/models/check_model_shape.py
```

**What it does**:
- Loads `models/cnn_baseline_functional.h5`
- Displays input shape, model type, and number of layers
- Validates model loads without errors

**When to use**:
- After converting or regenerating models
- Troubleshooting model loading issues
- Verifying model format

**Expected output**:
```
Checking cnn_baseline_functional.h5...
✓ Loaded successfully
  Input shape: (None, 224, 224, 3)
  Model type: <class 'keras.engine.functional.Functional'>
  Layers: 155
```

---

### check_tf_keras.py

**Purpose**: Verify TensorFlow and Keras imports work correctly.

**Usage**:
```bash
python scripts/models/check_tf_keras.py
```

**What it does**:
- Tests TensorFlow import and version
- Tests tf.keras import with legacy mode
- Validates environment configuration

**When to use**:
- After changing TensorFlow/Keras versions
- Troubleshooting import errors
- Verifying environment setup

---

### check_tf_version.py

**Purpose**: Simple TensorFlow version check.

**Usage**:
```bash
python scripts/models/check_tf_version.py
```

**What it does**:
- Prints installed TensorFlow version

**When to use**:
- Quick version verification
- Environment debugging

---

### convert_to_functional_api.py

**Purpose**: Convert Sequential wrapper model to Functional API with correct input shape.

**Usage**:
```bash
python scripts/models/convert_to_functional_api.py
```

**What it does**:
1. Loads Sequential wrapper model (`cnn_model_fixed.h5`)
2. Extracts pretrained MobileNetV2 base
3. Rebuilds as Functional API with 224x224 input
4. Saves as `cnn_baseline_functional.h5`
5. Creates backup of existing model

**When to use**:
- Converting Sequential models to Functional API
- Fixing input shape issues for Grad-CAM
- Model architecture refactoring

**Input requirements**:
- Source model: `models/cnn_model_fixed.h5`
- Model should contain MobileNetV2 architecture

**Output**:
- New model: `models/cnn_baseline_functional.h5`
- Backup: `models/cnn_baseline_functional.h5.backup`

**Expected output**:
```
======================================================================
CONVERTING TO FUNCTIONAL API MODEL
======================================================================

Step 1: Loading Sequential wrapper model...
  ✓ Loaded: <class 'keras.engine.sequential.Sequential'>
  ✓ Input shape: (None, 224, 224, 3)
  ✓ Output shape: (None, 1000)
  ✓ Layers: 158

Step 2: Extracting MobileNetV2 base...
  ✓ Found MobileNetV2 layer
  ✓ Base model has 155 layers

Step 3: Rebuilding as Functional API...
  ✓ Created new Functional model
  ✓ Input: (None, 224, 224, 3)
  ✓ Output: (None, 1000)

Step 4: Transferring weights...
  ✓ Weights transferred successfully

Step 5: Saving new model...
  ✓ Saved to models/cnn_baseline_functional.h5

✓ SUCCESS: Model converted to Functional API!
```

---

### fix_models_for_tf2.10.py

**Purpose**: Fix model compatibility for TensorFlow 2.10 and later versions.

**Usage**:
```bash
python scripts/models/fix_models_for_tf2.10.py
```

**What it does**:
1. Loads model from H5 format
2. Removes deprecated parameters (`safe_mode`, `batch_shape`)
3. Rebuilds model for TensorFlow 2.10+ compatibility
4. Regenerates both H5 and SavedModel formats
5. Validates loading and attributes

**When to use**:
- Upgrading from TensorFlow 2.9 or earlier
- Fixing `batch_shape` errors
- Fixing `safe_mode` deprecation warnings
- Regenerating models after TensorFlow upgrade

**Environment**:
- Requires TensorFlow 2.10+ with tf-keras

**Output**:
- Updated H5 model
- Regenerated SavedModel directory
- Backups of original files

---

### regenerate_savedmodel.py

**Purpose**: Fix SavedModel format that was created incorrectly and loads as `_UserObject`.

**Usage**:
```bash
python scripts/models/regenerate_savedmodel.py
```

**What it does**:
1. Loads H5 model (`cnn_model.h5`)
2. Saves properly as SavedModel format
3. Verifies SavedModel has Keras attributes (`.layers`, `.predict`, etc.)
4. Creates backup of old SavedModel
5. Validates `keras_metadata.pb` exists

**When to use**:
- SavedModel loads as `_UserObject` without Keras attributes
- Missing `.layers`, `.predict`, or `.get_layer` methods
- SavedModel created with `tf.saved_model.save()` instead of `model.save()`
- Deployment failures due to model format

**Input requirements**:
- Source model: `models/cnn_model.h5`

**Output**:
- Regenerated: `models/cnn_baseline_savedmodel/`
- Backup: `models/cnn_baseline_savedmodel.backup/`

**Expected output**:
```
======================================================================
SavedModel Regeneration Script
======================================================================

1. Loading H5 model from models/cnn_model.h5
   ✓ Model loaded: <class 'keras.engine.functional.Functional'>
   ✓ Layers: 155
   ✓ Input shape: (None, 224, 224, 3)
   ✓ Output shape: (None, 1000)

2. Backing up existing SavedModel to cnn_baseline_savedmodel.backup
   ✓ Backup created

3. Saving model as SavedModel to models/cnn_baseline_savedmodel
   ✓ Model saved successfully

4. Verifying SavedModel loads correctly
   ✓ Model loaded: <class 'keras.engine.functional.Functional'>
   ✓ Has 'layers' attribute: True
   ✓ Has 'predict' attribute: True
   ✓ Has 'get_layer' attribute: True
   ✓ Has 'inputs' attribute: True
   ✓ Has 'output' attribute: True

   ✓ Validation passed! Model has 155 layers
   ✓ keras_metadata.pb exists

======================================================================
✓ SUCCESS: SavedModel regenerated successfully!
======================================================================
```

## Medical Imaging Scripts

### download_isic_data.py

**Purpose**: Download and organize ISIC 2020 Skin Lesion dataset for medical model training.

**Usage**:
```bash
python scripts/models/download_isic_data.py [--output-dir data/isic2020] [--max-images 5000]
```

**What it does**:
1. Downloads ISIC 2020 training metadata CSV from ISIC S3
2. Downloads a balanced subset of melanoma + benign images via ISIC API
3. Organizes into train/val split with class subdirectories
4. Saves dataset_summary.json with class distribution

**Output structure**:
```
data/isic2020/
├── ISIC_2020_Training_GroundTruth.csv
├── dataset_summary.json
├── images/
├── train/
│   ├── benign/
│   └── melanoma/
└── val/
    ├── benign/
    └── melanoma/
```

**Note**: ISIC 2020 is licensed CC BY-NC 4.0 (non-commercial).

---

### train_medical.py

**Purpose**: Fine-tune EfficientNetB0 on ISIC 2020 for binary skin lesion classification.

**Usage**:
```bash
python scripts/models/train_medical.py [--data-dir data/isic2020] [--epochs-phase1 10] [--epochs-phase2 20]
```

**What it does**:
1. Builds EfficientNetB0 with classification head (ImageNet pretrained)
2. Phase 1: Trains head only (base frozen, LR=1e-3)
3. Phase 2: Fine-tunes top 20 EfficientNet layers (LR=1e-5)
4. Handles 55:1 class imbalance via class weights
5. Evaluates: accuracy, AUC, sensitivity, specificity
6. Saves model as `models/medical_model.h5`
7. Updates `models/medical_metadata.json` with metrics

**Arguments**:
| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data/isic2020` | ISIC dataset directory |
| `--epochs-phase1` | `10` | Max epochs for head training |
| `--epochs-phase2` | `20` | Max epochs for fine-tuning |
| `--fine-tune-layers` | `20` | Top EfficientNet layers to unfreeze |
| `--dropout` | `0.3` | Dropout rate in classification head |

---

## Common Workflows

### Training the Medical Imaging Model

```bash
# 1. Download ISIC 2020 dataset (balanced subset)
python scripts/models/download_isic_data.py --output-dir data/isic2020

# 2. Train EfficientNetB0
python scripts/models/train_medical.py --data-dir data/isic2020

# 3. Verify model
python scripts/models/check_model_shape.py
```

---

### After Upgrading TensorFlow

```bash
# 1. Check TensorFlow version
python scripts/models/check_tf_version.py

# 2. Verify Keras imports work
python scripts/models/check_tf_keras.py

# 3. Fix model compatibility
python scripts/models/fix_models_for_tf2.10.py

# 4. Verify model loads
python scripts/models/check_model_shape.py
```

### Fixing SavedModel Format Issues

```bash
# 1. Regenerate SavedModel from H5
python scripts/models/regenerate_savedmodel.py

# 2. Verify model loads correctly
python scripts/models/check_model_shape.py
```

### Converting Model Architecture

```bash
# 1. Convert Sequential to Functional API
python scripts/models/convert_to_functional_api.py

# 2. Verify conversion succeeded
python scripts/models/check_model_shape.py
```

## Troubleshooting

### Script Fails with Import Error

**Issue**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**: Install dependencies
```bash
pip install -r api/requirements.txt
```

### Model File Not Found

**Issue**: `ERROR: H5 model not found at models/...`

**Solution**: Check model exists in `models/` directory
```bash
ls -la models/
```

Ensure the script is run from repository root:
```bash
cd /path/to/insight_ai_app
python scripts/models/script_name.py
```

### Script Fails with Permission Denied

**Issue**: Permission error when creating backup

**Solution**: Check file permissions
```bash
chmod +w models/
```

### SavedModel Still Loads as _UserObject

**Issue**: After regeneration, model still missing attributes

**Solution**:
1. Verify H5 model is valid: `python scripts/models/check_model_shape.py`
2. Check TensorFlow version: `python scripts/models/check_tf_version.py`
3. Ensure using `TF_USE_LEGACY_KERAS=1` for TensorFlow 2.15+
4. See [docs/SAVEDMODEL_FIX.md](../docs/SAVEDMODEL_FIX.md) for detailed troubleshooting

## Environment Requirements

All scripts require:
- **Python 3.10**
- **TensorFlow 2.15.0** (or compatible version)
- **tf-keras 2.15.1** (for TensorFlow 2.15+)

Scripts automatically set these environment variables:
- `TF_USE_LEGACY_KERAS=1` - Use legacy Keras 2.x API
- `CUDA_VISIBLE_DEVICES=-1` - Force CPU execution
- `TF_CPP_MIN_LOG_LEVEL=2` - Reduce TensorFlow logging

## Related Documentation

- **[docs/MODEL_LOADING_FIXES.md](../docs/MODEL_LOADING_FIXES.md)** - Model loading troubleshooting
- **[docs/SAVEDMODEL_FIX.md](../docs/SAVEDMODEL_FIX.md)** - SavedModel format issues
- **[docs/DEPLOYMENT_SUMMARY.md](../docs/DEPLOYMENT_SUMMARY.md)** - Deployment and model history
- **[docs/SETUP.md](../docs/SETUP.md)** - Project setup guide

## Contributing

When adding new scripts:

1. Place in appropriate subdirectory (e.g., `scripts/models/`)
2. Add shebang (`#!/usr/bin/env python3`) to Python scripts
3. Include docstring with purpose and usage
4. Set environment variables at top of script
5. Add clear success/failure messages
6. Update this README with script documentation
7. Create backup files before modifying models
8. Validate changes before overwriting originals

## Script Template

```python
#!/usr/bin/env python3
"""
Brief description of what this script does.

Usage:
    python scripts/models/script_name.py

This script:
1. Step one
2. Step two
3. Step three
"""

import os
# Set environment variables first
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from pathlib import Path

def main():
    print("=" * 70)
    print("SCRIPT NAME")
    print("=" * 70)

    # Your script logic here

    print("✓ SUCCESS: Operation completed!")
    return 0

if __name__ == "__main__":
    exit(main())
```

## License

MIT License - See [LICENSE](../LICENSE)
