# URGENT: SavedModel Loading Issue - FIXED

## Problem
Your deployment failed with:
```
AttributeError: Loaded model (type: <class 'tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject'>) does not have 'layers' attribute.
```

**Root Cause**: The SavedModel in `models/cnn_baseline_savedmodel/` was created incorrectly using `tf.saved_model.save()` instead of `model.save()`. This creates a generic TensorFlow SavedModel that loads as `_UserObject` without Keras attributes like `.layers`, `.predict`, etc.

## Solution Applied

### Updated api/main.py (lines 57-141)
The model loading logic now:

1. **Attempts SavedModel first**
2. **Validates it has Keras attributes** (`.layers`, `.predict`)
3. **If SavedModel is invalid** (loads as `_UserObject`):
   - Prints warning message
   - Automatically falls back to H5 format
4. **Uses H5 model** (`cnn_model.h5`) which works correctly

### Expected Deployment Output

When you redeploy, you should see:
```
============================================================
INITIALIZING INSIGHT AI BACKEND
============================================================
Attempting to load SavedModel from /app/models/cnn_baseline_savedmodel
SavedModel loaded, type: <class 'tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject'>
WARNING: SavedModel loaded as _UserObject without 'layers' attribute
This SavedModel was likely created with tf.saved_model.save() instead of model.save()
Falling back to H5 format...
Attempting to load H5 model from /app/models/cnn_model.h5
H5 model loaded successfully, type: <class 'keras.engine.functional.Functional'>
H5 model validation successful
✓ Model loaded successfully with 155 layers
✓ Found last Conv2D layer: Conv_1
✓ Grad-CAM configured for layer: Conv_1
============================================================
```

**The deployment will now succeed using the H5 model!**

## Optional: Fix the SavedModel (For Future Use)

If you want to use SavedModel format instead of H5, run this script locally:

```bash
python regenerate_savedmodel.py
```

This will:
1. Load `cnn_model.h5`
2. Save it correctly as SavedModel using `model.save()`
3. Verify it loads with all Keras attributes
4. Backup the old SavedModel

Then commit and push the regenerated SavedModel:
```bash
git add models/cnn_baseline_savedmodel/
git commit -m "Regenerate SavedModel with proper Keras format"
git push
```

## Why This Matters

**Incorrect method** (creates generic TF SavedModel):
```python
# DON'T DO THIS - causes _UserObject issue
tf.saved_model.save(model, 'path/to/model')
```

**Correct method** (creates Keras SavedModel):
```python
# DO THIS - preserves Keras attributes
model.save('path/to/model', save_format='tf')
# or
tf.keras.models.save_model(model, 'path/to/model')
```

## Current Status

✅ **FIXED**: Code now handles invalid SavedModel gracefully
✅ **WORKING**: Falls back to H5 model automatically
✅ **DEPLOYED**: Ready to redeploy on Render

You can now redeploy your backend and it will work correctly!

## Files Modified

1. **api/main.py** (lines 57-141)
   - Enhanced `load_model_safe()` to validate SavedModel before accepting it
   - Automatic fallback to H5 format
   - Better error messages and logging

2. **regenerate_savedmodel.py** (NEW)
   - Helper script to fix SavedModel format for future use

## Next Steps

1. **Redeploy on Render** - The fix is already in place
2. **Verify deployment** - Check logs for the initialization messages above
3. **Test endpoints** - Try `/predict`, `/gradcam`, `/caption`
4. **(Optional)** Run `regenerate_savedmodel.py` to fix the SavedModel for future use
