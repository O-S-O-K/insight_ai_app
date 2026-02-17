# Model Loading Fixes for Render Deployment

## Issues Fixed

### 1. AttributeError: '_UserObject' object has no attribute 'layers'

**Root Cause**: When TensorFlow loads a SavedModel format, it sometimes returns a `_UserObject` wrapper that doesn't expose the standard Keras model attributes like `.layers`, `.get_layer()`, `.inputs`, `.output`, etc.

**Solution Implemented**:

#### A. Enhanced Model Loading with Validation (`api/main.py`)
- Added `load_model_safe()` function that:
  - Tries SavedModel format first
  - Falls back to H5 format if SavedModel fails
  - Validates the loaded model has required attributes (`.layers`, `.predict`)
  - Provides detailed error messages if loading fails
  - Uses `tf.keras.models.load_model()` explicitly for better compatibility

#### B. Improved Layer Detection (`api/main.py`)
- Added `find_last_conv_layer()` function that:
  - Validates model has `.layers` attribute before accessing it
  - Searches for Conv2D layers in the main model
  - Falls back to searching nested models (e.g., MobileNetV2 functional API)
  - Provides helpful error messages

#### C. Enhanced Grad-CAM Endpoint (`api/main.py`)
- Added validation before accessing model attributes:
  - Checks `model.get_layer()` method exists
  - Checks `model.inputs` attribute exists
  - Checks `model.output` attribute exists
- Added `try-except` for `get_layer()` calls
- Added gradient validation (checks if gradient is None)
- Added epsilon to prevent division by zero in heatmap normalization

#### D. Dockerfile Build-Time Validation (`api/Dockerfile`)
- Added comprehensive model creation test during Docker build:
  - Creates a test Sequential model
  - Validates all required attributes exist
  - Fails fast during build if TensorFlow/Keras is misconfigured
  - Prevents runtime errors from invalid installations

#### E. Enhanced Health Check Endpoint (`api/main.py`)
- Returns diagnostic information:
  - Model type
  - Whether model has `.layers` attribute
  - Number of layers
  - TensorFlow version
- Helps diagnose issues without triggering errors

## Files Modified

1. **api/main.py**
   - Lines 54-117: New `load_model_safe()` and `find_last_conv_layer()` functions
   - Lines 139-149: Enhanced health check endpoint
   - Lines 198-272: Enhanced Grad-CAM endpoint with validation

2. **api/Dockerfile**
   - Lines 28-53: Comprehensive TensorFlow/Keras validation during build

## Testing the Fixes

### Local Testing
```bash
# Build and run with Docker
docker-compose up

# Or run backend directly
cd api
uvicorn main:app --reload

# Check health endpoint
curl http://localhost:8000/
```

### Expected Health Check Response
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "<class 'keras.engine.functional.Functional'>",
  "has_layers": true,
  "num_layers": 155,
  "gradcam_layer": "Conv_1",
  "tensorflow_version": "2.10.1"
}
```

### Render Deployment
The deployment will now:
1. Fail fast during build if TensorFlow is misconfigured
2. Validate model loading at startup with detailed logs
3. Return helpful diagnostic info in the health endpoint
4. Gracefully handle model attribute errors with detailed error messages

## Additional Safety Features

1. **Comprehensive Error Messages**: All errors include the actual model type and missing attributes
2. **Fallback Mechanisms**: Tries multiple loading strategies before failing
3. **Nested Model Support**: Searches nested models for Conv2D layers
4. **Gradient Validation**: Skips classes with None gradients instead of crashing
5. **Division by Zero Protection**: Adds epsilon in heatmap normalization

## What to Do If Errors Persist

If you still see `_UserObject` errors after these fixes:

1. **Check the health endpoint** (`GET /`) to see model type and attributes
2. **Review build logs** - the Dockerfile will fail fast if model creation fails
3. **Verify model files**:
   - SavedModel should be a directory with `saved_model.pb` and variables
   - H5 model should be a single `.h5` file
4. **Try regenerating the SavedModel**:
   ```python
   model = tf.keras.models.load_model('cnn_model.h5')
   model.save('cnn_baseline_savedmodel', save_format='tf')
   ```

## Environment Variables

Ensure these are set in Render:
- `TF_USE_LEGACY_KERAS=1` (set in Dockerfile)
- `CUDA_VISIBLE_DEVICES=-1` (set in Dockerfile)
- `TF_CPP_MIN_LOG_LEVEL=2` (set in Dockerfile)
- `HF_TOKEN` (optional, for BLIP captions)

## Summary

These fixes ensure that:
- Model loading is robust with multiple fallback strategies
- All model attributes are validated before use
- Errors are caught early with descriptive messages
- The Dockerfile fails fast during build if TensorFlow is misconfigured
- Diagnostic information is available via the health endpoint
