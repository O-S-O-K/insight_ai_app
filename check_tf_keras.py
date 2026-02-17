import sys
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

try:
    import tensorflow as tf
    print(f'TensorFlow version: {tf.__version__}')

    # tf-keras should make tensorflow.keras available
    from tensorflow import keras
    print(f'tf.keras available: {keras.__version__}')

    # Check if it's the legacy keras
    import tf_keras
    print(f'tf-keras installed: {tf_keras.__version__}')
    sys.exit(0)
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
