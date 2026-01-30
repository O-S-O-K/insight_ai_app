# training/train_baseline.py

import sys
import os

# Add project root to sys.path for imports

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from utils.preprocessing import load_cifar10, get_data_augmentation, CLASS_NAMES

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


# ------------------------------
# 1. Load and preprocess data
# ------------------------------
x_train, y_train, x_test, y_test = load_cifar10(normalize=True, one_hot=True)

# Optional: define data augmentation
data_augmentation = get_data_augmentation()

# ------------------------------
# 2. Build the baseline CNN
# ------------------------------
def build_baseline_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Simple CNN for CIFAR-10
    Compatible with Grad-CAM because last layer is convolutional
    """
    model = models.Sequential()

    # Convolutional Block 1
    model.add(layers.Input(shape=input_shape))
    model.add(data_augmentation)  # Apply augmentation on the fly
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))

    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))

    # Convolutional Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

model = build_baseline_cnn()
model.summary()

# ------------------------------
# 3. Compile the model
# ------------------------------
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------
# 4. Define callbacks
# ------------------------------
checkpoint_path = "models/cnn_baseline.h5"
os.makedirs("models", exist_ok=True)

callbacks_list = [
    callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
]

# ------------------------------
# 5. Train the model
# ------------------------------
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks_list
)

# ------------------------------
# 6. Evaluate the model
# ------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
