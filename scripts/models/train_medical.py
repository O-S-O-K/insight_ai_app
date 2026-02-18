#!/usr/bin/env python3
"""
Medical Imaging Fine-tuning Script — EfficientNetB0 on ISIC 2020 Skin Lesion Dataset.

Usage:
    python scripts/models/train_medical.py [--data-dir data/isic2020] [--epochs 30]

This script:
1. Loads ISIC 2020 train/val split (prepared by download_isic_data.py)
2. Phase 1: Trains a classification head on frozen EfficientNetB0
3. Phase 2: Fine-tunes the top 20 layers with a lower learning rate
4. Saves the trained model as models/medical_model.h5
5. Updates models/medical_metadata.json with evaluation metrics

Architecture:
    EfficientNetB0 (ImageNet pretrained) → GlobalAveragePooling2D → Dropout(0.3)
    → Dense(128, relu) → Dense(2, softmax)

Binary classes: 0=Benign, 1=Melanoma
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Set environment variables before TF import
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def check_dependencies():
    missing = []
    try:
        import tensorflow as tf
    except ImportError:
        missing.append("tensorflow or tensorflow-cpu")
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        sys.exit(1)


def build_model(num_classes: int = 2, dropout_rate: float = 0.3):
    """Build EfficientNetB0 + classification head for transfer learning."""
    import tensorflow as tf

    # Use legacy Keras for TF 2.15 compatibility
    try:
        from tf_keras.applications import EfficientNetB0
        from tf_keras import layers, Model
    except ImportError:
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras import layers, Model

    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base.trainable = False  # Freeze for phase 1

    inputs = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="efficientnetb0_medical")
    print(f"  Model built: {model.name}")
    print(f"  Total params: {model.count_params():,}")
    print(f"  Trainable params: {sum(tf.size(v).numpy() for v in model.trainable_variables):,}")
    return model, base


def get_data_generators(data_dir: Path, batch_size: int = 32):
    """Create train/val image data generators with augmentation."""
    try:
        from tf_keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # EfficientNetB0 expects [0, 255] input (handles normalization internally)
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        brightness_range=[0.8, 1.2],
        zoom_range=0.15,
        fill_mode="reflect",
    )

    val_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        print(f"ERROR: Data directories not found: {train_dir}, {val_dir}")
        print("Run scripts/models/download_isic_data.py first.")
        sys.exit(1)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        classes=["benign", "melanoma"],
        shuffle=True,
    )

    val_flow = val_gen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        classes=["benign", "melanoma"],
        shuffle=False,
    )

    return train_flow, val_flow


def compute_class_weights(train_flow) -> dict:
    """Compute inverse-frequency class weights to handle imbalance."""
    import numpy as np

    labels = train_flow.classes
    n_total = len(labels)
    n_classes = len(train_flow.class_indices)
    weights = {}
    for cls_idx in range(n_classes):
        n_cls = (labels == cls_idx).sum()
        weights[cls_idx] = n_total / (n_classes * n_cls)

    print("  Class weights:")
    for cls_idx, w in weights.items():
        cls_name = list(train_flow.class_indices.keys())[cls_idx]
        print(f"    {cls_name}: {w:.4f}")
    return weights


def train_phase1(model, base, train_flow, val_flow, class_weights, epochs: int, output_dir: Path):
    """Phase 1: Train head only (base frozen)."""
    try:
        from tf_keras.optimizers import Adam
        from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
        from tf_keras.metrics import AUC
    except ImportError:
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.metrics import AUC

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=5, mode="max", restore_best_weights=True),
        ModelCheckpoint(
            str(output_dir / "phase1_best.h5"),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
        ),
    ]

    print(f"  Training {epochs} epochs (head only)...")
    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def train_phase2(model, base, train_flow, val_flow, class_weights, epochs: int, output_dir: Path, fine_tune_layers: int = 20):
    """Phase 2: Unfreeze top N layers and fine-tune."""
    try:
        from tf_keras.optimizers import Adam
        from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
        from tf_keras.metrics import AUC
    except ImportError:
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        from tensorflow.keras.metrics import AUC

    # Unfreeze top N layers
    base.trainable = True
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"  Fine-tuning top {trainable_count} EfficientNetB0 layers")

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=7, mode="max", restore_best_weights=True),
        ModelCheckpoint(
            str(output_dir / "phase2_best.h5"),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
        ),
    ]

    print(f"  Fine-tuning {epochs} epochs...")
    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_model(model, val_flow):
    """Evaluate model and return metrics dict."""
    import numpy as np

    results = model.evaluate(val_flow, verbose=0)
    metric_names = model.metrics_names
    metrics = dict(zip(metric_names, results))

    print("  Validation metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # Compute sensitivity / specificity
    import tensorflow as tf
    preds = model.predict(val_flow, verbose=0)
    pred_classes = preds.argmax(axis=1)
    true_classes = val_flow.classes

    tp = ((pred_classes == 1) & (true_classes == 1)).sum()
    fn = ((pred_classes == 0) & (true_classes == 1)).sum()
    tn = ((pred_classes == 0) & (true_classes == 0)).sum()
    fp = ((pred_classes == 1) & (true_classes == 0)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics["sensitivity"] = float(sensitivity)
    metrics["specificity"] = float(specificity)
    print(f"    sensitivity: {sensitivity:.4f}")
    print(f"    specificity: {specificity:.4f}")

    return metrics


def save_model(model, output_path: Path):
    """Save model in H5 format."""
    model.save(str(output_path))
    print(f"  Model saved: {output_path}")


def update_metadata(metrics: dict, metadata_path: Path):
    """Update medical_metadata.json with training results."""
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    meta["performance"] = {
        "val_accuracy": round(metrics.get("accuracy", 0), 4),
        "val_auc": round(metrics.get("auc", 0), 4),
        "val_sensitivity": round(metrics.get("sensitivity", 0), 4),
        "val_specificity": round(metrics.get("specificity", 0), 4),
    }

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata updated: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 on ISIC 2020")
    parser.add_argument("--data-dir", default="data/isic2020", help="ISIC dataset directory")
    parser.add_argument("--output-dir", default="models", help="Model output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-phase1", type=int, default=10, help="Phase 1 head epochs")
    parser.add_argument("--epochs-phase2", type=int, default=20, help="Phase 2 fine-tune epochs")
    parser.add_argument("--fine-tune-layers", type=int, default=20, help="Unfreeze top N EfficientNet layers")
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    print("=" * 70)
    print("Medical Imaging Training — EfficientNetB0 on ISIC 2020")
    print("=" * 70)
    print()

    check_dependencies()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Building model...")
    model, base = build_model(num_classes=2, dropout_rate=args.dropout)

    print("\nStep 2: Loading dataset...")
    train_flow, val_flow = get_data_generators(data_dir, batch_size=args.batch_size)
    print(f"  Train: {train_flow.samples} images")
    print(f"  Val:   {val_flow.samples} images")

    print("\nStep 3: Computing class weights...")
    class_weights = compute_class_weights(train_flow)

    print(f"\nStep 4: Phase 1 — Training head ({args.epochs_phase1} epochs max)...")
    train_phase1(model, base, train_flow, val_flow, class_weights, args.epochs_phase1, output_dir)

    print(f"\nStep 5: Phase 2 — Fine-tuning top {args.fine_tune_layers} layers ({args.epochs_phase2} epochs max)...")
    train_phase2(model, base, train_flow, val_flow, class_weights, args.epochs_phase2, output_dir, args.fine_tune_layers)

    print("\nStep 6: Evaluating final model...")
    metrics = evaluate_model(model, val_flow)

    print("\nStep 7: Saving final model...")
    model_path = output_dir / "medical_model.h5"
    save_model(model, model_path)

    print("\nStep 8: Updating metadata...")
    metadata_path = output_dir / "medical_metadata.json"
    if metadata_path.exists():
        update_metadata(metrics, metadata_path)

    print()
    print("=" * 70)
    print("SUCCESS: Medical model training complete!")
    print(f"  Model: {model_path}")
    print(f"  Val AUC: {metrics.get('auc', 0):.4f}")
    print(f"  Val Sensitivity: {metrics.get('sensitivity', 0):.4f}")
    print(f"  Val Specificity: {metrics.get('specificity', 0):.4f}")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Set MODEL_TYPE=medical in backend env vars")
    print("  2. Ensure models/medical_model.h5 is accessible to the API")
    print("  3. Restart the backend service")
    return 0


if __name__ == "__main__":
    sys.exit(main())
