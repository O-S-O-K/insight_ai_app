"""Download ImageNet labels for MobileNetV2"""
import json
import urllib.request

# ImageNet class labels (1000 classes)
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

print(f"Downloading ImageNet labels from {url}...")
with urllib.request.urlopen(url) as response:
    imagenet_labels = json.loads(response.read().decode())

# Convert to simple index -> label mapping
label_map = {}
for idx, (wordnet_id, label) in imagenet_labels.items():
    # Use the human-readable label (second item)
    label_map[str(idx)] = label.replace("_", " ").title()

# Create metadata file
metadata = {
    "model_name": "cnn_baseline_functional",
    "version": "v2.0_imagenet",
    "architecture": "MobileNetV2 (ImageNet pretrained)",
    "trained_on": "ImageNet-1K (1000 classes)",
    "last_updated": "2026-02-13",
    "classes": label_map
}

# Save to model_metadata.json
output_file = "model_metadata.json"
with open(output_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved {len(label_map)} ImageNet labels to {output_file}")
print(f"\nSample labels:")
for i in range(5):
    print(f"  {i}: {label_map[str(i)]}")
print("  ...")
print(f"  150: {label_map.get('150', 'N/A')}")
print(f"  250: {label_map.get('250', 'N/A')}")
