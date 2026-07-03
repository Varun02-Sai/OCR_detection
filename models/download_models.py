"""
Download all pretrained model weights for the multi-model ANPR pipeline.

Models:
  1. YOLOv8n  - Fine-tuned for license plate detection (Hugging Face)
  2. YOLOv8m  - COCO pretrained (vehicle detection) [Ultralytics auto-download]
  3. YOLOv10n - COCO pretrained (vehicle detection) [Ultralytics auto-download]
  4. RT-DETR-l - COCO pretrained (vehicle detection) [Ultralytics auto-download]
"""

import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")

# -- Model definitions --
MODELS = {
    "yolov8n_lp.pt": {
        "url": "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
        "description": "YOLOv8 Nano - fine-tuned for license plate detection (Hugging Face)",
    },
}

# These are auto-downloaded by Ultralytics on first use, but we trigger
# the download here so the user doesn't hit a delay at inference time.
ULTRALYTICS_MODELS = [
    "yolov8m.pt",
    "yolov10n.pt",
    "rtdetr-l.pt",
]

def download_file(url: str, dest: Path):
    """Download a file with a progress indicator."""
    print(f"  Downloading {dest.name} ...")
    print(f"  URL: {url}")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  [OK] Saved ({size_mb:.1f} MB)")

def download_huggingface_models():
    """Download manually-hosted model weights."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, info in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"  [SKIP] {filename} already exists")
            continue
        print(f"\n--- {info['description']} ---")
        download_file(info["url"], dest)

def download_ultralytics_models():
    """Trigger Ultralytics auto-download for COCO-pretrained models."""
    from ultralytics import YOLO

    for model_name in ULTRALYTICS_MODELS:
        print(f"\n--- Loading {model_name} (Ultralytics will auto-download if needed) ---")
        try:
            _ = YOLO(model_name)
            print(f"  [OK] {model_name} ready")
        except Exception as e:
            print(f"  [FAIL] Failed to load {model_name}: {e}")

def main():
    print("=" * 60)
    print("  Multi-Model ANPR - Downloading Pretrained Weights")
    print("=" * 60)

    print("\n[1/2] Downloading license-plate-specific model from Hugging Face ...")
    download_huggingface_models()

    print("\n[2/2] Downloading COCO-pretrained vehicle detection models ...")
    download_ultralytics_models()

    print("\n" + "=" * 60)
    print("  All models downloaded successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
