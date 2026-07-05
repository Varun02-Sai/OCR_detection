"""
Download pretrained model weights for the ANPR pipeline.

Models:
  1. YOLOv8n  - Fine-tuned for license plate detection (from HuggingFace)
  2. YOLOv8m  - COCO pretrained vehicle detection (Ultralytics auto-download)
  3. YOLO11s  - COCO pretrained vehicle detection (Ultralytics auto-download)
"""

import os
import urllib.request
from pathlib import Path

# Use __file__ for reliable path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent

HF_MODELS = {
    "yolov8n_lp.pt": {
        "url": "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
        "description": "YOLOv8 Nano — fine-tuned for license plate detection",
    },
}

ULTRALYTICS_MODELS = ["yolov8m.pt", "yolo11s.pt"]


def download_file(url: str, dest: Path):
    """Download a file with progress."""
    print(f"  Downloading {dest.name} ...")
    print(f"  URL: {url}")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, str(dest))
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [OK] Saved ({size_mb:.1f} MB)")
            return
        except Exception as e:
            print(f"  Attempt {attempt+1}/3 failed: {e}")
    print(f"  [FAIL] Could not download {dest.name}")


def download_hf_models():
    """Download models from HuggingFace."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, info in HF_MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"  [SKIP] {filename} already exists")
            continue
        print(f"\n--- {info['description']} ---")
        download_file(info["url"], dest)


def download_ultralytics_models():
    """Trigger Ultralytics auto-download for COCO models."""
    from ultralytics import YOLO
    
    # Ensure ultralytics models are downloaded into the project root
    # where the script expects them
    os.chdir(BASE_DIR)
    
    for model_name in ULTRALYTICS_MODELS:
        print(f"\n--- Loading {model_name} (Ultralytics auto-download) ---")
        try:
            _ = YOLO(model_name)
            print(f"  [OK] {model_name} ready")
        except Exception as e:
            print(f"  [WARN] Could not load {model_name}: {e}")


def main():
    print("=" * 60)
    print("  ANPR Pipeline — Downloading Model Weights")
    print("=" * 60)
    print(f"\nModels directory: {MODELS_DIR}")
    
    print("\n[1/2] License plate model (HuggingFace) ...")
    download_hf_models()
    
    print("\n[2/2] Vehicle detection models (Ultralytics) ...")
    download_ultralytics_models()
    
    print("\n" + "=" * 60)
    print("  All models ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
