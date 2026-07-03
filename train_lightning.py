"""
Training Script for Lightning AI
=================================

Fine-tune all 4 detection models on a license plate dataset so that each
model is *specifically* trained to detect plates (not just COCO vehicles).

How to use
----------
1. Upload this file to Lightning AI (https://lightning.ai).
2. Choose a GPU instance (A10G or better recommended).
3. Run:  python train_lightning.py
4. Download the trained weights from  runs/  and place them in the
   models/ directory of this project.

Dataset
-------
Uses the Roboflow "License Plate Recognition" dataset (open source).
The script auto-downloads it via the Roboflow API.
If you have your own dataset in YOLO format, set CUSTOM_DATA_YAML below.
"""

import os
from pathlib import Path

# ── Configuration ──
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
ROBOFLOW_WORKSPACE = "roboflow-universe-projects"
ROBOFLOW_PROJECT   = "license-plate-recognition-rxg4e"
ROBOFLOW_VERSION   = 4
DATA_FORMAT        = "yolov8"

# Set to a path if you have your own YOLO-format dataset
CUSTOM_DATA_YAML   = None   # e.g. "/data/plates/data.yaml"

EPOCHS  = 50
IMGSZ   = 640
BATCH   = 16
DEVICE  = "0"   # GPU id; set to "cpu" for CPU-only


def download_dataset():
    """Download the license plate dataset from Roboflow."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(DATA_FORMAT)
    return dataset.location + "/data.yaml"


def train_model(base_weights: str, run_name: str, data_yaml: str):
    """Fine-tune a single YOLO / RT-DETR model."""
    from ultralytics import YOLO
    print(f"\n{'='*60}")
    print(f"  Training: {run_name}  (base: {base_weights})")
    print(f"{'='*60}\n")

    model = YOLO(base_weights)
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        name=run_name,
        patience=10,
        save=True,
        plots=True,
    )
    # The best weights are at  runs/detect/<run_name>/weights/best.pt
    best = Path(f"runs/detect/{run_name}/weights/best.pt")
    print(f"\n✓ Best weights saved to: {best}")
    return best


def main():
    # 1. Get dataset
    if CUSTOM_DATA_YAML:
        data_yaml = CUSTOM_DATA_YAML
        print(f"Using custom dataset: {data_yaml}")
    else:
        print("Downloading license plate dataset from Roboflow ...")
        data_yaml = download_dataset()

    # 2. Train each model
    models_to_train = [
        ("yolov8n.pt",   "yolov8n_lp"),     # Nano — fast
        ("yolov8m.pt",   "yolov8m_lp"),     # Medium — accurate
        ("yolov10n.pt",  "yolov10n_lp"),    # v10 Nano — NMS-free
        ("rtdetr-l.pt",  "rtdetr_lp"),      # Transformer — highest accuracy
    ]

    results = {}
    for base, name in models_to_train:
        try:
            best = train_model(base, name, data_yaml)
            results[name] = str(best)
        except Exception as e:
            print(f"\n✗ Failed to train {name}: {e}")
            results[name] = "FAILED"

    # 3. Summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — Summary")
    print("=" * 60)
    for name, path in results.items():
        print(f"  {name:20s} → {path}")
    print()
    print("Copy these best.pt files to your project's models/ directory:")
    print("  models/yolov8n_lp.pt   ← runs/detect/yolov8n_lp/weights/best.pt")
    print("  models/yolov8m_lp.pt   ← runs/detect/yolov8m_lp/weights/best.pt")
    print("  models/yolov10n_lp.pt  ← runs/detect/yolov10n_lp/weights/best.pt")
    print("  models/rtdetr_lp.pt    ← runs/detect/rtdetr_lp/weights/best.pt")
    print()
    print("Then update Config.VEHICLE_MODELS in main.py to point to the new weights.")


if __name__ == "__main__":
    main()
