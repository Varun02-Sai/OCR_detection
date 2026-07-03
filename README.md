# Multi-Model Vehicle OCR Number Plate Detection

A robust, production-grade ANPR (Automatic Number Plate Recognition) pipeline
designed for **CCTV / traffic camera footage** (police project).

## Architecture

| Component | Models | Purpose |
|-----------|--------|---------|
| **Detection Ensemble** | YOLOv8n (LP fine-tuned), YOLOv8m, YOLOv10n, RT-DETR | Find license plates via Weighted Box Fusion |
| **OCR Ensemble** | EasyOCR, PaddleOCR, Tesseract | Read plate text via majority voting |
| **Image Enhancement** | CLAHE + Bilateral Denoise + Unsharp Mask | Handle low-quality / night CCTV footage |
| **Plate Tracker** | IoU-based frame-to-frame tracker | Stabilise text across frames |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download all pretrained models
python models/download_models.py

# 3. Run the pipeline on all videos in input_videos/
python main.py

# Or process a single video
python main.py --video input_videos/my_video.mp4
```

## Output

- **Annotated videos** → `output_videos/`  (bounding boxes + OCR text overlay)
- **CSV logs** → `output_csv/`  (frame-by-frame plate text, timestamps, per-engine votes)

## Training Your Own Models (Optional)

If the pretrained weights don't work well for your region's plates:

1. Get a [Roboflow](https://roboflow.com) API key
2. Upload `train_lightning.py` to [Lightning AI](https://lightning.ai)
3. Run it on a GPU instance
4. Copy the generated `best.pt` files into the `models/` directory

## Project Structure

```
OCR_detection/
├── main.py                    # Multi-model detection + OCR pipeline
├── train_lightning.py         # Training script for Lightning AI
├── requirements.txt           # Python dependencies
├── README.md
├── models/
│   ├── download_models.py     # Downloads all pretrained weights
│   └── yolov8n_lp.pt         # Fine-tuned LP detector (auto-downloaded)
├── input_videos/              # Place your CCTV footage here
├── output_videos/             # Annotated output videos
└── output_csv/                # Per-frame plate detection logs
```
