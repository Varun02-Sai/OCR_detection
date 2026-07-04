# Multi-Model Vehicle OCR Number Plate Detection

A robust, highly optimized ANPR (Automatic Number Plate Recognition) pipeline designed specifically for CCTV and traffic camera footage.

## Architecture

Our pipeline employs an advanced "multi-model ensemble" approach to guarantee maximum accuracy on fast-moving vehicles.

| Component | Models | Purpose |
|-----------|--------|---------|
| **Detection Ensemble** | YOLOv8m + YOLOv10n | Uses Weighted Box Fusion (WBF) to combine bounding boxes from multiple AI models, balancing speed (v10n) with high accuracy (v8m). |
| **Plate Localization** | YOLOv8n (Fine-tuned) | Specifically trained to crop license plates accurately. |
| **OCR Ensemble** | PaddleOCR + EasyOCR | Reads plate text using multiple optical engines and implements majority-voting to filter out errors. |
| **Image Enhancement** | CLAHE + Denoising | Dynamically handles low-quality and night-time CCTV footage. |
| **Plate Tracker** | History Buffer Algorithm | Stabilizes text across multiple frames so the output is locked onto the vehicle. |

## Quick Start on Google Colab (GPU Recommended)

Due to the heavy multi-model architecture, this project is designed to run seamlessly on a Cloud GPU.

1. Open the project in Google Colab.
2. Run the following setup commands:
```bash
!pip install -r requirements.txt
!python setup_dataset.py
!python gradio_app.py
```
3. Click the Gradio Web Server link that appears to upload videos and view real-time AI processing!

## Dataset
The system uses the `car-number-plate-video` dataset automatically provisioned via KaggleHub for high-quality testing data.

## Output
- **Annotated videos**: Real-time bounding boxes and OCR text overlay.
- **CSV logs**: Frame-by-frame plate text, timestamps, and per-engine votes.
