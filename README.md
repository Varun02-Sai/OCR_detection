# Multi-Model Automatic Number Plate Recognition (ANPR)

A production-grade, highly optimized ANPR pipeline designed for CCTV, toll gate, and traffic camera footage.

## Architecture

This pipeline uses a **3-model detection ensemble** combined with **GPU-accelerated PaddleOCR** for maximum accuracy and speed on fast-moving vehicles.

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Plate Detection** | YOLOv8n (Fine-tuned) | Specifically trained to crop license plates accurately. |
| **Vehicle Detection 1** | YOLOv8m (COCO) | Medium model: catches distant/small vehicles. |
| **Vehicle Detection 2** | YOLO11s (COCO) | Modern small model: highly accurate on small objects. |
| **OCR Engine** | PaddleOCR | Industry-standard OCR engine for accurate text extraction (GPU optimized). |
| **Image Enhancement** | CLAHE + Bilateral Denoising | Dynamically handles low-quality, glared, or night-time CCTV footage. |
| **Tracking Engine** | IoU + History Buffer | Tracks vehicles across frames and stabilises the recognized text. |

## Quick Start (Google Colab / Cloud GPU)

For the best performance, run this on a GPU instance (like a Google Colab T4).

1. Open `Run_on_Colab.ipynb` in Google Colab.
2. The notebook will automatically:
   - Install dependencies
   - Download the AI models
   - Download 3 high-quality test videos (toll gate, traffic camera, highway)
   - Launch the Gradio Web UI

## Local Installation

If you have a local GPU (Nvidia) or want to test on CPU:

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Download Models
python models/download_models.py

# 3. (Optional) Download Test Videos
python download_samples.py

# 4. Launch the Web Interface
python gradio_app.py
```

## Command Line Usage

To process a video without the UI:
```bash
python main.py --video path/to/your_video.mp4
```

The output will be saved in `output_videos/` and a detailed CSV log will be generated in `output_csv/`.
