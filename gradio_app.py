"""
Production ANPR — Gradio Web Interface
=======================================
Professional UI for uploading traffic videos and viewing
per-vehicle license plate detection results.
"""

import gradio as gr
import os
import shutil
import pandas as pd
from pathlib import Path

# Import the ANPR pipeline
from main import Config, VideoProcessor

# ── Configuration ──
cfg = Config()
cfg.OUTPUT_DIR = Path("temp_gradio_out")
cfg.CSV_DIR = Path("temp_gradio_csv")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CSV_DIR, exist_ok=True)

# ── Load models once at startup ──
print("\n🔧 Loading ANPR models into memory...")
processor = VideoProcessor(cfg)
print("✅ Models loaded. Ready to process videos.\n")


def process_video(video_path, progress=gr.Progress()):
    """
    Process an uploaded video and return results.

    Returns: (output_video_path, vehicles_dataframe, csv_path)
    """
    if not video_path:
        raise gr.Error("Please upload a video first.")

    progress(0.0, desc="Starting processing...")

    def progress_cb(pct, status):
        progress(pct, desc=status)

    try:
        result = processor.process_video(video_path, progress_callback=progress_cb)
    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")

    progress(1.0, desc="Done!")

    # ── Build vehicles table ──
    vehicles = result.get("vehicles", [])
    if vehicles:
        table_data = []
        for i, v in enumerate(vehicles, 1):
            table_data.append({
                "Vehicle #": i,
                "License Plate": v["plate_text"],
                "Confidence": f"{v['confidence']:.0%}",
                "First Seen": f"{v['first_seen_s']:.1f}s",
                "Last Seen": f"{v['last_seen_s']:.1f}s",
                "Frames": v["frame_count"],
            })
        df = pd.DataFrame(table_data)
    else:
        df = pd.DataFrame({
            "Vehicle #": [],
            "License Plate": [],
            "Confidence": [],
            "First Seen": [],
            "Last Seen": [],
            "Frames": [],
        })

    out_video = result.get("output_video")
    out_csv = result.get("output_csv")

    return out_video, df, out_csv


def clear_outputs():
    """Clear temp files."""
    for d in [cfg.OUTPUT_DIR, cfg.CSV_DIR]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
    return None, None, None


# =============================================================================
# Gradio UI — Dark, Professional Theme
# =============================================================================
custom_css = """
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}
.results-header {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.5rem;
    font-weight: 700;
}
.model-info {
    padding: 12px 16px;
    border-radius: 8px;
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.3);
    margin-bottom: 1rem;
    font-size: 0.9rem;
}
"""

with gr.Blocks(
    title="ANPR — License Plate Detection",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="emerald",
    ),
    css=custom_css,
) as demo:

    # ── Header ──
    gr.HTML('<h1 class="main-title">🚔 Automatic Number Plate Recognition</h1>')
    gr.HTML('<p class="subtitle">Upload any traffic video — toll gates, highways, CCTV — and get precise license plate readings</p>')

    gr.HTML("""
    <div class="model-info">
        <strong>🧠 Detection:</strong> YOLOv8n (LP fine-tuned) + YOLOv8m + YOLO11s &nbsp;|&nbsp;
        <strong>📝 OCR:</strong> PaddleOCR (GPU) &nbsp;|&nbsp;
        <strong>🔧 Enhancement:</strong> CLAHE + Bilateral Denoise + Unsharp Mask
    </div>
    """)

    with gr.Row(equal_height=True):
        # ── Left Column: Upload ──
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Video")
            input_video = gr.Video(
                label="Traffic / CCTV Video",
                height=350,
            )
            with gr.Row():
                process_btn = gr.Button(
                    "🚀 Detect License Plates",
                    variant="primary",
                    size="lg",
                )
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")

        # ── Right Column: Output Video ──
        with gr.Column(scale=1):
            gr.Markdown("### 🎬 Annotated Output")
            output_video = gr.Video(
                label="Detected Plates",
                height=350,
                interactive=False,
            )

    # ── Results Section ──
    gr.HTML('<h2 class="results-header">📋 Detection Results</h2>')

    with gr.Row():
        with gr.Column(scale=2):
            vehicles_table = gr.Dataframe(
                label="Detected Vehicles & License Plates",
                headers=["Vehicle #", "License Plate", "Confidence",
                         "First Seen", "Last Seen", "Frames"],
                interactive=False,
                wrap=True,
            )
        with gr.Column(scale=1):
            csv_download = gr.File(
                label="📥 Download Full CSV Log",
                interactive=False,
            )
            gr.Markdown("""
            **CSV contains:**
            - Frame-by-frame detections
            - Timestamps for each plate reading
            - Bounding box coordinates
            - Stabilised plate text
            """)

    # ── Usage Tips ──
    with gr.Accordion("💡 Tips for Best Results", open=False):
        gr.Markdown("""
        - **Video Quality**: Higher resolution = better OCR accuracy
        - **Camera Angle**: Plates visible at < 45° angle work best
        - **Lighting**: The pipeline auto-enhances dark/CCTV footage
        - **Speed**: Processing uses GPU if available (recommended: T4 or better)
        - **Multiple Vehicles**: Each vehicle gets a unique track ID
        - **Stabilisation**: Plate text is stabilised across frames for accuracy
        """)

    # ── Event Handlers ──
    process_btn.click(
        fn=process_video,
        inputs=[input_video],
        outputs=[output_video, vehicles_table, csv_download],
    )

    clear_btn.click(
        fn=clear_outputs,
        outputs=[output_video, vehicles_table, csv_download],
    )


if __name__ == "__main__":
    print("\n🌐 Launching ANPR Web Interface...\n")
    demo.launch(
        server_name="0.0.0.0",
        share=True,
        show_error=True,
    )
