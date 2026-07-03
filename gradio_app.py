import gradio as gr
import os
import shutil
from pathlib import Path

# Import our existing pipeline
from main import Config, VideoProcessor

# Configure temp directories for Gradio
cfg = Config()
cfg.OUTPUT_DIR = Path("temp_gradio_out")
cfg.CSV_DIR = Path("temp_gradio_csv")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.CSV_DIR, exist_ok=True)

# Load models globally so they don't reload on every upload
print("Loading models into memory... This might take a moment.")
processor = VideoProcessor(cfg)

def process_video_gradio(video_path):
    """Gradio handler: Takes uploaded video path, returns output paths."""
    if not video_path:
        return None, None
    
    print(f"\n[Gradio] Processing uploaded video: {video_path}")
    processor.process_video(video_path)
    
    # Find the generated outputs
    vid_name = Path(video_path).stem
    raw_vid = cfg.OUTPUT_DIR / f"detected_{vid_name}.mp4"
    out_csv = cfg.CSV_DIR / f"plates_{vid_name}.csv"
    
    if not raw_vid.exists() or not out_csv.exists():
        raise gr.Error("Processing failed or generated no output.")

    # Convert to browser-compatible H.264 using FFmpeg
    import subprocess
    web_vid = cfg.OUTPUT_DIR / f"web_{vid_name}.mp4"
    print("Converting video for web browser compatibility...")
    subprocess.run(["ffmpeg", "-y", "-i", str(raw_vid), "-vcodec", "libx264", str(web_vid)], capture_output=True)
    
    final_vid = str(web_vid) if web_vid.exists() else str(raw_vid)
        
    return final_vid, str(out_csv)

# ---------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------
with gr.Blocks(title="Multi-Model ANPR Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚓 Multi-Model Vehicle OCR Number Plate Detection
        Upload a CCTV traffic video. This system uses **4 detection models** (YOLOv8n, YOLOv8m, YOLOv10n, RT-DETR) 
        and **3 OCR engines** (EasyOCR, PaddleOCR, Tesseract) to find and transcribe license plates.
        """
    )
    
    with gr.Row():
        with gr.Column():
            in_video = gr.Video(label="Upload Traffic Video")
            submit_btn = gr.Button("Process Video", variant="primary")
            gr.Markdown("*Note: This is very compute-heavy. A GPU is highly recommended.*")
        
        with gr.Column():
            out_video = gr.Video(label="Annotated Output Video", interactive=False)
            out_csv = gr.File(label="Download CSV Log", interactive=False)
            
    submit_btn.click(
        fn=process_video_gradio,
        inputs=in_video,
        outputs=[out_video, out_csv]
    )

if __name__ == "__main__":
    # share=True creates a public gradio.live URL automatically
    print("Launching Gradio Web Server...")
    demo.launch(share=True, server_name="0.0.0.0")
