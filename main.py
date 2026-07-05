"""
Production ANPR Pipeline — Automatic Number Plate Recognition
==============================================================

Architecture:
  Detection : YOLOv8n (fine-tuned LP) + YOLOv8m (vehicle) + YOLO11s (vehicle)
  OCR       : PaddleOCR (GPU-accelerated)
  Enhance   : CLAHE + Bilateral Denoise + Unsharp Mask
  Tracking  : IoU-based plate tracker with text stabilisation

Usage:
  python main.py --video path/to/video.mp4
  python main.py                              # process all in sample_videos/
"""

import cv2
import os
import csv
import re
import argparse
import logging
import numpy as np
import torch
from collections import Counter
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration — auto-detects GPU, nothing hardcoded
# =============================================================================
class Config:
    """Central configuration with automatic GPU detection."""

    # ── Device ──
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FP16 = torch.cuda.is_available()

    # ── Model Paths ──
    BASE_DIR = Path(__file__).resolve().parent
    LP_MODEL_PATH = BASE_DIR / "models" / "yolov8n_lp.pt"
    VEHICLE_MODELS = {
        "yolov8m":  "yolov8m.pt",    # Medium — high accuracy
        "yolo11s":  "yolo11s.pt",    # YOLO11 Small — best small-object detection
    }

    # ── Detection Thresholds ──
    LP_CONF = 0.25            # license plate detector confidence
    VEHICLE_CONF = 0.40       # vehicle detector confidence
    NMS_IOU = 0.45            # IoU threshold for NMS dedup
    VEHICLE_CLASSES = {2, 3, 5, 7}  # COCO: car, motorcycle, bus, truck

    # ── Processing ──
    PROCESS_EVERY_N_FRAMES = 2   # skip frames for speed (1 = every frame)

    # ── OCR ──
    OCR_MIN_CONF = 0.30

    # ── Tracker ──
    TRACK_HISTORY = 15        # frames of text history per track
    TRACK_IOU = 0.30          # IoU threshold for matching plates across frames

    # ── Image Enhancement ──
    CLAHE_CLIP = 3.0
    CLAHE_GRID = (8, 8)

    # ── Paths ──
    INPUT_DIR = BASE_DIR / "sample_videos"
    OUTPUT_DIR = BASE_DIR / "output_videos"
    CSV_DIR = BASE_DIR / "output_csv"


# =============================================================================
# Image Enhancement — handles low-quality CCTV footage
# =============================================================================
class ImageEnhancer:
    """CLAHE contrast + bilateral denoising + unsharp masking."""

    def __init__(self, cfg: Config):
        self.clahe = cv2.createCLAHE(
            clipLimit=cfg.CLAHE_CLIP, tileGridSize=cfg.CLAHE_GRID
        )

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance a full video frame for better detection."""
        # 1. Bilateral denoise (preserves edges)
        dn = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        # 2. CLAHE on the L channel of LAB colour space
        lab = cv2.cvtColor(dn, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        lab = cv2.merge([self.clahe.apply(l_ch), a_ch, b_ch])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # 3. Unsharp mask for edge sharpening
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    def enhance_plate(self, crop: np.ndarray):
        """Enhance a plate crop for OCR. Returns (upscaled_bgr, enhanced_gray)."""
        if crop is None or crop.size == 0:
            return crop, None
        h, w = crop.shape[:2]
        # Upscale tiny plates so OCR has enough pixels
        if w < 200:
            scale = max(200 / w, 2.0)
            crop = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = self.clahe.apply(gray)
        return crop, gray


# =============================================================================
# Plate Detector — 3-model architecture with NMS dedup
# =============================================================================
class PlateDetector:
    """
    3-pass detection:
      Pass 1 — Fine-tuned YOLOv8n detects plates directly on the full frame.
      Pass 2 — YOLOv8m finds vehicles → LP model runs on each vehicle crop.
      Pass 3 — YOLO11s finds vehicles → LP model runs on each vehicle crop.
    All results are merged with IoU-based NMS to remove duplicates.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lp_model = None
        self.vehicle_models = {}
        self._load_models()

    def _load_models(self):
        from ultralytics import YOLO

        # Primary LP model (fine-tuned)
        lp_path = self.cfg.LP_MODEL_PATH
        if lp_path.exists():
            logger.info("Loading LP detector   : %s", lp_path.name)
            self.lp_model = YOLO(str(lp_path))
        else:
            logger.warning("LP model not found: %s — run models/download_models.py", lp_path)

        # Vehicle detectors (COCO pre-trained)
        for tag, weight_file in self.cfg.VEHICLE_MODELS.items():
            try:
                logger.info("Loading vehicle det   : %s", weight_file)
                self.vehicle_models[tag] = YOLO(weight_file)
            except Exception as exc:
                logger.warning("Could not load %s: %s", weight_file, exc)

        total = (1 if self.lp_model else 0) + len(self.vehicle_models)
        logger.info("Detection models ready: %d / 3", total)
        if self.lp_model is None:
            raise RuntimeError("LP detection model is required but not found!")

    def _run_lp(self, image: np.ndarray):
        """Run LP model on an image. Returns (boxes_px, scores)."""
        if self.lp_model is None:
            return [], []
        results = self.lp_model(
            image, conf=self.cfg.LP_CONF, verbose=False,
            half=self.cfg.USE_FP16, device=self.cfg.DEVICE,
        )
        boxes, scores = [], []
        for r in results:
            for b in r.boxes:
                boxes.append(b.xyxy[0].cpu().numpy().tolist())
                scores.append(float(b.conf[0]))
        return boxes, scores

    def _run_vehicle(self, frame: np.ndarray, model):
        """Return list of vehicle bboxes [x1, y1, x2, y2] (int)."""
        results = model(
            frame, conf=self.cfg.VEHICLE_CONF, verbose=False,
            half=self.cfg.USE_FP16, device=self.cfg.DEVICE,
        )
        out = []
        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) in self.cfg.VEHICLE_CLASSES:
                    coords = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    out.append(coords)
        return out

    @staticmethod
    def _iou(a, b):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        xi = max(a[0], b[0])
        yi = max(a[1], b[1])
        xa = min(a[2], b[2])
        ya = min(a[3], b[3])
        inter = max(0, xa - xi) * max(0, ya - yi)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _nms(self, detections):
        """Simple NMS to remove duplicate plate detections."""
        if not detections:
            return []
        dets = sorted(detections, key=lambda x: x["conf"], reverse=True)
        keep = []
        for d in dets:
            is_dup = False
            for k in keep:
                if self._iou(d["bbox"], k["bbox"]) > self.cfg.NMS_IOU:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(d)
        return keep

    def detect(self, frame: np.ndarray):
        """
        Run 3-pass detection on a frame.

        Returns
        -------
        list[dict]   Each dict: {'bbox': [x1,y1,x2,y2], 'conf': float}
        """
        h, w = frame.shape[:2]
        all_dets = []

        # Pass 1 — direct LP detection on full frame
        boxes, scores = self._run_lp(frame)
        for box, score in zip(boxes, scores):
            x1 = max(0, min(w, int(box[0])))
            y1 = max(0, min(h, int(box[1])))
            x2 = max(0, min(w, int(box[2])))
            y2 = max(0, min(h, int(box[3])))
            if x2 > x1 and y2 > y1:
                all_dets.append({"bbox": [x1, y1, x2, y2], "conf": score})

        # Pass 2–3 — vehicle detection → crop → LP detection
        for tag, model in self.vehicle_models.items():
            vehicles = self._run_vehicle(frame, model)
            for vx1, vy1, vx2, vy2 in vehicles:
                # Clamp vehicle box
                vx1 = max(0, vx1)
                vy1 = max(0, vy1)
                vx2 = min(w, vx2)
                vy2 = min(h, vy2)
                crop = frame[vy1:vy2, vx1:vx2]
                if crop.size == 0:
                    continue
                pb, ps = self._run_lp(crop)
                for (px1, py1, px2, py2), sc in zip(pb, ps):
                    # Map crop coords → frame coords
                    fx1 = max(0, min(w, int(px1 + vx1)))
                    fy1 = max(0, min(h, int(py1 + vy1)))
                    fx2 = max(0, min(w, int(px2 + vx1)))
                    fy2 = max(0, min(h, int(py2 + vy1)))
                    if fx2 > fx1 and fy2 > fy1:
                        all_dets.append({"bbox": [fx1, fy1, fx2, fy2], "conf": sc})

        # NMS merge
        return self._nms(all_dets)


# =============================================================================
# OCR — PaddleOCR (GPU-accelerated, single engine)
# =============================================================================
class PlateOCR:
    """PaddleOCR-based plate text recognition with GPU support."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ocr = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            use_gpu = (self.cfg.DEVICE == "cuda")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=use_gpu,
                show_log=False,
            )
            logger.info("OCR engine loaded     : PaddleOCR (GPU=%s)", use_gpu)
        except Exception as e:
            logger.error("PaddleOCR failed to load: %s", e)

    @staticmethod
    def _clean(text: str) -> str:
        """Remove non-alphanumeric characters and uppercase."""
        return re.sub(r"[^A-Z0-9]", "", text.upper().strip())

    def recognize(self, plate_bgr: np.ndarray, plate_gray: np.ndarray = None):
        """
        Recognize text from a plate crop.

        Returns
        -------
        dict  {'text': str, 'confidence': float}
        """
        if self.ocr is None or plate_bgr is None or plate_bgr.size == 0:
            return {"text": "", "confidence": 0.0}

        # Try BGR image first, then grayscale if BGR fails
        for img in [plate_bgr, plate_gray]:
            if img is None:
                continue
            try:
                result = self.ocr.ocr(img, cls=True)
                if result and result[0]:
                    texts, confs = [], []
                    for line in result[0]:
                        if line and len(line) >= 2:
                            txt = line[1][0]
                            conf = line[1][1]
                            if conf > self.cfg.OCR_MIN_CONF:
                                texts.append(txt)
                                confs.append(conf)
                    if texts:
                        full_text = self._clean(" ".join(texts))
                        if len(full_text) >= 2:
                            avg_conf = sum(confs) / len(confs)
                            return {"text": full_text, "confidence": avg_conf}
            except Exception:
                continue

        return {"text": "", "confidence": 0.0}


# =============================================================================
# Plate Tracker — IoU matching + text stabilisation
# =============================================================================
class PlateTracker:
    """Reduces OCR flicker by maintaining per-plate text history."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tracks = {}   # id → {bbox, text_history, last_seen, first_seen}
        self._next_id = 0

    @staticmethod
    def _iou(a, b):
        xi = max(a[0], b[0])
        yi = max(a[1], b[1])
        xa = min(a[2], b[2])
        ya = min(a[3], b[3])
        inter = max(0, xa - xi) * max(0, ya - yi)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def update(self, detections: list, frame_num: int):
        """Match detections to existing tracks, return tracked results."""
        # Evict stale tracks
        stale = [
            k for k, v in self._tracks.items()
            if frame_num - v["last_seen"] > self.cfg.TRACK_HISTORY
        ]
        for k in stale:
            del self._tracks[k]

        used = set()
        out = []

        for det in detections:
            best_id, best_iou = None, 0
            for tid, trk in self._tracks.items():
                if tid in used:
                    continue
                iou = self._iou(det["bbox"], trk["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_iou > self.cfg.TRACK_IOU and best_id is not None:
                # Matched existing track
                used.add(best_id)
                trk = self._tracks[best_id]
                trk["bbox"] = det["bbox"]
                trk["last_seen"] = frame_num
                if det.get("text"):
                    trk["text_history"].append(det["text"])
                    trk["text_history"] = trk["text_history"][-self.cfg.TRACK_HISTORY:]
                trk["frame_count"] = trk.get("frame_count", 0) + 1
                tid = best_id
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "bbox": det["bbox"],
                    "text_history": [det["text"]] if det.get("text") else [],
                    "last_seen": frame_num,
                    "first_seen": frame_num,
                    "frame_count": 1,
                }

            # Stabilise text: most common reading from history
            hist = [t for t in self._tracks[tid]["text_history"] if len(t) >= 2]
            stable = Counter(hist).most_common(1)[0][0] if hist else det.get("text", "")

            out.append({
                **det,
                "stable_text": stable,
                "track_id": tid,
                "first_seen": self._tracks[tid]["first_seen"],
                "frame_count": self._tracks[tid]["frame_count"],
            })

        return out

    def get_all_vehicles(self):
        """Return summary of all tracked vehicles (for Gradio results table)."""
        vehicles = []
        for tid, trk in self._tracks.items():
            hist = [t for t in trk["text_history"] if len(t) >= 2]
            if not hist:
                continue
            best_text, count = Counter(hist).most_common(1)[0]
            confidence = count / len(hist) if hist else 0
            vehicles.append({
                "track_id": tid,
                "plate_text": best_text,
                "confidence": confidence,
                "first_seen": trk["first_seen"],
                "last_seen": trk["last_seen"],
                "frame_count": trk.get("frame_count", 0),
            })
        return vehicles


# =============================================================================
# Video Processor — orchestrates the full pipeline
# =============================================================================
class VideoProcessor:
    """End-to-end video processing: detect → OCR → track → annotate."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        logger.info("=" * 60)
        logger.info("  Production ANPR Pipeline — Initialising")
        logger.info("  Device: %s | FP16: %s", cfg.DEVICE, cfg.USE_FP16)
        logger.info("=" * 60)
        self.enhancer = ImageEnhancer(cfg)
        self.detector = PlateDetector(cfg)
        self.ocr = PlateOCR(cfg)
        self.tracker = PlateTracker(cfg)

    @staticmethod
    def _draw(frame, results):
        """Draw bounding boxes and plate text on frame."""
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            txt = r.get("stable_text", r.get("text", ""))
            conf = r.get("conf", 0)
            tid = r.get("track_id", "?")
            label = f"Vehicle {tid}: {txt} ({conf:.0%})"

            # Green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(
                frame,
                (x1, y1 - th - 10),
                (x1 + tw + 4, y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                frame, label, (x1 + 2, y1 - 5),
                font, font_scale, (0, 0, 0), thickness,
            )
        return frame

    def process_video(self, video_path: str, progress_callback=None):
        """
        Process a video end-to-end.

        Parameters
        ----------
        video_path : str
            Path to the input video file.
        progress_callback : callable, optional
            Function(progress_float, status_str) for Gradio progress updates.

        Returns
        -------
        dict with keys: 'output_video', 'output_csv', 'vehicles'
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open: %s", video_path)
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output paths
        out_video_path = self.cfg.OUTPUT_DIR / f"detected_{video_path.stem}.mp4"
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h),
        )

        csv_path = self.cfg.CSV_DIR / f"plates_{video_path.stem}.csv"
        os.makedirs(self.cfg.CSV_DIR, exist_ok=True)

        logger.info("Processing : %s", video_path.name)
        logger.info("Resolution : %dx%d  FPS: %d  Frames: %d", w, h, fps, total)

        # Reset tracker for each new video
        self.tracker = PlateTracker(self.cfg)

        frame_num = 0
        last_tracked = []

        # Use `with` for safe CSV writing
        with open(csv_path, "w", newline="") as csv_f:
            csv_w = csv.writer(csv_f)
            csv_w.writerow([
                "frame", "time_s", "track_id", "plate_text",
                "stable_text", "confidence", "x1", "y1", "x2", "y2",
            ])

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_num += 1

                if frame_num % self.cfg.PROCESS_EVERY_N_FRAMES == 0:
                    # 1) Enhance frame
                    enhanced = self.enhancer.enhance_frame(frame)

                    # 2) Detect plates (3-model ensemble)
                    dets = self.detector.detect(enhanced)

                    # 3) OCR each plate
                    for d in dets:
                        x1, y1, x2, y2 = d["bbox"]
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            d["text"] = ""
                            continue
                        crop_bgr, gray = self.enhancer.enhance_plate(crop)
                        ocr_res = self.ocr.recognize(crop_bgr, gray)
                        d["text"] = ocr_res["text"]

                    # 4) Track plates across frames
                    last_tracked = self.tracker.update(dets, frame_num)

                    # 5) Write to CSV
                    ts = frame_num / fps
                    for t in last_tracked:
                        csv_w.writerow([
                            frame_num,
                            f"{ts:.2f}",
                            t["track_id"],
                            t.get("text", ""),
                            t["stable_text"],
                            f'{t["conf"]:.3f}',
                            *t["bbox"],
                        ])

                # 6) Annotate EVERY frame with latest detections
                writer.write(self._draw(frame.copy(), last_tracked))

                # Progress logging
                if frame_num % 30 == 0:
                    pct = frame_num / max(total, 1)
                    logger.info("  Frame %d / %d  (%.1f%%)", frame_num, total, pct * 100)
                    if progress_callback:
                        progress_callback(pct, f"Frame {frame_num}/{total}")

        cap.release()
        writer.release()

        # Convert to H.264 for browser playback
        web_video_path = self._convert_to_h264(out_video_path)

        vehicles = self.tracker.get_all_vehicles()
        # Add timestamp info
        for v in vehicles:
            v["first_seen_s"] = round(v["first_seen"] / fps, 2)
            v["last_seen_s"] = round(v["last_seen"] / fps, 2)

        logger.info("[OK] Video  → %s", web_video_path)
        logger.info("[OK] CSV    → %s", csv_path)
        logger.info("[OK] Vehicles detected: %d", len(vehicles))

        return {
            "output_video": str(web_video_path),
            "output_csv": str(csv_path),
            "vehicles": vehicles,
        }

    @staticmethod
    def _convert_to_h264(raw_path: Path) -> Path:
        """Convert mp4v to H.264 for web browser compatibility."""
        import subprocess
        web_path = raw_path.parent / f"web_{raw_path.name}"
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_path),
                 "-vcodec", "libx264", "-preset", "fast",
                 "-crf", "23", str(web_path)],
                capture_output=True, timeout=300,
            )
            if result.returncode == 0 and web_path.exists():
                raw_path.unlink(missing_ok=True)  # remove raw to save space
                return web_path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("FFmpeg not available — video may not play in browser")
        return raw_path

    def process_all(self):
        """Process all videos in the input directory."""
        input_dir = self.cfg.INPUT_DIR
        if not input_dir.exists():
            logger.error("Input directory not found: %s", input_dir)
            logger.info("Run: python download_samples.py  to get test videos")
            return

        extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        vids = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)
        if not vids:
            logger.error("No videos found in %s", input_dir)
            return

        logger.info("Found %d video(s)", len(vids))
        for i, v in enumerate(vids, 1):
            logger.info("\n" + "=" * 60)
            logger.info("  [%d/%d]  %s", i, len(vids), v.name)
            logger.info("=" * 60)
            self.process_video(str(v))


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Production ANPR — License Plate Detection & OCR"
    )
    ap.add_argument("--video", type=str, help="Path to a single video file")
    ap.add_argument("--input-dir", type=str, default=None,
                    help="Directory of videos to process")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Directory for output videos")
    args = ap.parse_args()

    cfg = Config()
    if args.input_dir:
        cfg.INPUT_DIR = Path(args.input_dir)
    if args.output_dir:
        cfg.OUTPUT_DIR = Path(args.output_dir)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CSV_DIR, exist_ok=True)

    proc = VideoProcessor(cfg)
    if args.video:
        result = proc.process_video(args.video)
        logger.info("\n📋 Detected Vehicles:")
        for v in result["vehicles"]:
            logger.info(
                "  Vehicle %d: %s (conf: %.0f%%, seen: %.1fs–%.1fs, %d frames)",
                v["track_id"], v["plate_text"],
                v["confidence"] * 100,
                v["first_seen_s"], v["last_seen_s"],
                v["frame_count"],
            )
    else:
        proc.process_all()


if __name__ == "__main__":
    main()
