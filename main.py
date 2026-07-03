"""
Multi-Model Vehicle License Plate Detection & OCR Pipeline
===========================================================

Architecture:
  - 4 Detection Models: YOLOv8n (LP fine-tuned), YOLOv8m, YOLOv10n, RT-DETR
  - 3 OCR Engines: EasyOCR, PaddleOCR, Tesseract (majority voting)
  - Image Enhancement: CLAHE + Denoising + Sharpening (for poor quality CCTV)
  - Plate Tracker: History buffer for text stabilization across frames

Usage:
  python main.py                        # Process all videos in input_videos/
  python main.py --video path/to.mp4    # Process a single video
"""

import cv2
import os
import sys
import csv
import re
import argparse
import logging
import numpy as np
from collections import Counter
from pathlib import Path

# --- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================
class Config:
    """Central configuration - nothing is hard-coded elsewhere."""

    # -- Paths --
    MODELS_DIR = Path("models")
    LP_MODEL_PATH = MODELS_DIR / "yolov8n_lp.pt"
    VEHICLE_MODELS = {
        # "yolov8m":  "yolov8m.pt",   # Disabled for speed
        "yolov10n": "yolov10n.pt",  # Ultra-fast NMS-free model
        # "rtdetr":   "rtdetr-l.pt",  # Disabled for speed (too heavy)
    }

    # -- Processing Speed --
    PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame to double speed

    # -- Detection thresholds --
    LP_CONF = 0.25          # license-plate detector confidence
    VEHICLE_CONF = 0.40     # vehicle detector confidence
    WBF_IOU = 0.50          # IoU threshold for Weighted Box Fusion
    WBF_SKIP = 0.01         # minimum score to keep after WBF

    # WBF weights: [direct_LP, yolov8m->LP, yolov10n->LP, rtdetr->LP]
    WBF_WEIGHTS = [3.0, 1.0, 1.0, 1.0]

    # COCO class IDs that represent vehicles
    VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    # -- OCR --
    OCR_MIN_CONF = 0.20

    # -- Tracker --
    TRACK_HISTORY = 15      # frames of text history to keep
    TRACK_IOU = 0.30        # IoU threshold for matching across frames

    # -- Enhancement --
    CLAHE_CLIP = 3.0
    CLAHE_GRID = (8, 8)

    # -- I/O --
    INPUT_DIR = Path("input_videos")
    OUTPUT_DIR = Path("output_videos")
    CSV_DIR = Path("output_csv")

# ==============================================================================
# Image Enhancement (handles low-quality CCTV footage)
# ==============================================================================
class ImageEnhancer:
    """CLAHE contrast + bilateral denoising + unsharp masking."""

    def __init__(self, cfg: Config):
        self.clahe = cv2.createCLAHE(
            clipLimit=cfg.CLAHE_CLIP, tileGridSize=cfg.CLAHE_GRID
        )

    # -- full-frame enhancement --
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        # 1. Bilateral denoise (preserves edges)
        dn = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        # 2. CLAHE on the L channel of LAB
        lab = cv2.cvtColor(dn, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge([self.clahe.apply(l), a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # 3. Unsharp mask
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # -- plate-crop enhancement for OCR --
    def enhance_plate(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        # Up-scale tiny plates so OCR has enough pixels
        if w < 200:
            crop = cv2.resize(crop, None, fx=200 / w, fy=200 / w,
                              interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        return self.clahe.apply(gray)

# ==============================================================================
# Detection Ensemble (4 models -> Weighted Box Fusion)
# ==============================================================================
class DetectionEnsemble:
    """
    Strategy
    --------
    Pass 1 - *Direct* LP detection on the full frame (fine-tuned YOLOv8n).
    Pass 2-4 - Each COCO vehicle detector finds vehicles; the LP model then
               runs on every vehicle crop.  This catches small / distant plates
               that the full-frame pass might miss.
    All plate boxes are merged with Weighted Box Fusion (ensemble-boxes lib).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lp_model = None
        self.vehicle_models = {}
        self._load()

    # -- loading --
    def _load(self):
        from ultralytics import YOLO

        # Primary LP model
        p = self.cfg.LP_MODEL_PATH
        if p.exists():
            logger.info("Loading LP detector  : %s", p)
            self.lp_model = YOLO(str(p))
        else:
            logger.warning("LP model not found at %s - run  models/download_models.py", p)

        # Vehicle detectors (COCO-pretrained)
        for tag, weight_file in self.cfg.VEHICLE_MODELS.items():
            try:
                logger.info("Loading vehicle det  : %s", weight_file)
                self.vehicle_models[tag] = YOLO(weight_file)
            except Exception as exc:
                logger.warning("Could not load %s: %s", weight_file, exc)

        total = (1 if self.lp_model else 0) + len(self.vehicle_models)
        logger.info("Detection models ready: %d", total)
        if total == 0:
            raise RuntimeError("No detection models available!")

    # -- helpers --
    def _run_lp(self, image: np.ndarray):
        """Return (boxes_px, scores) from the LP model."""
        if self.lp_model is None:
            return [], []
        results = self.lp_model(image, conf=self.cfg.LP_CONF, verbose=False)
        boxes, scores = [], []
        for r in results:
            for b in r.boxes:
                boxes.append(b.xyxy[0].cpu().numpy().tolist())
                scores.append(float(b.conf[0]))
        return boxes, scores

    def _run_vehicle(self, frame: np.ndarray, model):
        """Return list of vehicle bboxes [x1,y1,x2,y2] (int)."""
        results = model(frame, conf=self.cfg.VEHICLE_CONF, verbose=False)
        out = []
        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) in self.cfg.VEHICLE_CLASSES:
                    coords = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    out.append(coords)
        return out

    # -- main entry point --
    def detect(self, frame: np.ndarray):
        """
        Returns
        -------
        list[dict]   Each dict: {'bbox': [x1,y1,x2,y2], 'conf': float}
        """
        h, w = frame.shape[:2]
        boxes_lists, scores_lists, labels_lists, weights = [], [], [], []

        # Pass 1 - direct LP detection
        db, ds = self._run_lp(frame)
        if db:
            boxes_lists.append([[x / w, y / h, x2 / w, y2 / h]
                                for x, y, x2, y2 in db])
            scores_lists.append(ds)
            labels_lists.append([0] * len(db))
            weights.append(self.cfg.WBF_WEIGHTS[0])

        # Pass 2-4 - vehicle crop -> LP detection
        for idx, (tag, model) in enumerate(self.vehicle_models.items()):
            vehicles = self._run_vehicle(frame, model)
            mb, ms = [], []
            for vx1, vy1, vx2, vy2 in vehicles:
                crop = frame[vy1:vy2, vx1:vx2]
                if crop.size == 0:
                    continue
                pb, ps = self._run_lp(crop)
                for (px1, py1, px2, py2), sc in zip(pb, ps):
                    # map crop coords -> frame coords -> normalised
                    mb.append([(px1 + vx1) / w, (py1 + vy1) / h,
                               (px2 + vx1) / w, (py2 + vy1) / h])
                    ms.append(sc)
            if mb:
                boxes_lists.append(mb)
                scores_lists.append(ms)
                labels_lists.append([0] * len(mb))
                weights.append(self.cfg.WBF_WEIGHTS[idx + 1])

        if not boxes_lists:
            return []

        # -- Weighted Box Fusion --
        if len(boxes_lists) == 1:
            fused_b, fused_s = boxes_lists[0], scores_lists[0]
        else:
            try:
                from ensemble_boxes import weighted_boxes_fusion
                fused_b, fused_s, _ = weighted_boxes_fusion(
                    boxes_lists, scores_lists, labels_lists,
                    weights=weights,
                    iou_thr=self.cfg.WBF_IOU,
                    skip_box_thr=self.cfg.WBF_SKIP,
                )
            except ImportError:
                logger.warning("ensemble_boxes not installed - using direct detections only")
                fused_b, fused_s = boxes_lists[0], scores_lists[0]

        return [
            {"bbox": [int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h)],
             "conf": float(s)}
            for b, s in zip(fused_b, fused_s)
        ]

# ==============================================================================
# OCR Ensemble (3 engines -> majority vote)
# ==============================================================================
class OCREnsemble:
    """EasyOCR + PaddleOCR + Tesseract with majority-vote text selection."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engines = {}
        self._init()

    def _init(self):
        # -- EasyOCR --
        try:
            import easyocr
            self.engines["easyocr"] = easyocr.Reader(["en"], gpu=False)
            logger.info("OCR engine loaded    : EasyOCR")
        except Exception as e:
            logger.warning("EasyOCR unavailable  : %s", e)

        # -- PaddleOCR --
        try:
            from paddleocr import PaddleOCR
            self.engines["paddleocr"] = PaddleOCR(
                use_angle_cls=True, lang="en", show_log=False
            )
            logger.info("OCR engine loaded    : PaddleOCR")
        except Exception as e:
            logger.warning("PaddleOCR unavailable: %s", e)

        # -- Tesseract --
        # Disabled for speed (runs entirely on CPU and bottlenecks the GPU pipeline)
        # try:
        #     import pytesseract
        #     pytesseract.get_tesseract_version()
        #     self.engines["tesseract"] = pytesseract
        #     logger.info("OCR engine loaded    : Tesseract")
        # except Exception as e:
        #     logger.warning("Tesseract unavailable: %s", e)

        if not self.engines:
            logger.error("[WARN] No OCR engines loaded - text recognition disabled")

    # -- helpers --
    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper().strip())

    def _easy(self, img):
        r = self.engines["easyocr"]
        try:
            res = r.readtext(img)
            return self._clean(" ".join(t for _, t, c in res if c > self.cfg.OCR_MIN_CONF))
        except Exception:
            return ""

    def _paddle(self, img):
        ocr = self.engines["paddleocr"]
        try:
            res = ocr.ocr(img, cls=True)
            if res and res[0]:
                return self._clean(" ".join(
                    line[1][0] for line in res[0]
                    if line and len(line) >= 2 and line[1][1] > self.cfg.OCR_MIN_CONF
                ))
        except Exception:
            pass
        return ""

    def _tess(self, img):
        tess = self.engines["tesseract"]
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            return self._clean(tess.image_to_string(gray, config="--psm 7 --oem 3"))
        except Exception:
            return ""

    # -- main entry --
    def recognize(self, plate_bgr: np.ndarray, plate_gray: np.ndarray = None):
        """
        Returns
        -------
        dict  {'text': str, 'confidence': float, 'votes': dict}
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return {"text": "", "confidence": 0.0, "votes": {}}

        votes = {}
        if "easyocr" in self.engines:
            votes["easyocr"] = self._easy(plate_bgr)
        if "paddleocr" in self.engines:
            votes["paddleocr"] = self._paddle(plate_bgr)
        if "tesseract" in self.engines:
            img = plate_gray if plate_gray is not None else plate_bgr
            votes["tesseract"] = self._tess(img)

        valid = [t for t in votes.values() if len(t) >= 2]
        if not valid:
            best = max(votes.values(), key=len, default="")
            return {"text": best, "confidence": 0.3, "votes": votes}

        best, count = Counter(valid).most_common(1)[0]
        return {"text": best, "confidence": count / len(valid), "votes": votes}

# ==============================================================================
# Plate Tracker (IoU matching + text stabilisation)
# ==============================================================================
class PlateTracker:
    """Reduces OCR flicker by maintaining a per-plate text history."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tracks = {}   # id -> {bbox, text_history, last_seen}
        self._next = 0

    @staticmethod
    def _iou(a, b):
        xi = max(a[0], b[0]); yi = max(a[1], b[1])
        xa = min(a[2], b[2]); ya = min(a[3], b[3])
        inter = max(0, xa - xi) * max(0, ya - yi)
        union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
        return inter / union if union else 0

    def update(self, detections: list, frame_num: int):
        # evict stale tracks
        stale = [k for k, v in self._tracks.items()
                 if frame_num - v["last_seen"] > self.cfg.TRACK_HISTORY]
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
                used.add(best_id)
                trk = self._tracks[best_id]
                trk["bbox"] = det["bbox"]
                trk["last_seen"] = frame_num
                if det.get("text"):
                    trk["text_history"].append(det["text"])
                    trk["text_history"] = trk["text_history"][-self.cfg.TRACK_HISTORY:]
                tid = best_id
            else:
                tid = self._next; self._next += 1
                self._tracks[tid] = {
                    "bbox": det["bbox"],
                    "text_history": [det["text"]] if det.get("text") else [],
                    "last_seen": frame_num,
                }

            hist = [t for t in self._tracks[tid]["text_history"] if len(t) >= 2]
            stable = Counter(hist).most_common(1)[0][0] if hist else det.get("text", "")
            out.append({**det, "stable_text": stable, "track_id": tid})
        return out

# ==============================================================================
# Video Processor (orchestrator)
# ==============================================================================
class VideoProcessor:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        logger.info("=" * 60)
        logger.info("  Multi-Model ANPR Pipeline - Initialising")
        logger.info("=" * 60)
        self.enhancer = ImageEnhancer(cfg)
        self.detector = DetectionEnsemble(cfg)
        self.ocr      = OCREnsemble(cfg)
        self.tracker   = PlateTracker(cfg)

    # -- drawing --
    @staticmethod
    def _draw(frame, results):
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            txt = r.get("stable_text", r.get("text", ""))
            conf = r.get("conf", 0)
            label = f"{txt} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return frame

    # -- single video --
    def process_video(self, video_path: str):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open: %s", video_path)
            return

        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = self.cfg.OUTPUT_DIR / f"detected_{video_path.stem}.mp4"
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        csv_path = self.cfg.CSV_DIR / f"plates_{video_path.stem}.csv"
        os.makedirs(self.cfg.CSV_DIR, exist_ok=True)
        csv_f = open(csv_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame", "time_s", "track_id", "raw_text",
                        "stable_text", "confidence", "x1", "y1", "x2", "y2",
                        "easyocr", "paddleocr", "tesseract"])

        logger.info("Processing : %s", video_path.name)
        logger.info("Resolution : %dx%d  FPS: %d  Frames: %d", w, h, fps, total)

        n = 0
        last_tracked = []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            n += 1

            if n % self.cfg.PROCESS_EVERY_N_FRAMES == 0:
                # 1) enhance
                enhanced = self.enhancer.enhance_frame(frame)

                # 2) detect (multi-model ensemble)
                dets = self.detector.detect(enhanced)

                # 3) OCR each plate
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        d["text"] = ""
                        d["votes"] = {}
                        continue
                    gray = self.enhancer.enhance_plate(crop)
                    ocr_res = self.ocr.recognize(crop, gray)
                    d["text"] = ocr_res["text"]
                    d["votes"] = ocr_res["votes"]

                # 4) track
                last_tracked = self.tracker.update(dets, n)

                # 5) CSV log
                ts = n / fps
                for t in last_tracked:
                    v = t.get("votes", {})
                    csv_w.writerow([
                        n, f"{ts:.2f}", t["track_id"], t["text"],
                        t["stable_text"], f'{t["conf"]:.3f}', *t["bbox"],
                        v.get("easyocr", ""), v.get("paddleocr", ""),
                        v.get("tesseract", ""),
                    ])

            # 6) annotate + write EVERY frame (using latest tracked boxes)
            writer.write(self._draw(frame.copy(), last_tracked))

            if n % 30 == 0:
                pct = 100 * n / max(total, 1)
                logger.info("  Frame %d / %d  (%.1f%%)", n, total, pct)

        cap.release()
        writer.release()
        csv_f.close()
        logger.info("[OK] Video  -> %s", out_path)
        logger.info("[OK] CSV    -> %s", csv_path)

    # -- batch --
    def process_all(self):
        vids = sorted(
            p for p in self.cfg.INPUT_DIR.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        )
        if not vids:
            logger.error("No videos in %s", self.cfg.INPUT_DIR)
            return
        logger.info("Found %d video(s)", len(vids))
        for i, v in enumerate(vids, 1):
            logger.info("\n" + "=" * 60)
            logger.info("  [%d/%d]  %s", i, len(vids), v.name)
            logger.info("=" * 60)
            self.process_video(v)

# ==============================================================================
# CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Multi-Model Vehicle License Plate Detection & OCR")
    ap.add_argument("--video",      type=str, help="Single video file to process")
    ap.add_argument("--input-dir",  type=str, default="input_videos")
    ap.add_argument("--output-dir", type=str, default="output_videos")
    args = ap.parse_args()

    cfg = Config()
    cfg.INPUT_DIR  = Path(args.input_dir)
    cfg.OUTPUT_DIR = Path(args.output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CSV_DIR, exist_ok=True)

    proc = VideoProcessor(cfg)
    if args.video:
        proc.process_video(args.video)
    else:
        proc.process_all()

if __name__ == "__main__":
    main()
