"""
main.py
-------
Flask API server that integrates the V4 Faster R-CNN helmet-detection
pipeline so the safesight-web frontend can trigger processing via HTTP.

Endpoints:
    POST  /api/upload          – upload a video, start processing
    GET   /api/status/<job_id> – poll for progress
    GET   /api/download/<job_id> – download the annotated result video
    GET   /api/results/<job_id>  – get violation summary JSON

CLI fallback:
    python main.py --helmet   → launches legacy Streamlit app
    python main.py --v2       → launches legacy V2 script
    python main.py            → starts Flask API server (default)
"""

import os
import sys
import uuid
import time
import json
import glob
import threading
import argparse
import subprocess

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Project root (where this file lives) ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Make the V2/scripts directory importable ──────────────────────────────────
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'helmet_withoutyolo', 'V2', 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# Now import the V4 pipeline components
import torch
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from no_helmet_detection import draw_detections

# ── V4 Pipeline Constants ─────────────────────────────────────────────────────
NUM_CLASSES    = 4
CLASS_NAMES    = ['__background__', 'helmet', 'head', 'person']
THRESHOLD      = 0.5
DEVICE         = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
INFERENCE_SIZE = (320, 320)
FRAME_SKIP     = 2
BATCH_SIZE     = 4

MODEL_PATH = os.path.join(PROJECT_ROOT, 'helmet_withoutyolo', 'savedmodel', 'best_model_v4.pth')

# ── Upload / output directories ──────────────────────────────────────────────
UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'uploads')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'v4_results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── In-memory job tracker ─────────────────────────────────────────────────────
# { job_id: { status, progress, current_step, message, violations, ... } }
jobs = {}

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Vite dev server

# ── Model loading (lazy singleton) ───────────────────────────────────────────
_model = None
_model_lock = threading.Lock()


def get_model():
    """Load model once and cache it."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        print(f"[main] Loading V4 model on {DEVICE}...")
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        model = model.to(DEVICE)

        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("[main] Model weights loaded successfully.")
        else:
            print(f"[main] WARNING: Weights not found at {MODEL_PATH}")

        use_amp = DEVICE.type == 'cuda'
        if use_amp:
            model.half()
            model.float()
        model.eval()
        _model = model
        return _model


# ── Helper: convert frame to tensor ──────────────────────────────────────────
def frame_to_tensor(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INFERENCE_SIZE)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.to(DEVICE)


# ── Helper: run batch inference ──────────────────────────────────────────────
def run_batch(model, batch_tensors):
    use_amp = DEVICE.type == 'cuda'
    with torch.inference_mode():
        if use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(batch_tensors)
        else:
            outputs = model(batch_tensors)
    return outputs


# ── Core processing function (runs in a thread) ─────────────────────────────
def process_video_job(job_id, video_path, output_path):
    """
    Process a single video through the V4 pipeline.
    Updates the global `jobs` dict with real-time progress.
    """
    job = jobs[job_id]
    job['status'] = 'processing'
    job['current_step'] = 0
    job['message'] = 'Loading model...'

    try:
        model = get_model()

        # Step 1: Open video
        job['current_step'] = 0
        job['message'] = 'Opening video and preparing frames...'
        job['progress'] = 5

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            job['status'] = 'error'
            job['message'] = 'Failed to open video file'
            return

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scale_x = width  / INFERENCE_SIZE[0]
        scale_y = height / INFERENCE_SIZE[1]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Step 2: Running detection
        job['current_step'] = 1
        job['message'] = 'Running Faster R-CNN detection...'
        job['progress'] = 15

        frame_idx      = 0
        frames_buf     = []
        tensors_buf    = []
        write_queue    = {}
        next_write_idx = 0

        last_boxes  = np.empty((0, 4))
        last_scores = np.empty((0,))
        last_labels = np.empty((0,), dtype=int)

        # Violation tracking
        violation_frames = []   # list of { frame_idx, timestamp, detections[] }
        total_no_helmet  = 0

        def flush_write_queue():
            nonlocal next_write_idx
            while next_write_idx in write_queue:
                annotated = write_queue.pop(next_write_idx)
                out.write(annotated)
                next_write_idx += 1
            return True

        def flush_batch():
            nonlocal last_boxes, last_scores, last_labels, total_no_helmet
            if not tensors_buf:
                return True
            outputs = run_batch(model, tensors_buf)
            for (fidx, buf_frame), output in zip(frames_buf, outputs):
                last_boxes  = output['boxes'].cpu().numpy()
                last_scores = output['scores'].cpu().numpy()
                last_labels = output['labels'].cpu().numpy()

                # Collect no-helmet detection scores before drawing
                no_helmet_scores = []
                for score, label in zip(last_scores, last_labels):
                    if score > THRESHOLD and CLASS_NAMES[label] == 'head':
                        no_helmet_scores.append(float(score))

                if no_helmet_scores:
                    total_no_helmet += len(no_helmet_scores)
                    timestamp_sec = fidx / fps if fps > 0 else 0
                    mins = int(timestamp_sec // 60)
                    secs = int(timestamp_sec % 60)
                    violation_frames.append({
                        'frame': fidx,
                        'time': f'{mins:02d}:{secs:02d}',
                        'scores': no_helmet_scores,
                    })

                buf_frame = draw_detections(buf_frame, last_boxes, last_scores,
                                            last_labels, scale_x, scale_y)
                write_queue[fidx] = buf_frame
            frames_buf.clear()
            tensors_buf.clear()
            return flush_write_queue()

        t0 = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP == 0:
                frames_buf.append((frame_idx, frame))
                tensors_buf.append(frame_to_tensor(frame))
                if len(tensors_buf) == BATCH_SIZE:
                    flush_batch()
            else:
                annotated = draw_detections(frame.copy(), last_boxes, last_scores,
                                            last_labels, scale_x, scale_y)
                write_queue[frame_idx] = annotated
                flush_write_queue()

            frame_idx += 1

            # Update progress (15% → 85% over the frame loop)
            if total > 0:
                pct = 15 + int((frame_idx / total) * 70)
                job['progress'] = min(pct, 85)
                job['message'] = f'Processing frame {frame_idx}/{total}...'

        flush_batch()
        flush_write_queue()

        cap.release()
        out.release()
        elapsed = time.time() - t0

        # Step 3: Compiling results
        job['current_step'] = 2
        job['message'] = 'Compiling violation report...'
        job['progress'] = 90

        # Build violation summary, grouped by timestamp
        grouped_by_time = {}
        for v in violation_frames:
            time_str = v['time']
            if time_str not in grouped_by_time:
                grouped_by_time[time_str] = []
            grouped_by_time[time_str].extend(v['scores'])

        def parse_time(t_str):
            m, s = map(int, t_str.split(':'))
            return m * 60 + s

        def format_time(t_sec):
            m = t_sec // 60
            s = t_sec % 60
            return f'{m:02d}:{s:02d}'

        sorted_times = sorted(grouped_by_time.keys(), key=parse_time)
        violations = []

        if sorted_times:
            current_range_start = parse_time(sorted_times[0])
            current_range_end = current_range_start
            current_scores = list(grouped_by_time[sorted_times[0]])

            for t_str in sorted_times[1:]:
                t_sec = parse_time(t_str)
                if t_sec == current_range_end + 1:
                    current_range_end = t_sec
                    current_scores.extend(grouped_by_time[t_str])
                else:
                    # Save previous range
                    avg_score = sum(current_scores) / len(current_scores)
                    time_label = format_time(current_range_start) if current_range_start == current_range_end else f"{format_time(current_range_start)} - {format_time(current_range_end)}"
                    violations.append({
                        'time': time_label,
                        'type': 'No Helmet',
                        'severity': 'high',
                        'confidence': round(avg_score * 100, 1),
                    })
                    
                    # Start new range
                    current_range_start = t_sec
                    current_range_end = t_sec
                    current_scores = list(grouped_by_time[t_str])
                    
            # Save the last range
            avg_score = sum(current_scores) / len(current_scores)
            time_label = format_time(current_range_start) if current_range_start == current_range_end else f"{format_time(current_range_start)} - {format_time(current_range_end)}"
            violations.append({
                'time': time_label,
                'type': 'No Helmet',
                'severity': 'high',
                'confidence': round(avg_score * 100, 1),
            })

        # Step 4: Done
        job['current_step'] = 3
        job['progress'] = 100
        job['status'] = 'done'
        job['message'] = 'Processing complete'
        job['elapsed'] = round(elapsed, 1)
        job['total_frames'] = total
        job['violations'] = violations
        job['output_path'] = output_path
        job['total_no_helmet'] = total_no_helmet

        print(f"[main] Job {job_id} done in {elapsed:.1f}s — "
              f"{len(violations)} violations, output: {output_path}")

    except Exception as e:
        job['status'] = 'error'
        job['message'] = str(e)
        print(f"[main] Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Accept a video file, start background processing, return a job_id."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save uploaded file
    job_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or '.mp4'
    safe_name = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_DIR, f'{job_id}_{safe_name}')
    file.save(input_path)

    output_name = f'{job_id}_output{ext}'
    output_path = os.path.join(OUTPUT_DIR, output_name)

    # Initialise job entry
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'current_step': -1,
        'message': 'Queued for processing',
        'filename': file.filename,
        'input_path': input_path,
        'output_path': output_path,
        'violations': [],
        'total_no_helmet': 0,
        'elapsed': 0,
    }

    # Start processing in a background thread
    thread = threading.Thread(
        target=process_video_job,
        args=(job_id, input_path, output_path),
        daemon=True,
    )
    thread.start()

    return jsonify({'job_id': job_id, 'message': 'Processing started'}), 202


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Return current processing status for a job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'current_step': job['current_step'],
        'message': job['message'],
        'filename': job.get('filename', ''),
    })


@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Return the violation report for a completed job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job['status'] != 'done':
        return jsonify({'error': 'Job not yet complete', 'status': job['status']}), 400

    return jsonify({
        'job_id': job_id,
        'filename': job.get('filename', ''),
        'elapsed': job.get('elapsed', 0),
        'total_frames': job.get('total_frames', 0),
        'total_no_helmet': job.get('total_no_helmet', 0),
        'violations': job.get('violations', []),
    })


@app.route('/api/download/<job_id>', methods=['GET'])
def download_result(job_id):
    """Download the annotated output video."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job['status'] != 'done':
        return jsonify({'error': 'Job not yet complete'}), 400

    out_path = job.get('output_path', '')
    if not os.path.exists(out_path):
        return jsonify({'error': 'Output file not found'}), 404

    return send_file(out_path, as_attachment=True,
                     download_name=f"safesight_{job_id}_output.mp4")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SafeSight — CV Pipeline & API Server")
    parser.add_argument("--helmet", action="store_true", help="Run legacy Streamlit helmet app")
    parser.add_argument("--v2", action="store_true", help="Run legacy V2 script")
    parser.add_argument("--port", type=int, default=5000, help="API server port (default: 5000)")
    args = parser.parse_args()

    if args.helmet:
        print("Launching legacy Helmet Detection (Streamlit)...")
        subprocess.run(["streamlit", "run", "helmet_withoutyolo/app.py"])

    elif args.v2:
        print("Launching legacy V2 Project...")
        subprocess.run([sys.executable, "V2/scripts/app.py"])

    else:
        print("=" * 60)
        print("  SafeSight API Server")
        print(f"  Device  : {DEVICE}")
        print(f"  Model   : {MODEL_PATH}")
        print(f"  Port    : {args.port}")
        print("=" * 60)
        print(f"\n  Upload endpoint: http://localhost:{args.port}/api/upload")
        print(f"  Status endpoint: http://localhost:{args.port}/api/status/<job_id>")
        print()

        # Pre-load model at startup so first request is faster
        get_model()

        app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == "__main__":
    main()