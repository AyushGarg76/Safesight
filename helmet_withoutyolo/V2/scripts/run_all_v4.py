"""
run_all_v4.py
-------------
Batch-processes all test_video*.mp4 files in the scripts/ directory through
the V4 Faster R-CNN detection pipeline and writes annotated outputs to:
    helmet_withoutyolo/V2/outputs_v4/
"""

import os
import glob
import time
import torch
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.cuda.amp import autocast
from no_helmet_detection import draw_detections

# ── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES    = 4
CLASS_NAMES    = ['__background__', 'helmet', 'head', 'person']
THRESHOLD      = 0.5
DEVICE         = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
INFERENCE_SIZE = (320, 320)
FRAME_SKIP     = 2
BATCH_SIZE     = 4

CLASS_COLORS = {
    'helmet': (0, 255, 0),
    'head':   (255, 0,   0),
    'person': (0,   0, 255),
}

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, '..', '..', 'savedmodel', 'best_model_v4.pth')

# All 6 test videos live in the same scripts/ directory
VIDEO_PATTERN = os.path.join(SCRIPT_DIR, 'test_video*.mp4')

# Output folder: V2/outputs_v4/
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs_v4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHOW_PREVIEW = False


# ── Model helpers ─────────────────────────────────────────────────────────────
def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def frame_to_tensor(frame, inference_size):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, inference_size)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.to(DEVICE)


# def draw_detections(frame, boxes, scores, labels, scale_x, scale_y):
#     for box, score, label in zip(boxes, scores, labels):
#         if score > THRESHOLD:
#             xmin, ymin, xmax, ymax = (int(box[0]*scale_x), int(box[1]*scale_y),
#                                        int(box[2]*scale_x), int(box[3]*scale_y))
#             class_name = CLASS_NAMES[label]
#             color = CLASS_COLORS.get(class_name, (200, 200, 200))
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#             cv2.putText(frame, f"{class_name} {score:.2f}",
#                         (xmin, max(ymin - 5, 0)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     return frame


def run_batch(model, batch_tensors, use_amp):
    with torch.inference_mode():
        if use_amp and DEVICE.type == 'cuda':
            with autocast():
                outputs = model(batch_tensors)
        else:
            outputs = model(batch_tensors)
    return outputs


# ── Per-video processing ──────────────────────────────────────────────────────
def process_video(model, video_path, out_path, use_amp):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale_x = width  / INFERENCE_SIZE[0]
    scale_y = height / INFERENCE_SIZE[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"  >> {os.path.basename(video_path)}  ({total} frames @ {fps:.1f} fps)")

    frame_idx      = 0
    frames_buf     = []
    tensors_buf    = []
    write_queue    = {}
    next_write_idx = 0

    last_boxes  = np.empty((0, 4))
    last_scores = np.empty((0,))
    last_labels = np.empty((0,), dtype=int)

    def flush_write_queue():
        nonlocal next_write_idx
        while next_write_idx in write_queue:
            annotated = write_queue.pop(next_write_idx)
            out.write(annotated)
            if SHOW_PREVIEW:
                cv2.imshow('Helmet Detection V4', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False
            next_write_idx += 1
        return True

    def flush_batch():
        nonlocal last_boxes, last_scores, last_labels
        if not tensors_buf:
            return True
        outputs = run_batch(model, tensors_buf, use_amp)
        for (fidx, buf_frame), output in zip(frames_buf, outputs):
            last_boxes  = output['boxes'].cpu().numpy()
            last_scores = output['scores'].cpu().numpy()
            last_labels = output['labels'].cpu().numpy()
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
            tensors_buf.append(frame_to_tensor(frame, INFERENCE_SIZE))
            if len(tensors_buf) == BATCH_SIZE:
                if not flush_batch():
                    break
        else:
            annotated = draw_detections(frame.copy(), last_boxes, last_scores,
                                        last_labels, scale_x, scale_y)
            write_queue[frame_idx] = annotated
            if not flush_write_queue():
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"     {frame_idx}/{total} frames  ({elapsed:.1f}s elapsed)")

    flush_batch()
    flush_write_queue()

    cap.release()
    out.release()
    elapsed = time.time() - t0
    print(f"     Done in {elapsed:.1f}s  ->  {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Output : {os.path.abspath(OUTPUT_DIR)}\n")

    use_amp = DEVICE.type == 'cuda'

    model = create_model(NUM_CLASSES).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded V4 model weights.\n")
    else:
        print(f"[WARNING] Weights not found at {MODEL_PATH}. Predictions may be random.\n")

    if use_amp:
        model.half()
        model.float()
    model.eval()

    video_files = sorted(glob.glob(VIDEO_PATTERN))
    if not video_files:
        print(f"No videos matched pattern: {VIDEO_PATTERN}")
        return

    print(f"Found {len(video_files)} video(s) to process:\n")
    total_start = time.time()

    for i, video_path in enumerate(video_files, 1):
        name     = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{name}_output_v4.mp4")
        print(f"[{i}/{len(video_files)}]", end=" ")
        process_video(model, video_path, out_path, use_amp)
        print()

    total_elapsed = time.time() - total_start
    print(f"All done in {total_elapsed:.1f}s  --  outputs in: {os.path.abspath(OUTPUT_DIR)}")

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
