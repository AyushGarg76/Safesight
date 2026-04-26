import os
import torch
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.cuda.amp import autocast

# Constants
NUM_CLASSES = 4
CLASS_NAMES = ['__background__', 'helmet', 'head', 'person']
THRESHOLD = 0.5
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Inference size — larger = more accurate, smaller = faster (128 is fast, 320 is balanced)
INFERENCE_SIZE = (320, 320)

# Process every Nth frame; skip frames reuse last detections (1 = every frame, 2 = every other, etc.)
FRAME_SKIP = 2

# Batch size: how many frames to infer at once (increase if VRAM allows)
BATCH_SIZE = 4

# Paths
MODEL_PATH = os.path.join('..', '..', 'savedmodel', 'best_model_v4.pth')
VIDEO_PATH = os.path.join('..', '..', 'test_video6.mp4')
OUT_PATH   = os.path.join('..', '..', 'output_v4.mp4')

# Show live preview window (set False for maximum speed — no GUI overhead)
SHOW_PREVIEW = False

# Colors per class
CLASS_COLORS = {
    'helmet': (0, 255, 0),
    'head':   (255, 0,   0),
    'person': (0,   0, 255),
}


def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def frame_to_tensor(frame, inference_size):
    """Convert BGR frame → normalised float tensor on DEVICE."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, inference_size)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.to(DEVICE)


def draw_detections(frame, boxes, scores, labels, scale_x, scale_y):
    for box, score, label in zip(boxes, scores, labels):
        if score > THRESHOLD:
            xmin = int(box[0] * scale_x)
            ymin = int(box[1] * scale_y)
            xmax = int(box[2] * scale_x)
            ymax = int(box[3] * scale_y)
            class_name = CLASS_NAMES[label]
            color = CLASS_COLORS.get(class_name, (200, 200, 200))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, f"{class_name} {score:.2f}",
                        (xmin, max(ymin - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def run_batch(model, batch_tensors, use_amp):
    """Run inference on a list of tensors, return list of output dicts."""
    with torch.inference_mode():
        if use_amp and DEVICE.type == 'cuda':
            with autocast():
                outputs = model(batch_tensors)
        else:
            outputs = model(batch_tensors)
    return outputs


def main():
    print(f"Using device: {DEVICE}")
    use_amp = DEVICE.type == 'cuda'

    model = create_model(NUM_CLASSES).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Loaded V4 model weights.")
    else:
        print(f"Weights not found at {MODEL_PATH}. Predictions may be random.")

    # FP16 for speed on GPU
    if use_amp:
        model.half()
        model.float()   # keep BN in float; AMP handles mixed precision per-op

    model.eval()

    # Resolve video path
    video_input = VIDEO_PATH if os.path.exists(VIDEO_PATH) else 'test_video6.mp4'
    if not os.path.exists(video_input):
        print("No video found. Please provide a valid video path.")
        return

    cap = cv2.VideoCapture(video_input)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale_x = width  / INFERENCE_SIZE[0]
    scale_y = height / INFERENCE_SIZE[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (width, height))

    print(f"Processing video: {video_input}  ({total} frames, {fps:.1f} fps)")
    print(f"Inference size: {INFERENCE_SIZE}, Frame skip: {FRAME_SKIP}, Batch: {BATCH_SIZE}, AMP: {use_amp}")

    frame_idx   = 0
    frames_buf  = []   # (frame_index, raw BGR frame) for infer batch
    tensors_buf = []   # preprocessed tensors for infer batch

    # Write-queue: holds (frame_index, annotated BGR frame) in order
    write_queue = {}
    next_write_idx = 0  # next frame index to flush sequentially

    # Cache last detections for skipped frames
    last_boxes  = np.empty((0, 4))
    last_scores = np.empty((0,))
    last_labels = np.empty((0,), dtype=int)

    def flush_write_queue():
        """Write all consecutively-ready frames to disk in order."""
        nonlocal next_write_idx
        while next_write_idx in write_queue:
            annotated = write_queue.pop(next_write_idx)
            out.write(annotated)
            if SHOW_PREVIEW:
                cv2.imshow('Helmet Detection V4', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False   # signal stop
            next_write_idx += 1
        return True

    def flush_batch():
        """Run inference on buffered batch, store results in write_queue."""
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        needs_infer = (frame_idx % FRAME_SKIP == 0)

        if needs_infer:
            frames_buf.append((frame_idx, frame))
            tensors_buf.append(frame_to_tensor(frame, INFERENCE_SIZE))

            # Run inference when batch is full
            if len(tensors_buf) == BATCH_SIZE:
                if not flush_batch():
                    break
        else:
            # Skipped frame — reuse last detections immediately
            annotated = draw_detections(frame.copy(), last_boxes, last_scores,
                                        last_labels, scale_x, scale_y)
            write_queue[frame_idx] = annotated
            if not flush_write_queue():
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total} frames...")

    # Flush any remaining partial batch then drain the write queue
    flush_batch()
    flush_write_queue()

    cap.release()
    out.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print(f"\nDone! Output saved to: {OUT_PATH}")


if __name__ == '__main__':
    main()
