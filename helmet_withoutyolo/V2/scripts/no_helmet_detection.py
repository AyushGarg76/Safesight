"""
no_helmet_detection.py
----------------------
Drop-in replacement for draw_detections() in run_all_v4.py.

No retraining needed. Uses the existing 3-class model output
(helmet, head, person) to derive "person without helmet" detections.

TWO STRATEGIES run together on every frame:

  Strategy 1 — Direct:
    The model's "head" class already means "bare head, no helmet".
    Every "head" detection above threshold is immediately flagged.

  Strategy 2 — Cross-check (IoU):
    For every "person" box, look at whether any "helmet" box overlaps
    with the upper 40% of that person box.
    If nothing overlaps → that person has no visible helmet.
    This catches workers whose heads are small / partially occluded.

The "probability inversion" idea you mentioned maps to Strategy 2:
  confidence(no_helmet | person) = 1.0 - max_helmet_overlap_iou
  i.e. if no helmet overlaps at all, score = 1.0 (certain no helmet)
       if a helmet fully covers the head region, score ≈ 0.0
"""

import cv2
import numpy as np

# ── Colours ───────────────────────────────────────────────────────────────────
COLOR_HELMET    = (0, 200, 0)      # green  — helmet on
COLOR_HEAD      = (0, 0, 255)      # red    — bare head (no helmet)
COLOR_PERSON    = (180, 180, 180)  # gray   — person body box
COLOR_NO_HELMET = (0, 0, 255)      # red    — no helmet alert

# Tune these to your scene
THRESHOLD          = 0.50   # minimum model score to consider any detection
HEAD_AREA_FRACTION = 0.40   # top X% of a person box = "head region"
OVERLAP_IOU_MIN    = 0.10   # IoU between helmet and person-head-region to count as "covered"


# ── IoU helper ────────────────────────────────────────────────────────────────
def iou(boxA, boxB):
    """
    Intersection-over-Union for two boxes [x1, y1, x2, y2].
    Returns a float in [0, 1].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


# ── Main function (replaces draw_detections in run_all_v4.py) ─────────────────
def draw_detections(frame, boxes, scores, labels, scale_x, scale_y,
                    class_names=None):
    """
    Replaces the original draw_detections() in run_all_v4.py.

    Args:
        frame       : BGR numpy array (full resolution)
        boxes       : numpy array of shape (N, 4) in INFERENCE_SIZE coords
        scores      : numpy array of shape (N,)
        labels      : numpy array of shape (N,) with integer class indices
        scale_x     : frame_width  / INFERENCE_SIZE[0]
        scale_y     : frame_height / INFERENCE_SIZE[1]
        class_names : list like ['__background__', 'helmet', 'head', 'person']

    Returns:
        Annotated frame.
    """
    if class_names is None:
        class_names = ['__background__', 'helmet', 'head', 'person']

    # ── 1. Scale all boxes to full-resolution frame coords ────────────────────
    scaled_boxes = []
    for box, score, label in zip(boxes, scores, labels):
        if score < THRESHOLD:
            continue
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)
        name = class_names[label]
        scaled_boxes.append((name, score, x1, y1, x2, y2))

    # Separate by class for cross-matching
    helmet_boxes = [(x1,y1,x2,y2) for (n,s,x1,y1,x2,y2) in scaled_boxes if n == 'helmet']
    head_boxes   = [(s,x1,y1,x2,y2) for (n,s,x1,y1,x2,y2) in scaled_boxes if n == 'head']
    person_boxes = [(s,x1,y1,x2,y2) for (n,s,x1,y1,x2,y2) in scaled_boxes if n == 'person']

    no_helmet_detections = []   # list of (x1,y1,x2,y2, confidence, source)

    # ── Strategy 1: "head" class = direct no-helmet signal ────────────────────
    for (score, x1, y1, x2, y2) in head_boxes:
        no_helmet_detections.append((x1, y1, x2, y2, score, 'head'))

    # ── Strategy 2: person box with no overlapping helmet ─────────────────────
    for (p_score, px1, py1, px2, py2) in person_boxes:
        ph = py2 - py1

        # Define the "head region" = top HEAD_AREA_FRACTION of the person box
        head_region = (px1, py1, px2, py1 + int(ph * HEAD_AREA_FRACTION))

        # Compute max IoU between the head region and every detected helmet
        max_overlap = 0.0
        for hbox in helmet_boxes:
            overlap = iou(head_region, hbox)
            if overlap > max_overlap:
                max_overlap = overlap

        # "no_helmet confidence" = inverted overlap, as you described
        no_helmet_conf = 1.0 - max_overlap

        if max_overlap < OVERLAP_IOU_MIN:
            # No helmet found over this person's head region
            # Use the head-region box for the alert (not the full body box)
            no_helmet_detections.append(
                (head_region[0], head_region[1],
                 head_region[2], head_region[3],
                 no_helmet_conf, 'person_check')
            )

    # ── Draw: person boxes (light, background) ────────────────────────────────
    for (p_score, px1, py1, px2, py2) in person_boxes:
        cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_PERSON, 1)

    # ── Draw: helmet boxes ────────────────────────────────────────────────────
    for (hx1, hy1, hx2, hy2) in helmet_boxes:
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), COLOR_HELMET, 2)
        cv2.putText(frame, "HELMET",
                    (hx1, max(0, hy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HELMET, 2)

    # ── Draw: no-helmet alerts ────────────────────────────────────────────────
    for (nx1, ny1, nx2, ny2, conf, source) in no_helmet_detections:
        cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), COLOR_NO_HELMET, 3)
        label_text = f"NO HELMET {conf:.0%}"
        cv2.putText(frame, label_text,
                    (nx1, max(0, ny1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_NO_HELMET, 2)

    # ── Optional: frame-level alert banner ────────────────────────────────────
    if no_helmet_detections:
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (fw, 48), (0, 0, 140), -1)
        cv2.putText(frame,
                    f"ALERT: {len(no_helmet_detections)} person(s) without helmet",
                    (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return frame