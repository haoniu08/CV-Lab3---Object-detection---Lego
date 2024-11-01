# evaluation.py
from collections import defaultdict
import numpy as np
import torch


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection)


def evaluate_batch(predictions, targets, iou_threshold=0.5):
    """Evaluate a batch of predictions"""
    metrics = defaultdict(list)

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        true_boxes = target['boxes'].cpu().numpy()

        # Count detections above threshold
        detections_count = (pred_scores > 0.5).sum()
        metrics['detections'].append(detections_count)

        if len(pred_boxes) == 0 or len(true_boxes) == 0:
            continue

        # Calculate IoUs for each prediction-target pair
        ious = []
        for pred_box in pred_boxes:
            box_ious = [calculate_iou(pred_box, true_box) for true_box in true_boxes]
            ious.append(max(box_ious))

        metrics['ious'].extend([iou for iou in ious if iou > iou_threshold])

    return {
        'avg_iou': np.mean(metrics['ious']) if metrics['ious'] else 0,
        'detections': metrics['detections']
    }