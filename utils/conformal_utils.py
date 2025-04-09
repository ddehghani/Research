from typing import Union, Callable
from models import Instance
from pathlib import Path
from tqdm import tqdm
from constants import COCO_LABELS, IOU_THRESHOLD
from .bbox_utils import iou
from .annotation_utils import filter_annotations

def compute_nonconformity_scores(images: list[str], predictions: list, get_annotations: Callable) -> list[float]:
    """Compute nonconformity scores for a set of predictions and ground truth annotations."""
    scores = []
    for image, detections in tqdm(zip(images, predictions), total=len(images)):
        image_id = Path(image).stem
        gt_annotations = filter_annotations(get_annotations(image_id))
        for detection in filter_annotations(detections):
            nonconformity = 1 - get_true_class_probability(detection, gt_annotations)
            scores.append(nonconformity)
    return scores

def get_true_class_probability(annotation: Instance, gt_annotations: list[Instance]) -> float:
    """
    Returns the probability assigned to the closest ground truth class based on IoU.
    """
    best_iou = -1
    class_name = ''

    for gt in gt_annotations:
        iou_val = iou(annotation.bbox, gt.bbox)
        if iou_val > best_iou:
            class_name = gt.name
            best_iou = iou_val

    if best_iou >= IOU_THRESHOLD:
        class_id = COCO_LABELS.index(class_name)
        return annotation.probs[class_id]
    
    return 0.0

def get_prediction_sets(annotations: Union[Instance, list], qhat: float) -> Union[list[str], list[list[str]]]:
    """
    Returns the prediction sets for annotations at a given confidence threshold (qhat).
    """
    if not annotations:
        return []

    if isinstance(annotations, Instance):
        indexes = 1 - annotations.probs <= qhat
        return [lbl for lbl, include in zip(COCO_LABELS, indexes) if include]

    if isinstance(annotations[0], Instance):
        result = []
        for inst in annotations:
            indexes = 1 - inst.probs <= qhat
            result.append([lbl for lbl, include in zip(COCO_LABELS, indexes) if include])
        return result

    return [get_prediction_sets(sublist, qhat) for sublist in annotations]