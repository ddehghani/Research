from typing import List
import copy

def iou(boxA_orig: List[float], boxB_orig: List[float]) -> float:
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Boxes are in [x, y, width, height] format.

    Args:
        boxA: The first bounding box [x, y, w, h].
        boxB: The second bounding box [x, y, w, h].

    Returns:
        IoU score as a float between 0 and 1.
    """
    boxA = copy.deepcopy(boxA_orig)
    boxB = copy.deepcopy(boxB_orig)

    boxA[2] += boxA[0]
    boxA[3] += boxA[1]
    boxB[2] += boxB[0]
    boxB[3] += boxB[1]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / (boxAArea + boxBArea - interArea)

def contains(boxA: List[float], boxB: List[float]) -> bool:
    """
    Check if one bounding box (boxA) completely contains another (boxB).

    Args:
        boxA: The container box [x, y, w, h].
        boxB: The box to check for containment [x, y, w, h].

    Returns:
        True if boxA contains boxB, False otherwise.
    """
    return (
        boxA[0] < boxB[0] and
        boxA[1] < boxB[1] and
        boxB[0] + boxB[2] < boxA[0] + boxA[2] and
        boxB[1] + boxB[3] < boxA[1] + boxA[3]
    )