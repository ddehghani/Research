import copy
import shutil
from pathlib import Path
from models import Model
from .image_utils import load_image
from .annotation_utils import filter_annotations
from .packing_utils import pack, gridify
from .bbox_utils import contains, iou
from constants import CROP_PADDING, ACCEPTANCE_MARGIN, PACKING_PADDING, GRID_WIDTH, GRID_HEIGHT, REMOVE_LABELS
from PIL import Image

def apply_gt_corrections(results: list, offload_set: list[tuple[int, int]], gt_annotations: list, iou_threshold: float) -> list:
    """
    Applies ground truth corrections to detection results using IoU matching.

    Parameters:
    - results: List of detection results (list of lists of instances)
    - offload_set: List of tuples (image_index, instance_index) to be corrected
    - gt_annotations: List of ground truth annotations per image
    - iou_threshold: IoU threshold for assigning a ground truth label to a prediction

    Returns:
    - A deep copy of the corrected results list
    """
    corrected = copy.deepcopy(results)
    offload_set.sort(key=lambda x: (x[0], -x[1]))
    for image_index, instance_index in offload_set:
        pred = corrected[image_index][instance_index]
        for gt_ant in gt_annotations[image_index]:
            if iou(pred.bbox, gt_ant.bbox) > iou_threshold:
                pred.name = gt_ant.name
                break
        else:
            corrected[image_index].pop(instance_index)

    return corrected

def apply_cloud_corrections_with_packing(
    results: list,
    offload_set: list[tuple[int, int]],
    calibration_images: list[str],
    cloud_model: Model
) -> list:
    """
    Applies cloud-based corrections to detection results using packed crops for batch processing.

    Parameters:
    - results: List of detection results (list of lists of instances)
    - offload_set: List of tuples (image_index, instance_index) to be corrected
    - calibration_images: List of file paths to corresponding images
    - cloud_model: The cloud model used for correction

    Returns:
    - A deep copy of the corrected results list with labels updated based on cloud predictions
    """
    batch_size = GRID_WIDTH * GRID_HEIGHT
    corrected = copy.deepcopy(results)
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    cropped_info = []

    # Sort offload_set to ensure consistent processing
    offload_set.sort(key=lambda x: (x[0], -x[1]))

    for img_idx, inst_idx in offload_set:
        image_path = calibration_images[img_idx]
        image = load_image(image_path)
        bbox = results[img_idx][inst_idx].bbox

        left = max(bbox[0] - CROP_PADDING, 0)
        top = max(bbox[1] - CROP_PADDING, 0)
        right = min(bbox[0] + bbox[2] + CROP_PADDING, image.width)
        bottom = min(bbox[1] + bbox[3] + CROP_PADDING, image.height)

        cropped_img = image.crop((left, top, right, bottom))
        crop_path = temp_dir / f"crop_{img_idx}_{inst_idx}.jpg"
        cropped_img.save(crop_path)

        relative_bbox = (
            bbox[0] - left,
            bbox[1] - top,
            bbox[2],
            bbox[3]
        )

        cropped_info.append((str(crop_path), relative_bbox, (img_idx, inst_idx), image.size))

    # Batch process cropped images
    for i in range(0, len(cropped_info), batch_size):
        batch = cropped_info[i:i + batch_size]
        paths = [entry[0] for entry in batch]
        annotations, packed_img = pack(paths, GRID_WIDTH, GRID_HEIGHT, PACKING_PADDING, 255)
        packed_path = temp_dir / f"cloud_input_packed_{i // batch_size}.jpg"
        packed_img.save(packed_path)
        
        cloud_preds = filter_annotations(cloud_model.detect([str(packed_path)]))[0]
        cloud_preds = [pred for pred in cloud_preds if pred.bbox[2] > 4 and pred.bbox[3] > 4] 

        for path, rel_bbox, (img_idx, inst_idx), img_size in batch:
            file_suffix = f"{img_idx}_{inst_idx}.jpg"
            matching_ann = next((ann for ann in annotations if ann.name.endswith(file_suffix)), None)
            if matching_ann is None:
                print(f"[WARN] Could not find matching annotation for {file_suffix}")
                continue

            bbox = matching_ann.bbox
            bbox[0] += rel_bbox[0]
            bbox[1] += rel_bbox[1]
            bbox[2], bbox[3] = rel_bbox[2], rel_bbox[3]

            # Define acceptance box
            accept_left = bbox[0] - ACCEPTANCE_MARGIN
            accept_top = bbox[1] - ACCEPTANCE_MARGIN
            accept_right = bbox[0] + bbox[2] + ACCEPTANCE_MARGIN
            accept_bottom = bbox[1] + bbox[3] + ACCEPTANCE_MARGIN
            accept_bbox = (accept_left, accept_top, accept_right - accept_left, accept_bottom - accept_top)

            candidates = [pred for pred in cloud_preds if iou(accept_bbox, pred.bbox) > 0.3]

            if candidates:
                best_pred = max(candidates, key=lambda p: p.bbox[2] * p.bbox[3])
                corrected[img_idx][inst_idx].name = best_pred.name
            elif REMOVE_LABELS:
                corrected[img_idx].pop(inst_idx)

    shutil.rmtree(temp_dir)
    return corrected


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def apply_cloud_corrections(
    results: list,
    offload_set: list[tuple[int, int]],
    calibration_images: list[str],
    cloud_model: Model
) -> list:
    """
    Applies cloud-based corrections to detection results by cropping each instance individually (no packing).

    Parameters:
    - results: List of detection results (list of lists of instances)
    - offload_set: List of tuples (image_index, instance_index) to be corrected
    - calibration_images: List of file paths to corresponding images
    - cloud_model: The cloud model used for correction

    Returns:
    - A deep copy of the corrected results list with labels updated based on cloud predictions
    """
    corrected = copy.deepcopy(results)
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Sort to ensure consistency, and prepare for safe removal
    offload_set_sorted = sorted(offload_set, key=lambda x: (x[0], -x[1]))
    removal_list = []  # Track (image_idx, inst_idx) for label removal
    
    for image_index, instance_index in offload_set_sorted:
        image_path = calibration_images[image_index]
        image = load_image(image_path)
        bbox = results[image_index][instance_index].bbox

        # Compute padded crop bounds
        left = max(bbox[0] - CROP_PADDING, 0)
        top = max(bbox[1] - CROP_PADDING, 0)
        right = min(bbox[0] + bbox[2] + CROP_PADDING, image.width)
        bottom = min(bbox[1] + bbox[3] + CROP_PADDING, image.height)
        cropped = image.crop((left, top, right, bottom))

        temp_path = temp_dir / f"cloud_input_{image_index}_{instance_index}.jpg"
        cropped.save(temp_path)

        # Compute bbox position relative to crop
        rel_x = bbox[0] - left
        rel_y = bbox[1] - top

        # Define acceptance box around the original bbox inside the crop
        accept_left = rel_x - ACCEPTANCE_MARGIN
        accept_top = rel_y - ACCEPTANCE_MARGIN
        accept_right = rel_x + bbox[2] + ACCEPTANCE_MARGIN
        accept_bottom = rel_y + bbox[3] + ACCEPTANCE_MARGIN
        acceptance_bbox = (
            max(accept_left, 0),
            max(accept_top, 0),
            accept_right - max(accept_left, 0),
            accept_bottom - max(accept_top, 0)
        )

        # Cloud prediction
        cloud_result = filter_annotations(cloud_model.detect([str(temp_path)]))[0]
        filtered_preds = [pred for pred in cloud_result if contains(acceptance_bbox, pred.bbox)]

        if filtered_preds:
            final_pred = max(filtered_preds, key=lambda p: p.bbox[2] * p.bbox[3])
            corrected[image_index][instance_index].name = final_pred.name
        elif REMOVE_LABELS:
            removal_list.append((image_index, instance_index))

    # Remove labels after processing to avoid indexing issues
    for image_index, instance_index in removal_list:
        if instance_index < len(corrected[image_index]):
            corrected[image_index].pop(instance_index)

    shutil.rmtree(temp_dir)
    return corrected