import copy
import shutil
from pathlib import Path
from models import Model
from .image_utils import load_image
from .annotation_utils import filter_annotations
from .packing_utils import pack
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

def apply_cloud_corrections_with_packing(results: list, offload_set: list[tuple[int, int]], calibration_images: list[str], cloud_model: Model) -> list:
    """
    Applies cloud-based corrections to detection results using packed crops for batch processing.

    Parameters:
    - results: List of detection results (list of lists of instances)
    - offload_set: List of tuples (image_index, instance_index) to be corrected
    - calibration_images: List of file paths to corresponding images

    Returns:
    - A deep copy of the corrected results list with labels updated based on cloud predictions
    """
    batch_size = GRID_WIDTH * GRID_HEIGHT

    corrected = copy.deepcopy(results)
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    cropped_info = []

    # Sort to process consistent order of bboxes
    offload_set.sort(key=lambda x: (x[0], -x[1]))

    # Crop bounding boxes from calibration images
    for img_idx, inst_idx in offload_set:
        image_path = calibration_images[img_idx]
        image = load_image(image_path)
        bbox = results[img_idx][inst_idx].bbox

        left = max(bbox[0] - CROP_PADDING, 0)
        top = max(bbox[1] - CROP_PADDING, 0)
        right = min(bbox[0] + bbox[2] + CROP_PADDING, image.width)
        bottom = min(bbox[1] + bbox[3] + CROP_PADDING, image.height)

        cropped_img = image.crop((left, top, right, bottom))
        crop_path = temp_dir / f"cloud_input_{img_idx}_{inst_idx}.jpg"
        cropped_img.save(crop_path)

        relative_bbox = (
            bbox[0] - left,
            bbox[1] - top,
            bbox[2],
            bbox[3]
        )

        cropped_info.append((str(crop_path), relative_bbox, (img_idx, inst_idx), image.size))

    # Batch process cropped images
    for i in range(0, len(cropped_info),  batch_size):
        batch = cropped_info[i:i +  batch_size]
        paths = [entry[0] for entry in batch]
        annotations, packed_img = pack(paths, GRID_WIDTH, GRID_HEIGHT, PACKING_PADDING, 0)
        packed_path = temp_dir / f"cloud_input_packed_{i //  batch_size}.jpg"
        packed_img.save(packed_path)

        cloud_preds = filter_annotations(cloud_model.detect([str(packed_path)]))[0]
        # Utils.annotateAndSave(packed_path, cloud_preds, str(temp_dir), f"cloud_input_packed_{i // BATCH_SIZE}_annotated.jpg")

        for path, rel_bbox, (img_idx, inst_idx), img_size in batch:
            matching_ann = next((ann for ann in annotations if ann.name == path), None)

            bbox = matching_ann.bbox
            bbox[0] += rel_bbox[0]
            bbox[1] += rel_bbox[1]
            bbox[2], bbox[3] = rel_bbox[2], rel_bbox[3]

            # Define acceptance box with margin
            accept_left = bbox[0] - ACCEPTANCE_MARGIN
            accept_top = bbox[1] - ACCEPTANCE_MARGIN
            accept_right = bbox[0] + bbox[2] + ACCEPTANCE_MARGIN
            accept_bottom = bbox[1] + bbox[3] + ACCEPTANCE_MARGIN
            accept_bbox = (accept_left, accept_top, accept_right - accept_left, accept_bottom - accept_top)

            # Select valid cloud predictions
            candidates = [pred for pred in cloud_preds if contains(accept_bbox, pred.bbox)]
            # Utils.annotateAndSave(packed_path, candidates, str(temp_dir), f"cloud_input_packed_{i // BATCH_SIZE}_annotated.jpg")
            if candidates:
                best_pred = max(candidates, key=lambda p: p.bbox[2] * p.bbox[3])
                corrected[img_idx][inst_idx].name = best_pred.name
            elif REMOVE_LABELS:
                corrected[img_idx].pop(inst_idx)
    shutil.rmtree(temp_dir)
    return corrected

def apply_cloud_corrections(results: list, offload_set: list[tuple[int, int]], calibration_images: list[str], cloud_model: Model) -> list:
    """
    Applies cloud-based corrections to detection results by cropping each instance individually.

    Parameters:
    - results: List of detection results (list of lists of instances)
    - offload_set: List of tuples (image_index, instance_index) to be corrected
    - calibration_images: List of file paths to corresponding images

    Returns:
    - A deep copy of the corrected results list with labels updated based on cloud predictions
    """
    corrected = copy.deepcopy(results)
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    offload_set.sort(key=lambda x: (x[0], -x[1]))
    for image_index, instance_index in offload_set:
        image_path = calibration_images[image_index]
        image = load_image(image_path)
        bbox = results[image_index][instance_index].bbox

        left = max(bbox[0] - CROP_PADDING, 0)
        top = max(bbox[1] - CROP_PADDING, 0)
        right = min(bbox[0] + bbox[2] + CROP_PADDING, image.width)
        bottom = min(bbox[1] + bbox[3] + CROP_PADDING, image.height)
        cropped = image.crop((left, top, right, bottom))
        temp_path = temp_dir / f"cloud_input_{image_index}_{instance_index}.jpg"
        
        blank_width = 1280
        blank_height = 720
        x_offset = (blank_width - cropped.width) // 2
        y_offset = (blank_height - cropped.height) // 2

        blank_image = Image.new("RGB", (blank_width, blank_height), (0, 0, 0))
        blank_image.paste(cropped, (x_offset, y_offset))


        # cropped.save(temp_path)
        blank_image.save(temp_path)
        # Utils.annotateAndSave(temp_path, [results[image_index][instance_index]], str(temp_dir), f"cloud_input_{image_index}_{instance_index}_annotated.jpg")

        # accept results withing the original bbox with a slight padding for error
        left = max(bbox[0] - ACCEPTANCE_MARGIN, 0) + x_offset
        top = max(bbox[1] - ACCEPTANCE_MARGIN, 0) + y_offset
        right = min(bbox[0] + bbox[2] + ACCEPTANCE_MARGIN, image.width)
        bottom = min(bbox[1] + bbox[3] + ACCEPTANCE_MARGIN, image.height)
        acceptance_bbox = (left, top, right - left, bottom - top)

        cloud_result = filter_annotations(cloud_model.detect([str(temp_path)]))[0]
        filtered_preds = [pred for pred in cloud_result if contains(acceptance_bbox, pred.bbox)]

        if filtered_preds:
            final_pred = max(filtered_preds, key=lambda p: p.bbox[2] * p.bbox[3])
            corrected[image_index][instance_index].name = final_pred.name
        elif REMOVE_LABELS:
            corrected[image_index].pop(instance_index)
    
    shutil.rmtree(temp_dir)
    return corrected