import copy
import shutil
from pathlib import Path
from models import Cloud
import Utils

def apply_gt_corrections(results: list, offload_set: list[tuple[int, int]], gt_annotations: list, iou: float) -> list:
    corrected = copy.deepcopy(results)
    offload_set.sort(key=lambda x: (x[0], -x[1]))
    for image_index, instance_index in offload_set:
        pred = corrected[image_index][instance_index]
        for gt_ant in gt_annotations[image_index]:
            if Utils.iou(pred.bbox, gt_ant.bbox) > iou:
                pred.name = gt_ant.name
                break
        else:
            corrected[image_index].pop(instance_index)

    return corrected

def apply_cloud_corrections_with_packing(results: list, offload_set: list[tuple[int, int]], calibration_images: list[str]) -> list:
    """Apply cloud-based corrections by cropping and packing bounding boxes for batch detection."""
    
    ACCEPTANCE_MARGIN = 40
    CROP_PADDING = 200
    PACKING_PADDING = 35
    GRID_WIDTH, GRID_HEIGHT = 2, 2
    BATCH_SIZE = GRID_WIDTH * GRID_HEIGHT

    corrected = copy.deepcopy(results)
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    cropped_info = []

    # Sort to process consistent order of bboxes
    offload_set.sort(key=lambda x: (x[0], -x[1]))

    # Crop bounding boxes from calibration images
    for img_idx, inst_idx in offload_set:
        image_path = calibration_images[img_idx]
        image = Utils.load_image(image_path)
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
    for i in range(0, len(cropped_info), BATCH_SIZE):
        batch = cropped_info[i:i + BATCH_SIZE]
        paths = [entry[0] for entry in batch]
        annotations, packed_img = Utils.pack(paths, GRID_WIDTH, GRID_HEIGHT, PACKING_PADDING, 0)
        packed_path = temp_dir / f"cloud_input_packed_{i // BATCH_SIZE}.jpg"
        packed_img.save(packed_path)

        cloud_preds = Utils.filter_annotations(Cloud().detect([str(packed_path)]))[0]
        # Utils.annotateAndSave(packed_path, cloud_preds, str(temp_dir), f"cloud_input_packed_{i // BATCH_SIZE}_annotated.jpg")

        for path, rel_bbox, (img_idx, inst_idx), img_size in batch:
            matching_ann = next((ann for ann in annotations if ann.name == path), None)

            bbox = matching_ann.bbox
            bbox[0] += rel_bbox[0]
            bbox[1] += rel_bbox[1]
            bbox[2], bbox[3] = rel_bbox[2], rel_bbox[3]

            # Define acceptance box with margin
            accept_left = max(bbox[0] - ACCEPTANCE_MARGIN, 0)
            accept_top = max(bbox[1] - ACCEPTANCE_MARGIN, 0)
            accept_right = min(bbox[0] + bbox[2] + ACCEPTANCE_MARGIN, img_size[0])
            accept_bottom = min(bbox[1] + bbox[3] + ACCEPTANCE_MARGIN, img_size[1])
            accept_bbox = (accept_left, accept_top, accept_right - accept_left, accept_bottom - accept_top)

            # Select valid cloud predictions
            candidates = [pred for pred in cloud_preds if Utils.contains(accept_bbox, pred.bbox)]
            # Utils.annotateAndSave(packed_path, candidates, str(temp_dir), f"cloud_input_packed_{i // BATCH_SIZE}_annotated.jpg")
            if candidates:
                best_pred = max(candidates, key=lambda p: p.bbox[2] * p.bbox[3])
                corrected[img_idx][inst_idx].name = best_pred.name
            # else:
            #     corrected[image_index].pop(instance_index)
    # shutil.rmtree(temp_dir)
    return corrected

def apply_cloud_corrections(results: list, offload_set: list[tuple[int, int]], calibration_images: list[str]) -> list:
    crop_padding = 400
    acceptance_margin = 40
    corrected = copy.deepcopy(results)
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    offload_set.sort(key=lambda x: (x[0], -x[1]))
    for image_index, instance_index in offload_set:
        image_path = calibration_images[image_index]
        image = Utils.load_image(image_path)
        bbox = results[image_index][instance_index].bbox

        left = max(bbox[0] - crop_padding, 0)
        top = max(bbox[1] - crop_padding, 0)
        right = min(bbox[0] + bbox[2] + crop_padding, image.width)
        bottom = min(bbox[1] + bbox[3] + crop_padding, image.height)
        cropped = image.crop((left, top, right, bottom))
        temp_path = temp_dir / f"cloud_input_{image_index}_{instance_index}.jpg"
        cropped.save(temp_path)
        # Utils.annotateAndSave(temp_path, [results[image_index][instance_index]], str(temp_dir), f"cloud_input_{image_index}_{instance_index}_annotated.jpg")

        # accept results withing the original bbox with a slight padding for error
        left = max(bbox[0] - acceptance_margin, 0)
        top = max(bbox[1] - acceptance_margin, 0)
        right = min(bbox[0] + bbox[2] + acceptance_margin, image.width)
        bottom = min(bbox[1] + bbox[3] + acceptance_margin, image.height)
        acceptance_bbox = (left, top, right - left, bottom - top)

        cloud_result = Utils.filter_annotations(Cloud().detect([str(temp_path)]))[0]
        filtered_preds = [pred for pred in cloud_result if Utils.contains(acceptance_bbox, pred.bbox)]

        if filtered_preds:
            final_pred = max(filtered_preds, key=lambda p: p.bbox[2] * p.bbox[3])
            corrected[image_index][instance_index].name = final_pred.name
        # else:
        #     corrected[image_index].pop(instance_index)
    
    
    shutil.rmtree(temp_dir)
    return corrected

