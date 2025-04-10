from pathlib import Path
import json
from models import YOLO
from typing import Dict
from .annotation_utils import get_coco_annotations, get_voc_annotations
from .download_util import download_coco, download_voc, download_open_images
from models import Instance

MAIN_DIR = Path(__file__).resolve().parent.parent

def load_dataset(dataset_name: str, dataset_dir: str) -> Dict:
    def get_images(input_dir: str) -> list[str]:
        """Retrieve all image paths from the input directory."""
        return sorted([str(p) for p in Path(input_dir).glob("*.jpg")])  # Modify for other formats if needed
    # download the dataset if it doesn't exist
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "coco":
        download_coco(dataset_dir / "coco")
        annotations = dataset_dir / "coco/annotations/instances_train2017.json"
        data_dir = dataset_dir / "coco/images/train2017"

        with open(annotations, 'r') as f:
            gt = json.load(f)
        return {
            "images": get_images(data_dir),
            "get_annotations": lambda img_id: get_coco_annotations(gt, img_id + ".jpg"),
            "edge_model": YOLO(MAIN_DIR / "models/yolov5nu.pt"),
            "cloud_model": YOLO(MAIN_DIR / "models/yolo11x.pt"),
        }
    elif dataset_name == "voc":
        download_voc(dataset_dir / "voc")
        return {
            "images": get_images(dataset_dir / "voc/VOCdevkit/VOC2012/JPEGImages/"),
            "get_annotations": lambda img_id: get_voc_annotations(img_id, dataset_dir / "voc/VOCdevkit/VOC2012/Annotations/"),
            "edge_model": YOLO(MAIN_DIR / "models/yolov5nu_voc_trained_min.pt"), 
            "cloud_model": YOLO(MAIN_DIR / "models/yolo11x_voc_trained.pt"),
        }
    elif dataset_name == "open-images":
        import os
        from PIL import Image

        dataset = download_open_images() # cant set dir
        gt = {}
        for sample in dataset:
            image_id = os.path.splitext(os.path.basename(sample.filepath))[0]
            img = Image.open(sample.filepath)
            w, h = img.width, img.height
            gt[image_id] = [Instance(detection.label, 1.0, [
                detection.bounding_box[0] * w,
                detection.bounding_box[1] * h,
                detection.bounding_box[2] * w,
                detection.bounding_box[3] * h,
            ]) for detection in sample.ground_truth.detections]
        
        return {
            "images": [sample.filepath for sample in dataset],
            "get_annotations": lambda img_id: gt[img_id],
            "edge_model": YOLO(MAIN_DIR / "models/yolov5nu_open_images_trained.pt"),
            "cloud_model": YOLO(MAIN_DIR / "models/yolo11x_open_images_trained.pt"),
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


