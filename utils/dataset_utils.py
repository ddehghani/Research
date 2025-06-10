from pathlib import Path
import json
from models import YOLO
from typing import Dict
from .annotation_utils import get_coco_annotations, get_voc_annotations
from .download_util import download_coco, download_voc, download_open_images
from models import Instance
import sys
import os

MAIN_DIR = Path(__file__).resolve().parent.parent

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from constants import OPEN_IMAGES_LABELS_MAP as mapping


def load_dataset(dataset_name: str, dataset_dir: str, download: bool = False) -> Dict:
    def get_images(input_dir: str) -> list[str]:
        """Retrieve all image paths from the input directory."""
        return sorted([str(p) for p in Path(input_dir).glob("*.jpg")])  # Modify for other formats if needed
    # download the dataset if it doesn't exist
    dataset_dir_path: Path = Path(dataset_dir)
    if not dataset_dir_path.exists():
        dataset_dir_path.mkdir(parents=True, exist_ok=True)

    if dataset_name == "coco":
        if download:
            print("Downloading COCO dataset...")
            download_coco(str(dataset_dir_path / "coco"))
        annotations = dataset_dir_path / "coco/annotations/instances_train2017.json"
        data_dir = dataset_dir_path / "coco/images/train2017"

        with open(annotations, 'r') as f:
            gt = json.load(f)
        return {
            "images": get_images(str(data_dir)),
            "get_annotations": lambda img_id: get_coco_annotations(gt, img_id + ".jpg"),
            "edge_model": YOLO(str(MAIN_DIR / "models/coco_edge.pt")),
            "cloud_model": YOLO(str(MAIN_DIR / "models/coco_cloud.pt")),
        }
    elif dataset_name == "voc":
        if download:
            print("Downloading VOC dataset...")
            download_voc(str(dataset_dir_path / "voc"))
        return {
            "images": get_images(str(dataset_dir_path / "voc/VOCdevkit/VOC2012/JPEGImages/")),
            "get_annotations": lambda img_id: get_voc_annotations(img_id, str(dataset_dir_path / "voc/VOCdevkit/VOC2012/Annotations/")),
            "edge_model": YOLO(str(MAIN_DIR / "models/voc_edge.pt")),
            "cloud_model": YOLO(str(MAIN_DIR / "models/voc_cloud.pt")),
        }
    elif dataset_name == "open-images":
        import os
        from PIL import Image

        dataset = download_open_images() # cant set dir
        gt = {}
        unique = set(mapping.values())
        unique.discard(None)
        classes = sorted(list(unique))
        
        for sample in dataset:
            image_id = os.path.splitext(os.path.basename(sample.filepath))[0]
            img = Image.open(sample.filepath)
            w, h = img.width, img.height
            gt[image_id] = [Instance(mapping[detection.label], 1.0, [
                detection.bounding_box[0] * w,
                detection.bounding_box[1] * h,
                detection.bounding_box[2] * w,
                detection.bounding_box[3] * h,
            ]) for detection in sample.ground_truth.detections if mapping[detection.label] is not None]
        
        return {
            "images": [sample.filepath for sample in dataset],
            "get_annotations": lambda img_id: gt[img_id],
            "edge_model": YOLO(str(MAIN_DIR / "models/yolov5nu_map2.pt")),
            "cloud_model": YOLO(str(MAIN_DIR / "models/oiv7_cloud.pt")),
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


