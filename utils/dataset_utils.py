from constants import COCO_PATH, VOC_PATH, COCO_ANNOTATIONS_PATH
from utils.annotation_utils import get_coco_annotations, get_voc_annotations
from pathlib import Path
import json
from models import YOLO

def load_dataset(dataset_name: str):
    if dataset_name == "coco":
        with open(COCO_ANNOTATIONS_PATH, 'r') as f:
            gt = json.load(f)
        return {
            "data_path": Path(COCO_PATH),
            "get_annotations": lambda img_id: get_coco_annotations(gt, img_id + ".jpg"),
            "edge_model": YOLO("./models/yolov5nu.pt"),
            "cloud_model": YOLO("./models/yolo11x.pt"),
        }
    elif dataset_name == "voc":
        return {
            "data_path": Path(VOC_PATH),
            "get_annotations": get_voc_annotations,
            "edge_model": YOLO("./models/yolov5nu_voc_trained_min.pt"), 
            "cloud_model": YOLO("./models/yolo11x_voc_trained.pt"),
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    

