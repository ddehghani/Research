import fiftyone as fo
import fiftyone.zoo as foz
import json
from fiftyone.core.labels import Detections
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import OPEN_IMAGES_LABELS_MAP


def main():
    supers = sorted({v for v in OPEN_IMAGES_LABELS_MAP.values() if v is not None})
    leaf_classes = [k for k, v in OPEN_IMAGES_LABELS_MAP.items() if v is not None]
    
    for split in ("train", "validation"):
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=leaf_classes,
            only_matching=True,
            max_samples=300_000 if split == "train" else 30_000,
        )
    
        for s in ds:
            for det in s.ground_truth.detections:
                det.label = OPEN_IMAGES_LABELS_MAP[det.label]
            s.save()
    
        ds.export(
            export_dir="/home/ubuntu/openImages/Research/old_files/datasets/modified-oiv7-map3",
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=supers,
            split="train" if split == "train" else "val",
        )

# def main():
#     unique = set(map.values())
#     unique.discard(None)
#     new_classes = sorted(list(unique))
#     mapped_classes = [k for k, v in map.items() if v is not None]
#     for split in ["train", "validation"]:
#         # Load the Open Images V7 dataset
#         dataset = foz.load_zoo_dataset(
#             "open-images-v7",
#             split=split,
#             label_types=["detections"],
#             classes = mapped_classes,
#             max_samples=1_000_000 if split == "train" else 50_000,  # Adjust this number based on your needs
#         )
        
#         for sample in dataset:
#             filtered_detections = []
#             detections = sample["ground_truth"].detections
#             for detection in detections:
#                 original_label = detection.label
#                 converted_label = map[original_label]
#                 if converted_label is not None:
#                     detection.label = converted_label
#                     filtered_detections.append(detection)
        
#             sample["ground_truth"] = Detections(detections=filtered_detections)
#             sample.save()
        
        
#         # Export the dataset to YOLO format
#         dataset.export(
#             export_dir="/home/ubuntu/openImages/Research/old_files/datasets/modified-oiv7-map1",
#             dataset_type=fo.types.YOLOv5Dataset,
#             label_field="ground_truth",
#             classes=new_classes,
#             split="train"  if split == "train" else "val"
#         )

if __name__ == "__main__":
    main()

