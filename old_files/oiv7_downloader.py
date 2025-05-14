import fiftyone as fo
import fiftyone.zoo as foz
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import OPEN_IMAGES_LABELS_MAP

def main():
    classes = [ k for k,v in OPEN_IMAGES_LABELS_MAP.items() if v != None]
    for split in ["train", "validation"]:
        # Load the Open Images V7 dataset
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            max_samples=100000 if split == "train" else 10000,  # Adjust this number based on your needs
        )
    
        # convert the labels to get supercategories only
        for sample in dataset:
            detections = sample["ground_truth"].detections
            for detection in detections:
                original_label = detection.label
                detection.label = OPEN_IMAGES_LABELS_MAP[original_label]
        
            # Save modified detections back into sample and save
            sample["ground_truth"].detections = detections
            sample.save()
    
       
        # Export the dataset to YOLO format
        dataset.export(
            export_dir="/data/dehghani/EfficientVideoQueryUsingCP/modified-oiv7",
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=classes,
            split="train"  if split == "train" else "val"
        )

if __name__ == "__main__":
    main()

