import fiftyone as fo
import fiftyone.zoo as foz
import json

def convert_label(label_name, hierarchy, super_candidate=None):

    # label name is entity
    if label_name == "Entity":
        return "Entity"
    
    # make a list of super categories
    supercategory_nodes = [category for category in hierarchy['Subcategory']]
    for node in supercategory_nodes:
        found = find_supercategory(label_name, node)
        if found:
            return found
    return None
   

def find_supercategory(label_name, node, current_super=None):
    # Base case: if this node is the label we're searching for
    if node.get("LabelName") == label_name:
        return current_super or label_name

    if "Subcategory" in node:
        for sub in node["Subcategory"]:
            found = find_supercategory(label_name, sub, current_super or node["LabelName"])
            if found:
                return found
    return None

def main():
    with open('/home/ubuntu/fiftyone/open-images-v7/train/metadata/hierarchy.json', 'r') as f:
        hierarchy = json.load(f)
    classes = []
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
                supercategory = convert_label(original_label, hierarchy)
                detection.label = supercategory
        
                if split == "train" and supercategory not in classes:
                    classes.append(supercategory)
        
            # Save modified detections back into sample and save
            sample["ground_truth"].detections = detections
            sample.save()
    
       
        # Export the dataset to YOLO format
        dataset.export(
            export_dir="/home/ubuntu/openImages/Research/old_files/datasets/modified-oiv7",
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=classes,
            split="train"  if split == "train" else "val"
        )

if __name__ == "__main__":
    main()

