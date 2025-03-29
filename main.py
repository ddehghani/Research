import argparse
import copy
import os
import shutil
from pathlib import Path
import random
from models import Edge, Cloud
import Utils
import json
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def get_images(input_dir):
    """Retrieve all image paths from the input directory."""
    return sorted([str(p) for p in Path(input_dir).glob("*.jpg")])  # Modify for other formats if needed

def split_images(images, calibration_ratio):
    """Split images into calibration and detection sets based on the specified ratio."""
    # random.shuffle(images)
    split_index = int(len(images) * calibration_ratio)
    return images[:split_index], images[split_index:]

def calculate_conf(model, ground_truth, image_set, desired_recall, starting_value = 0.25, adjustment_rate = 0.3, iou=0.5):
    """Calculated confidence threshold based on the desired recall value"""
    conf = starting_value
    if (conf < 0.05):
        print("The desired recall can't be achieved")
        return conf
    results = model.detect(image_set, conf = conf)

    tp = 0
    tp_and_fn = 0
    for image, detections in tqdm(zip(image_set,  results), total=len(image_set)):
        gt_annotations = Utils.filter_annotations(Utils.get_annotations(ground_truth, image))
        for gt_ant in gt_annotations:
            size = gt_ant.bbox[2] * gt_ant.bbox[3]
            tp_and_fn += 1
            for detection in detections:
                if Utils.iou(detection.bbox, gt_ant.bbox) > iou: # only the bbox and not the label
                    tp += 1
                    break
    current_recall = tp / tp_and_fn

    if current_recall > desired_recall:
        return conf
    else:
        
        conf -= max(adjustment_rate * (desired_recall - current_recall), 0.05)
        print(f"Current recall: {current_recall}")
        print(f"Lowering confidence threshold to: {conf}")
        return calculate_conf(model, ground_truth, image_set, desired_recall, conf, adjustment_rate, iou)



def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for selective cloud offloading in object detection.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save split datasets.")
    parser.add_argument("--annotations_json", type=str, default="/data/dehghani/EfficientVideoQueryUsingCP/coco/annotations/instances_train2017.json", help="Path to ground truth annotations file.")
    parser.add_argument("--calibration_ratio", type=float, default=0.05, help="Ratio of images for calibration (default: 0.05).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha value for the conformal prediction (object classification).")
    parser.add_argument("--qhat", type=float, help="Preset qhat for object classification (if set, alpha argument is ignored and the conformal prediction is skipped).")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold to match predicted and ground truth boxes (default: 0.5).")
    parser.add_argument("--conf", type=float, help="Confidence threshold used on the edge for initial oject localization/detection (if set recall argument is ignored)")
    parser.add_argument("--recall", type=float, default=0.75, help="Desired recall")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Get image paths and split them into calibration and detection sets
    print("Loading images...")
  
    images = get_images(args.input_dir)
    if not images:
        print("No images found in the input directory!")
        return

    calibration_images, detection_images = split_images(images, args.calibration_ratio)

    print(f"Dataset prepared for Selective Cloud Offloading:")
    print(f"  Calibration set: {len(calibration_images)} images")
    print(f"  Detection set: {len(detection_images)} images")
    
    print("Loading ground truth annotations ...")
    with open(args.annotations_json, 'r') as f:
        gt = json.load(f)
    
    if not gt:
        print("No ground truth annotations found in the annotations file!")
        return
    

    if args.conf:
        print(f"Using preset confidence threshold of {args.conf}")
        conf = args.conf
    else:
        print(f"Calculating confidence threshold based on the desired recall ...")
        conf = calculate_conf(Edge(), gt, calibration_images, args.recall, iou=args.iou)
        print(f"Confidence threshold: {conf}")
    
    
    print(f"Performing initial detection using edge model with confidence threshold: {conf}")
    edge_low_conf_results = Edge().detect(calibration_images, conf = conf)

   
    if args.qhat:
        print(f"Using preset qhat: {args.qhat}")
        qhat = args.qhat
    else:
        # conformal prediction
        print("Performing conformal prediction ...")
        # nonconformity calculation
        nonconformity_scores = []
        for image, detections in tqdm(zip(calibration_images,  edge_low_conf_results), total=len(calibration_images)):
            gt_annotations = Utils.filter_annotations(Utils.get_annotations(gt, image))
            for detection in Utils.filter_annotations(detections):
                # Nonconformity = 1 - probability assigned to true class
                nonconformity = 1 - Utils.get_true_class_probability(detection, gt_annotations)
                nonconformity_scores.append(nonconformity)

        
        qhat = np.quantile(nonconformity_scores, (1 - args.alpha) * (1 + 1/len(nonconformity_scores)))
        print(f"QHat: {qhat}")
        # alphas = [i * 0.01 for i in range(1, 100)]
        # qhats = [np.quantile(nonconformity_scores, (1 - alpha) * (1 + 1/len(nonconformity_scores))) for alpha in alphas]
        # sns.scatterplot(x=alphas, y=qhats)
        # plt.xlabel("Alpha")
        # plt.ylabel("QHat")
        # plt.title("Qhat vs Alpha for Image Classification")
        # plt.savefig("qhat_vs_alpha.png", dpi=300, bbox_inches='tight')

        # sns.histplot(nonconformity_scores, bins=15, kde=False)
        # plt.title("Frequency Histogram")
        # plt.xlabel("Non-conformity Scores")
        # plt.ylabel("Frequency")
        # plt.savefig("nonconformity_scores_hist.png", dpi=300, bbox_inches='tight')

        # qhat = float(input("Enter your desired QHat: "))
        
        # for different values of alpha, show cost, and accuracy gained
    
    # at this point qhat is calculated (maybe we want to do this with multiple qhats, but that's a different story)
    
    count = 0
    offload_set = []
    for image_index, result, in enumerate(edge_low_conf_results):
        count += len(result)
        for instance_index, prediction in enumerate(result):
            pred_set = Utils.get_prediction_sets(prediction, qhat)
            if len(pred_set) != 1: # offload bboxes with 0 (very rare) or 2 or more labels in prediction set
                offload_set.append((image_index, instance_index))
                break
        
    print(f"Number of bboxes offloaded: {len(offload_set)}, {len(offload_set)*100/count:.2f}% of all bboxes")



    # make a deep copy of predictions
    edge_low_conf_results_gt_corrected = copy.deepcopy(edge_low_conf_results)

    # for each offloaded bbox, find the matching ground truth bbox, if found, change label, else remove it
    gt_annot_calibration_images = [Utils.filter_annotations(Utils.get_annotations(gt, img)) for img in calibration_images]
    for image_index, instance_index in offload_set:
        pred = edge_low_conf_results_gt_corrected[image_index][instance_index]
        for gt_ant in gt_annot_calibration_images[image_index]:
            matches = False
            if Utils.iou(pred.bbox, gt_ant.bbox) > args.iou:
                pred.name = gt_ant.name
                matches = True
                break
        if not matches:
            edge_low_conf_results_gt_corrected[image_index].pop(instance_index)


    Utils.calculate_performance([edge_low_conf_results, edge_low_conf_results_gt_corrected], ["Edge", "GT Corrected"], gt_annot_calibration_images)



    # make a deep copy of predictions
    edge_low_conf_results_cloud_corrected = copy.deepcopy(edge_low_conf_results)
    # for each offloaded bbox, send to cloud, if classified as object, change label, else remove it (or use local label)

    





if __name__ == "__main__":
    main()


"""
args:




things to do:
2. compare filtering the result and detecting same images again using new conf thresh, if same same, use filteration method, otherwise use new detection
3. extract bboxes perform confromal prediction on object classification on these to using second alpha user provided
4. detect and show results
5. run the experiments (maybe a seperate command for this)

"""