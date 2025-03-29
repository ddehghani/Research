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
    random.shuffle(images)
    split_index = int(len(images) * calibration_ratio)
    return images[:split_index], images[split_index:]

def compute_conf_recall(model, ground_truth, image_set, iou=0.5):
    """Calculated confidence threshold based on the desired recall value"""
    confs = [0.05 * i + 0.05 for i in range(19)]
    recalls = []
    for conf in tqdm(confs):
        results = model.detect(image_set, conf = conf)
        
        tp = 0
        tp_and_fn = 0
        for image, detections in zip(image_set,  results):
            gt_annotations = Utils.filter_annotations(Utils.get_annotations(ground_truth, image))
            for gt_ant in gt_annotations:
                size = gt_ant.bbox[2] * gt_ant.bbox[3]
                tp_and_fn += 1
                for detection in detections:
                    if Utils.iou(detection.bbox, gt_ant.bbox) > iou: # only the bbox and not the label
                        tp += 1
                        break
        recall = tp / tp_and_fn
        recalls.append(recall)
    return confs, recalls
     
def compute_nonconformity_scores(images, predictions, ground_truth):
    """Compute nonconformity scores for a set of predictions and ground truth annotations."""
    scores = []
    for image, detections in tqdm(zip(images, predictions), total=len(images)):
        gt_annotations = Utils.filter_annotations(Utils.get_annotations(ground_truth, image))
        for detection in Utils.filter_annotations(detections):
            nonconformity = 1 - Utils.get_true_class_probability(detection, gt_annotations)
            scores.append(nonconformity)
    return scores

def plot_conformal_metrics(alphas, metrics_dict, output_dir="."):
    """
    Plots conformal metrics against alpha.

    Parameters:
    - alphas: list of alpha values
    - metrics_dict: dictionary where keys are metric names (e.g., "Qhat", "Cost") and values are lists of corresponding values
    - output_dir: directory to save plots
    - prefix: prefix for saved filenames
    """
    os.makedirs(output_dir, exist_ok=True) 

    for metric_name, values in metrics_dict.items():
        sns.scatterplot(x=alphas, y=values)
        plt.xlabel("Alpha")
        plt.ylabel(metric_name)
        plt.title(f"Alpha vs {metric_name} for Image Classification")
        plt.savefig(os.path.join(output_dir, f"alpha_{metric_name.lower()}.png"), dpi=300, bbox_inches='tight')
        plt.clf()
    

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for selective cloud offloading in object detection.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save split datasets.")
    parser.add_argument("--annotations_json", type=str, default="/data/dehghani/EfficientVideoQueryUsingCP/coco/annotations/instances_train2017.json", help="Path to ground truth annotations file.")
    parser.add_argument("--calibration_ratio", type=float, default=0.05, help="Ratio of images for calibration (default: 0.05).")
    parser.add_argument("--alpha", type=float, help="Alpha value for the conformal prediction (object classification).")
    parser.add_argument("--qhat", type=float, help="Preset qhat for object classification (if set, alpha argument is ignored and the conformal prediction is skipped).")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold to match predicted and ground truth boxes (default: 0.5).")
    parser.add_argument("--conf", type=float, help="Confidence threshold used on the edge for initial oject localization/detection")
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
    else:
        print(f"Drawing confidence threshold vs recall plot ...")
        confs, recalls = compute_conf_recall(Edge(), gt, calibration_images, iou=args.iou)
        sns.scatterplot(x=confs, y=recalls)
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Recall")
        plt.title(f"Confidence Threshold vs Recall")
        plt.savefig(os.path.join(args.output_dir, f"plots/conf_recall.png"), dpi=300, bbox_inches='tight')
        plt.clf()
        args.conf = float(input("Based on the plot, choose your desired Confidence Threshold: "))
    
    
    print(f"Performing initial detection using edge model with confidence threshold: {args.conf}")
    edge_low_conf_results = Edge().detect(calibration_images, conf = args.conf)

   
    if args.qhat:
        print(f"Using preset qhat: {args.qhat}")
    else:
        # conformal prediction
        print("Performing conformal prediction ...")
        nonconformity_scores = compute_nonconformity_scores(calibration_images, edge_low_conf_results, gt)
        if args.alpha:
            print(f"Using preset alpha: {args.alpha}")
        else:
            alphas = [i * 0.01 for i in range(1, 100)]
            qhats = [np.quantile(nonconformity_scores, (1 - alpha) * (1 + 1/len(nonconformity_scores))) for alpha in alphas]
            offload_sets = [[] for _ in qhats]
            count = 0

            for image_index, result in enumerate(edge_low_conf_results):
                count += len(result)
                for instance_index, prediction in enumerate(result):
                    for index, qhat in enumerate(qhats):
                        pred_set = Utils.get_prediction_sets(prediction, qhat)
                        if len(pred_set) != 1:
                            offload_sets[index].append((image_index, instance_index))
            
            costs = [len(s) for s in offload_sets]

            plot_conformal_metrics( alphas=alphas, 
                metrics_dict={"Qhat": qhats, "Cost": costs,},  # will add accuracy later
                output_dir=os.path.join(args.output_dir, 'plots'),
            )

            args.alpha = float(input("Based on the plot, choose your desired Alpha: "))
        
        # args.alpha is either preset or chosen
        args.qhat = np.quantile(nonconformity_scores, (1 - args.alpha) * (1 + 1/len(nonconformity_scores)))
        print(f"Q_hat: {args.qhat}")
    
    # args.qhat is either preset or chosen
    # for different values of alpha, show accuracy gained
    
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


    # make a deep copy of predictions
    edge_low_conf_results_cloud_corrected = copy.deepcopy(edge_low_conf_results)
    # for each offloaded bbox, send to cloud, if classified as object, change label, else remove it (or use local label)

    for image_index, instance_index in offload_set:
        image_path = calibration_images[image_index]
        image = Utils.load_image(image_path)  # Load full image as PIL
        bbox = edge_low_conf_results[image_index][instance_index].bbox

        # Crop and pad the image
        padding_width = 25
        left = max(bbox[0] - padding_width, 0)
        top = max(bbox[1] - padding_width, 0)
        right = min(bbox[0] + bbox[2] + padding_width, image.width)
        bottom = min(bbox[1] + bbox[3] + padding_width, image.height)
        cropped_image = image.crop((left, top, right, bottom))

        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        # Save image to temp file
        temp_path = temp_dir / f"cloud_input_{image_index}_{instance_index}.jpg"
        cropped_image.save(temp_path)

        # Run cloud model
        cloud_result = Cloud().detect([str(temp_path)])[0]
        print(len(cloud_result))

        # Filter predictions within original bbox
        original_bbox = edge_low_conf_results[image_index][instance_index].bbox
        filtered_preds = [pred for pred in cloud_result if Utils.contains(original_bbox, pred.bbox)]

        # If there are overlaps, keep only the largest one (by area)
        def bbox_area(bbox):
            return bbox[2] * bbox[3]
        
        if filtered_preds:
            final_pred = filtered_preds[0]
            for pred in filtered_preds[1:]:
                if bbox_area(pred.bbox) > bbox_area(final_pred.bbox):
                    final_pred = pred  # Replace with larger one

            edge_low_conf_results_cloud_corrected[image_index][instance_index].name = final_pred.name
        else:
            # If no cloud detection, remove the prediction
            edge_low_conf_results_cloud_corrected[image_index].pop(instance_index)

    
    Utils.calculate_performance([edge_low_conf_results, edge_low_conf_results_gt_corrected, edge_low_conf_results_cloud_corrected], 
                                ["Edge", "GT Corrected", "Cloud Corrected"], gt_annot_calibration_images)


if __name__ == "__main__":
    main()


"""

things to do:
2. compare filtering the result and detecting same images again using new conf thresh, if same same, use filteration method, otherwise use new detection
3. extract bboxes perform confromal prediction on object classification on these to using second alpha user provided
4. detect and show results
5. run the experiments (maybe a seperate command for this)

"""