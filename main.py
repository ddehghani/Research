import argparse
import os
from pathlib import Path
from models import Edge, Cloud
import Utils
import json
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from conformal_utils import *
from correction_utils import *

def get_images(input_dir: str) -> list[str]:
    """Retrieve all image paths from the input directory."""
    return sorted([str(p) for p in Path(input_dir).glob("*.jpg")])  # Modify for other formats if needed

def split_images(images: list[str], calibration_ratio: float) -> tuple[list[str], list[str]]:
    """Split images into calibration and detection sets based on the specified ratio."""
    # random.shuffle(images)
    split_index = int(len(images) * calibration_ratio)
    return images[:split_index], images[split_index:]

def select_confidence_threshold(images: list[str], ground_truth: dict, output_dir: str, iou: float) -> float:
    print(f"Drawing confidence threshold vs recall plot ...")
    confs = [0.05 * i + 0.05 for i in range(19)]
    recalls = []
    for conf in tqdm(confs):
        results = Edge().detect(images, conf = conf)
        
        tp = 0
        tp_and_fn = 0
        for image, detections in zip(images,  results):
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
    

    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    sns.scatterplot(x=confs, y=recalls)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Recall")
    plt.title("Confidence Threshold vs Recall")
    plt.savefig(os.path.join(output_dir, "plots/conf_recall.png"), dpi=300, bbox_inches='tight')
    print(f"Saved plot to {os.path.join(output_dir, 'plots/conf_recall.png')}")
    plt.clf()
    return float(input("Based on the plot, choose your desired Confidence Threshold: "))

def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description="Prepare datasets for selective cloud offloading in object detection.")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--annotations_json", type=str, default="/data/dehghani/EfficientVideoQueryUsingCP/coco/annotations/instances_train2017.json")
    parser.add_argument("--calibration_ratio", type=float, default=0.05)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--qhat", type=float)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--conf", type=float)
    args = parser.parse_args()

    if not os.path.exists(args.annotations_json):
        raise FileNotFoundError(f"Annotation file {args.annotations_json} not found.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    images = get_images(args.input_dir)
    if not images:
        print("No images found!")
        return

    calibration_images, detection_images = split_images(images, args.calibration_ratio)

    with open(args.annotations_json, 'r') as f:
        gt = json.load(f)

    if not args.conf:
        args.conf = select_confidence_threshold(calibration_images, gt, args.output_dir, args.iou)
    else:
        print(f"Using preset confidence threshold of {args.conf}")

    edge_results = Utils.filter_annotations(Edge().detect(calibration_images, conf=args.conf))

    if not args.qhat:
        nonconformity_scores = compute_nonconformity_scores(calibration_images, edge_results, gt)
        if not args.alpha:
            args.alpha = select_alpha(calibration_images, edge_results, nonconformity_scores, gt, args.output_dir)
        else:
            print(f"Using preset alpha: {args.alpha}")
        args.qhat = np.quantile(nonconformity_scores, (1 - args.alpha) * (1 + 1 / len(nonconformity_scores)))
        print(f"Using qhat: {args.qhat}")
    else:
        print(f"Using preset qhat: {args.qhat}")

    total = sum(len(res) for res in edge_results)
    offload_set = [(i, j) for i, result in enumerate(edge_results)
                   for j, pred in enumerate(result)
                   if len(Utils.get_prediction_sets(pred, args.qhat)) != 1]

    print(f"Number of bboxes offloaded: {len(offload_set)}, {len(offload_set)*100/total:.2f}%")
    
    gt_annotations = [Utils.filter_annotations(Utils.get_annotations(gt, img)) for img in calibration_images]

    random_offload_set = select_random_bboxes(edge_results, 30, 2)
    gt_corrected_random = apply_gt_corrections(edge_results, random_offload_set, gt_annotations, args.iou)
    cloud_corrected_random = apply_cloud_corrections(edge_results, random_offload_set, calibration_images)
    cloud_corrected_random_with_packing = apply_cloud_corrections_with_packing(edge_results, random_offload_set, calibration_images)

    print(f"Number of random bboxes offloaded: {len(random_offload_set)}, {len(random_offload_set)*100/total:.2f}%")

    gt_corrected = apply_gt_corrections(edge_results, offload_set, gt_annotations, args.iou)
    cloud_corrected = apply_cloud_corrections(edge_results, offload_set, calibration_images)
    cloud_corrected_with_packing = apply_cloud_corrections_with_packing(edge_results, offload_set, calibration_images)
    cloud_prediction = Utils.filter_annotations(Cloud().detect(calibration_images))
    Utils.calculate_performance([edge_results, gt_corrected, cloud_corrected, cloud_corrected_with_packing, gt_corrected_random, cloud_corrected_random, cloud_corrected_random_with_packing, cloud_prediction,],
                                ["Edge", "GT Corrected", "Cloud Corrected", "Cloud Corrected with Packing", "GT Corrected (Random Sample)", "Cloud Corrected (Randome Sample)", "Cloud Corrected with Packing (Randome Sample)", "Cloud Prediction",],  gt_annotations)

if __name__ == "__main__":
    main()