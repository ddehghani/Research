import argparse
import os
from pathlib import Path
from models import Edge, Cloud
import json
import numpy as np
from tqdm import tqdm
import random
from constants import IOU_THRESHOLD
from utils.bbox_utils import iou
from utils.image_utils import annotateAndSave
from utils.annotation_utils import filter_annotations, get_annotations
from utils.plotting_utils import calculate_performance, plot_scatterplot
from utils.conformal_utils import get_prediction_sets, compute_nonconformity_scores
from utils.correction_utils import apply_gt_corrections, apply_cloud_corrections, apply_cloud_corrections_with_packing


def select_alpha(predictions: list, nonconformity_scores: list, output_dir: str) -> tuple[float, float]:
    alphas = [i * 0.01 for i in range(1, 100)]
    qhats = [np.quantile(nonconformity_scores, (1 - alpha) * (1 + 1 / len(nonconformity_scores))) for alpha in alphas]
    offload_sets = [[] for _ in qhats]

    for image_index, result in enumerate(predictions):
        for instance_index, prediction in enumerate(result):
            for index, qhat in enumerate(qhats):
                pred_set = get_prediction_sets(prediction, qhat)
                if len(pred_set) > 1:
                    offload_sets[index].append((image_index, instance_index))


    costs = [len(s) for s in offload_sets]

    plot_scatterplot(   x_values_dict = {"Alpha" : alphas},
                        y_values_dict={"Qhat": qhats, "Cost": costs},
                        output_dir=os.path.join(output_dir, 'plots'))

    selected_alpha = float(input("Based on the plot, choose your desired Alpha: "))
    return selected_alpha

def select_random_bboxes(edge_results, num_outer, num_inner):
    selected_tuples = []

    outer_indices = random.sample(range(len(edge_results)), min(num_outer, len(edge_results)))

    for outer_idx in outer_indices:
        sublist = edge_results[outer_idx]
        if not sublist:
            continue  # skip empty sublists

        num_inner_to_select = min(num_inner, len(sublist))
        inner_indices = random.sample(range(len(sublist)), num_inner_to_select)
        
        for inner_idx in inner_indices:
            selected_tuples.append((outer_idx, inner_idx))

    return selected_tuples

def get_images(input_dir: str) -> list[str]:
    """Retrieve all image paths from the input directory."""
    return sorted([str(p) for p in Path(input_dir).glob("*.jpg")])  # Modify for other formats if needed

def split_images(images: list[str], calibration_ratio: float) -> tuple[list[str], list[str]]:
    """Split images into calibration and detection sets based on the specified ratio."""
    # random.shuffle(images)
    split_index = int(len(images) * calibration_ratio)
    return images[:split_index], images[split_index:]

def select_confidence_threshold(images: list[str], ground_truth: dict, output_dir: str, iou_threshold: float) -> float:
    print(f"Drawing confidence threshold vs recall plot ...")
    confs = [0.05 * i + 0.05 for i in range(19)]
    recalls = []
    for conf in tqdm(confs):
        results = Edge().detect(images, conf = conf)
        
        tp = 0
        tp_and_fn = 0
        for image, detections in zip(images,  results):
            gt_annotations = filter_annotations(get_annotations(ground_truth, image))
            for gt_ant in gt_annotations:
                tp_and_fn += 1
                for detection in detections:
                    if iou(detection.bbox, gt_ant.bbox) > iou_threshold: # only the bbox and not the label
                        tp += 1
                        break
        recall = tp / tp_and_fn
        recalls.append(recall)
    
    plot_scatterplot(   x_values_dict = {"Confidence Threshold" : confs},
                        y_values_dict={"Recall": recalls},
                        output_dir=os.path.join(output_dir, 'plots'))

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
        args.conf = select_confidence_threshold(calibration_images, gt, args.output_dir, IOU_THRESHOLD)
    else:
        print(f"Using preset confidence threshold of {args.conf}")

    edge_results = filter_annotations(Edge().detect(calibration_images, conf=args.conf))

    if not args.qhat:
        nonconformity_scores = compute_nonconformity_scores(calibration_images, edge_results, gt)
        
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(nonconformity_scores, bins=10)
        plt.title("Histogram of Nonconformity Scores")
        plt.xlabel("Nonconformity Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "plots", "nonconformity_scores_histogram.png"))
        plt.close()

        if not args.alpha:
            args.alpha = select_alpha(edge_results, nonconformity_scores, args.output_dir)
        else:
            print(f"Using preset alpha: {args.alpha}")
        args.qhat = np.quantile(nonconformity_scores, (1 - args.alpha) * (1 + 1 / len(nonconformity_scores)))
        print(f"Using qhat: {args.qhat}")
    else:
        print(f"Using preset qhat: {args.qhat}")

    total = sum(len(res) for res in edge_results)
    offload_set = [(i, j) for i, result in enumerate(edge_results)
                   for j, pred in enumerate(result)
                   if len(get_prediction_sets(pred, args.qhat)) != 1]

    print(f"Number of bboxes offloaded: {len(offload_set)}, {len(offload_set)*100/total:.2f}%")
    
    gt_annotations = [filter_annotations(get_annotations(gt, img)) for img in calibration_images]

    random_offload_set = select_random_bboxes(edge_results, 30, 2)
    gt_corrected_random = apply_gt_corrections(edge_results, random_offload_set, gt_annotations, IOU_THRESHOLD)
    cloud_corrected_random = apply_cloud_corrections(edge_results, random_offload_set, calibration_images)
    cloud_corrected_random_with_packing = apply_cloud_corrections_with_packing(edge_results, random_offload_set, calibration_images)

    print(f"Number of random bboxes offloaded: {len(random_offload_set)}, {len(random_offload_set)*100/total:.2f}%")

    gt_corrected = apply_gt_corrections(edge_results, offload_set, gt_annotations, IOU_THRESHOLD)
    cloud_corrected = apply_cloud_corrections(edge_results, offload_set, calibration_images)
    cloud_corrected_with_packing = apply_cloud_corrections_with_packing(edge_results, offload_set, calibration_images)
    cloud_prediction = filter_annotations(Cloud().detect(calibration_images))
    calculate_performance([edge_results, gt_corrected, cloud_corrected, cloud_corrected_with_packing, gt_corrected_random, cloud_corrected_random, cloud_corrected_random_with_packing, cloud_prediction,],
                                ["Edge", "GT Corrected", "Cloud Corrected", "Cloud Corrected with Packing", "GT Corrected (Random Sample)", "Cloud Corrected (Randome Sample)", "Cloud Corrected with Packing (Randome Sample)", "Cloud Prediction",],  gt_annotations)

if __name__ == "__main__":
    main()