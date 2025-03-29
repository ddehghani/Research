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

def get_images(input_dir: str) -> list[str]:
    """Retrieve all image paths from the input directory."""
    return sorted([str(p) for p in Path(input_dir).glob("*.jpg")])  # Modify for other formats if needed

def split_images(images: list[str], calibration_ratio: float) -> tuple[list[str], list[str]]:
    """Split images into calibration and detection sets based on the specified ratio."""
    random.shuffle(images)
    split_index = int(len(images) * calibration_ratio)
    return images[:split_index], images[split_index:]
     
def compute_nonconformity_scores(images: list[str], predictions: list, ground_truth: dict) -> list[float]:
    """Compute nonconformity scores for a set of predictions and ground truth annotations."""
    scores = []
    for image, detections in tqdm(zip(images, predictions), total=len(images)):
        gt_annotations = Utils.filter_annotations(Utils.get_annotations(ground_truth, image))
        for detection in Utils.filter_annotations(detections):
            nonconformity = 1 - Utils.get_true_class_probability(detection, gt_annotations)
            scores.append(nonconformity)
    return scores

def plot_conformal_metrics(alphas: list[float], metrics_dict: dict[str, list[float]], output_dir: str = ".") -> None:
    """
    Plots conformal metrics against alpha.

    Parameters:
    - alphas: list of alpha values
    - metrics_dict: dictionary where keys are metric names (e.g., "Qhat", "Cost") and values are lists of corresponding values
    - output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True) 

    for metric_name, values in metrics_dict.items():
        sns.scatterplot(x=alphas, y=values)
        plt.xlabel("Alpha")
        plt.ylabel(metric_name)
        plt.title(f"Alpha vs {metric_name} for Image Classification")
        plt.savefig(os.path.join(output_dir, f"alpha_{metric_name.lower()}.png"), dpi=300, bbox_inches='tight')
        print(f"Saved plot to {os.path.join(output_dir, f'alpha_{metric_name.lower()}.png')}")
        plt.clf()

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

def select_alpha(images: list[str], predictions: list, nonconformity_scores: list, ground_truth: dict, output_dir: str) -> tuple[float, float]:
    alphas = [i * 0.01 for i in range(1, 100)]
    qhats = [np.quantile(nonconformity_scores, (1 - alpha) * (1 + 1 / len(nonconformity_scores))) for alpha in alphas]
    offload_sets = [[] for _ in qhats]

    for image_index, result in enumerate(predictions):
        for instance_index, prediction in enumerate(result):
            for index, qhat in enumerate(qhats):
                pred_set = Utils.get_prediction_sets(prediction, qhat)
                if len(pred_set) != 1:
                    offload_sets[index].append((image_index, instance_index))

    costs = [len(s) for s in offload_sets]

    plot_conformal_metrics(alphas=alphas,
                           metrics_dict={"Qhat": qhats, "Cost": costs},
                           output_dir=os.path.join(output_dir, 'plots'))

    selected_alpha = float(input("Based on the plot, choose your desired Alpha: "))
    return selected_alpha


def apply_gt_corrections(results: list, offload_set: list[tuple[int, int]], gt_annotations: list, iou: float) -> list:
    corrected = copy.deepcopy(results)
    for image_index, instance_index in offload_set:
        pred = corrected[image_index][instance_index]
        for gt_ant in gt_annotations[image_index]:
            if Utils.iou(pred.bbox, gt_ant.bbox) > iou:
                pred.name = gt_ant.name
                break
        else:
            pass
            # corrected[image_index].pop(instance_index)
    return corrected


def apply_cloud_corrections(results: list, offload_set: list[tuple[int, int]], calibration_images: list[str]) -> list:
    corrected = copy.deepcopy(results)
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    for image_index, instance_index in offload_set:
        image_path = calibration_images[image_index]
        image = Utils.load_image(image_path)
        bbox = results[image_index][instance_index].bbox

        padding = 25
        left = max(bbox[0] - padding, 0)
        top = max(bbox[1] - padding, 0)
        right = min(bbox[0] + bbox[2] + padding, image.width)
        bottom = min(bbox[1] + bbox[3] + padding, image.height)
        cropped = image.crop((left, top, right, bottom))
        temp_path = temp_dir / f"cloud_input_{image_index}_{instance_index}.jpg"
        cropped.save(temp_path)

        cloud_result = Cloud().detect([str(temp_path)])[0]
        filtered_preds = [pred for pred in cloud_result if Utils.contains(bbox, pred.bbox)]

        if filtered_preds:
            final_pred = max(filtered_preds, key=lambda p: p.bbox[2] * p.bbox[3])
            corrected[image_index][instance_index].name = final_pred.name
        else:
            pass
            # corrected[image_index].pop(instance_index)
    
    shutil.rmtree(temp_dir)
    return corrected


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

    edge_results = Edge().detect(calibration_images, conf=args.conf)

    if not args.qhat:
        nonconformity_scores = compute_nonconformity_scores(calibration_images, edge_results, gt)
        if not args.alpha:
            args.alpha = select_alpha(calibration_images, edge_results, nonconformity_scores, gt, args.output_dir)
        else:
            print(f"Using preset alpha: {args.alpha}")
        args.qhat = np.quantile(nonconformity_scores, (1 - args.alpha) * (1 + 1 / len(nonconformity_scores)))
    else:
        print(f"Using preset qhat: {args.qhat}")

    print(f"Using Q_hat: {args.qhat}")

    total = sum(len(res) for res in edge_results)
    offload_set = [(i, j) for i, result in enumerate(edge_results)
                   for j, pred in enumerate(result)
                   if len(Utils.get_prediction_sets(pred, args.qhat)) != 1]

    print(f"Number of bboxes offloaded: {len(offload_set)}, {len(offload_set)*100/total:.2f}%")
    
    gt_annotations = [Utils.filter_annotations(Utils.get_annotations(gt, img)) for img in calibration_images]

    gt_corrected = apply_gt_corrections(edge_results, offload_set, gt_annotations, args.iou)
    cloud_corrected = apply_cloud_corrections(edge_results, offload_set, calibration_images)

    Utils.calculate_performance([edge_results, gt_corrected, cloud_corrected],
                                ["Edge", "GT Corrected", "Cloud Corrected"])

if __name__ == "__main__":
    main()