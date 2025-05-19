import argparse
import os
from pathlib import Path
from models import Model
import numpy as np
from tqdm import tqdm
import random
from constants import IOU_THRESHOLD
# from utils.bbox_utils import iou
from utils import (
    load_dataset, iou, annotateAndSave, filter_annotations, 
    calculate_performance, plot_scatterplot, 
    get_prediction_sets, compute_nonconformity_scores,
    apply_gt_corrections, apply_cloud_corrections, apply_cloud_corrections_with_packing)


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

def split_images(images: list[str], calibration_ratio: float) -> tuple[list[str], list[str]]:
    """Split images into calibration and detection sets based on the specified ratio."""
    random.shuffle(images)
    split_index = int(len(images) * calibration_ratio)
    return images[:split_index], images[split_index:]

def select_confidence_threshold(images: list[str], get_annotations, output_dir: str, iou_threshold: float, edge_model: Model) -> float:
    print(f"Drawing confidence threshold vs recall plot ...")
    confs = [0.05 * i + 0.05 for i in range(19)]
    recalls = []
    for conf in tqdm(confs):
        results = edge_model.detect(images, conf = conf)
        
        tp = 0
        tp_and_fn = 0
        for image, detections in zip(images,  results):
            image_id = Path(image).stem
            gt_annotations = filter_annotations(get_annotations(image_id))
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
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--dataset", type=str,  default="coco", choices=["coco", "voc", "open-images"], help="Dataset to use: coco or voc, or open-images")
    parser.add_argument("--calibration_ratio", type=float, default=0.05)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--qhat", type=float)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--datasets_dir", type=str, default="./datasets", help="Where to store datasets")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.datasets_dir)
    get_annotations = dataset["get_annotations"]
    images = dataset["images"]
    edge_model = dataset["edge_model"]
    cloud_model = dataset["cloud_model"]

    if not images:
        print("No images found!")
        return
    
    calibration_images, detection_images = split_images(images, args.calibration_ratio)
    print(f"Calibration images: {len(calibration_images)}, Detection images: {len(detection_images)}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    # for testing purposes only 
    # for i in range(10):
    #     predictions = edge_model.detect(calibration_images[i])[0]
    #     annotateAndSave(calibration_images[i], predictions, args.output_dir, f'model{i}.jpg')
    # 
    #     image_id = os.path.splitext(os.path.basename(calibration_images[i]))[0]
    #     gt_predictions = get_annotations(image_id)
    #     annotateAndSave(calibration_images[i], gt_predictions, args.output_dir, f'gt{i}.jpg')
    #     print(i)
    #     for pred in predictions:
    #         print(pred.name)
    #         
    #     print(gt_predictions)
    # return

    if not args.conf:
        args.conf = select_confidence_threshold(calibration_images, get_annotations, args.output_dir, IOU_THRESHOLD, edge_model)
    else:
        print(f"Using preset confidence threshold of {args.conf}")

    edge_results = filter_annotations(edge_model.detect(calibration_images, conf=args.conf))

    if not args.qhat:
        nonconformity_scores = compute_nonconformity_scores(calibration_images, edge_results, get_annotations)
        
        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.hist(nonconformity_scores, bins=10)
        # plt.title("Histogram of Nonconformity Scores")
        # plt.xlabel("Nonconformity Score")
        # plt.ylabel("Frequency")
        # plt.grid(True)
        # plt.savefig(os.path.join(args.output_dir, "plots", "nonconformity_scores_histogram.png"))
        # plt.close()

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
    
    gt_annotations = [filter_annotations(get_annotations(Path(img).stem)) for img in calibration_images]
    random_offload_set = select_random_bboxes(edge_results, len(offload_set), 1)

    
    gt_corrected_random = apply_gt_corrections(edge_results, random_offload_set, gt_annotations, IOU_THRESHOLD)
    cloud_corrected_random = apply_cloud_corrections(edge_results, random_offload_set, calibration_images, cloud_model)
    cloud_corrected_random_with_packing = apply_cloud_corrections_with_packing(edge_results, random_offload_set, calibration_images, cloud_model)

    print(f"Number of random bboxes offloaded: {len(random_offload_set)}, {len(random_offload_set)*100/total:.2f}%")

    gt_corrected = apply_gt_corrections(edge_results, offload_set, gt_annotations, IOU_THRESHOLD)
    cloud_corrected = apply_cloud_corrections(edge_results, offload_set, calibration_images, cloud_model)
    cloud_corrected_with_packing = apply_cloud_corrections_with_packing(edge_results, offload_set, calibration_images, cloud_model)
    cloud_prediction = filter_annotations(cloud_model.detect(calibration_images))
    calculate_performance([edge_results, gt_corrected, cloud_corrected, cloud_corrected_with_packing, gt_corrected_random, cloud_corrected_random, cloud_corrected_random_with_packing, cloud_prediction,],
                                ["Edge", "GT Corrected", "Cloud Corrected", "Cloud Corrected with Packing", "GT Corrected (Random Sample)", "Cloud Corrected (Randome Sample)", "Cloud Corrected with Packing (Randome Sample)", "Cloud Prediction",],  gt_annotations)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    # for different alphas ?
    
    # Sample data
    strategies = ['Full Edge', 'Smart Offloading (CP)', 'Smart Offloading (CP + Packing)', 'Random', 'Full Cloud']
    offload_costs = [0, 20, 15, 20, 50]  # hypothetical # of API calls
    precisions = [0.45, 0.65, 0.67, 0.55, 0.75]
    recalls = []
    accuraccies = []
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=offload_costs, y=precisions, marker='o')
    for i, label in enumerate(strategies):
        plt.text(offload_costs[i], precisions[i]+0.01, label, fontsize=9)
    
    plt.xlabel('Offloading Cost (API Calls)')
    plt.ylabel('Precision')
    plt.title('Precision vs. Offloading Cost')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{args.dataset}-precision_vs_cost.png", dpi=300, bbox_inches='tight')
    

if __name__ == "__main__":
    main()

