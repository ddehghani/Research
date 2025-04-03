import os
import random
import Utils
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

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

def select_alpha(images: list[str], predictions: list, nonconformity_scores: list, ground_truth: dict, output_dir: str) -> tuple[float, float]:
    alphas = [i * 0.01 for i in range(1, 100)]
    qhats = [np.quantile(nonconformity_scores, (1 - alpha) * (1 + 1 / len(nonconformity_scores))) for alpha in alphas]
    offload_sets = [[] for _ in qhats]

    for image_index, result in enumerate(predictions):
        for instance_index, prediction in enumerate(result):
            for index, qhat in enumerate(qhats):
                pred_set = Utils.get_prediction_sets(prediction, qhat)
                if len(pred_set) > 1:
                    offload_sets[index].append((image_index, instance_index))


    costs = [len(s) for s in offload_sets]

    plot_conformal_metrics(alphas=alphas,
                           metrics_dict={"Qhat": qhats, "Cost": costs},
                           output_dir=os.path.join(output_dir, 'plots'))

    selected_alpha = float(input("Based on the plot, choose your desired Alpha: "))
    return selected_alpha
