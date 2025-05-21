import seaborn as sns
import matplotlib.pyplot as plt
import os
from .bbox_utils import iou

def calculate_performance(model_preds: list, model_names: list, gt_annotations: list, iou_threshold: float = 0.5):
    """
    Calculates performance metrics (Recall, Precision, Accuracy) for multiple models based on their predictions and ground truth annotations.

    Parameters:
    - model_preds: list of model predictions, where each element corresponds to predictions from one model
    - model_names: list of model names
    - gt_annotations: list of ground truth annotations per image
    - iou_threshold: IoU threshold for considering a prediction as a true positive

    Prints:
    - A table displaying the Recall, Precision, and Accuracy for each model
    """
    tp = [0] * len(model_preds)
    fp = [0] * len(model_preds)

    for model_idx, model_pred in enumerate(model_preds):
        for img_idx, annotations in enumerate(model_pred):
            for pred in annotations:
                fp[model_idx] += 1
                for gt in gt_annotations[img_idx]:
                    if iou(gt.bbox, pred.bbox) > iou_threshold and gt.name == pred.name:
                        tp[model_idx] += 1
                        fp[model_idx] -= 1
                        break

    gt_total = sum(len(anns) for anns in gt_annotations)

    data = []
    for idx, name in enumerate(model_names):
        recall = tp[idx] / gt_total if gt_total else 0
        precision = tp[idx] / (tp[idx] + fp[idx]) if (tp[idx] + fp[idx]) else 0
        accuracy = tp[idx] / (gt_total + fp[idx]) if (gt_total + fp[idx]) else 0
        data.append([name, recall, precision, accuracy])

    return data

def plot_scatterplot(x_values_dict: dict[str, list[float]], y_values_dict: dict[str, list[float]], output_dir: str = ".") -> None:
    """
    Generates scatter plots for all combinations of x and y variables.

    Parameters:
    - x_values_dict: dictionary where keys are variable names for x-axis (e.g., "Epoch", "Alpha") and values are lists of x values
    - y_values_dict: dictionary where keys are variable names for y-axis (e.g., "Accuracy", "Loss") and values are lists of y values
    - output_dir: directory to save the generated plots
    """
    os.makedirs(output_dir, exist_ok=True)

    for x_name, x_values in x_values_dict.items():
        for y_name, y_values in y_values_dict.items():
            if len(x_values) != len(y_values):
                raise ValueError(f"Length mismatch between x '{x_name}' and y '{y_name}'")

            sns.scatterplot(x=x_values, y=y_values)
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.title(f"{x_name} vs {y_name}")
            filename = f"{x_name.lower()}_vs_{y_name.lower()}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
            plt.clf()
