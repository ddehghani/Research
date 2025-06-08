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
    matched_gt = [0] * len(model_preds)

   
    
    for model_idx, model_pred in enumerate(model_preds):
        matched_gt_idxs = set()
        for img_idx, annotations in enumerate(model_pred):
            for pred in annotations:
                match_found = False
                for gt_idx, gt in enumerate(gt_annotations[img_idx]):
                    # if (img_idx, gt_idx) in matched_gt_idxs:
                    #     continue  # Already matched this GT box
                    
                    if iou(gt.bbox, pred.bbox) > iou_threshold and gt.name == pred.name:
                        tp[model_idx] += 1
                        matched_gt_idxs.add((img_idx, gt_idx))
                        match_found = True
                        break
                if not match_found:
                    fp[model_idx] += 1
        matched_gt[model_idx] = len(matched_gt_idxs)

    gt_total = sum(len(anns) for anns in gt_annotations)

    data = []
    for idx, name in enumerate(model_names):
        recall = matched_gt[idx] / gt_total if gt_total else 0
        # recall = tp[idx] / gt_total if gt_total else 0
        precision = tp[idx] / (tp[idx] + fp[idx]) if (tp[idx] + fp[idx]) else 0
        accuracy = tp[idx] / (gt_total + fp[idx]) if (gt_total + fp[idx]) else 0
        data.append([name, recall, precision, accuracy])

    return data
    
def plot_lineplot(x_values_dict: dict[str, list[float]], y_values_dict: dict[str, list[float]], output_dir: str = ".") -> None:
    """
    Generates smooth line plots for all combinations of x and y variables.

    Parameters:
    - x_values_dict: dictionary where keys are variable names for x-axis (e.g., "Epoch", "Alpha") and values are lists of x values
    - y_values_dict: dictionary where keys are variable names for y-axis (e.g., "Accuracy", "Loss") and values are lists of y values
    - output_dir: directory to save the generated plots
    """
    os.makedirs(output_dir, exist_ok=True)
    # Set Seaborn + Matplotlib style with large fonts
    sns.set(style="whitegrid", font_scale=1.6, rc={
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.titlesize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
    
    for x_name, x_values in x_values_dict.items():
        for y_name, y_values in y_values_dict.items():
            if len(x_values) != len(y_values):
                raise ValueError(f"Length mismatch between x '{x_name}' and y '{y_name}'")
    
            # Plot
            plt.figure(figsize=(6, 4), dpi=300)
            sns.lineplot(x=x_values, y=y_values, marker="o")
    
            # Labels and styling
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.grid(True, linestyle='--', linewidth=0.6)
            # plt.legend(loc='best', frameon=False)
    
            # Save
            filename = f"{x_name.lower()}_vs_{y_name.lower()}.pdf"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
            plt.close()

def plot_multi_lineplot(x_values_dict: dict[str, list[float]], 
                        y_values_dict: dict[str, list[float]], 
                        output_dir: str = ".") -> None:
    """
    Plots multiple lines (from different y-variables) on a single plot for each x-variable.

    Parameters:
    - x_values_dict: dict where keys are x-axis labels (e.g., "Confidence Threshold"), values are x data (list of floats).
    - y_values_dict: dict where keys are y labels (e.g., "Recall", "Accuracy"), values are y data (list of floats).
    - output_dir: path to save plots.
    """
    sns.set(style="whitegrid", font_scale=1.6, rc={
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.titlesize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
    
    os.makedirs(output_dir, exist_ok=True)
    
    for x_name, x_values in x_values_dict.items():
        plt.figure(figsize=(6, 4), dpi=300)
    
        for y_name, y_values in y_values_dict.items():
            if len(x_values) != len(y_values):
                raise ValueError(f"Length mismatch between x '{x_name}' and y '{y_name}'")
    
            sns.lineplot(x=x_values, y=y_values, label=y_name, marker="o")
    
        plt.xlabel(x_name)
        plt.ylabel("Performance Metrics")
        # Removed plt.title for consistency with prior plots
        plt.legend(loc='best', frameon=False)
        plt.grid(True, linestyle='--', linewidth=0.6)
    
        filename = f"{x_name.lower().replace(' ', '_')}_vs_performance_metrics.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved plot to {filepath}")
        plt.close()
        
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
            filename = f"{x_name.lower()}_vs_{y_name.lower()}.pdf"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight')
            print(f"Saved plot to {filepath}")
            plt.clf()
