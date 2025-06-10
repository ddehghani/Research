import argparse
import os
from pathlib import Path
from models import Model
import numpy as np
from tqdm import tqdm
import random
from constants import IOU_THRESHOLD, GRID_WIDTH, GRID_HEIGHT, COCO_LABELS, VOC_LABELS, OPEN_IMAGES_LABELS
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from utils import (
    load_dataset, iou, annotateAndSave, filter_annotations, pack, gridify, annotateAndSave,
    calculate_performance, plot_scatterplot, plot_lineplot, plot_multi_lineplot,
    get_prediction_sets, compute_nonconformity_scores,
    apply_gt_corrections, apply_cloud_corrections, apply_cloud_corrections_with_packing)


def select_random_bboxes(edge_results, total_to_select):
    """
    Selects exactly `total_to_select` random (image_idx, instance_idx) pairs
    from edge_results, ignoring empty sublists.
    """
    all_indices = [
        (img_idx, inst_idx)
        for img_idx, instances in enumerate(edge_results)
        for inst_idx in range(len(instances))
    ]

    return random.sample(all_indices, min(total_to_select, len(all_indices)))


def split_images(images: list[str], calibration_ratio: float) -> tuple[list[str], list[str]]:
    """Split images into calibration and detection sets based on the specified ratio."""
    random.shuffle(images)
    split_index = int(len(images) * calibration_ratio)
    return images[1000:split_index+1000], images[split_index:]

def sweep_over_conf(
    images: list[str], 
    gt_annotations, 
    output_dir: str, 
    iou_threshold: float, 
    edge_model: Model,
    cloud_model: Model,
    qhat
) -> None:
    confs = [0.05 * i + 0.05 for i in range(19)]
    recalls = []
    final_recalls = []
    final_precisions = []
    final_accuracies = []
    cloud_results = filter_annotations(cloud_model.detect(images))
    for conf in tqdm(confs):
        edge_results = filter_annotations(edge_model.detect(images, conf = conf))
        
        tp = 0
        tp_and_fn = 0
        for detections, gt_annots in zip(edge_results, gt_annotations):
            for gt_ant in gt_annots:
                tp_and_fn += 1
                for detection in detections:
                    if iou(detection.bbox, gt_ant.bbox) > iou_threshold: # only the bbox and not the label
                        tp += 1
                        break
        recall = tp / tp_and_fn
        recalls.append(recall)
        performance_data = run_experiment(
            images,
            edge_results,
            gt_annotations,
            cloud_model,
            qhat,
            include_baselines = False,
            include_fc = False,
            cloud_results = cloud_results
        )[0]
        final_recalls.append(performance_data[1])
        final_precisions.append(performance_data[2])
        final_accuracies.append(performance_data[3])
    

    plot_lineplot(   x_values_dict = {"Confidence Threshold" : confs},
                        y_values_dict={"Recall": recalls},
                        output_dir=os.path.join(output_dir, 'plots'))

    
    plot_multi_lineplot( x_values_dict = {"Confidence Threshold" : confs},
                         y_values_dict={
                            "Recall": final_recalls,
                            "Precision": final_precisions,
                            # "Accuracy": final_accuracies
                         },
                         output_dir=os.path.join(output_dir, 'plots'))

def run_experiment(
    calibration_images,
    edge_results,
    gt_annotations,
    cloud_model,
    qhat,
    cloud_results = None,
    include_baselines = True,
    include_randoms = True,
    include_gt = True,
    include_fc = True,
    include_fe = False,
    plot = False,
):
    total = sum(len(res) for res in edge_results)

    offload_set = [
        (i, j)
        for i, result in enumerate(edge_results)
        for j, pred in enumerate(result)
        if len(get_prediction_sets(pred, qhat)) > 1
    ]

    strategies_labels = []
    strategies = []
    offload_costs = []

    if include_fe:
        strategies_labels.append('FE')
        strategies.append(edge_results)
        offload_costs.append(0)
    
    cloud_corrected_with_packing = apply_cloud_corrections_with_packing(edge_results, offload_set, calibration_images, cloud_model)
    strategies_labels.append('SO-P')
    strategies.append(cloud_corrected_with_packing)
    offload_costs.append(len(offload_set)// (GRID_WIDTH * GRID_HEIGHT) + 1)
    
    if include_baselines:
        cloud_corrected = apply_cloud_corrections(edge_results, offload_set, calibration_images, cloud_model)
        strategies_labels.append('SO-NP')
        strategies.append(cloud_corrected)
        offload_costs.append(len(offload_set))
        
        if include_randoms:
            random_offload_set = select_random_bboxes(edge_results, len(offload_set))
            cloud_corrected_random = apply_cloud_corrections(edge_results, random_offload_set, calibration_images, cloud_model)
            cloud_corrected_random_with_packing = apply_cloud_corrections_with_packing(edge_results, random_offload_set, calibration_images, cloud_model)
            strategies_labels.append('RO-P')
            strategies.append(cloud_corrected_random_with_packing)
            offload_costs.append(len(random_offload_set)// (GRID_WIDTH * GRID_HEIGHT) + 1)
            strategies_labels.append('RO-NP')
            strategies.append(cloud_corrected_random)
            offload_costs.append(len(random_offload_set))

        if include_gt:
            gt_corrected = apply_gt_corrections(edge_results, offload_set, gt_annotations, IOU_THRESHOLD)
            gt_corrected_random = apply_gt_corrections(edge_results, random_offload_set, gt_annotations, IOU_THRESHOLD)
            strategies_labels.append('SO-GT')
            strategies.append(gt_corrected)
            offload_costs.append(0)
            strategies_labels.append('RO-GT')
            strategies.append(gt_corrected_random)
            offload_costs.append(0)
    
    if include_fc:
        if cloud_results == None:
            cloud_prediction = filter_annotations(cloud_model.detect(calibration_images))
        else:
            cloud_prediction = cloud_results
        strategies_labels.append('FC')
        strategies.append(cloud_prediction)
        offload_costs.append(len(calibration_images))

    performance_data = calculate_performance(
        strategies,
        strategies_labels,
        gt_annotations,
    )
    
    if plot:
        plots = ['Recall', 'Precision', 'Accuracy']
        markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', '<', '>']  # Fallback markers
        colors = cm.get_cmap('tab10', len(strategies))
        
        # Indices for special strategies
        fe_index = strategies_labels.index('FE') if 'FE' in strategies_labels else None
        fc_index = strategies_labels.index('FC') if 'FC' in strategies_labels else None
        sop_index = strategies_labels.index('SO-P') if 'SO-P' in strategies_labels else None
        
        for index, plot in enumerate(plots):
            data = [row[index + 1] for row in performance_data]  # offset by 1 because Name is at index 0
            
            plt.figure(figsize=(5, 3), dpi=300)
        
            for i in range(len(strategies)):
                if strategies_labels[i] == 'SO-P':
                    plt.scatter(offload_costs[i], data[i],
                                color='red',
                                marker='*',
                                s=90,  # make it bigger for visibility
                                label='SO-P')
                else:
                    plt.scatter(offload_costs[i], data[i],
                                color=colors(i),
                                marker=markers[i % len(markers)],
                                s=50,
                                label=strategies_labels[i])

            # Dashed line between FE and FC
            if fe_index is not None and fc_index is not None:
                x_vals = [offload_costs[fe_index], offload_costs[fc_index]]
                y_vals = [data[fe_index], data[fc_index]]
                plt.plot(x_vals, y_vals, linestyle='--', linewidth=1.5, color='black', label='FE-FC Line')
        
            plt.legend(fontsize=14, loc='best', ncol=2)
            plt.xlabel('Offloading Cost (API Calls)', fontsize=18)
            plt.ylabel(plot, fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True)
        
            plt.savefig(f"{args.output_dir}/plots/{args.dataset}-{plot.lower()}_vs_cost.pdf", bbox_inches='tight')
            plt.close()
            
    return performance_data

def sweep_over_alphas(
    nonconformity_scores,
    calibration_images,
    edge_results,
    gt_annotations,
    cloud_model,
    output_dir
):
    alphas = [i * 0.01 for i in range(1, 50)]
    qhats = [
        np.quantile(nonconformity_scores, (1 - alpha) * (1 + 1 / len(nonconformity_scores)))
        for alpha in alphas
    ]

    first_high_index = next((i for i, val in enumerate(qhats) if val <= 0.99), 1)
    first_low_index = next((i for i, val in enumerate(qhats) if val <= 0.01), len(qhats) - 1)

    alphas = alphas[first_high_index - 1:first_low_index]
    qhats = qhats[first_high_index - 1:first_low_index]

    # Determine offload sets (used only for filtering by cost)
    offload_sets = [[] for _ in qhats]
    total_boxes = sum(len(r) for r in edge_results)

    for img_idx, result in enumerate(edge_results):
        for inst_idx, prediction in enumerate(result):
            for i, qhat in enumerate(qhats):
                pred_set = get_prediction_sets(prediction, qhat)
                if len(pred_set) > 1:
                    offload_sets[i].append((img_idx, inst_idx))

    costs = [len(s) for s in offload_sets]
    first_cost_cutoff = next((i for i, val in enumerate(costs) if val <= total_boxes / 50), len(costs) - 1)

    # Final alpha/qhat/cost sets
    alphas = alphas[:first_cost_cutoff]
    qhats = qhats[:first_cost_cutoff]
    costs = costs[:first_cost_cutoff]

    print("[Filtered Qhats]", qhats)
    print("[Filtered Costs]", costs)

    # Run experiments
    final_recalls = []
    final_precisions = []
    final_accuracies = []
    
    for qhat in tqdm(qhats):
        perf_data = run_experiment(
            calibration_images,
            edge_results,
            gt_annotations,
            cloud_model,
            qhat,
            include_baselines = True,
            include_fc = True,
        )[0]
        final_recalls.append(perf_data[1])
        final_precisions.append(perf_data[2])
        final_accuracies.append(perf_data[3])
        
    plot_lineplot(  x_values_dict = {"Alpha" : alphas},
                    y_values_dict={
                            "Qhat": qhats, 
                            "Cost": costs,
                            "Recall": final_recalls,
                            "Precision": final_precisions,
                        },
                        output_dir=os.path.join(args.output_dir, 'plots')
                     )

    
    plot_multi_lineplot( x_values_dict = {"Alpha" : alphas},
                             y_values_dict={
                                "Recall": final_recalls,
                                "Precision": final_precisions,
                             },
                             output_dir=os.path.join(args.output_dir, 'plots')
                           )

    plot_multi_lineplot( x_values_dict = {"Cost": costs},
                             y_values_dict={
                                "Recall": final_recalls,
                                "Precision": final_precisions,
                             },
                             output_dir=os.path.join(args.output_dir, 'plots')
                           )
    
    
def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, args.datasets_dir)
    get_annotations = dataset["get_annotations"]
    images = dataset["images"]
    edge_model = dataset["edge_model"]
    cloud_model = dataset["cloud_model"]

    print(f"Experimenting with {GRID_WIDTH} by {GRID_HEIGHT}")
    # Split images
    calibration_images, detection_images = split_images(images, args.calibration_ratio)
    print(f"Calibration images: {len(calibration_images)}, Detection images: {len(detection_images)}")

    # Load GT annotations once
    gt_annotations = [filter_annotations(get_annotations(Path(img).stem)) for img in calibration_images]

    
    if not args.conf:
        sweep_over_conf(calibration_images, gt_annotations, args.output_dir, IOU_THRESHOLD, edge_model, cloud_model, args.qhat)
        return
    
    # Run edge inference once
    print(f"Using preset confidence threshold of {args.conf}")
    edge_results = filter_annotations(edge_model.detect(calibration_images, conf=args.conf))

    # Select dataset-specific label set
    label_map = {
        "coco": COCO_LABELS,
        "voc": VOC_LABELS,
        "open-images": ["Airplane", "Apple", "Bag" , "Ball", "Banana", "Bed", "Bicycle", "Bird", 
                            "Bottle", "Car", "Chair", "Cup", "Flower", "Helmet", "Laptop", 
                            "Motorcycle", "Person", "TV", "Table", "Wheel"],
    }

    if args.qhat is None:
        nonconformity_scores = compute_nonconformity_scores(
            calibration_images, edge_results, get_annotations, label_map[args.dataset]
        )
        if args.alpha is None:
            sweep_over_alphas(
                nonconformity_scores,
                calibration_images,
                edge_results,
                gt_annotations,
                cloud_model,
                args.output_dir
            )
            return
        
        args.qhat = np.quantile(nonconformity_scores, (1 - args.alpha) * (1 + 1 / len(nonconformity_scores)))
        print(f"Using qhat: {args.qhat}")
    else:
        print(f"Using preset qhat: {args.qhat}")
        
    performance_data = run_experiment(
                calibration_images,
                edge_results,
                gt_annotations,
                cloud_model,
                args.qhat,
                include_gt = False,
                include_fe = True,
                plot = True
            )
    print(tabulate(performance_data, headers=["Name", "Recall", "Precision", "Accuracy"], tablefmt="grid"))
        

if __name__ == "__main__":
    # load_dotenv()
    
    parser = argparse.ArgumentParser(description="Prepare datasets for selective cloud offloading in object detection.")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--dataset", type=str,  default="coco", choices=["coco", "voc", "open-images"], help="Dataset to use: coco or voc, or open-images")
    parser.add_argument("--calibration_ratio", type=float, default=0.05)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--qhat", type=float)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--datasets_dir", type=str, default="./datasets", help="Where to store datasets")
    args = parser.parse_args()
    main(args)

