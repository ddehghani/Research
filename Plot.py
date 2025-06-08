
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import matplotlib.cm as cm
from utils import (
    load_dataset, iou, annotateAndSave, filter_annotations, pack, gridify, annotateAndSave,
    calculate_performance, plot_scatterplot, plot_lineplot, plot_multi_lineplot,
    get_prediction_sets, compute_nonconformity_scores,
    apply_gt_corrections, apply_cloud_corrections, apply_cloud_corrections_with_packing)

# Set theme and matplotlib defaults
plt.figure(figsize=(5, 3), dpi=300)
sns.set_theme(font_scale=1.5, context='paper', style='white', palette='deep')
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
default_configs = dict(
    linewidth=4,
    markersize=10,
)




# plot_scatterplot(x_values_dict: dict[str, list[float]], y_values_dict: dict[str, list[float]], output_dir: str = ".") -> None:

sns.set(style="whitegrid", font_scale=1.6, rc={
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.titlesize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })


x_values = [1, 4, 9, 16, 25 , 49, 81, 121]
# x_values = [ 500/x for x in x_values]
recalls = [0.924277, 0.915029, 0.904624, 0.895376 , 0.89422 , 0.882659, 0.882659, 0.883237]
pers = [0.782673, 0.774841, 0.76603, 0.758199, 0.75722 ,0.74743, 0.74743, 0.74792]

fe = 0.883237
fc = 0.978613
fc_per = 0.885924
fe_per = 0.74792
# Create the plot
plt.figure(figsize=(5, 3), dpi=300)

sns.lineplot(x=x_values, y=recalls, marker='o', label='Recall')
sns.lineplot(x=x_values, y=pers, marker='s', label='Precision')



plt.xlabel('Maximum BBoxes Per Image')
plt.ylabel('Metrics')
plt.grid(True, linewidth=0.5, linestyle='--')

# Legend
plt.legend(loc='best', ncol=1, frameon=False)

# Save
plt.savefig('double_plot.pdf', bbox_inches='tight')
plt.close()



y_values = [ 500/x for x in x_values]
# Create the plot
plt.figure(figsize=(5, 3), dpi=300)

# Main line
sns.lineplot(x=x_values, y=y_values, marker='o')

# Add horizontal dashed lines
# plt.axhline(y=fe, color='black', linestyle='--', linewidth=1.2, label='FE Baseline')
# plt.axhline(y=fc, color='gray', linestyle='--', linewidth=1.2, label='FC Baseline')

# Labels and grid
plt.xlabel('Maximum BBoxes Per Image')
plt.ylabel('Cost')
plt.grid(True, linewidth=0.5, linestyle='--')

# Legend
plt.legend(loc='best', ncol=1, frameon=False)

# Save figure
plt.savefig('cost_vs_max.pdf', bbox_inches='tight')
plt.close()




# Create the plot
plt.figure(figsize=(5, 3), dpi=300)

# Main line
sns.lineplot(x=x_values, y=pers, marker='o', color='blue', label='Smart Offloading')

# Add horizontal dashed lines
plt.axhline(y=fe_per, color='black', linestyle='--', linewidth=1.2, label='FE Baseline')
# plt.axhline(y=fc_per, color='gray', linestyle='--', linewidth=1.2, label='FC Baseline')

# Labels and grid
plt.xlabel('Maximum BBoxes Per Image')
plt.ylabel('Precision')
plt.grid(True, linewidth=0.5, linestyle='--')

# Legend
plt.legend(loc='best', ncol=1, frameon=False)

# Save figure
plt.savefig('precision.pdf', bbox_inches='tight')
plt.close()