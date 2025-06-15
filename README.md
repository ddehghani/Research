# Selective Cloud Offloading for Object Detection

This repository accompanies the paper:  
**"Selective Cloud Offloading for Accurate and Efficient Object Detection"**

## 🌐 Overview

Deploying high-performance object detection models on edge devices (e.g., drones, smartphones, IoT sensors) is challenging due to computational and memory constraints. Full cloud offloading of image data addresses this, but introduces high latency, bandwidth limitations, and cost.

This project presents a **selective cloud offloading framework** that intelligently balances **prediction accuracy and offloading cost** using two core innovations:

- **Conformal Prediction** for statistically principled uncertainty quantification
- **Packing-Based Image Stitching (PBIS)** to reduce cloud invocation overhead by grouping uncertain regions before offloading

Our framework enables:

- High-accuracy detection via edge-cloud collaboration
- Tunable control over accuracy-cost tradeoffs
- Significant reduction in cloud inference cost with minimal performance degradation

## 🧠 Key Features

- Edge detection with YOLOv5n
- Cloud refinement with YOLOv11x
- Conformal prediction to flag uncertain detections
- Packing algorithm to reduce cloud API usage
- Modular experiment runner with baselines and plots

## 📁 Project Structure

```
.
├── main.py                   # Entry point for experiments
├── models/                  # Model definitions and wrappers
├── utils.py                 # Core utilities: iou, packing, plotting
├── constants.py             # Thresholds, label maps, and configs
├── datasets/                # Dataset directory
├── plots/                   # Auto-generated output plots
└── SelectiveCloudOffloading.pdf     # Full research paper
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

Supported datasets:

- [COCO](https://cocodataset.org/#download)
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Open Images (simplified)](https://storage.googleapis.com/openimages/web/index.html)

The default dataset directory is `./datasets`.

You can automatically download datasets using the `--download` flag. This flag can be added to any experiment command to ensure the dataset is downloaded before processing. For example:

```bash
python main.py outputs/ --dataset coco --qhat 0.9 --download
```

---

### 3. Run an Experiment

To see the effect of the parameter confidence threshold on our results:

```bash
python main.py outputs/ --dataset coco --qhat 0.9
```

To see the effect of the parameter alpha on our results:

```bash
python main.py outputs/ --dataset voc --conf 0.2
```

Fixed alpha and confidence:

```bash
python main.py outputs/ --dataset open-images --conf 0.2 --alpha 0.2
```

---

## ⚙️ Parameters

| Argument              | Description                                       | Default      |
| --------------------- | ------------------------------------------------- | ------------ |
| `--dataset`           | Dataset name: `coco`, `voc`, `open-images`        | `coco`       |
| `--datasets_dir`      | Directory containing datasets                     | `./datasets` |
| `--output_dir`        | Output directory for results and plots            | `outputs/`   |
| `--conf`              | Confidence threshold for edge model               | `None`       |
| `--alpha`             | Significance level for conformal prediction (0–1) | `None`       |
| `--qhat`              | Quantile threshold (overrides alpha if provided)  | `None`       |
| `--calibration_ratio` | Ratio of images used for calibration              | `0.05`       |

---

## 📊 Output

- Precision, Recall, Accuracy vs. offloading cost
- Auto-generated plots (`outputs/plots/`)
- Tabulated result summaries

---

## 📈 Example Plots

- Confidence vs Recall
- Alpha vs Offloading Cost
- Precision/Recall vs API Calls
- Comparison: Full Edge vs Selective vs Full Cloud
