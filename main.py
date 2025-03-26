import argparse
import os
import shutil
from pathlib import Path
import random
from models import Edge
import Utils
import json
import numpy as np
from tqdm import tqdm


PATH_TO_ANNOTATIONS = '/data/dehghani/EfficientVideoQueryUsingCP/coco/annotations/instances_train2017.json'
IOU_THRESH = 0.5
def get_images(input_dir):
    """Retrieve all image paths from the input directory."""
    return sorted([str(p) for p in Path(input_dir).glob("*.jpg")])  # Modify for other formats if needed

def split_images(images, calibration_ratio):
    """Split images into calibration and detection sets based on the specified ratio."""
    # random.shuffle(images)
    split_index = int(len(images) * calibration_ratio)
    return images[:split_index], images[split_index:]


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for selective cloud offloading in object detection.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save split datasets.")
    parser.add_argument("--calibration_ratio", type=float, default=0.05, help="Ratio of images for calibration (default: 0.05).")
    parser.add_argument("--alpha1", type=float, default=0.05, help="Alpha value for the first conformal prediction (object localization).")
    parser.add_argument("--alpha2", type=float, default=0.05, help="Alpha value for the second conformal prediction (object classification).")
    parser.add_argument("--qhat1", type=float, help="Preset qhat1 for object localization (if set, alpha1 is ignored and first conformal prediction is skipped).")
    parser.add_argument("--qhat2", type=float, help="Preset qhat2 for object classification (if set, alpha2 is ignored and second conformal prediction is skipped).")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold to match predicted and ground truth boxes (default: 0.5).")
    
    args = parser.parse_args()

    images = get_images(args.input_dir)
    if not images:
        print("No images found in the input directory!")
        return

    calibration_images, detection_images = split_images(images, args.calibration_ratio)


    print(f"Dataset prepared for Selective Cloud Offloading:")
    print(f"  Calibration set: {len(calibration_images)} images")
    print(f"  Detection set: {len(detection_images)} images")
    
    qhat1 = 0

    if args.qhat1:
        print(f"Using preset qhat1: {args.qhat1} (Skipping first conformal prediction)")
        qhat1 = args.qhat1
    else:
        print(f"Using alpha1 = {args.alpha1} for first conformal prediction ...")
        results = Edge().detect(calibration_images, conf = 0) 
        with open(PATH_TO_ANNOTATIONS, 'r') as f:
            gt = json.load(f)
        
        scores_by_class = {0: [], 1: []}
        # nonconformity_scores = []
        for image, detections in tqdm(zip(calibration_images,  results), total=len(calibration_images)):
            gt_annotations = Utils.get_annotations(gt, image)
            for detection in detections:
                # find GT binary label (objects exists or not)
                gt_label = 0
                for gt_ant in gt_annotations:
                    if Utils.iou(gt_ant.bbox, detection.bbox) > args.iou:
                        gt_label = 1
                        break
                # Nonconformity = 1 - probability assigned to true class


                if gt_label:
                    scores_by_class[1].append(1 - detection.confidence)
                else:
                    scores_by_class[0].append(detection.confidence)
                # nonconformity_score = 1 - detection.confidence if gt_label else detection.confidence # or objectness score but normalized
                # nonconformity_scores.append(nonconformity_score)
        
        # def predict_mcp(prob, scores_by_class, alpha=0.1):
        # 
        #     prediction_set = []
            
        #     for y in [0, 1]:
        #         score = 1 - prob[y]
        #         scores_y = scores_by_class[y]
        #         p_val = (np.sum(np.array(scores_y) >= score) + 1) / (len(scores_y) + 1)
                
        #         if p_val > alpha:
        #             prediction_set.append(y)
        #     return prediction_set

        # print(sorted(nonconformity_scores)[:20])
        # print(sorted(nonconformity_scores)[-20:])
        # qhat1 = np.quantile(nonconformity_scores, 1 - args.alpha1, method='higher')

        # print(f"Using qhat1: {qhat1} (Resulted from the first conformal prediction)")

        results = Edge().detect(detection_images[:25], conf = 0) 


        true_count = 0
        false_count = 0
        unsure_count = 0

        for image, detections in tqdm(zip(detection_images[:25],  results), total = 25):
            gt_annotations = Utils.get_annotations(gt, image)
            for detection in detections:
                # find GT binary label (objects exists or not)
                gt_label = 0
                for gt_ant in gt_annotations:
                    if Utils.iou(gt_ant.bbox, detection.bbox) > args.iou:
                        gt_label = 1
                        break
                
                # Mondrian Conformal Prediction
                prob = {1: detection.confidence, 0: 1 - detection.confidence}
                prediction_set = []
                for y in [0, 1]:
                    score = 1 - prob[y]
                    scores_y = scores_by_class[y]
                    p_val = (np.sum(np.array(scores_y) >= score) + 1) / (len(scores_y) + 1)
                    
                    if p_val > args.alpha1:
                        prediction_set.append(y)
                
                if len(prediction_set) == 1:
                    if prediction_set[0] == gt_label:
                        true_count += 1
                    else:
                        false_count += 1
                else:
                    unsure_count += 1

                # Inductive Conformal Prediction (ICP)
                # if (detection.confidence > qhat1 and gt_label) or (detection.confidence <= qhat1 and not gt_label):
                #     true_count += 1
                # else:
                #     false_count += 1
               
        
        print(f"Precision: {(true_count + unsure_count) / (true_count + false_count + unsure_count)} > 1 - alpha ({1 - args.alpha1})?")

        # print(f"Precision: {true_count / (true_count + false_count)} > 1 - alpha ({1 - args.alpha1})? {true_count / (true_count + false_count) > 1 - args.alpha1}")


    
    if args.qhat2:
        print(f"Using preset qhat2: {args.qhat2} (Skipping second conformal prediction)")
    else:
        print(f"Using alpha2 for second conformal prediction: {args.alpha2}")

if __name__ == "__main__":
    main()


"""
args:




things to do:
2. compare filtering the result and detecting same images again using new conf thresh, if same same, use filteration method, otherwise use new detection
3. extract bboxes perform confromal prediction on object classification on these to using second alpha user provided
4. detect and show results
5. run the experiments (maybe a seperate command for this)

"""