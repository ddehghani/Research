from models import *
from Utils import *
import json
import os
import shutil
from tqdm import tqdm
import os
from random import sample

"""
Evaluates the performance of models with certain constraints applied to the original COCO dataset.
Saves annotated images with bounding boxes for each model's predictions.

Note: This script does not evaluate the models themselves, but rather their performance under specific constraints.
"""

PATH_TO_ANNOTATED_IMAGES = '/home/dehghani/EfficientVideoQueryUsingCP/annotated_images'
PATH_TO_DATA_SOURCE = '/home/dehghani/EfficientVideoQueryUsingCP/data/coco/train2017'
PATH_TO_ANNOTATIONS = '/home/dehghani/EfficientVideoQueryUsingCP/data/coco/annotations/instances_train2017.json'

SAMPLE_SIZE = 50
MIN_BBOX_SIZE = 2000 # 929 experimentally minimum
# ALLOWED_LABELS = ['person', 'chair', 'motorcycle']

def main():
    
    shutil.rmtree(PATH_TO_ANNOTATED_IMAGES) 

    with open(PATH_TO_ANNOTATIONS, 'r') as f:
        annotations = json.load(f)
    

    total_gt_bboxes = 0
    # tp_google_bboxes = 0
    # tp_amazon_bboxes = 0
    # tp_microsoft_bboxes = 0
    tp_faster_rcnn_bboxes = 0
    tp_yolo_light_bboxes = 0
    tp_yolo_heavy_bboxes = 0

    fp_faster_rcnn_bboxes = 0
    fp_yolo_light_bboxes = 0
    fp_yolo_heavy_bboxes = 0


    image_list = os.listdir(PATH_TO_DATA_SOURCE)
    for image in tqdm(range(SAMPLE_SIZE)):
        image = sample(image_list, 1)[0]
        if image.endswith(".jpg"):
            img_path = os.path.join(PATH_TO_DATA_SOURCE, image)
            image_annotations = get_annotations(annotations, image)

            # Ground Truth
            gt_annotations = []
            for annotation in image_annotations:
                bbox_size = annotation.bbox[3] * annotation.bbox[2]
                if bbox_size > MIN_BBOX_SIZE:
                # if True:
                    annotation.bbox[2] += annotation.bbox[0]
                    annotation.bbox[3] += annotation.bbox[1]
                    gt_annotations.append(annotation)
                    
            annotateAndSave(img_path, gt_annotations, PATH_TO_ANNOTATED_IMAGES, image)
                
            
            # # Google
            # google_annotations = detect_using_google(img_path)
            # annotateAndSave(img_path, google_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{image}_google.jpg" )
           
                
            # Yolo (light)
            yolo_light_annotations = detect_using_yolo_light(img_path)
            annotateAndSave(img_path, yolo_light_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{image}_yolo_light.jpg" )
            
            # Yolo (heavy)
            yolo_heavy_annotations = detect_using_yolo_heavy(img_path)
            annotateAndSave(img_path, yolo_heavy_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{image}_yolo_heavy.jpg" )
            
            
            # Faster RCNN
            faster_rcnn_annotations = detect_using_faster_rcnn(img_path)
            annotateAndSave(img_path, faster_rcnn_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{image}_faster_rcnn.jpg" )
            
            # # Amazon
            # amazon_annotations = detect_using_amazon(img_path)
            # annotateAndSave(img_path, amazon_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{image}_amazon.jpg" )
            
            
            # # Microsoft
            # microsoft_bboxes, microsoft_labels, _ = detectObjectsConstrained( img_path=img_path,
            #                                                             method=detect_using_microsoft,
            #                                                             annotate=True,
            #                                                             annotate_dest=PATH_TO_ANNOTATED_IMAGES, 
            #                                                             annotate_name_extension='microsoft')
            
            # calculate accuracy
            total_gt_bboxes += len(gt_annotations)

            # # Google accuracy
            # for annotation in google_annotations:
            #     for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            #         if iou(gt_bbox, annotation.bbox) > 0.5 and gt_label == annotation.name:
            #             tp_google_bboxes += 1
            #             break

            # Yolo light accuracy
            for annotation in yolo_light_annotations:
                tp = False
                for gt_annotation in gt_annotations:
                    if iou(gt_annotation.bbox, annotation.bbox) > 0.5 and gt_annotation.name == annotation.name:
                        tp_yolo_light_bboxes += 1
                        tp = True
                        break
                if not tp:
                    fp_yolo_light_bboxes += 1

            # Yolo heavy accuracy
            for annotation in yolo_heavy_annotations:
                tp = False
                for gt_annotation in gt_annotations:
                    if iou(gt_annotation.bbox, annotation.bbox) > 0.5 and gt_annotation.name == annotation.name:
                        tp_yolo_heavy_bboxes += 1
                        tp = True
                        break
                if not tp:
                    fp_yolo_heavy_bboxes += 1

            # FasterRCNN accuracy
            for annotation in faster_rcnn_annotations:
                tp = False
                for gt_annotation in gt_annotations:
                    if iou(gt_annotation.bbox, annotation.bbox) > 0.5 and gt_annotation.name == annotation.name:
                        tp_faster_rcnn_bboxes += 1
                        tp = True
                        break
                if not tp:
                    fp_faster_rcnn_bboxes += 1
            
            # # Amazon accuracy
            # for amazon_bbox, amazon_label in zip (amazon_bboxes, amazon_labels):
            #     for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            #         if iou(gt_bbox, amazon_bbox) > 0.5 and gt_label == amazon_label:
            #             tp_amazon_bboxes += 1
            #             break

            # # Microsoft accuracy
            # for microsoft_bbox, microsoft_label in zip (microsoft_bboxes, microsoft_labels):
            #     for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            #         if iou(gt_bbox, microsoft_bbox) > 0.5 and gt_label == microsoft_label:
            #             tp_microsoft_bboxes += 1
            #             break
    
    
    print('*' * 50)
    print('Recall')
    print('*' * 50)
    print(f'YOLO Light: {tp_yolo_light_bboxes / total_gt_bboxes}')
    print(f'YOLO Heavy: {tp_yolo_heavy_bboxes / total_gt_bboxes}')
    print(f'FasterRCNN: {tp_faster_rcnn_bboxes / total_gt_bboxes}')
    print('*' * 50)
    print('Percision')
    print('*' * 50)
    print(f'YOLO Light: {tp_yolo_light_bboxes / (tp_yolo_light_bboxes + fp_yolo_light_bboxes)}')
    print(f'YOLO Heavy: {tp_yolo_heavy_bboxes / (tp_yolo_heavy_bboxes + fp_yolo_heavy_bboxes) }')
    print(f'FasterRCNN: {tp_faster_rcnn_bboxes / (tp_faster_rcnn_bboxes + fp_faster_rcnn_bboxes)}')
    print('*' * 50)
    print('Accuracy')
    print('*' * 50)
    print(f'YOLO Light: {tp_yolo_light_bboxes / (total_gt_bboxes + fp_yolo_light_bboxes)}')
    print(f'YOLO Heavy: {tp_yolo_heavy_bboxes / (total_gt_bboxes + fp_yolo_heavy_bboxes) }')
    print(f'FasterRCNN: {tp_faster_rcnn_bboxes / (total_gt_bboxes + fp_faster_rcnn_bboxes)}')
    print('*' * 50)
    # print(f'Google Recall: {tp_google_bboxes / total_gt_bboxes}') 
    # print(f'Amazon Recall: {tp_amazon_bboxes / total_gt_bboxes}')
    # print(f'Microsoft Recall: {tp_microsoft_bboxes / total_gt_bboxes}')


if __name__ == '__main__':
    main()