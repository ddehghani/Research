from ObjectDetection import *
from PIL import Image
from tqdm import tqdm
import os
from Utils import *
from random import sample
import time


PATH_TO_CROPPED_IMAGES = "/home/dehghani/EfficientVideoQueryUsingCP/data/coco/cropped"
PATH_TO_ANNOTATED_IMAGES = '/home/dehghani/EfficientVideoQueryUsingCP/annotated_packed_images'
MIN_SIZE = 4000
IOU_THRESHOLD = 0.5
SAMPLE_SIZE = 5

GRID_HEIGHT = 10
GRID_WIDTH = 10
PADDING = 5
PADDING_COLOR = 0 # black


def main():
    """ This class evaluates the performace of Object Detection models on cropped Images of coco with constrain pieced together """
    
    batch_size = GRID_HEIGHT * GRID_WIDTH

    # filter images that are too small
    image_list_unfiltered = sample(os.listdir(PATH_TO_CROPPED_IMAGES), GRID_HEIGHT * GRID_WIDTH * SAMPLE_SIZE * 10)
    image_list = []
    for image in tqdm(image_list_unfiltered):
        if image.endswith(".jpg"):
            img_path = os.path.join(PATH_TO_CROPPED_IMAGES, image)
            image_file = Image.open(img_path)
            image_size = image_file.width * image_file.height
            if image_size >= MIN_SIZE:
                image_list.append(img_path)

    tp_yolo_light_bboxes = 0
    tp_yolo_heavy_bboxes = 0
    fp_yolo_light_bboxes = 0
    fp_yolo_heavy_bboxes = 0
    total_time_grid = 0
    total_time_pack = 0
    total_time_individual = 0
    total_packing_time = 0
    total_griding_time = 0

    packed_tp_yolo_light_bboxes = 0
    packed_tp_yolo_heavy_bboxes = 0
    packed_fp_yolo_light_bboxes = 0
    packed_fp_yolo_heavy_bboxes = 0
    total_gt_bboxes = GRID_HEIGHT * GRID_WIDTH * SAMPLE_SIZE
    # gridify images into a grid, detect and annotate them
    for index in tqdm(range(SAMPLE_SIZE)):
        images = sample(image_list, batch_size)

        # for timing purposes only
        start = time.time()
        for image in images:
            no_overlap(filter_annotation(detect_using_yolo_heavy(image)))
            
        end = time.time()
        total_time_individual += end - start

        start = time.time()
        gt_annotations, grid_image = gridify(images, GRID_WIDTH, GRID_HEIGHT, PADDING, PADDING_COLOR)
        end = time.time()
        total_griding_time += end - start
        start = time.time()
        packed_gt_annotations, packed_image = pack(images, GRID_WIDTH, GRID_HEIGHT, PADDING, PADDING_COLOR)
        end = time.time()
        total_packing_time += end - start
        
        packed_image_path = os.path.join(PATH_TO_ANNOTATED_IMAGES, f'{index}_pack.jpg')
        packed_image.save(packed_image_path)

        annotateAndSave(packed_image_path, packed_gt_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{index}_packed_annotated.jpg" )
        
        # Save the original grid image
        grid_image_path = os.path.join(PATH_TO_ANNOTATED_IMAGES, f'{index}_grid.jpg')
        grid_image.save(grid_image_path)

        # annotateAndSave(grid_image_path, gt_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{index}_grid_annotated.jpg" )

        # Yolo (light)
        yolo_light_annotations = no_overlap(filter_annotation(detect_using_yolo_light(grid_image)))
        annotateAndSave(grid_image_path, yolo_light_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{index}_yolo_light.jpg" )

        packed_yolo_light_annotations = no_overlap(filter_annotation(detect_using_yolo_light(packed_image)))
        annotateAndSave(packed_image_path, packed_yolo_light_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{index}_packed_yolo_light.jpg" )
            
            
        # Yolo (heavy)
        start = time.time()
        yolo_heavy_annotations = no_overlap(filter_annotation(detect_using_yolo_heavy(grid_image)))
        end = time.time()
        total_time_grid += end - start
        annotateAndSave(grid_image_path, yolo_heavy_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{index}_yolo_heavy.jpg" )
            
        start = time.time()
        packed_yolo_heavy_annotations = no_overlap(filter_annotation(detect_using_yolo_heavy(packed_image)))
        end = time.time()
        total_time_pack += end - start
        annotateAndSave(packed_image_path, packed_yolo_heavy_annotations, PATH_TO_ANNOTATED_IMAGES, file_name=f"{index}_packed_yolo_heavy.jpg" )
         

        # Yolo light accuracy
        for annotation in yolo_light_annotations:
            tp = False
            for gt_annotation in gt_annotations:
                if iou(gt_annotation.bbox, annotation.bbox) > IOU_THRESHOLD and gt_annotation.name == annotation.name:
                    tp_yolo_light_bboxes += 1
                    tp = True
                    break
            if not tp:
                fp_yolo_light_bboxes += 1

        for annotation in packed_yolo_light_annotations:
            tp = False
            for gt_annotation in packed_gt_annotations:
                if iou(gt_annotation.bbox, annotation.bbox) > IOU_THRESHOLD and gt_annotation.name == annotation.name:
                    packed_tp_yolo_light_bboxes += 1
                    tp = True
                    break
            if not tp:
                packed_fp_yolo_light_bboxes += 1

        for annotation in yolo_heavy_annotations:
            tp = False
            for gt_annotation in gt_annotations:
                if iou(gt_annotation.bbox, annotation.bbox) > IOU_THRESHOLD and gt_annotation.name == annotation.name:
                    tp_yolo_heavy_bboxes += 1
                    tp = True
                    break
            if not tp:
                fp_yolo_heavy_bboxes += 1

        for annotation in packed_yolo_heavy_annotations:
            tp = False
            for gt_annotation in packed_gt_annotations:
                if iou(gt_annotation.bbox, annotation.bbox) > IOU_THRESHOLD and gt_annotation.name == annotation.name:
                    packed_tp_yolo_heavy_bboxes += 1
                    tp = True
                    break
            if not tp:
                packed_fp_yolo_heavy_bboxes += 1
    print("GRID RESULTS:")
    print('*' * 50)
    print('Recall')
    print('*' * 50)
    print(f'YOLO Light: {tp_yolo_light_bboxes / total_gt_bboxes}')
    print(f'YOLO Heavy: {tp_yolo_heavy_bboxes / total_gt_bboxes}')
    # print(f'FasterRCNN: {tp_faster_rcnn_bboxes / total_gt_bboxes}')
    print('*' * 50)
    print('Percision')
    print('*' * 50)
    print(f'YOLO Light: {tp_yolo_light_bboxes / (tp_yolo_light_bboxes + fp_yolo_light_bboxes)}')
    print(f'YOLO Heavy: {tp_yolo_heavy_bboxes / (tp_yolo_heavy_bboxes + fp_yolo_heavy_bboxes) }')
    # print(f'FasterRCNN: {tp_faster_rcnn_bboxes / (tp_faster_rcnn_bboxes + fp_faster_rcnn_bboxes)}')
    print('*' * 50)
    print('Accuracy')
    print('*' * 50)
    print(f'YOLO Light: {tp_yolo_light_bboxes / (total_gt_bboxes + fp_yolo_light_bboxes)}')
    print(f'YOLO Heavy: {tp_yolo_heavy_bboxes / (total_gt_bboxes + fp_yolo_heavy_bboxes) }')
    # print(f'FasterRCNN: {tp_faster_rcnn_bboxes / (total_gt_bboxes + fp_faster_rcnn_bboxes)}')
    print('*' * 50)

    print("\n\n\n\nPACKED RESULTS:")
    print('*' * 50)
    print('Recall')
    print('*' * 50)
    print(f'YOLO Light: {packed_tp_yolo_light_bboxes / total_gt_bboxes}')
    print(f'YOLO Heavy: {packed_tp_yolo_heavy_bboxes / total_gt_bboxes}')
    # print(f'FasterRCNN: {tp_faster_rcnn_bboxes / total_gt_bboxes}')
    print('*' * 50)
    print('Percision')
    print('*' * 50)
    print(f'YOLO Light: {packed_tp_yolo_light_bboxes / (packed_tp_yolo_light_bboxes + packed_fp_yolo_light_bboxes)}')
    print(f'YOLO Heavy: {packed_tp_yolo_heavy_bboxes / (packed_tp_yolo_heavy_bboxes + packed_fp_yolo_heavy_bboxes) }')
    # print(f'FasterRCNN: {tp_faster_rcnn_bboxes / (tp_faster_rcnn_bboxes + fp_faster_rcnn_bboxes)}')
    print('*' * 50)
    print('Accuracy')
    print('*' * 50)
    print(f'YOLO Light: {packed_tp_yolo_light_bboxes / (total_gt_bboxes + packed_fp_yolo_light_bboxes)}')
    print(f'YOLO Heavy: {packed_tp_yolo_heavy_bboxes / (total_gt_bboxes + packed_fp_yolo_heavy_bboxes) }')
    # print(f'FasterRCNN: {tp_faster_rcnn_bboxes / (total_gt_bboxes + fp_faster_rcnn_bboxes)}')
    print('*' * 50)

    print(f"Average time for grid detection: {1000 * total_time_grid/SAMPLE_SIZE :.1f} seconds per 1000 batches")
    print(f"Average time for packed detection: {1000 * total_time_pack/SAMPLE_SIZE :.1f} seconds per 1000 batches")
    print(f"Average time for individual images detection: {1000 * total_time_individual/SAMPLE_SIZE :.1f} seconds per 1000 batches")

    print()
    print(f"Average griding time: {1000 * total_griding_time/SAMPLE_SIZE :.1f} seconds per 1000 batches")
    print(f"Average packing time: {1000 * total_packing_time/SAMPLE_SIZE :.1f} seconds per 1000 batches")


if __name__ == '__main__':
    main()