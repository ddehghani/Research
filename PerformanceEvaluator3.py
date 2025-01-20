from Utils import *
from ObjectDetection import *
from PIL import Image
from tqdm import tqdm
from random import sample
import os

""" This class evaluates the performace of Object Detection models on cropped Images of coco with constrain """

PATH_TO_CROPPEDIMAGES = '/home/dehghani/EfficientVideoQueryUsingCP/data/coco/cropped'
MIN_SIZE = 2000

def main():
    SAMPLE_SIZE = 50

    tp_yolo_light = 0
    tp_yolo_heavy = 0
    tp_faster_rcnn = 0

    fp_yolo_light = 0
    fp_yolo_heavy = 0
    fp_faster_rcnn = 0

    image_list = os.listdir(PATH_TO_CROPPEDIMAGES)
    for image in tqdm(range(SAMPLE_SIZE)):
        image = sample(image_list, 1)[0]
        if image.endswith(".jpg"):
            img_path = os.path.join(PATH_TO_CROPPEDIMAGES, image)
            image_file = Image.open(img_path)
            image_size = image_file.width * image_file.height
            if image_size < MIN_SIZE:
                SAMPLE_SIZE -= 1
                continue
            gt_label = image.split('.')[0].split('_')[-1]
            faster_rcnn_result = detect_using_faster_rcnn(img_path)
            for instance in faster_rcnn_result:
                if instance.name == gt_label:
                    tp_faster_rcnn += 1
                else:
                    fp_faster_rcnn += 1

            yolo_light_result = detect_using_yolo_light(img_path)
            for instance in yolo_light_result:
                if instance.name == gt_label:
                    tp_yolo_light += 1
                else:
                    fp_yolo_light += 1
            yolo_heavy_result = detect_using_yolo_heavy(img_path)
            for instance in yolo_heavy_result:
                if instance.name == gt_label:
                    tp_yolo_heavy += 1
                else:
                    fp_yolo_heavy += 1
            annotateAndSave(img_path, yolo_heavy_result)
    
    
    print('*' * 50)
    print('Recall')
    print('*' * 50)
    print(f'YOLO Light: {tp_yolo_light / SAMPLE_SIZE}')
    print(f'YOLO Heavy: {tp_yolo_heavy / SAMPLE_SIZE}')
    print(f'FasterRCNN: {tp_faster_rcnn / SAMPLE_SIZE}')
    print('*' * 50)
    print('Percision')
    print('*' * 50)
    print(f'YOLO Light: {tp_yolo_light / (tp_yolo_light + fp_yolo_light)}')
    print(f'YOLO Heavy: {tp_yolo_heavy / (tp_yolo_heavy + fp_yolo_heavy) }')
    print(f'FasterRCNN: {tp_faster_rcnn / (tp_faster_rcnn + fp_faster_rcnn)}')
    print('*' * 50)
    print('Accuracy')
    print('*' * 50)
    print(f'YOLO Light: {tp_yolo_light / (SAMPLE_SIZE + fp_yolo_light)}')
    print(f'YOLO Heavy: {tp_yolo_heavy / (SAMPLE_SIZE + fp_yolo_heavy) }')
    print(f'FasterRCNN: {tp_faster_rcnn / (SAMPLE_SIZE + fp_faster_rcnn)}')
    print('*' * 50)



if __name__ == '__main__':
    main()
