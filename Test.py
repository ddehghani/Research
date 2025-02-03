import ObjectDetection
import os
import Utils
import random

PATH_TO_SOURCE = '/home/dehghani/EfficientVideoQueryUsingCP/images'
DEST = '/home/dehghani/EfficientVideoQueryUsingCP'

def main():
    counter = 1
    files = os.listdir(PATH_TO_SOURCE)
    while (input('Enter Q to quit: ') != 'Q'):
        
        choice = random.choice(files)
        image = os.path.join(PATH_TO_SOURCE,choice)

        Utils.annotateAndSave(image,ObjectDetection.detect_using_yolo_light([image])[0], DEST, f'{counter}_tiny.jpg')
        Utils.annotateAndSave(image,ObjectDetection.detect_using_faster_rcnn(image), DEST,f'{counter}_faster.jpg')
        Utils.annotateAndSave(image,ObjectDetection.detect_using_yolo_heavy([image])[0], DEST, f'{counter}_tiny.jpg')
        counter += 1

if __name__ == "__main__":
    main()
