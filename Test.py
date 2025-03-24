import ObjectDetection
import os
import Utils
import random
from PIL import Image

PATH_TO_SOURCE = '/data/dehghani/EfficientVideoQueryUsingCP/coco/cropped/'
DEST = '/home/dehghani/EfficientVideoQueryUsingCP'

def main():
    MIN_SIZE = 2000
    MAX_SIZE = 10000
    counter = 1
    files = os.listdir(PATH_TO_SOURCE)
    while (input('Enter Q to quit: ') != 'Q'):
        
        # choice = random.choice(files)
        choices = []

        while (len(choices) < 9):
            path = os.path.join(PATH_TO_SOURCE, random.choice(files))
            img = Image.open(path)
            size = img.width * img.height
            if size > MIN_SIZE and size < MAX_SIZE:
                choices.append(path)
        
        _, image_grid = Utils.gridify(choices, 3, 3, 5, 0)
        _, image_pack = Utils.pack(choices, 3, 3, 5, 0)
        grid_path = os.path.join(DEST, f"image_grid_{counter}.jpg")
        pack_path = os.path.join(DEST, f"image_pack_{counter}.jpg")
        image_grid.save(grid_path)
        image_pack.save(pack_path)
        # image = os.path.join(PATH_TO_SOURCE,choice)

        # Utils.annotateAndSave(image,ObjectDetection.detect_using_yolo_light([image])[0], DEST, f'{counter}_tiny.jpg')
        # Utils.annotateAndSave(image,ObjectDetection.detect_using_faster_rcnn(image), DEST,f'{counter}_faster.jpg')
        # Utils.annotateAndSave(image,ObjectDetection.detect_using_yolo_heavy([image])[0], DEST, f'{counter}_heavy.jpg')

        Utils.annotateAndSave(pack_path,ObjectDetection.detect_using_yolo_light([pack_path])[0], DEST, f'pack_{counter}_tiny.jpg')
        Utils.annotateAndSave(pack_path,ObjectDetection.detect_using_faster_rcnn(pack_path), DEST,f'pack_{counter}_faster.jpg')
        Utils.annotateAndSave(pack_path,ObjectDetection.detect_using_yolo_heavy([pack_path])[0], DEST, f'pack_{counter}_heavy.jpg')

        Utils.annotateAndSave(grid_path,ObjectDetection.detect_using_yolo_light([grid_path])[0], DEST, f'grid_{counter}_tiny.jpg')
        Utils.annotateAndSave(grid_path,ObjectDetection.detect_using_faster_rcnn(grid_path), DEST,f'grid_{counter}_faster.jpg')
        Utils.annotateAndSave(grid_path,ObjectDetection.detect_using_yolo_heavy([grid_path])[0], DEST, f'grid_{counter}_heavy.jpg')
        
        counter += 1

if __name__ == "__main__":
    main()
