from PIL import Image
import json
import os
import shutil
import math
from tqdm import tqdm
from utils import get_annotations, classToId, annotateAndSave, add_padding
import supervision as sv
import numpy as np
import cv2
import rpack

# Define the paths to the data source and annotations
PATH_TO_SOURCE = '/data/dehghani/EfficientVideoQueryUsingCP/coco/train2017'
PATH_TO_ANNOTATIONS = '/data/dehghani/EfficientVideoQueryUsingCP/coco/annotations/instances_train2017.json'
PATH_TO_DESTINATION = '/data/dehghani/EfficientVideoQueryUsingCP/coco/packed'

# Config constants
PADDING = 20 # padding arround each bbox in pixels
PACK_SIZE = 9
GRID_WIDTH = 3
GRID_HEIGHT = 3
PACKING_PADDING = 15

image_counter = 0
def main():
    # list all the files in the directory
    files = os.listdir(PATH_TO_SOURCE)
    # remove the destination directory if it exists and create a new one
    path_to_images = os.path.join(PATH_TO_DESTINATION, 'images')
    path_to_labels = os.path.join(PATH_TO_DESTINATION, 'labels')
    if os.path.exists(PATH_TO_DESTINATION):
        shutil.rmtree(PATH_TO_DESTINATION)
    os.makedirs(PATH_TO_DESTINATION)
    os.makedirs(os.path.join(PATH_TO_DESTINATION, 'train/images'))
    os.makedirs(os.path.join(PATH_TO_DESTINATION, 'val/images'))
    os.makedirs(os.path.join(PATH_TO_DESTINATION, 'train/labels'))
    os.makedirs(os.path.join(PATH_TO_DESTINATION, 'val/labels'))

    # load the annotations from the json file
    with open(PATH_TO_ANNOTATIONS, 'r') as f:
        annotations = json.load(f)
    
    # initialize an empty list to store the cropped images and their corresponding labels
    cropped_images = []

    # loop through each file
    for file in tqdm(files):
        # check if the file is an image file
        if file.endswith('.jpg') or file.endswith('.png'):
            image_annotations = get_annotations(annotations, file)
            image_file = Image.open(os.path.join(PATH_TO_SOURCE, file)) # open the image for processing 
            # annotateAndSave(os.path.join(PATH_TO_SOURCE, file), image_annotations, '/home/dehghani/EfficientVideoQueryUsingCP/', 'original.jpg')
            for instance in image_annotations:
                bbox = instance.bbox
                # add padding to the bounding box but save the orignal bbox for label
                # label format: class x_center y_center width height (has to be shifted and normalized later)
                x_offset = max(bbox[0] - PADDING, 0) - bbox[0]
                y_offset = max(bbox[1] - PADDING, 0) - bbox[1]
                x = bbox[0] + x_offset
                y = bbox[1] + y_offset
                width = min(-x_offset + bbox[2] + PADDING, image_file.width - x)
                height = min(-y_offset + bbox[3] + PADDING, image_file.height - y)
                image_label = [ classToId(instance.name),
                                bbox[2]/2 - x_offset,
                                bbox[3]/2 - y_offset,
                                bbox[2],
                                bbox[3]
                              ]
                try:
                    cropped_image = image_file.crop([x, y, x+width, y+height])
                    cropped_images.append((cropped_image, image_label))
                    if len(cropped_images) == PACK_SIZE:
                        generate_packed_image(cropped_images)
                        cropped_images.clear()
                    
                except Exception as e:
                    print(e)
                    cropped_images.clear()
                    print(f'an error occured while processing image:{file}')
        
            # class x_center y_center width height (normalized)

def generate_packed_image(cropped_images: list):
    global image_counter
    # calculate total area and find the max width and height of packed image
    image_sizes = []
    total_area = 0
    for img,_ in cropped_images:
        width , height = img.size
        width += 2 * PACKING_PADDING
        height += 2 * PACKING_PADDING
        total_area += width * height
        image_sizes.append((width, height))

    total_area *= 1.50 # accounting for lower packing ratio
    ratio = math.ceil(math.sqrt(total_area / (GRID_WIDTH * GRID_HEIGHT)))
    positions = rpack.pack(image_sizes, GRID_WIDTH * ratio, GRID_HEIGHT * ratio)

    # find packed image's actual width and height
    packed_height = 0
    packed_width = 0
    for image_size, position in zip(image_sizes, positions):
        packed_height = max(packed_height, position[1] + image_size[1])
        packed_width = max(packed_width, position[0] + image_size[0])

    # paste cropped images onto the packed image and save it as a new image
    pack_image = Image.new('RGB', (packed_width, packed_height))
    train_val_dir = 'train/' if image_counter % 10 < 8 else 'val/'
    with open(os.path.join(PATH_TO_DESTINATION, train_val_dir, f"labels/pack{image_counter}.txt"), "w") as f:
        for index, (img, label) in enumerate(cropped_images):
            padded_img = add_padding(img, PACKING_PADDING, PACKING_PADDING, PACKING_PADDING, PACKING_PADDING, 0)
            label[1] += positions[index][0] + PACKING_PADDING
            label[2] += positions[index][1] + PACKING_PADDING
            label[1] /= pack_image.width
            label[2] /= pack_image.height
            label[3] /= pack_image.width
            label[4] /= pack_image.height
            f.write('{}\t{:.6f} {:.6f} {:.6f} {:.6f}\n'.format(*label))

            pack_image.paste(padded_img, positions[index])
    

    pack_image.save(os.path.join(PATH_TO_DESTINATION, train_val_dir, f'images/pack{image_counter}.jpg'))
    
    image_counter += 1
    # uncomment to view image
    # image = cv2.imread('/data/dehghani/temp/images/pack0.jpg')
    # detections = sv.Detections(
    #     xyxy=np.array([[label[1] - label[3]/2,
    #             label[2] - label[4]/2,
    #             label[1] + label[3]/2,
    #             label[2] + label[4]/2,
    #             ] for _,label in cropped_images]),
    #     class_id=np.array([label[0] for _,label in cropped_images]),
    #     confidence=np.array([1 for _ in cropped_images])
    #     )

    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = box_annotator.annotate(
    #         scene=image.copy(),
    #         detections=detections
    #     )
    # with sv.ImageSink(target_dir_path=PATH_TO_DESTINATION,overwrite=False) as sink:
    #     sink.save_image(image=annotated_frame, image_name = f'pack_annotated.jpg')
    

if __name__ == '__main__':
    main()