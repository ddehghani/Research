from PIL import Image
import json
import os
import shutil

""" Crop ms coco images to the bounding boxes given by their annotations """

def main():
    PATH_TO_DATA_SOURCE = '/home/dehghani/EfficientVideoQueryUsingCP/data/coco/train2017'
    PATH_TO_ANNOTATIONS = '/home/dehghani/EfficientVideoQueryUsingCP/data/coco/annotations/instances_train2017.json'
    PATH_TO_DESTINATION = '/home/dehghani/EfficientVideoQueryUsingCP/data/coco/cropped'
    
    if os.path.exists(PATH_TO_DESTINATION):
        shutil.rmtree(PATH_TO_DESTINATION)
    os.makedirs(PATH_TO_DESTINATION)

    with open(PATH_TO_ANNOTATIONS, 'r') as f:
        annotations = json.load(f)
    
    for image in os.listdir(PATH_TO_DATA_SOURCE):
        if image.endswith(".jpg"):
            image_annotations = get_annotations(annotations, image)
            print(image)
            print(image_annotations)
            return
            image_path = os.path.join(PATH_TO_DATA_SOURCE, image)
            image_file = Image.open(image_path)
            for index, (bbox, category) in enumerate(image_annotations):
                bbox[3] += bbox[1]
                bbox[2] += bbox[0]
                cropped_image = image_file.crop(bbox)
                cropped_image.save(os.path.join(PATH_TO_DESTINATION, f'{image[:-4]}_{index}_{category}.jpg'))

    
def get_annotations(annotations: dict, filename: str) -> list[tuple]:

    # get image id
    for image in annotations['images']:
        if image['file_name'] == filename:
            image_id = image['id']
            break
    
    # find all instances(annotated objects) that match 'image_id'
    instances = [instance for instance in annotations['annotations'] if instance['image_id'] == image_id]

    # add category name to each instance
    for instance in instances:
        instance.pop('segmentation')
        instance.pop('area')
        instance.pop('iscrowd')
        instance.pop('image_id')
        instance.pop('id')
        
        for category in annotations['categories']:
            if category['id'] == instance['category_id']:
                instance['category_name'] = category['name']
                instance.pop('category_id')
                break
            
    return instances

def get_category(annotations: dict, category_id: int) -> str:
     for category in annotations['categories']:
        if category['id'] == category_id:
            return category['name']

if __name__ == '__main__':
    main()