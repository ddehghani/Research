from models import Instance
from constants import COCO_LABELS, MIN_BBOX_SIZE 


def filter_annotations(annotations: list) -> list:
    """
    Filters out annotations with bounding boxes smaller than MIN_BBOX_SIZE.
    Supports both flat and nested list structures.
    """
    if not annotations:
        return []
    
    if isinstance(annotations[0], Instance):
        return [ann for ann in annotations if (ann.bbox[2] * ann.bbox[3]) >= MIN_BBOX_SIZE]

    return [filter_annotations(sublist) for sublist in annotations]

def get_annotations(annotations: dict, filename: str) -> list[Instance]:
    """
    Retrieves and standardizes all instances corresponding to a given filename from a COCO-style annotation dict.
    """
    filename = filename.split('/')[-1]
    
    # get image id
    image_id = None
    for image in annotations['images']:
        if image['file_name'] == filename:
            image_id = image['id']
            break
    if image_id is None:
        return []

    # find instances that match this image_id
    instances = [inst for inst in annotations['annotations'] if inst['image_id'] == image_id]

    # attach category names
    standard_instances = []
    for instance in instances:
        for category in annotations['categories']:
            if category['id'] == instance['category_id']:
                instance['category_name'] = category['name']
                break
        standard_instances.append(
            Instance(instance['category_name'].lower(), 1, instance['bbox'])
        )

    return standard_instances

def coco_class_to_id(class_name: str) -> int:
    """
    Converts a class name to its corresponding index in COCO_LABELS.
    """
    return COCO_LABELS.index(class_name)