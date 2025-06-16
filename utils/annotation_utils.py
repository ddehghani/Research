from models import Instance
import xml.etree.ElementTree as ET
from pathlib import Path
from constants import MIN_BBOX_SIZE


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

def get_coco_annotations(annotations: dict, filename: str) -> list[Instance]:
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

def get_voc_annotations(image_id: str, annotations_dir: str) -> list[Instance]:
    """Converts XML annotations from VOC format to standard format by extracting bounding boxes."""

    def convert_box( box):
        return [ box[0], box[2], box[1] - box[0], box[3] - box[2]]

    in_file = open(Path(annotations_dir) / f"{image_id}.xml")
    tree = ET.parse(in_file)
    root = tree.getroot()

    instances = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if int(obj.find("difficult").text) != 1:
            xmlbox = obj.find("bndbox")
            bb = convert_box([float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")])
            instances.append(Instance(name=cls, confidence=1.0, bbox=bb))
    return instances
    
