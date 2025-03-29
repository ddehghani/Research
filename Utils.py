import supervision as sv
import copy
import cv2
import numpy as np
from models import Instance
from PIL import Image
import rpack
import math
from tabulate import tabulate

MIN_BBOX_SIZE = 4000
IOU_THRESHOLD = 0.1
COCO_LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def load_image(path: str):
    return Image.open(path)

def no_overlap(annotations_original):
    annotations = copy.deepcopy(annotations_original)
    no_overlap = []
    while len(annotations) > 0:
        in_group = []
        out_group = []
        in_group.append(annotations[0])
        for ant in annotations[1:]:
            in_group.append(ant) if iou(ant.bbox, in_group[0].bbox) > 0 else out_group.append(ant)
        no_overlap.append(max(in_group, key= lambda i: i.bbox[2] * i.bbox[3], default=Instance('None', 0, [0,0,0,0])))
        annotations = out_group

    return no_overlap

def no_overlap_bulk(annotations: list):
    return [no_overlap(annotation) for annotation in annotations]

def filter_annotations(annotations: list):
    if not annotations:
        return []
    
    if isinstance(annotations[0], Instance):
        return [ann for ann in annotations if (ann.bbox[2] * ann.bbox[3]) >= MIN_BBOX_SIZE]

    return [filter_annotations(sublist) for sublist in annotations]

def iou(boxA_original, boxB_original):
    # convert xywh to xyxy
    boxA = copy.deepcopy(boxA_original)
    boxB = copy.deepcopy(boxB_original)
    boxA[2] += boxA[0]
    boxA[3] += boxA[1]
    boxB[2] += boxB[0]
    boxB[3] += boxB[1]
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou

def contains(boxA, boxB):
    # check if boxA contains boxB
    return  boxA[0] < boxB[0] and boxA[1] < boxB[1] and \
            boxB[0] + boxB[2] <  boxA[0] + boxA[2] and \
            boxB[1] + boxB[3] <  boxA[1] + boxA[3]

def annotateAndSave(source_image_path: str, instances: list, dest_dir_path: str, file_name: str = None):
    if file_name is None:
        file_name = source_image_path.split('/')[-1]
    
    if len(instances) == 0:
        return
    image = cv2.imread(source_image_path)
 
    classes = [abs(hash(instance.name)) for instance in instances]
    confidences = [instance.confidence for instance in instances]
    bboxes = []
    for instance in instances:
        bbox = copy.deepcopy(instance.bbox)
        # convert xywh to xyxy
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        bboxes.append(bbox)
 
    detections = sv.Detections(
        xyxy=np.array(bboxes),
        class_id=np.array(classes),
        confidence=np.array(confidences)
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    with sv.ImageSink(target_dir_path=dest_dir_path,overwrite=False) as sink:
        sink.save_image(image=annotated_frame, image_name = file_name)

def get_annotations(annotations: dict, filename: str) -> list[tuple]:
    filename = filename.split('/')[-1]
    # get image id
    for image in annotations['images']:
        if image['file_name'] == filename:
            image_id = image['id']
            break
    
    # find all instances(annotated objects) that match 'image_id'
    instances = [instance for instance in annotations['annotations'] if instance['image_id'] == image_id]

    # add category name to each instance and standardize
    standard_instances = []
    for instance in instances:
        for category in annotations['categories']:
            if category['id'] == instance['category_id']:
                instance['category_name'] = category['name']
                break
        standard_instances.append(Instance(instance['category_name'].lower(), 1, instance['bbox'] ))

    return standard_instances

def pack(images:list, grid_width: int, grid_height: int, padding_width: float, padding_color: int):
    image_sizes = []
    total_area = 0
    for img_path in images:
        img = Image.open(img_path)
        width , height = img.size
        width += 2 * padding_width
        height += 2 * padding_width
        total_area += width * height
        image_sizes.append((width, height))

    total_area *= 1.70 # accounting for lower packing ratio
    ratio = math.ceil(math.sqrt(total_area / (grid_width * grid_height)))
    positions = rpack.pack(image_sizes, grid_width * ratio, grid_height * ratio)

    # find max width and height
    max_height = 0
    max_width = 0
    for image_size, position in zip(image_sizes, positions):
        max_height = max(max_height, position[1] + image_size[1])
        max_width = max(max_width, position[0] + image_size[0])

    annotations = []
    pack_image = Image.new('RGB', (max_width, max_height))
    for index, img_path in enumerate(images):
        img = Image.open(img_path)
        padded_img = add_padding(img, padding_width, padding_width, padding_width, padding_width, padding_color)
        annotations.append(Instance(name=img_path.split('.')[0].split('_')[-1],
                                    confidence=1,
                                    bbox = np.array([positions[index][0] + padding_width, positions[index][1] + padding_width,img.width, img.height])
                                    ))
        pack_image.paste(padded_img, positions[index])
    
    return annotations, pack_image

def gridify(images:list, grid_width:int, grid_height: int, padding_width: float, padding_color: int):
    # find max heigh and width
    max_height = 0
    max_width = 0
    for img_path in images:
        img = Image.open(img_path)
        max_height = max(img.height, max_height)
        max_width = max(img.width, max_width)
    
    # add some extra padding to avoid images to touch
    max_height = max_height + 2 * padding_width
    max_width = max_width + 2 * padding_width
        
    # add padding and annotations
    annotations = []
    grid_image_size = (max_width * grid_width, max_height * grid_height)
    grid_image = Image.new('RGB', grid_image_size)
    for index, img_path in enumerate(images):
        img = Image.open(img_path)
        top = (max_height - img.height) // 2
        right = (max_width - img.width) // 2
        padded_img = add_padding(img, top, right, top, right, padding_color)
        row_index = index // grid_width
        col_index = index % grid_height
        annotations.append(Instance(name=img_path.split('.')[0].split('_')[-1],
                                    confidence=1,
                                    bbox = np.array([col_index * max_width + right, row_index * max_height + top, img.width, img.height])
                                    ))
        grid_image.paste(padded_img, (col_index * max_width, row_index * max_height))

    return annotations, grid_image

def add_padding(pil_img, top = 10, right = 10, bottom = 10, left = 10, color = 0):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_true_class_probability(annotation, gt_annotations):
    iou_max = -1
    class_name = ''
    for gt_annotation in gt_annotations:
        iou_value = iou(annotation.bbox, gt_annotation.bbox)
        if iou_value > iou_max:
            class_name = gt_annotation.name
            iou_max = iou_value
    # print(f'{class_name}: {iou_max}')
    if iou_max >= IOU_THRESHOLD:
        return annotation.probs[classToId(class_name)]
    else:
        return 0.0
    
def classToId(class_name):
    return COCO_LABELS.index(class_name)

def calculate_performance(model_preds: list, model_names: list, gt_annotations: list, iou_threshold: float = 0.5):
    tp = [0] * len(model_preds)
    fp = [0] * len(model_preds)
    for model_index, model_pred in enumerate(model_preds):
        for instance_index, annotations in enumerate(model_pred):
            for annotation in annotations:
                fp[model_index] += 1 # assume it is fp
                for gt_annotation in gt_annotations[instance_index]:
                    if iou(gt_annotation.bbox, annotation.bbox) > iou_threshold and gt_annotation.name == annotation.name:
                        tp[model_index] += 1 # true positive
                        fp[model_index] -= 1
                        break

    gt_total = sum([len(annotations) for annotations in gt_annotations])

    # names , recal, percision, accuracy 
    data = []
    for index, model_name in enumerate(model_names):
        recall = tp[index] / gt_total if gt_total!= 0 else 0
        precision = tp[index] / (tp[index] + fp[index]) if (tp[index] + fp[index])!= 0 else 0
        accuracy = tp[index] / (gt_total + fp[index]) if (gt_total + fp[index])!= 0 else 0
        data.append([model_name, recall, precision, accuracy])

    print(tabulate(data, headers=['Name', 'Recall', 'Precision', 'Accuracy'], tablefmt='grid'))

def get_prediction_sets(annotations, qhat):
    if not annotations:
        return []
    
    if isinstance(annotations, Instance):
        indexes = 1 - annotations.probs <= qhat
        return [lbl for lbl, index in zip(COCO_LABELS,indexes) if index]
    
    if isinstance(annotations[0], Instance):
        result = []
        for instance in annotations:
            indexes = 1 - instance.probs <= qhat
            result.append([lbl for lbl, index in zip(COCO_LABELS,indexes) if index])
        return result

    return [get_prediction_sets(sublist, qhat) for sublist in annotations]