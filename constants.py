# Dataset paths and constants
COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

VOC_LABELS = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

COCO_PATH = "/data/dehghani/EfficientVideoQueryUsingCP/coco/train2017"
COCO_ANNOTATIONS_PATH = "/data/dehghani/EfficientVideoQueryUsingCP/coco/annotations/instances_train2017.json"

VOC_PATH = "/data/dehghani/EfficientVideoQueryUsingCP/VOCdevkit/VOC2012/JPEGImages"

# IoU threshold for matching two bboxes
IOU_THRESHOLD = 0.5

# Minimum size of the bounding box to be considered
MIN_BBOX_SIZE = 4000

# Constants for cloud correction
CROP_PADDING = 50  # Padding for cropping around in bbox from the original image
ACCEPTANCE_MARGIN = 40 # Margin for accepting a prediction from the cloud

# Constants for packing crops
PACKING_PADDING = 35
GRID_WIDTH, GRID_HEIGHT = 3, 3

# Whether to remove labels from the edge results if the cloud prediction does not include them
REMOVE_LABELS = False