# Initialization
from dataclasses import dataclass
from PIL import Image 
from typing import Optional

# YOLO 
import torch
from ultralytics import YOLO
import Ultralytics_patch

yolo_light = YOLO("yolov5nu.pt")
yolo_heavy = YOLO("yolo11x.pt")

# FasterRCNN
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

# for more model: https://pytorch.org/vision/stable/models.html
# weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
# model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.9)
model.eval()
preprocess = weights.transforms()

# Microsoft
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

microsoft_endpoint = 'https://researchvisionyorku.cognitiveservices.azure.com/'
microsoft_key = '861ea6c3e535421faa4217ccbeba4864'
microsoft_client = ImageAnalysisClient(
    endpoint=microsoft_endpoint,
    credential=AzureKeyCredential(microsoft_key)
)

# Amazon
import boto3
amazon_client = boto3.client('rekognition')

# Google
from google.cloud import vision
google_client = vision.ImageAnnotatorClient()

@dataclass
class Instance:
    name: str
    confidence: float
    bbox: list # xywh format
    probs: Optional[list] = None

def detect_using_yolo_light(image_paths, **kwargs):
    predictions = yolo_light.predict(image_paths, verbose=False, **kwargs)
    result = []
    for pred in predictions:
        classes = [pred.names[id] for id in pred.boxes.data[:, 5].cpu().numpy()]
        confidences = pred.boxes.data[:, 4].cpu().numpy()
        bboxes = pred.boxes.data[:, :4].cpu().numpy() # in xyxy format
        # convert xyxy to xywh format
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]

        probs = pred.boxes.data[:, 6:]
        probs_normalized = (probs / probs.sum(dim=0, keepdim=True)).cpu().numpy()
        
        instances = [ Instance(name, conf, bbox, prob) for name, conf, bbox, prob in zip(classes, confidences, bboxes, probs_normalized)]
        result.append(instances)
    return result

def detect_using_yolo_heavy(image_paths, **kwargs):
    predictions = yolo_heavy.predict(image_paths, verbose=False, **kwargs)[0]
    result = []
    for pred in predictions:
        classes = [pred.names[id] for id in pred.boxes.data[:, 5].cpu().numpy()]
        confidences = pred.boxes.data[:, 4].cpu().numpy()
        bboxes = pred.boxes.data[:, :4].cpu().numpy() # in xyxy format
        # convert xyxy to xywh format
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]

        probs = pred.boxes.data[:, 6:]
        probs_normalized = (probs / probs.sum(dim=0, keepdim=True)).cpu().numpy()
        
        instances = [ Instance(name, conf, bbox, prob) for name, conf, bbox, prob in zip(classes, confidences, bboxes, probs_normalized)]
        result.append(instances)
    return result

# Ultralytics without patch 
# def detect_using_yolo_heavy(image_path, **kwargs):
#     pred = yolo_heavy.predict(image_path, verbose=False, **kwargs)[0]
#     bboxes = pred.boxes.cpu().numpy()
#     classes = [pred.names[id] for id in bboxes.cls]
#     instances = []
#     for cls, conf, box_dim in zip(classes, bboxes.conf, bboxes.xywh):
#         box_dim[0] -= box_dim[2] / 2
#         box_dim[1] -= box_dim[3] / 2
#         instances.append(Instance(name=cls, 
#                                 confidence=conf,
#                                 bbox=box_dim,
#                                 probs = []))
#     return instances

def detect_using_faster_rcnn(image_path):
    img = read_image(image_path)
    batch = [preprocess(img)]
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    bboxes = prediction['boxes'].detach().numpy()
    scores = prediction['scores'].detach().numpy()
    instances = []
   
    for label, bbox, score in zip(labels, bboxes, scores):
        instances.append(Instance(  name=label, 
                                    confidence=score,
                                    bbox=[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3]-bbox[1]]
                                    ))
    return instances

def detect_using_microsoft(image_path):
    with open(image_path, "rb") as data:
        result = microsoft_client.analyze(
            image_data=data.read(),
            visual_features=[VisualFeatures.OBJECTS],
            language="en"
        )
    instances = []
    for instance in result['objectsResult']['values']:
        tag = instance['tags'][0]
        bbox = instance['boundingBox']
        instances.append(Instance(name=tag['name'], 
                                confidence=tag['confidence'],
                                bbox=[bbox['x'], bbox['y'], bbox['w'], bbox['h']]
                                ))
    return instances

def detect_using_amazon(image_path):
    # get image 
    img = Image.open(image_path) 
    with open(image_path, 'rb') as data:
        result = amazon_client.detect_labels(
            Image={'Bytes': data.read(),},
            # MaxLabels=123,
            # MinConfidence=...,
            Features=['GENERAL_LABELS'],
        )
    instances = []
    for label in result['Labels']:
        for instance in label['Instances']:
            instances.append(
                Instance(name = label['Name'], 
                        confidence = label['Confidence'],
                        bbox = [instance['BoundingBox']['Left'] * img.width,
                                instance['BoundingBox']['Top'] * img.height,
                                instance['BoundingBox']['Width'] * img.width,
                                instance['BoundingBox']['Height'] * img.height]
                        )
            )

    return instances

def detect_using_google(image_path):
    # get image 
    img = Image.open(image_path) 
  
    with open(image_path, "rb") as data:
        image = vision.Image(content=data.read())
        result = google_client.object_localization(image=image).localized_object_annotations
    
    instances = []
    for instance in result:
        bpoly = instance.bounding_poly.normalized_vertices
        x = bpoly[0].x * img.width
        y = bpoly[0].y * img.height
        width = (bpoly[1].x - bpoly[0].x) * img.width
        height = (bpoly[3].y - bpoly[0].y) * img.height
        bbox = [x,y,width,height]
        instances.append(Instance(
            name= instance.name,
            confidence= instance.score,
            bbox=bbox
        ))
    return instances
