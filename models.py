from dataclasses import dataclass
from typing import Optional, List
from ultralytics import YOLO as ultralytics_YOLO
import ultralytics_patch
import torch
# from torchvision.io.image import read_image
# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
# import boto3
# from google.cloud import vision
# from azure.ai.vision.imageanalysis import ImageAnalysisClient
# from azure.ai.vision.imageanalysis.models import VisualFeatures
# from azure.core.credentials import AzureKeyCredential

@dataclass(slots=True)
class Instance:
    name: str
    confidence: float
    bbox: list  # xywh format
    probs: Optional[list] = None
    objectness_score: float | None = None

class Model:
    def detect(self, image_path: str) -> list[List[Instance]]:
        raise NotImplementedError("Subclasses must implement the detect method")

class YOLO(Model):
    def __init__(self, model_path: str):
        self.model = ultralytics_YOLO(model_path)

    def detect(self, image_paths, *, batch=8, half=True, imgsz=640, **kwargs) -> list[List[Instance]]:
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            predictions = self.model.predict(
                image_paths, 
                verbose=False, 
                stream=True,
                batch=batch,
                half=half,
                imgsz=640,
                **kwargs)
        
        names = self.model.names
        result = []
        for pred in predictions:
            pred = pred.cpu()
            
            classes = [pred.names[id] for id in pred.boxes.data[:, 5].numpy()]
            confidences = pred.boxes.data[:, 4].numpy()
            bboxes = pred.boxes.data[:, :4].numpy()
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]
            probs = pred.boxes.data[:, 6:]
            instances = [
                Instance(name, conf, bbox, prob.numpy()) 
                for name, conf, bbox, prob, in zip(classes, confidences, bboxes, probs)
            ]
            
            
            
            # boxes = pred.boxes.xywh.numpy()
            # cls_ids = pred.boxes.cls.numpy().astype(int)
            # confs = pred.boxes.conf.numpy()
            # probs = pred.probs.numpy() if pred.probs is not None else [None] * len(boxes)
            # instances = [
            #     Instance(names[cid], conf, box.tolist(), pb)
            #     for box, cid, conf, pb in zip(boxes, cls_ids, confs, probs)
            # ]
            result.append(instances)
            del pred
            torch.cuda.empty_cache()
        
        return result

# class FasterRCNN(Model):
#     def __init__(self):
#         if not hasattr(self, "model"):
#             self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
#             self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.9)
#             self.model.eval()
#             self.preprocess = self.weights.transforms()

#     def detect(self, image_path):
#         img = read_image(image_path)
#         batch = [self.preprocess(img)]
#         prediction = self.model(batch)[0]
#         labels = [self.weights.meta["categories"][i] for i in prediction["labels"]]
#         bboxes = prediction['boxes'].detach().numpy()
#         scores = prediction['scores'].detach().numpy()
#         instances = [Instance(label, score, [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]) for label, bbox, score in zip(labels, bboxes, scores)]
#         return instances

# class MicrosoftDetector(Model):
#     def __init__(self):
#         if not hasattr(self, "client"):
#             self.endpoint = 'https://researchvisionyorku.cognitiveservices.azure.com/'
#             self.key = '861ea6c3e535421faa4217ccbeba4864'
#             self.client = ImageAnalysisClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))

#     def detect(self, image_path):
#         with open(image_path, "rb") as data:
#             result = self.client.analyze(image_data=data.read(), visual_features=[VisualFeatures.OBJECTS], language="en")
#         instances = [Instance(instance['tags'][0]['name'], instance['tags'][0]['confidence'], [instance['boundingBox']['x'], instance['boundingBox']['y'], instance['boundingBox']['w'], instance['boundingBox']['h']]) for instance in result['objectsResult']['values']]
#         return instances

# class AmazonDetector(Model):
#     def __init__(self):
#         if not hasattr(self, "client"):
#             self.client = boto3.client('rekognition')

#     def detect(self, image_path):
#         img = Image.open(image_path)
#         with open(image_path, 'rb') as data:
#             result = self.client.detect_labels(Image={'Bytes': data.read()}, Features=['GENERAL_LABELS'])
#         instances = [Instance(label['Name'], label['Confidence'], [inst['BoundingBox']['Left'] * img.width, inst['BoundingBox']['Top'] * img.height, inst['BoundingBox']['Width'] * img.width, inst['BoundingBox']['Height'] * img.height]) for label in result['Labels'] for inst in label.get('Instances', [])]
#         return instances

# class GoogleDetector(Model):
#     def __init__(self):
#         if not hasattr(self, "client"):
#             self.client = vision.ImageAnnotatorClient()

#     def detect(self, image_path):
#         img = Image.open(image_path)
#         with open(image_path, "rb") as data:
#             image = vision.Image(content=data.read())
#             result = self.client.object_localization(image=image).localized_object_annotations
#         instances = [Instance(inst.name, inst.score, [inst.bounding_poly.normalized_vertices[0].x * img.width, inst.bounding_poly.normalized_vertices[0].y * img.height, (inst.bounding_poly.normalized_vertices[1].x - inst.bounding_poly.normalized_vertices[0].x) * img.width, (inst.bounding_poly.normalized_vertices[3].y - inst.bounding_poly.normalized_vertices[0].y) * img.height]) for inst in result]
#         return instances





