from dataclasses import dataclass
from typing import Optional, List
from ultralytics import YOLO as ultralytics_YOLO
import ultralytics_patch
import torch

@dataclass(slots=True)
class Instance:
    name: str
    confidence: float
    bbox: list  # xywh format
    probs: Optional[list] = None
    objectness_score: float | None = None

class Model:
    def detect(self, image_paths: list[str], *, batch: int = 8, half: bool = True, imgsz: int = 640, **kwargs) -> list[List[Instance]]:
        raise NotImplementedError("Subclasses must implement the detect method")

class YOLO(Model):
    def __init__(self, model_path: str):
        self.model = ultralytics_YOLO(model_path)

    def detect(self, image_paths: list[str], *, batch: int = 8, half: bool = True, imgsz: int = 640, **kwargs) -> list[List[Instance]]:
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
            
            result.append(instances)
            del pred
            torch.cuda.empty_cache()
        
        return result