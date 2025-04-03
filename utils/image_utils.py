from PIL import Image
import cv2
import numpy as np
import supervision as sv

def load_image(path: str):
    """
    Loads an image from the specified file path.

    Parameters:
    - path: Path to the image file

    Returns:
    - A PIL.Image object representing the loaded image
    """
    return Image.open(path)

def add_padding(pil_img, top=10, right=10, bottom=10, left=10, color=0):
    """
    Adds padding around a PIL image.

    Parameters:
    - pil_img: The original image
    - top, right, bottom, left: Amount of padding to add on each side (in pixels)
    - color: Background color of the padding

    Returns:
    - A new PIL.Image object with padding applied
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def annotateAndSave(source_image_path: str, instances: list, dest_dir_path: str, file_name: str = None):
    """
    Annotates an image with bounding boxes from a list of instances and saves the annotated image.

    Parameters:
    - source_image_path: Path to the original image
    - instances: List of instance objects containing name, confidence, and bbox attributes
    - dest_dir_path: Directory where the annotated image will be saved
    - file_name: Optional custom filename for the saved image (defaults to original image name)

    Returns:
    - None
    """
    if file_name is None:
        file_name = source_image_path.split('/')[-1]
    
    if len(instances) == 0:
        return

    image = cv2.imread(source_image_path)

    classes = [abs(hash(instance.name)) for instance in instances]
    confidences = [instance.confidence for instance in instances]
    bboxes = []
    for instance in instances:
        bbox = instance.bbox.copy()
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

    with sv.ImageSink(target_dir_path=dest_dir_path, overwrite=False) as sink:
        sink.save_image(image=annotated_frame, image_name=file_name)