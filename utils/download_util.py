from ultralytics.utils.downloads import download
from pathlib import Path

def download_coco(path: str):
    dir = Path(path)  # dataset root dir
    urls = ["http://images.cocodataset.org/annotations/annotations_trainval2017.zip"] # annotations
    download(urls, dir=dir)
    # Download data
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
        "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
    ]
    download(urls, dir=dir / "images")

def download_voc(path: str):
    dir = Path(path)  # dataset root dir
    urls = ["http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"]
    download(urls, dir=dir)
   
def download_open_images():
    import fiftyone.zoo as foz
    import fiftyone as fo
    import sys
    import os
    
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)
    
    from constants import OPEN_IMAGES_LABELS_MAP as mapping
    
    # Set fraction of the dataset you want to download
    fraction = 0.0001
    
    mapped_classes = [k for k, v in mapping.items() if v is not None]

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        max_samples=int(1743042 * fraction),  # total samples in train split
        classes=mapped_classes,  # specify the classes you want
        overwrite=False
    )

    return dataset