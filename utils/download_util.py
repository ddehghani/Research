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
    download(urls, dir=dir / "images", threads=3)

def download_voc(path: str):
    dir = Path(path)  # dataset root dir
    urls = ["http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"]
    download(urls, dir=dir)
   
def download_open_images():
    import fiftyone.zoo as foz
    import fiftyone as fo

    # Set fraction of the dataset you want to download
    fraction = 0.01

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        max_samples=int(1743042 * fraction),  # total samples in train split
        classes=['Aircraft', 'Airplane', 'Ambulance', 'Apple', 'Backpack', 
                 'Banana', 'Bat (Animal)', 'Bed', 'Bicycle', 'Bicycle helmet', 
                 'Bicycle wheel', 'Billiard table', 'Bird', 'Bottle', 'Boy', 
                 'Bus', 'Car', 'Chair', 'Chicken', 'Coffee cup', 'Coffee table', 
                 'Computer monitor', 'Duck', 'Eagle', 'Flower', 'Football', 
                 'Football helmet', 'Girl', 'Golf cart', 'Handbag', 'Helmet', 
                 'Kitchen & dining room table', 'Laptop', 'Luggage and bags', 'Man', 
                 'Motorcycle', 'Mug', 'Owl', 'Parrot', 'Penguin', 'Person', 'Sofa bed', 
                 'Stationary bicycle', 'Stool', 'Suitcase', 'Swan', 'Table', 'Television', 
                 'Truck', 'Van', 'Vehicle', 'Volleyball (Ball)', 'Wheel', 'Woman'],
        overwrite=False
    )

    return dataset
