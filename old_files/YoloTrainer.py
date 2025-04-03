from ultralytics import YOLO

model = YOLO("yolo11n.pt")

train_results = model.train(
            data="/home/dehghani/EfficientVideoQueryUsingCP/pack.yaml", 
            batch=8,
            epochs=3,
            workers=1,
            device='cuda:0')

metrics = model.val()

path = model.export()

print(path)
