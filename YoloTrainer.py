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


# vldb
# introduction
# related work in area
# why doesn't it work
# why is this need (offloading etc.)
# why conformal prediction (more recent citation in more common papers)


# icdm event heatmap

# related work

