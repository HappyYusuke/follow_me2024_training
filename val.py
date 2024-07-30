from ultralytics import YOLOv10

model = YOLOv10('train6/weights/best.pt')

model.val(data='/home/demulab/follow_me_dataset2024/data.yaml',
            batch=16)
