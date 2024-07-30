import os
from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('jameslahm/yolov10x')

model.train(data='/home/demulab/follow_me_dataset2024/data.yaml',
            epochs=500,
            batch=10,
            imgsz=700,
            single_cls=True,
            save_dir='/home/demulab/follow_me2024_training')
