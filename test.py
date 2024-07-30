from ultralytics import YOLOv10

model = YOLOv10('train6/weights/best.pt')

model.predict()
