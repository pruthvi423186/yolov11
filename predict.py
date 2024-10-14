from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source = "http://192.168.1.12:4747/video", show=True, save=True)
