from ultralytics import YOLO # type: ignore
from ultralytics.yolo.v8.detect.predict import DetectionPredictor # type: ignore

import cv2 # type: ignore

model = YOLO("weights/y8best.pt")

results = model.predict(source="demo.mp4", show=True)
print(results)