import os
import sys
import ultralytics
from ultralytics import YOLO
model = YOLO("yolov8n.yaml")

# Use the model
results = model.train(data="/Users/user/Documents/GitHub/Food-Detection/dataset/yolo.yaml", epochs=50, pretrained=True, iou=0.5, visualize=True, patience=0)  # train the model
results = model.val()

# need to retrieve the model for future use
# test the model accuracy