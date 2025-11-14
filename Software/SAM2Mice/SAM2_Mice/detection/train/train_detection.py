import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from ultralytics import YOLO

# python SAM2_Mice/detection/train/train_detection.py

model = YOLO(model='SAM2_Mice/detection/train/cfg/yolo11.yaml')
model.load("yolo11l.pt")

train_results = model.train(
    data="SAM2_Mice/detection/train/cfg/Airscope_five_mouse.yaml",  
    epochs=100, 
    imgsz=1024,  
    device="1",  
    amp=True,
    batch=128, 
    workers=64,
    project="SAM2_Mice/detection/train/exp",  
    name="yolo11_five_mouse_exp1",               
)

metrics = model.val()
