from ultralytics import YOLO

model = YOLO(model='SAM2_Mice/detection/train/cfg/yolo11.yaml')
model.load("/home/gpu_0/BBNC/PICO_code/yolo_v11/weights/yolo11l.pt")

train_results = model.train(
    data="SAM2_Mice/detection/train/cfg/Airscope_five_mouse.yaml",  # 数据集配置文件路径
    epochs=100,  # 训练周期数
    imgsz=2048,  # 训练图像尺寸
    device="cuda:0",  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
    amp=False,
    batch=8, 
    workers=64,
)

metrics = model.val()
