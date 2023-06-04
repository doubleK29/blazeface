from models.metrics import FaceDetectMetric
import torch
from models.model import YOLOV5n_05, YOLOv5_BlazeFace
from data_loader.FaceLoader import FaceDataLoader
import yaml
from models.metrics import FaceDetectMetric
from utils.autoanchor import check_anchors
from models.loss import LossYolov5

hyp = yaml.load(open("config/hyp.scratch.yaml"), Loader=yaml.FullLoader)
# path = "data/widerface/val"
device = "cuda:0"
# model = YOLOV5n_05(in_channels=3, num_classes=1).to(device)
model = YOLOv5_BlazeFace().to(device)
model.hyp = hyp
model.gr = 1.0
weights = "runs/train/YOLO_exp13/weights/YOLO_exp-best.pt"
path = "data/combine/train/train.txt"
loader = FaceDataLoader(path, 320, 4, 32, workers=2, square=True)
dataset = loader.get_dataset()
check_anchors(dataset, model, thr=4.0, imgsz=320)
