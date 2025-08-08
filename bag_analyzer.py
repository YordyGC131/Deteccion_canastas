# bag_analyzer.py
#Encapsula la inferencia YOLO.
import cv2
import numpy as np
from ultralytics import YOLO
from config import Config
from typing import Tuple

class BagAnalyzer:
    def __init__(self, model_path: str = Config.MODEL_PATH):
        self.model = YOLO(model_path)

    def analyze(self, img_path: str) -> Tuple[int, float, np.ndarray, np.ndarray]:
        img = cv2.imread(img_path)
        results = self.model(img, conf=Config.CONF_THRESH)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        count = len(boxes)
        avg_conf = float(confs.mean()) if count else 0.0
        return count, avg_conf, boxes, confs
