# basket_detector.py
#Detecta la región de la canasta vía HSV y contornos.
import cv2
import numpy as np
from config import Config
from typing import Tuple

class BasketDetector:
    def __init__(self):
        self.lower = np.array(Config.HSV_LOWER)
        self.upper = np.array(Config.HSV_UPPER)
        self.kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    def detect(self, frame: np.ndarray) -> Tuple[bool, Tuple[int,int,int,int]]:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kern)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kern)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            if cv2.contourArea(cnt) < Config.MIN_BASKET_AREA:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x,y,w,h = cv2.boundingRect(approx)
                return True, (x,y,w,h)

        return False, (0,0,0,0)
