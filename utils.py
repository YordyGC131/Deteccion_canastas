# utils.py
#Funciones de dibujo reutilizables.
import cv2
import os
from typing import Tuple
import numpy as np

def draw_boxes_with_confidence(
    img: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    box_color: Tuple[int,int,int] = (0,255,0),
    font_scale: float = 0.5,
    thickness: int = 1
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x1,y1,x2,y2), c in zip(boxes, confs):
        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        cv2.rectangle(img, (x1,y1), (x2,y2), box_color, thickness)
        lbl = f"{c:.2f}"
        (tw,th), base = cv2.getTextSize(lbl, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1-th-base), (x1+tw, y1), box_color, -1)
        cv2.putText(img, lbl, (x1, y1-4), font, font_scale, (0,0,0), thickness)

def draw_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int,int],
    bg_color: Tuple[int,int,int],
    text_color: Tuple[int,int,int] = (255,255,255),
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1,
    thickness: int = 2
) -> None:
    (tw,th), base = cv2.getTextSize(text, font, font_scale, thickness)
    x,y = org
    cv2.rectangle(img, (x, y-th-base), (x+tw, y+base), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
