# config.py
#Centraliza toda la configuración como constantes y tipos.
from typing import Tuple

class Config:
    VIDEO_PATH: str           = "VID-20250314-WA0009.mp4"
    MODEL_PATH: str           = "bestcanasta.pt"
    RAW_FOLDER: str           = "raw_snapshots"
    ERROR_FOLDER: str         = "error_images"
    LOG_FILE: str             = "event_log.txt"
    CYCLES: int               = 5
    CONF_THRESH: float        = 0.25
    DELAY_RANGE: Tuple[float, float] = (0.1, 0.5)
    MIN_BASKET_AREA: int      = 10000
    HSV_LOWER: Tuple[int,int,int] = (100, 150, 50)
    HSV_UPPER: Tuple[int,int,int] = (140, 255, 255)
