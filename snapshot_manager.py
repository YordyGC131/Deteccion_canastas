# snapshot_manager.py
import os
import cv2
from datetime import datetime
from config import Config
from typing import Dict
import numpy as np

class SnapshotManager:
    def __init__(self):
        os.makedirs(Config.RAW_FOLDER,   exist_ok=True)
        self.raw_id_map: Dict[str,int] = {}
        self._counter = 0

    @property
    def counter(self) -> int:
        return self._counter

    def save(self, frame: np.ndarray) -> str:
        self._counter += 1
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        name = f"CAN{ts}.jpg"
        path = os.path.join(Config.RAW_FOLDER, name)
        cv2.imwrite(path, frame)
        self.raw_id_map[path] = self._counter
        return path


