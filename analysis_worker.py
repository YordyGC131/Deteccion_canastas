# analysis_worker.py
import threading
import time
import random
import os
import cv2
from config import Config
from logger import EventLogger
from utils import draw_boxes_with_confidence, draw_text
from bag_analyzer import BagAnalyzer
from snapshot_manager import SnapshotManager

class AnalysisWorker(threading.Thread):
    def __init__(
        self,
        snapshots: SnapshotManager,
        stop_event: threading.Event,
        logger: EventLogger
    ):
        super().__init__(daemon=True)
        self.snapshots  = snapshots
        self.stop_event = stop_event
        self.logger     = logger
        self.analyzer   = BagAnalyzer()
        self.fail_counts = {}
        self.pass_counts = {}
        os.makedirs(Config.ERROR_FOLDER, exist_ok=True)

    def run(self):
        while not self.stop_event.is_set():
            paths = [
                os.path.join(Config.RAW_FOLDER, f)
                for f in os.listdir(Config.RAW_FOLDER)
                if f.lower().endswith((".jpg", ".png"))
            ]
            random.shuffle(paths)

            for img_path in paths:
                if self.stop_event.is_set():
                    return
                self._process(img_path)
                time.sleep(random.uniform(*Config.DELAY_RANGE))

            time.sleep(1)

    def _process(self, img_path: str):
        count, avg_conf, boxes, confs = self.analyzer.analyze(img_path)
        passed = (count == 22 and avg_conf >= Config.CONF_THRESH)

        self.fail_counts[img_path] = 0 if passed else self.fail_counts.get(img_path, 0) + 1
        self.pass_counts[img_path] = self.pass_counts.get(img_path, 0) + 1 if passed else 0

        if self.fail_counts[img_path] >= Config.CYCLES:
            self._handle_fail(img_path, count, avg_conf, boxes, confs)

        if self.pass_counts[img_path] >= Config.CYCLES:
            self._handle_pass(img_path, count, avg_conf)

    def _handle_fail(
        self,
        img_path: str,
        count: int,
        avg_conf: float,
        boxes,
        confs
    ):
        img = cv2.imread(img_path)
        draw_boxes_with_confidence(img, boxes, confs)
        draw_text(img, f"Bolsas: {count}", (550,40), bg_color=(0,255,0))
        base = os.path.splitext(os.path.basename(img_path))[0]
        outp = os.path.join(Config.ERROR_FOLDER, f"{base}_{count}.jpg")
        cv2.imwrite(outp, img)

        cid = self.snapshots.raw_id_map.get(img_path, "desconocido")
        self.logger.write(cid, count, avg_conf)

        self.fail_counts[img_path] = 0

    def _handle_pass(self, img_path: str, count: int, avg_conf: float):
        cid = self.snapshots.raw_id_map.get(img_path, "desconocido")
        self.logger.write(cid, count, avg_conf)

        try:
            os.remove(img_path)
        except OSError:
            pass

        for d in (self.fail_counts, self.pass_counts, self.snapshots.raw_id_map):
            d.pop(img_path, None)
