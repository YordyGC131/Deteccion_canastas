# capture_worker.py
import cv2
from config import Config
from basket_detector import BasketDetector
from snapshot_manager import SnapshotManager
import threading

class CaptureWorker:
    def __init__(
        self,
        snapshots: SnapshotManager,
        stop_event: threading.Event
    ):
        self.snapshots  = snapshots
        self.detector   = BasketDetector()
        self.stop_event = stop_event

    def run(self):
        cap = cv2.VideoCapture(Config.VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = int(1000 / fps)
        prev_found = False

        print(f"→ FPS: {fps:.2f}, delay/frame: {delay} ms")
        print("▶️ Iniciando captura de canastas…")

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            found, bbox = self.detector.detect(frame)

            if found and not prev_found:
                self.snapshots.save(frame)

            prev_found = found

            if found:
                x,y,w,h = bbox
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

            cv2.putText(
                frame,
                f"Canastas: {self.snapshots.counter}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,255,255), 2, cv2.LINE_AA
            )

            cv2.imshow("Detección de canastas", frame)
            if cv2.waitKey(delay) & 0xFF == 27:
                self.stop_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()
