# main.py
#Orquesta todos los módulos e inicia los workers.
import threading
from logger import EventLogger
from snapshot_manager import SnapshotManager
from analysis_worker import AnalysisWorker
from capture_worker import CaptureWorker

def main():
    stop_event = threading.Event()
    logger     = EventLogger()
    snapshots  = SnapshotManager()

    analysis = AnalysisWorker(snapshots, stop_event, logger)
    analysis.start()

    capture = CaptureWorker(snapshots, stop_event)
    capture.run()

    analysis.join()

if __name__ == "__main__":
    main()
