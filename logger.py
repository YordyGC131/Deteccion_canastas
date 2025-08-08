# logger.py
from datetime import datetime
from config import Config

class EventLogger:
    def __init__(self, path: str = Config.LOG_FILE):
        self.path = path

    def write(self, canasta_id: str, bag_count: int, avg_conf: float, timestamp: datetime = None) -> None:
        ts = timestamp or datetime.now()
        line = (
            f"Fecha:{ts.strftime('%Y-%m-%d %H:%M:%S')};"
            f"Canasta:{canasta_id};"
            f"Bolsas:{bag_count};"
            f"Confianza:{avg_conf:.2f}\n"
        )
        with open(self.path, "a") as f:
            f.write(line)
