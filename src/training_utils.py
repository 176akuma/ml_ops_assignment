import json, logging, sqlite3, time
from pathlib import Path

def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mlops"); logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger

def ensure_sqlite(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path); cur = con.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS predictions
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, payload TEXT, prediction REAL)"""
    ); con.commit(); con.close()

def log_prediction(db_path: Path, payload: dict, prediction: float):
    con = sqlite3.connect(db_path); cur = con.cursor()
    cur.execute(
        "INSERT INTO predictions (ts, payload, prediction) VALUES (?, ?, ?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(payload), float(prediction)),
    ); con.commit(); con.close()
