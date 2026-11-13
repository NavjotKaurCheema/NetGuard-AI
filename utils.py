"""
utils.py
========
Database helpers: init, log predictions, fetch metrics for dashboard.
"""

import sqlite3
import json
import os

DB_PATH     = 'nids.db'
METRICS_PATH = os.path.join('models', 'metrics.json')

# ─────────────────────────────────────────────
# Database init
# ─────────────────────────────────────────────
def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT,
            email    TEXT UNIQUE,
            address  TEXT,
            mobile   TEXT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now','localtime')),
            src_ip    TEXT,
            dst_ip    TEXT,
            score     REAL,
            label     TEXT
        )
    """)
    db.commit()
    db.close()

# ─────────────────────────────────────────────
# Log a prediction to DB
# ─────────────────────────────────────────────
def log_prediction(src_ip: str, dst_ip: str, score: float, label: str):
    try:
        db = sqlite3.connect(DB_PATH)
        db.execute(
            "INSERT INTO predictions (src_ip,dst_ip,score,label) VALUES (?,?,?,?)",
            (src_ip, dst_ip, score, label)
        )
        db.commit()
        db.close()
    except Exception as e:
        print(f"[WARN] DB log error: {e}")

# ─────────────────────────────────────────────
# Fetch metrics for the Analysis dashboard
# ─────────────────────────────────────────────
def get_metrics_data() -> dict:
    """Returns training metrics from metrics.json + live DB stats."""

    # Training metrics
    metrics = {}
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0,
            'confusion_matrix': [[0,0],[0,0]],
            'roc': {}, 'pr': {}, 'top_features': []
        }

    # Live DB stats
    try:
        db    = sqlite3.connect(DB_PATH)
        total = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        attacks = db.execute("SELECT COUNT(*) FROM predictions WHERE label='Attack'").fetchone()[0]

        # Top attacking IPs
        top_ips = db.execute("""
            SELECT src_ip, COUNT(*) as hits
            FROM predictions WHERE label='Attack'
            GROUP BY src_ip ORDER BY hits DESC LIMIT 5
        """).fetchall()

        db.close()
        metrics['total_flows']   = total
        metrics['total_attacks'] = attacks
        metrics['top_ips']       = [{'ip': r[0], 'hits': r[1]} for r in top_ips]
    except Exception as e:
        print(f"[WARN] Could not fetch live DB stats: {e}")

    return metrics
