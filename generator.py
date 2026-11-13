"""
generator.py
============
Replays rows from the CIC-IDS-2017 CSV as live network flows
to the Flask-SocketIO server for real-time detection.

Usage:
    python generator.py --csv data/CICIDS2017_subset.csv --speed 1
    python generator.py --speed 2          (2x faster)
    python generator.py --rows 2000        (send 2000 sampled flows)
    python generator.py --rows 0 --loop    (stream all rows forever, chunked)
"""

import pandas as pd
import time
import socketio
import argparse
import random

# ─────────────────────────────────────────────
# Socket.IO client setup
# ─────────────────────────────────────────────
sio = socketio.Client()

@sio.event
def connect():
    print("[INFO] Connected to Flask server.")

@sio.event
def disconnect():
    print("[INFO] Disconnected from server.")

@sio.event
def connect_error(data):
    print(f"[ERROR] Could not connect to server:", data)

# ─────────────────────────────────────────────
# Feature columns to send
# Loaded dynamically from models/feature_keys.json (set by train.py)
# Falls back to original 15 if file not found
# ─────────────────────────────────────────────
import json as _json, os as _os
_FEAT_PATH = _os.path.join('models', 'feature_keys.json')
if _os.path.exists(_FEAT_PATH):
    with open(_FEAT_PATH) as _ff:
        EXPECTED_COLS = _json.load(_ff)['selected_features']
    print(f'[INFO] Generator using {len(EXPECTED_COLS)} research-selected features.')
else:
    EXPECTED_COLS = [
        'Protocol', 'Flow Duration',
        'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
        'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean',
        'Fwd PSH Flags', 'Bwd URG Flags', 'Bwd UGR Flags',
        'Fwd UGR Flags',
    ]
    print('[WARN] models/feature_keys.json not found. Using 15 fallback features.')

# ─────────────────────────────────────────────
# Random IP generator (simulates real IPs)
# ─────────────────────────────────────────────
SAMPLE_IPS = [
    "195.70.2.112","147.40.239.59","37.71.168.237","150.106.58.240",
    "154.60.4.197","104.125.69.45","123.88.174.186","106.219.6.100",
    "117.208.138.18","45.192.31.57","90.182.19.77","20.169.200.162",
    "208.236.84.39","209.79.138.249","10.68.27.141","74.119.30.86",
    "219.95.243.108","91.110.33.85","48.47.88.191","31.159.1.16",
    "104.88.137.108","80.155.101.105","155.243.188.20","168.19.215.206",
    "102.130.156.145","82.16.153.26","152.104.207.244","134.231.153.58",
    "149.87.61.118","31.25.69.24","91.113.9.24",
    "168.170.186.74","126.70.202.51","207.82.168.128","129.18.134.235",
    "188.140.229.18"
]

def random_ip():
    return f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"

# ─────────────────────────────────────────────
# Fast sampled load — never loads full 2.8M rows
# Uses reservoir sampling logic via skiprows
# ─────────────────────────────────────────────
def load_sample(csv_path: str, n_rows: int) -> pd.DataFrame:
    print(f"[INFO] Counting rows in CSV (fast scan)...")
    with open(csv_path, 'r') as f:
        total_lines = sum(1 for _ in f) - 1  # minus header
    print(f"[INFO] CSV has {total_lines:,} data rows. Sampling {n_rows:,}...")

    keep_count = min(n_rows, total_lines)
    skip_indices = []
    kept = 0
    rng = random.Random(42)  # reproducible seed
    for idx in range(1, total_lines + 1):
        remaining = total_lines - idx + 1
        still_need = keep_count - kept
        if still_need <= 0:
            skip_indices.append(idx)
        elif rng.random() < (still_need / remaining):
            kept += 1
        else:
            skip_indices.append(idx)

    df = pd.read_csv(csv_path, skiprows=skip_indices)
    df.columns = df.columns.str.strip()
    return df

# ─────────────────────────────────────────────
# Chunked streaming — for --rows 0 mode
# ─────────────────────────────────────────────
def stream_chunks(csv_path: str, chunk_size: int = 500):
    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        yield chunk

# ─────────────────────────────────────────────
# Emit a single flow row via SocketIO
# ─────────────────────────────────────────────
def emit_flow(row, cols, flow_num: int, total: int):
    flow = {col: row[col] for col in cols if col in row.index}
    flow['src_ip'] = random.choice(SAMPLE_IPS)
    flow['dst_ip'] = random.choice(SAMPLE_IPS)
    sio.emit('flow_data', flow)
    if flow_num % 100 == 0 or flow_num <= 5:
        total_str = f"/{total}" if total else ""
        print(f"[SENT] Flow {flow_num}{total_str}")

# ─────────────────────────────────────────────
# Main replay function
# ─────────────────────────────────────────────
def replay(csv_path: str, server: str = 'http://localhost:5000',
           speed: float = 1.0, loop: bool = False, rows: int = 1000):

    print(f"[INFO] Loading CSV: {csv_path}")

    # Peek at columns without full load — keep RAW names for usecols matching
    peek = pd.read_csv(csv_path, nrows=2)
    raw_cols = list(peek.columns)                        # e.g. [' Label', ' Flow Duration', ...]
    peek.columns = peek.columns.str.strip()              # stripped for our logic

    # Fast row count
    with open(csv_path, 'r') as f:
        total_rows = sum(1 for _ in f) - 1
    print(f"[INFO] CSV loaded. Shape: ({total_rows}, {len(peek.columns)})")

    # Quick label stats — must pass the RAW column name to usecols
    label_col_stripped = next((c for c in peek.columns if 'label' in c.lower()), None)
    label_col_raw      = next((c for c in raw_cols if 'label' in c.lower()), None)
    if label_col_raw:
        benign_count = attack_count = 0
        for chunk in pd.read_csv(csv_path, usecols=[label_col_raw], chunksize=50000):
            chunk.columns = chunk.columns.str.strip()   # strip after load
            vals = chunk[label_col_stripped].str.strip().str.upper()
            benign_count += (vals == 'BENIGN').sum()
            attack_count += (vals != 'BENIGN').sum()
        print(f"[INFO] Dataset: {benign_count:,} benign | {attack_count:,} attacks")

    print(f"[INFO] Connecting to {server} ...")
    # Force WebSocket transport — skips the polling handshake phase entirely.
    # 'Too many packets in payload' error happens during long-polling (the
    # default initial transport). WebSocket mode bypasses polling completely.
    sio.connect(server, wait_timeout=10, transports=['websocket'])

    delay = 0.05 / speed  # base 50ms between flows
    stream_all = (rows == 0)

    if stream_all:
        # ── Chunked full-dataset streaming ──────────
        print(f"[INFO] Streaming ALL {total_rows:,} rows in chunks at {speed}x speed...")
        print(f"[INFO] Estimated time: ~{total_rows * delay / 60:.1f} minutes")
        run = True
        while run:
            flow_num = 0
            cols_ready = False
            usable_cols = EXPECTED_COLS
            for chunk in stream_chunks(csv_path):
                if not cols_ready:
                    missing = [c for c in EXPECTED_COLS if c not in chunk.columns]
                    if missing:
                        print(f"[WARN] Missing columns: {missing}. Filling with zeros.")
                        for c in missing:
                            chunk[c] = 0
                    usable_cols = [c for c in EXPECTED_COLS if c in chunk.columns]
                    cols_ready = True
                for _, row in chunk.iterrows():
                    flow_num += 1
                    emit_flow(row, usable_cols, flow_num, total_rows)
                    time.sleep(delay)
            if not loop:
                run = False
            else:
                print("[INFO] Looping replay...")

    else:
        # ── Sampled mode (default — fast!) ──────────
        actual_rows = min(rows, total_rows)
        print(f"[INFO] Sampling {actual_rows:,} flows from {total_rows:,} total rows...")
        df = load_sample(csv_path, actual_rows)

        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        if missing:
            print(f"[WARN] Missing columns: {missing}. Filling with zeros.")
            for c in missing:
                df[c] = 0
        usable_cols = [c for c in EXPECTED_COLS if c in df.columns]

        est_secs = len(df) * delay
        print(f"[INFO] Starting replay of {len(df):,} sampled flows at {speed}x speed...")
        print(f"[INFO] Estimated time: ~{est_secs:.0f} seconds ({est_secs/60:.1f} min)")

        run = True
        while run:
            for i, (_, row) in enumerate(df.iterrows(), 1):
                emit_flow(row, usable_cols, i, len(df))
                time.sleep(delay)
            if not loop:
                run = False
            else:
                print("[INFO] Looping replay...")

    print("[INFO] Replay complete.")
    sio.disconnect()

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIDS Traffic Generator')
    parser.add_argument('--csv',    default='data/CICIDS2017_subset.csv', help='CSV to replay')
    parser.add_argument('--server', default='http://localhost:5000',       help='Flask server URL')
    parser.add_argument('--speed',  type=float, default=1.0,              help='Replay speed multiplier')
    parser.add_argument('--loop',   action='store_true',                   help='Loop replay forever')
    parser.add_argument('--rows',   type=int,   default=1000,
                        help='Rows to sample and replay (default: 1000). '
                             'Use 0 to stream ALL rows in chunks (very slow on 2.8M rows).')
    args = parser.parse_args()

    replay(args.csv, args.server, args.speed, args.loop, args.rows)
