import os
import json
import time
import traceback
import threading

import pandas as pd
import numpy as np
import sqlite3
import geoip2.database

from flask import Flask, render_template, request, redirect, url_for, jsonify, session, g
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from threading import Lock
from functools import wraps
from utils import init_db, log_prediction, get_metrics_data
from inference import predict_flow, set_detector_params

# ─────────────────────────────────────────────
# Flask & SocketIO Setup
# ─────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-key-please-change')
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=1e8,
    ping_timeout=60,
    ping_interval=25,
    engineio_logger=False,
    logger=False
)
init_db()

# ─────────────────────────────────────────────
# GeoIP Setup
# ─────────────────────────────────────────────
GEOIP_DB = 'GeoLite2-City.mmdb'
try:
    geoip_reader = geoip2.database.Reader(GEOIP_DB)
    print(f"[INFO] GeoIP DB loaded: {GEOIP_DB}")
except Exception:
    geoip_reader = None
    print("[WARN] GeoIP DB not found. Map features disabled.")

# ─────────────────────────────────────────────
# Auth Helpers
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        name     = data.get('name', '').strip()
        email    = data.get('email', '').strip()
        address  = data.get('address', '').strip()
        mobile   = data.get('mobile', '').strip()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        if not all([name, email, username, password]):
            return render_template('register.html', error='All fields required.')
        hashed = generate_password_hash(password)
        db = sqlite3.connect('nids.db')
        try:
            db.execute(
                "INSERT INTO users (name,email,address,mobile,username,password) VALUES (?,?,?,?,?,?)",
                (name, email, address, mobile, username, hashed)
            )
            db.commit()
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists.')
        finally:
            db.close()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        db = sqlite3.connect('nids.db')
        row = db.execute("SELECT id,username,password FROM users WHERE username=?", (username,)).fetchone()
        db.close()
        if row and check_password_hash(row[2], password):
            session['user_id']   = row[0]
            session['username']  = row[1]
            return redirect(url_for('prediction'))
        return render_template('login.html', error='Invalid credentials.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/model_evaluation')
def model_evaluation():
    return render_template('model_evaluation.html')

@app.route('/prediction')
@login_required
def prediction():
    return render_template('prediction.html', username=session.get('username'))

@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html', username=session.get('username'))

@app.route('/ingest_flow', methods=['POST'])
def ingest_flow():
    """
    Receives a flow dict from generator.py via plain HTTP POST.
    Runs inference, logs to DB, and broadcasts result via Socket.IO.
    No Socket.IO client version issues this way.
    """
    try:
        data = request.get_json(force=True)
        result = predict_flow(data)
        src_ip = data.get('src_ip', '0.0.0.0')
        dst_ip = data.get('dst_ip', '0.0.0.0')

        # GeoIP lookup
        geo = {}
        if geoip_reader:
            try:
                r   = geoip_reader.city(src_ip)
                geo = dict(lat=r.location.latitude, lon=r.location.longitude,
                           city=r.city.name or '', country=r.country.name or '')
            except Exception:
                pass

        payload = dict(
            src_ip=src_ip, dst_ip=dst_ip,
            score=round(result['score'], 3),
            label=result['label'],
            shap=result.get('shap', []),
            geo=geo,
            timestamp=time.strftime('%H:%M:%S')
        )
        log_prediction(src_ip, dst_ip, payload['score'], payload['label'])
        # Broadcast to all connected browser clients
        socketio.emit('flow_result', payload)
        return jsonify(payload)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

# ─────────────────────────────────────────────
# REST API
# ─────────────────────────────────────────────
@app.route('/set_params', methods=['POST'])
@login_required
def set_params():
    data = request.get_json(force=True)
    alpha  = float(data.get('alpha',  0.5))
    beta   = float(data.get('beta',   0.4))
    thresh = float(data.get('thresh', 0.5))
    set_detector_params(alpha, beta, thresh)
    print(f"[PARAMS] Updated: {data}")
    return jsonify(status='ok', alpha=alpha, beta=beta, thresh=thresh)

@app.route('/get_metrics')
@login_required
def get_metrics():
    return jsonify(get_metrics_data())

@app.route('/get_history')
@login_required
def get_history():
    db  = sqlite3.connect('nids.db')
    rows = db.execute(
        "SELECT timestamp, src_ip, dst_ip, score, label FROM predictions ORDER BY id DESC LIMIT 200"
    ).fetchall()
    db.close()
    return jsonify([
        dict(timestamp=r[0], src_ip=r[1], dst_ip=r[2], score=r[3], label=r[4])
        for r in rows
    ])

# ─────────────────────────────────────────────
# SocketIO Events
# ─────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    print("[SOCKET] Client connected.")

@socketio.on('disconnect')
def on_disconnect():
    print("[SOCKET] Client disconnected.")

@socketio.on('flow_data')
def on_flow(data):
    """
    Receives a flow dict from generator.py,
    runs inference, logs, and broadcasts result.
    """
    try:
        result = predict_flow(data)
        src_ip = data.get('src_ip', '0.0.0.0')
        dst_ip = data.get('dst_ip', '0.0.0.0')

        # GeoIP lookup
        geo = {}
        if geoip_reader:
            try:
                r   = geoip_reader.city(src_ip)
                geo = dict(lat=r.location.latitude, lon=r.location.longitude,
                           city=r.city.name or '', country=r.country.name or '')
            except Exception:
                pass

        payload = dict(
            src_ip=src_ip, dst_ip=dst_ip,
            score=round(result['score'], 3),
            label=result['label'],
            shap=result.get('shap', []),
            geo=geo,
            timestamp=time.strftime('%H:%M:%S')
        )
        log_prediction(src_ip, dst_ip, payload['score'], payload['label'])
        socketio.emit('flow_result', payload)
    except Exception as e:
        socketio.emit('flow_error', {'error': str(e)})
        traceback.print_exc()

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("[INFO] NIDS Flask-SocketIO Server Running at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
