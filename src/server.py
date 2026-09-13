"""
MÓDULO: server.py

PROPÓSITO:
    Servidor web del Live Caption.

FUNCIONES:
    - Mostrar la interfaz.
    - Recibir transcripciones.
    - Mantener historial.
    - Recibir controles manuales.
    - Mostrar estado de captura/Whisper.
"""

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
)

from flask_cors import CORS

import os
import threading


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "templates",
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
)

CORS(app)


# ============================================================
# HISTORIAL
# ============================================================

recent_transcriptions = []

max_transcriptions = 50


# ============================================================
# ESTADO GENERAL
# ============================================================

status = {
    "state": "starting",
    "message": "Iniciando...",
    "segment_id": None,
    "segment_duration": 0.0,
    "queue_size": 0,
    "queue_max": 30,
    "updated_at": "",
}


# ============================================================
# CONTROLES
# ============================================================

controls = {
    "force_cut": False,
    "paused": False,
}

control_lock = threading.Lock()


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# HISTORIAL
# ============================================================

@app.route(
    "/api/transcriptions",
    methods=["GET"],
)
def get_transcriptions():

    return jsonify(
        recent_transcriptions
    )


# ============================================================
# NUEVA TRANSCRIPCIÓN
# ============================================================

@app.route(
    "/api/transcribe",
    methods=["POST"],
)
def add_transcription():

    data = request.get_json(
        silent=True
    ) or {}

    text = data.get(
        "text",
        "",
    )

    timestamp = data.get(
        "timestamp",
        "",
    )

    duration = data.get(
        "duration",
        0,
    )

    segment_id = data.get(
        "segment_id",
        None,
    )

    if not text:

        return jsonify(
            {
                "status": "error",
                "message": "No text provided",
            }
        ), 400

    item = {
        "text": text,
        "timestamp": timestamp,
        "duration": duration,
        "segment_id": segment_id,
    }

    recent_transcriptions.append(
        item
    )

    if (
        len(recent_transcriptions)
        > max_transcriptions
    ):

        recent_transcriptions.pop(0)

    return jsonify(
        {
            "status": "ok"
        }
    )


# ============================================================
# ESTADO
# ============================================================

@app.route(
    "/api/status",
    methods=["GET"],
)
def get_status():

    return jsonify(status)


@app.route(
    "/api/status",
    methods=["POST"],
)
def update_status():

    data = request.get_json(
        silent=True
    ) or {}

    status.update(data)

    return jsonify(
        {
            "status": "ok"
        }
    )


# ============================================================
# CONTROLES
# ============================================================

@app.route(
    "/api/control",
    methods=["POST"],
)
def control():

    data = request.get_json(
        silent=True
    ) or {}

    action = data.get(
        "action"
    )

    with control_lock:

        if action == "force_cut":

            controls["force_cut"] = True

        elif action == "pause":

            controls["paused"] = True

        elif action == "resume":

            controls["paused"] = False

        else:

            return jsonify(
                {
                    "status": "error",
                    "message": "Unknown action",
                }
            ), 400

    return jsonify(
        {
            "status": "ok"
        }
    )


@app.route(
    "/api/control",
    methods=["GET"],
)
def get_control():

    with control_lock:

        current = controls.copy()

        # force_cut es de tipo "evento":
        # una vez leído, se consume.
        controls["force_cut"] = False

    return jsonify(current)


# ============================================================
# SERVIDOR
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "SERVIDOR WEB DE LIVE CAPTION"
    )
    print("=" * 60)

    print(
        "Abre tu navegador en:"
    )

    print(
        "http://localhost:5000"
    )

    print(
        "Presiona Ctrl+C para detener."
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
