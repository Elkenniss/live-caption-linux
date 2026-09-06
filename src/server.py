"""
MÓDULO: server.py
PROPÓSITO: Servidor web que muestra transcripciones en tiempo real.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os

# Obtener la ruta absoluta del directorio del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

# Almacenar últimas transcripciones
recent_transcriptions = []
max_transcriptions = 50

@app.route('/')
def index():
    """
    Página principal: muestra las transcripciones en tiempo real.
    """
    return render_template('index.html')

@app.route('/api/transcriptions')
def get_transcriptions():
    """
    API: Retorna las últimas transcripciones.
    """
    return jsonify(recent_transcriptions)

@app.route('/api/transcribe', methods=['POST'])
def add_transcription():
    """
    API: Recibe una nueva transcripción desde main.py.
    """
    data = request.json
    text = data.get('text', '')
    timestamp = data.get('timestamp', '')
    
    if text:
        # Agregar a la lista
        recent_transcriptions.append({
            'text': text,
            'timestamp': timestamp
        })
        
        # Mantener solo las últimas N
        if len(recent_transcriptions) > max_transcriptions:
            recent_transcriptions.pop(0)
        
        return jsonify({'status': 'ok'})
    
    return jsonify({'status': 'error', 'message': 'No text provided'}), 400

if __name__ == '__main__':
    print("=" * 60)
    print("SERVIDOR WEB DE TRANSCRIPCIONES")
    print("=" * 60)
    print("Abre tu navegador en: http://localhost:5000")
    print("Presiona Ctrl+C para detener el servidor")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
