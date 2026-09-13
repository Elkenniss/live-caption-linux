"""
MÓDULO: main.py

PROPÓSITO:
    Coordinar captura continua, VAD, cola, Whisper y servidor web.

FLUJO:

    PipeWire
       ↓
    captura continua
       ↓
    Silero VAD
       ↓
    segmento
       ↓
    queue
       ↓
    Whisper
       ↓
    server.py
       ↓
    navegador
"""

import sys
import os
import time
import requests

from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# RUTA
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# ============================================================
# CONFIG
# ============================================================

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    VAD_MIN_SILENCE_MS,
    DEBUG,
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    SEGMENT_QUEUE_MAXSIZE,
)


# ============================================================
# MÓDULOS
# ============================================================

from audio_capture import (
    SpeechSegmentCapture,
    save_wav,
)

from transcriber import (
    load_model,
    transcribe_audio,
)


# ============================================================
# SERVIDOR
# ============================================================

SERVER_BASE_URL = (
    "http://localhost:5000"
)

SERVER_TRANSCRIBE_URL = (
    f"{SERVER_BASE_URL}/api/transcribe"
)

SERVER_STATUS_URL = (
    f"{SERVER_BASE_URL}/api/status"
)

SERVER_CONTROL_URL = (
    f"{SERVER_BASE_URL}/api/control"
)


# ============================================================
# ZONA HORARIA
# ============================================================

LOCAL_TIMEZONE = ZoneInfo(
    "America/Managua"
)


def get_local_timestamp():
    """
    Devuelve hora local de Nicaragua
    en formato de 12 horas.
    """

    now = datetime.now(
        LOCAL_TIMEZONE
    )

    return now.strftime(
        "%I:%M:%S %p"
    )


# ============================================================
# ENVIAR ESTADO
# ============================================================

def update_server_status(
    state,
    message,
    segment_id=None,
    segment_duration=0.0,
    queue_size=0,
):

    try:

        requests.post(
            SERVER_STATUS_URL,
            json={
                "state": state,
                "message": message,
                "segment_id": segment_id,
                "segment_duration": round(
                    segment_duration,
                    2,
                ),
                "queue_size": queue_size,
                "queue_max": SEGMENT_QUEUE_MAXSIZE,
                "updated_at": get_local_timestamp(),
            },
            timeout=0.5,
        )

    except requests.RequestException:

        # El servidor web puede estar apagado.
        # No debemos detener el transcriptor por esto.
        pass


# ============================================================
# ENVIAR TRANSCRIPCIÓN
# ============================================================

def send_to_server(
    text,
    timestamp,
    duration,
    segment_id,
):

    try:

        response = requests.post(

            SERVER_TRANSCRIBE_URL,

            json={
                "text": text,
                "timestamp": timestamp,
                "duration": round(
                    duration,
                    2,
                ),
                "segment_id": segment_id,
            },

            timeout=2,
        )

        if response.status_code != 200:

            print(
                "⚠️ Error enviando "
                f"al servidor: "
                f"{response.status_code}"
            )

    except requests.RequestException as error:

        print(
            "⚠️ No se pudo conectar "
            f"al servidor web: {error}"
        )


# ============================================================
# LEER CONTROLES DEL NAVEGADOR
# ============================================================

def read_controls():

    try:

        response = requests.get(
            SERVER_CONTROL_URL,
            timeout=0.3,
        )

        if response.status_code != 200:
            return None

        return response.json()

    except requests.RequestException:

        return None


# ============================================================
# APLICAR CONTROLES
# ============================================================

def handle_controls(
    capturer,
):

    controls = read_controls()

    if not controls:
        return

    # --------------------------------------------------------
    # PAUSA
    # --------------------------------------------------------

    if controls.get(
        "paused",
        False,
    ):

        if not capturer.is_paused():

            capturer.pause()

            update_server_status(
                "paused",
                "Escucha pausada por el usuario.",
                queue_size=(
                    capturer.get_queue_size()
                ),
            )

    else:

        if capturer.is_paused():

            capturer.resume()

            update_server_status(
                "listening",
                "Escuchando nuevamente.",
                queue_size=(
                    capturer.get_queue_size()
                ),
            )

    # --------------------------------------------------------
    # CORTE MANUAL
    # --------------------------------------------------------

    if controls.get(
        "force_cut",
        False,
    ):

        capturer.force_cut()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "TRANSCRIPTOR DE AUDIO EN TIEMPO REAL"
    )
    print("=" * 80)

    print(
        f"Sample rate: "
        f"{AUDIO_SAMPLE_RATE} Hz"
    )

    print(
        f"Canales: "
        f"{AUDIO_CHANNELS}"
    )

    print(
        f"Silencio para cerrar: "
        f"{VAD_MIN_SILENCE_MS} ms"
    )

    print("-" * 80)

    print(
        f"Modelo Whisper: "
        f"{WHISPER_MODEL}"
    )

    print(
        f"Idioma: "
        f"{WHISPER_LANGUAGE or 'auto-detect'}"
    )

    print(
        f"Dispositivo: "
        f"{WHISPER_DEVICE}"
    )

    print(
        f"Compute type: "
        f"{WHISPER_COMPUTE_TYPE}"
    )

    print("-" * 80)

    print(
        "🌐 Abre tu navegador en:"
        " http://localhost:5000"
    )

    print(
        "Presiona Ctrl+C para salir"
    )

    print()


    # ========================================================
    # MODELO
    # ========================================================

    print(
        "Cargando modelo Whisper..."
    )

    try:

        model = load_model()

    except Exception as error:

        print(
            "\n❌ No se pudo cargar "
            "Whisper."
        )

        print(
            f"Error: {error}"
        )

        if DEBUG:

            import traceback

            traceback.print_exc()

        return


    print(
        "Modelo cargado correctamente."
    )


    # ========================================================
    # CAPTURADOR
    # ========================================================

    capturer = SpeechSegmentCapture()

    capturer.start()


    # ========================================================
    # ESTADO INICIAL
    # ========================================================

    update_server_status(
        "listening",
        "Escuchando...",
        queue_size=(
            capturer.get_queue_size()
        ),
    )


    segment_id = 0


    # ========================================================
    # LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # CONTROLES DEL NAVEGADOR
            # ------------------------------------------------

            handle_controls(
                capturer
            )


            # ------------------------------------------------
            # LEER SIGUIENTE SEGMENTO
            # ------------------------------------------------

            audio = capturer.get_segment(
                timeout=0.2
            )

            if audio is None:

                if capturer.is_paused():

                    update_server_status(
                        "paused",
                        "Escucha pausada.",
                        queue_size=(
                            capturer.get_queue_size()
                        ),
                    )

                elif capturer.is_speech_active():

                    update_server_status(
                        "listening",
                        "Voz detectada.",
                        segment_duration=(
                            capturer.get_current_duration()
                        ),
                        queue_size=(
                            capturer.get_queue_size()
                        ),
                    )

                else:

                    update_server_status(
                        "listening",
                        "Escuchando...",
                        queue_size=(
                            capturer.get_queue_size()
                        ),
                    )

                continue


            # ------------------------------------------------
            # NUEVO SEGMENTO
            # ------------------------------------------------

            segment_id += 1

            duration = (
                len(audio)
                / AUDIO_SAMPLE_RATE
            )

            timestamp = get_local_timestamp()


            print()
            print(
                "=" * 80
            )

            print(
                f"SEGMENTO #{segment_id}"
            )

            print(
                f"Hora: {timestamp}"
            )

            print(
                f"Duración: "
                f"{duration:.2f} segundos"
            )

            print(
                "=" * 80
            )


            # ------------------------------------------------
            # ESTADO: TRANSCRIBIENDO
            # ------------------------------------------------

            update_server_status(
                "transcribing",
                "Whisper está transcribiendo...",
                segment_id=segment_id,
                segment_duration=duration,
                queue_size=(
                    capturer.get_queue_size()
                ),
            )


            # ------------------------------------------------
            # GUARDAR WAV
            # ------------------------------------------------

            wav_file = save_wav(
                audio
            )


            try:

                # ------------------------------------------------
                # TRANSCRIBIR
                # ------------------------------------------------

                text = transcribe_audio(
                    model,
                    wav_file,
                )


                # ------------------------------------------------
                # RESULTADO
                # ------------------------------------------------

                if text:

                    print()
                    print(
                        f"[{timestamp}] "
                        f"{text}"
                    )


                    send_to_server(
                        text=text,
                        timestamp=timestamp,
                        duration=duration,
                        segment_id=segment_id,
                    )


                    # --------------------------------------------
                    # LISTO / VOLVER A ESCUCHAR
                    # --------------------------------------------

                    update_server_status(
                        "listening",
                        "Escuchando...",
                        queue_size=(
                            capturer.get_queue_size()
                        ),
                    )

                else:

                    print(
                        "⚠️ Segmento sin texto."
                    )

                    update_server_status(
                        "listening",
                        "Escuchando...",
                        queue_size=(
                            capturer.get_queue_size()
                        ),
                    )


            finally:

                if os.path.exists(
                    wav_file
                ):

                    os.remove(
                        wav_file
                    )


    except KeyboardInterrupt:

        print(
            "\n\nPrograma detenido "
            "por el usuario."
        )


    except Exception as error:

        print(
            f"\n❌ Error: {error}"
        )

        update_server_status(
            "error",
            str(error),
            queue_size=(
                capturer.get_queue_size()
            ),
        )

        if DEBUG:

            import traceback

            traceback.print_exc()


    finally:

        capturer.stop()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()
