"""
MÓDULO: main.py
PROPÓSITO:
    Capturar audio continuamente, detectar segmentos de voz,
    transcribirlos y enviarlos a la página web.

FLUJO:

    captura continua
          ↓
       Silero VAD
          ↓
    persona habla
          ↓
    persona hace pausa
          ↓
      2 segundos
          ↓
    cerrar segmento
          ↓
      Whisper
          ↓
    enviar caption
          ↓
    continuar capturando
"""

import sys
import os
import time
import requests

from datetime import datetime


# ============================================================
# RUTA DEL PROYECTO
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# CONFIGURACIÓN
# ============================================================

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_VAD_CHUNK_DURATION,
    VAD_MIN_SILENCE_MS,
    DEBUG,
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)


# ============================================================
# MÓDULOS DEL PROYECTO
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
# SERVIDOR WEB
# ============================================================

SERVER_URL = (
    "http://localhost:5000/api/transcribe"
)


# ============================================================
# ENVIAR AL SERVIDOR
# ============================================================

def send_to_server(
    text,
    timestamp,
):
    """
    Envía un caption completo al servidor web.
    """

    try:

        response = requests.post(

            SERVER_URL,

            json={
                "text": text,
                "timestamp": timestamp,
            },

            timeout=2,
        )

        if response.status_code != 200:

            print(
                "⚠️ Error enviando al "
                f"servidor: "
                f"{response.status_code}"
            )

    except requests.exceptions.RequestException as error:

        print(
            "⚠️ No se pudo conectar "
            f"al servidor web: {error}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "TRANSCRIPTOR DE AUDIO "
        "EN TIEMPO REAL"
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
        f"Análisis VAD cada: "
        f"{AUDIO_VAD_CHUNK_DURATION} s"
    )

    print(
        f"Silencio para cerrar segmento: "
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
    # CARGAR WHISPER
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
        "Modelo cargado."
    )

    print(
        "Iniciando captura continua...\n"
    )


    # ========================================================
    # INICIAR CAPTURADOR
    # ========================================================

    capturer = (
        SpeechSegmentCapture()
    )

    capturer.start()


    # ========================================================
    # PROCESAMIENTO
    # ========================================================

    try:

        segment_count = 0

        while True:

            # Esperar siguiente segmento.
            audio = capturer.get_segment(
                timeout=1
            )

            if audio is None:
                continue


            segment_count += 1

            print(
                "\n"
                + "=" * 80
            )

            print(
                f"SEGMENTO #{segment_count}"
            )

            print(
                "Procesando bloque completo..."
            )

            print(
                "=" * 80
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

                texto = transcribe_audio(
                    model,
                    wav_file,
                )


                # ------------------------------------------------
                # RESULTADO
                # ------------------------------------------------

                if texto:

                    timestamp = (
                        datetime.now()
                        .strftime("%H:%M:%S")
                    )

                    print(
                        f"\n[{timestamp}]"
                    )

                    print(
                        texto
                    )

                    send_to_server(
                        texto,
                        timestamp,
                    )

                else:

                    print(
                        "⚠️ No se obtuvo "
                        "texto del segmento."
                    )


            finally:

                # ------------------------------------------------
                # BORRAR WAV TEMPORAL
                # ------------------------------------------------

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
