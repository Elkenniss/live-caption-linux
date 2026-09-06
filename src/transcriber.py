"""
MÓDULO: transcriber.py
PROPÓSITO:
    Convertir segmentos WAV a texto usando faster-whisper.
"""

import os
import sys

from faster_whisper import WhisperModel


# ============================================================
# RUTA DEL PROYECTO
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# ============================================================
# CONFIGURACIÓN
# ============================================================

from config import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_VAD_FILTER,
    VAD_THRESHOLD,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    WHISPER_BEAM_SIZE,
    WHISPER_TEMPERATURE,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_WORD_TIMESTAMPS,
    WHISPER_NO_SPEECH_THRESHOLD,
)


# ============================================================
# CARGAR MODELO
# ============================================================

def load_model():
    """
    Carga Whisper una sola vez.
    """

    print(
        f"Cargando modelo Whisper "
        f"'{WHISPER_MODEL}' "
        f"en {WHISPER_DEVICE}..."
    )

    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    print(
        "Modelo cargado correctamente."
    )

    return model


# ============================================================
# TRANSCRIBIR
# ============================================================

def transcribe_audio(
    model,
    audio_path,
):
    """
    Transcribe un segmento completo de audio.

    El segmento ya fue separado por VAD antes de llegar aquí.
    """

    if not os.path.exists(
        audio_path
    ):

        raise FileNotFoundError(
            f"El archivo '{audio_path}' "
            f"no existe."
        )

    print(
        f"Transcribiendo: {audio_path}"
    )

    segments, info = model.transcribe(

        audio_path,

        language=WHISPER_LANGUAGE,

        task="transcribe",

        beam_size=WHISPER_BEAM_SIZE,

        temperature=WHISPER_TEMPERATURE,

        vad_filter=WHISPER_VAD_FILTER,

        vad_parameters={
            "threshold": VAD_THRESHOLD,
            "min_silence_duration_ms":
                VAD_MIN_SILENCE_MS,
            "speech_pad_ms":
                VAD_SPEECH_PAD_MS,
        },

        condition_on_previous_text=(
            WHISPER_CONDITION_ON_PREVIOUS_TEXT
        ),

        word_timestamps=(
            WHISPER_WORD_TIMESTAMPS
        ),

        no_speech_threshold=(
            WHISPER_NO_SPEECH_THRESHOLD
        ),
    )

    # ========================================================
    # IMPORTANTE
    # ========================================================
    #
    # faster-whisper devuelve un GENERADOR.
    #
    # Al recorrerlo aquí hacemos que la transcripción se
    # ejecute completamente.
    #

    text_parts = []

    for segment in segments:

        text_parts.append(
            segment.text
        )

    text = " ".join(
        text_parts
    ).strip()

    print(
        f"Idioma detectado: "
        f"{info.language} "
        f"(confianza: "
        f"{info.language_probability:.2f})"
    )

    return text


# ============================================================
# PRUEBA DESDE TERMINAL
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Uso:"
        )

        print(
            "python src/transcriber.py "
            "<archivo.wav>"
        )

        return

    audio_file = sys.argv[1]

    model = load_model()

    try:

        text = transcribe_audio(
            model,
            audio_file,
        )

        print(
            "\n=== TRANSCRIPCIÓN ==="
        )

        print(
            text
        )

        print(
            "====================\n"
        )

    except Exception as error:

        print(
            f"\n❌ Error durante "
            f"la transcripción: {error}"
        )


if __name__ == "__main__":
    main()
