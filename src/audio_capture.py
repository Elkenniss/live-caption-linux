"""
MÓDULO: audio_capture.py
PROPÓSITO:
    Capturar audio continuamente desde PipeWire/PulseAudio y
    crear segmentos de voz automáticamente usando Silero VAD.

FUNCIONAMIENTO:

    Audio continuo
          ↓
    pequeños bloques
          ↓
       Silero VAD
          ↓
      ¿hay voz?
       /     \
     NO       SÍ
              ↓
       comenzar segmento
              ↓
        seguir acumulando
              ↓
       2 segundos sin voz
              ↓
        cerrar segmento
              ↓
        enviar a la cola
              ↓
          Whisper

VENTAJA:
La captura continúa mientras Whisper procesa otro segmento.
"""

import os
import sys
import time
import wave
import queue
import threading
import subprocess

import numpy as np


# ============================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


from config import (
    AUDIO_DEVICE,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_VAD_CHUNK_DURATION,
    AUDIO_VAD_ENABLED,
    VAD_THRESHOLD,
    VAD_MIN_SPEECH_DURATION_MS,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    AUDIO_PRE_ROLL_MS,
    AUDIO_MAX_SPEECH_SEGMENT_SECONDS,
)


# ============================================================
# SILERO VAD DE FASTER-WHISPER
# ============================================================

from faster_whisper.vad import (
    VadOptions,
    get_speech_timestamps,
)


# ============================================================
# UTILIDADES
# ============================================================

BYTES_PER_SAMPLE = 2  # int16
PRE_ROLL_SAMPLES = int(
    AUDIO_SAMPLE_RATE * AUDIO_PRE_ROLL_MS / 1000
)

CHUNK_SAMPLES = max(
    512,
    int(
        AUDIO_SAMPLE_RATE
        * AUDIO_VAD_CHUNK_DURATION
    )
)

CHUNK_SAMPLES = (
    CHUNK_SAMPLES // 512
) * 512

CHUNK_BYTES = (
    CHUNK_SAMPLES
    * AUDIO_CHANNELS
    * BYTES_PER_SAMPLE
)


class SpeechSegmentCapture:
    """
    Capturador continuo de audio con detección automática de voz.

    La captura ocurre en un hilo independiente para que
    Whisper pueda procesar segmentos sin detener la captura.
    """

    def __init__(self):
        self.process = None
        self.thread = None

        self.stop_event = threading.Event()

        # Cola donde se depositan los segmentos terminados.
        self.segment_queue = queue.Queue(maxsize=5)

    # ========================================================
    # INICIAR
    # ========================================================

    def start(self):
        """
        Inicia parec y el hilo de captura.
        """

        if self.thread and self.thread.is_alive():
            return

        print(
            f"🎤 Iniciando captura continua desde:\n"
            f"   {AUDIO_DEVICE}"
        )

        cmd = [
            "parec",
            "--raw",
            f"--rate={AUDIO_SAMPLE_RATE}",
            f"--channels={AUDIO_CHANNELS}",
            "--format=s16le",
            f"-d={AUDIO_DEVICE}",
        ]

        try:

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )

        except FileNotFoundError as error:
            raise RuntimeError(
                "No se encontró 'parec'. "
                "Verifica que PulseAudio/PipeWire esté instalado."
            ) from error

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )

        self.thread.start()

    # ========================================================
    # BUCLE DE CAPTURA
    # ========================================================

    def _capture_loop(self):
        """
        Captura continuamente desde parec.
        """

        audio_buffer = np.array(
            [],
            dtype=np.float32,
        )

        pre_roll_buffer = np.array(
            [],
            dtype=np.float32,
        )

        speech_active = False

        segment_buffer = np.array(
            [],
            dtype=np.float32,
        )

        max_segment_samples = int(
            AUDIO_SAMPLE_RATE
            * AUDIO_MAX_SPEECH_SEGMENT_SECONDS
        )

        # Configuración VAD
        vad_options = VadOptions(
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )

        print(
            "✅ Captura continua iniciada."
        )

        while not self.stop_event.is_set():

            # ------------------------------------------------
            # LEER AUDIO
            # ------------------------------------------------

            try:

                raw_data = self.process.stdout.read(
                    CHUNK_BYTES
                )

            except Exception:
                break

            if not raw_data:
                break

            audio_int16 = np.frombuffer(
                raw_data,
                dtype=np.int16,
            )

            if audio_int16.size == 0:
                continue

            # Convertir int16 → float32 [-1, 1]
            audio_float = (
                audio_int16.astype(np.float32)
                / 32768.0
            )

            audio_buffer = np.concatenate(
                (audio_buffer, audio_float)
            )

            # ------------------------------------------------
            # MODO SIN VAD
            # ------------------------------------------------

            if not AUDIO_VAD_ENABLED:

                self._put_segment(
                    audio_buffer.copy()
                )

                audio_buffer = np.array(
                    [],
                    dtype=np.float32,
                )

                continue

            # ------------------------------------------------
            # MANTENER PRE-ROLL
            # ------------------------------------------------

            if not speech_active:

                pre_roll_buffer = np.concatenate(
                    (
                        pre_roll_buffer,
                        audio_float,
                    )
                )

                if (
                    len(pre_roll_buffer)
                    > PRE_ROLL_SAMPLES
                ):
                    pre_roll_buffer = (
                        pre_roll_buffer[
                            -PRE_ROLL_SAMPLES:
                        ]
                    )

                # Solo analizamos el buffer corto
                # para detectar el inicio de voz.
                analysis_audio = pre_roll_buffer

                if len(analysis_audio) < 512:
                    continue

                speech = get_speech_timestamps(
                    analysis_audio,
                    vad_options,
                    sampling_rate=AUDIO_SAMPLE_RATE,
                )

                if speech:

                    speech_active = True

                    # Comenzar segmento incluyendo
                    # el pequeño pre-roll.
                    segment_buffer = (
                        pre_roll_buffer.copy()
                    )

                    # Limpiar audio acumulado general.
                    audio_buffer = np.array(
                        [],
                        dtype=np.float32,
                    )

                    print(
                        "🗣️ Voz detectada. "
                        "Comenzando segmento..."
                    )

                continue

            # ------------------------------------------------
            # YA ESTAMOS DENTRO DE UN SEGMENTO
            # ------------------------------------------------

            segment_buffer = np.concatenate(
                (
                    segment_buffer,
                    audio_float,
                )
            )

            # ------------------------------------------------
            # SEGURIDAD: SEGMENTO MUY LARGO
            # ------------------------------------------------

            if (
                len(segment_buffer)
                >= max_segment_samples
            ):

                print(
                    "⚠️ Segmento alcanzó el límite "
                    f"de {AUDIO_MAX_SPEECH_SEGMENT_SECONDS} s."
                )

                self._finish_segment(
                    segment_buffer
                )

                segment_buffer = np.array(
                    [],
                    dtype=np.float32,
                )

                pre_roll_buffer = np.array(
                    [],
                    dtype=np.float32,
                )

                speech_active = False

                continue

            # ------------------------------------------------
            # VENTANA DE ANÁLISIS
            # ------------------------------------------------
            #
            # Para saber si llevamos ~2 segundos sin voz,
            # no necesitamos analizar todo el segmento.
            #
            # Solo observamos una ventana reciente.
            #

            vad_window_seconds = max(
                4.0,
                (VAD_MIN_SILENCE_MS / 1000.0) + 1.0,
            )

            vad_window_samples = int(
                AUDIO_SAMPLE_RATE
                * vad_window_seconds
            )

            analysis_audio = segment_buffer[
                -vad_window_samples:
            ]

            if len(analysis_audio) < 512:
                continue

            # ------------------------------------------------
            # EJECUTAR VAD
            # ------------------------------------------------

            speech = get_speech_timestamps(
                analysis_audio,
                vad_options,
                sampling_rate=AUDIO_SAMPLE_RATE,
            )

            # ------------------------------------------------
            # COMPROBAR SILENCIO
            # ------------------------------------------------

            if speech:

                last_speech = speech[-1]

                last_speech_end = last_speech[
                    "end"
                ]

                silence_samples = (
                    len(analysis_audio)
                    - last_speech_end
                )

                silence_ms = (
                    silence_samples
                    / AUDIO_SAMPLE_RATE
                    * 1000
                )

                if silence_ms >= VAD_MIN_SILENCE_MS:

                    print(
                        f"🔇 {silence_ms:.0f} ms "
                        "sin voz detectada."
                    )

                    self._finish_segment(
                        segment_buffer
                    )

                    segment_buffer = np.array(
                        [],
                        dtype=np.float32,
                    )

                    pre_roll_buffer = np.array(
                        [],
                        dtype=np.float32,
                    )

                    speech_active = False

            else:

                # No hay voz en la ventana reciente.
                #
                # Si el segmento ya tiene suficiente duración,
                # consideramos que terminó.

                if (
                    len(segment_buffer)
                    >= (
                        AUDIO_SAMPLE_RATE
                        * VAD_MIN_SILENCE_MS
                        / 1000
                    )
                ):

                    print(
                        "🔇 Segmento terminado "
                        "por ausencia de voz."
                    )

                    self._finish_segment(
                        segment_buffer
                    )

                    segment_buffer = np.array(
                        [],
                        dtype=np.float32,
                    )

                    pre_roll_buffer = np.array(
                        [],
                        dtype=np.float32,
                    )

                    speech_active = False

        # ----------------------------------------------------
        # FINAL DEL HILO
        # ----------------------------------------------------

        if (
            speech_active
            and len(segment_buffer) > 0
        ):
            self._finish_segment(
                segment_buffer
            )

    # ========================================================
    # FINALIZAR SEGMENTO
    # ========================================================

    def _finish_segment(self, audio):
        """
        Limpia silencios extremos y añade el segmento a la cola.
        """

        if audio is None:
            return

        if len(audio) == 0:
            return

        # Ejecutar VAD una última vez para localizar
        # exactamente la zona de voz.
        try:

            final_options = VadOptions(
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=VAD_MIN_SPEECH_DURATION_MS,
                min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                speech_pad_ms=VAD_SPEECH_PAD_MS,
            )

            speech = get_speech_timestamps(
                audio,
                final_options,
                sampling_rate=AUDIO_SAMPLE_RATE,
            )

        except Exception as error:

            print(
                f"⚠️ Error ejecutando VAD final: {error}"
            )

            speech = []

        if speech:

            start = speech[0]["start"]
            end = speech[-1]["end"]

            audio = audio[start:end]

        # Evitar mandar segmentos diminutos.
        minimum_samples = int(
            AUDIO_SAMPLE_RATE
            * VAD_MIN_SPEECH_DURATION_MS()
        ) if False else int(
            AUDIO_SAMPLE_RATE
            * VAD_MIN_SPEECH_DURATION_MS
            / 1000
        )

        if len(audio) < minimum_samples:
            return

        duration = (
            len(audio)
            / AUDIO_SAMPLE_RATE
        )

        print(
            f"📝 Segmento listo: "
            f"{duration:.2f} segundos"
        )

        self._put_segment(
            audio.copy()
        )

    # ========================================================
    # PONER EN COLA
    # ========================================================

    def _put_segment(self, audio):
        """
        Añade un segmento a la cola.
        """

        while not self.stop_event.is_set():

            try:

                self.segment_queue.put(
                    audio,
                    timeout=0.5,
                )

                return

            except queue.Full:

                continue

    # ========================================================
    # OBTENER SEGMENTO
    # ========================================================

    def get_segment(self, timeout=None):
        """
        Obtiene el siguiente segmento de voz.

        Retorna:
            numpy.ndarray
        """

        try:

            return self.segment_queue.get(
                timeout=timeout
            )

        except queue.Empty:

            return None

    # ========================================================
    # DETENER
    # ========================================================

    def stop(self):
        """
        Detiene captura y libera parec.
        """

        print(
            "\n🛑 Deteniendo captura..."
        )

        self.stop_event.set()

        if self.process:

            try:
                self.process.terminate()

            except Exception:
                pass

        if self.thread:

            self.thread.join(
                timeout=2
            )

        print(
            "✅ Captura detenida."
        )


# ============================================================
# GUARDAR WAV
# ============================================================

def save_wav(audio_array, filename=None):
    """
    Guarda un arreglo numpy como archivo WAV.
    """

    if filename is None:

        import tempfile

        fd, filename = tempfile.mkstemp(
            suffix=".wav",
            prefix="audio_segment_",
        )

        os.close(fd)

    audio_int16 = (
        np.clip(
            audio_array,
            -1.0,
            1.0,
        )
        * 32767
    ).astype(np.int16)

    with wave.open(
        filename,
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            AUDIO_CHANNELS
        )

        wav_file.setsampwidth(
            BYTES_PER_SAMPLE
        )

        wav_file.setframerate(
            AUDIO_SAMPLE_RATE
        )

        wav_file.writeframes(
            audio_int16.tobytes()
        )

    return filename

# ============================================================
# PRUEBA DIRECTA DEL MÓDULO
# ============================================================

def main():
    """
    Permite probar únicamente la captura.
    """

    print("=" * 80)
    print("CAPTURADOR CONTINUO + SILERO VAD")
    print("=" * 80)

    print(
        f"Fuente: {AUDIO_DEVICE}"
    )

    print(
        f"Sample rate: {AUDIO_SAMPLE_RATE} Hz"
    )

    print(
        f"Chunk VAD: {AUDIO_VAD_CHUNK_DURATION} s"
    )

    print(
        f"Umbral VAD: {VAD_THRESHOLD}"
    )

    print(
        f"Silencio para cortar: "
        f"{VAD_MIN_SILENCE_MS} ms"
    )

    print("=" * 80)

    capturer = SpeechSegmentCapture()

    capturer.start()

    try:

        count = 0

        while True:

            audio = capturer.get_segment(
                timeout=1
            )

            if audio is None:
                continue

            count += 1

            print(
                f"\n--- SEGMENTO #{count} ---"
            )

            filename = save_wav(
                audio
            )

            duration = (
                len(audio)
                / AUDIO_SAMPLE_RATE
            )

            print(
                f"Duración: {duration:.2f} s"
            )

            print(
                f"WAV: {filename}"
            )

    except KeyboardInterrupt:

        print(
            "\nPrueba detenida."
        )

    finally:

        capturer.stop()


if __name__ == "__main__":
    main()
