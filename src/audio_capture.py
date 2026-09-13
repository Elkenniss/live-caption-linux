"""
MÓDULO: audio_capture.py

PROPÓSITO:
    Capturar audio continuamente desde PipeWire/PulseAudio,
    detectar segmentos de voz mediante Silero VAD y permitir
    control manual desde la interfaz web.

FUNCIONAMIENTO:

    Audio continuo
          ↓
       Silero VAD
          ↓
    ¿hay voz?
      /       \
    NO         SÍ
               ↓
        comenzar segmento
               ↓
        seguir acumulando
               ↓
      silencio de 2 segundos
               ↓
        cerrar segmento
               ↓
          QUEUE
               ↓
          Whisper

CONTROLES MANUALES:

    force_cut()
        Cierra inmediatamente el segmento actual.

    pause()
        Ignora nueva voz hasta que se solicite resume().

    resume()
        Reanuda la detección de voz.

IMPORTANTE:
    "Pausar" no mata parec.
    El proceso de audio continúa abierto para evitar huecos.
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
    SEGMENT_QUEUE_MAXSIZE,
)


# ============================================================
# SILERO VAD
# ============================================================

from faster_whisper.vad import (
    VadOptions,
    get_speech_timestamps,
)


# ============================================================
# CONSTANTES
# ============================================================

BYTES_PER_SAMPLE = 2

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

PRE_ROLL_SAMPLES = int(
    AUDIO_SAMPLE_RATE
    * AUDIO_PRE_ROLL_MS
    / 1000
)


# ============================================================
# CAPTURA CONTINUA
# ============================================================

class SpeechSegmentCapture:
    """
    Capturador continuo con Silero VAD.

    Permite:
        - captura continua
        - detección automática de segmentos
        - corte manual
        - pausa manual
        - reanudación
        - cola de segmentos
    """

    def __init__(self):

        self.process = None
        self.thread = None

        self.stop_event = threading.Event()

        # Control manual.
        self.pause_event = threading.Event()
        self.force_cut_event = threading.Event()

        # Estado actual.
        self.speech_active = False
        self.current_segment_duration = 0.0

        # Lock para estados compartidos.
        self.state_lock = threading.Lock()

        # Cola de segmentos.
        self.segment_queue = queue.Queue(
            maxsize=SEGMENT_QUEUE_MAXSIZE
        )

    # ========================================================
    # INICIAR
    # ========================================================

    def start(self):

        if (
            self.thread
            and self.thread.is_alive()
        ):
            return

        print(
            "🎤 Iniciando captura continua desde:"
        )

        print(
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
                "Verifica que PulseAudio/PipeWire "
                "esté instalado."
            ) from error

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )

        self.thread.start()

    # ========================================================
    # ESTADO
    # ========================================================

    def is_paused(self):
        return self.pause_event.is_set()

    def is_speech_active(self):

        with self.state_lock:
            return self.speech_active

    def get_current_duration(self):

        with self.state_lock:
            return self.current_segment_duration

    def get_queue_size(self):

        return self.segment_queue.qsize()

    # ========================================================
    # PAUSAR
    # ========================================================

    def pause(self):
        """
        Pausa la detección de nuevos segmentos.

        parec sigue abierto.
        """

        self.pause_event.set()

        with self.state_lock:

            self.speech_active = False
            self.current_segment_duration = 0.0

        print(
            "🔵 Captura pausada por el usuario."
        )

    # ========================================================
    # REANUDAR
    # ========================================================

    def resume(self):
        """
        Reanuda la detección de voz.

        Se limpia cualquier segmento parcialmente acumulado.
        """

        self.force_cut_event.clear()

        self.pause_event.clear()

        with self.state_lock:

            self.speech_active = False
            self.current_segment_duration = 0.0

        print(
            "🟢 Captura reanudada."
        )

    # ========================================================
    # CORTE MANUAL
    # ========================================================

    def force_cut(self):
        """
        Solicita que el segmento actual termine inmediatamente.
        """

        self.force_cut_event.set()

        print(
            "🔴 Solicitud de corte manual recibida."
        )

    # ========================================================
    # BUCLE DE CAPTURA
    # ========================================================

    def _capture_loop(self):

        pre_roll_buffer = np.array(
            [],
            dtype=np.float32,
        )

        segment_buffer = np.array(
            [],
            dtype=np.float32,
        )

        speech_active = False

        max_segment_samples = int(
            AUDIO_SAMPLE_RATE
            * AUDIO_MAX_SPEECH_SEGMENT_SECONDS
        )

        vad_options = VadOptions(
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=(
                VAD_MIN_SPEECH_DURATION_MS
            ),
            min_silence_duration_ms=(
                VAD_MIN_SILENCE_MS
            ),
            speech_pad_ms=(
                VAD_SPEECH_PAD_MS
            ),
        )

        print(
            "✅ Captura continua iniciada."
        )

        while not self.stop_event.is_set():

            # =================================================
            # LEER AUDIO
            # =================================================

            try:

                raw_data = (
                    self.process.stdout.read(
                        CHUNK_BYTES
                    )
                )

            except Exception as error:

                print(
                    f"⚠️ Error leyendo audio: {error}"
                )

                break

            if not raw_data:
                break

            audio_int16 = np.frombuffer(
                raw_data,
                dtype=np.int16,
            )

            if audio_int16.size == 0:
                continue

            audio_float = (
                audio_int16.astype(
                    np.float32
                )
                / 32768.0
            )

            # =================================================
            # PAUSA MANUAL
            # =================================================

            if self.pause_event.is_set():

                # No acumulamos absolutamente nada.
                speech_active = False

                segment_buffer = np.array(
                    [],
                    dtype=np.float32,
                )

                pre_roll_buffer = np.array(
                    [],
                    dtype=np.float32,
                )

                with self.state_lock:

                    self.speech_active = False
                    self.current_segment_duration = 0.0

                continue

            # =================================================
            # CORTE MANUAL
            # =================================================

            if self.force_cut_event.is_set():

                if (
                    speech_active
                    and len(segment_buffer) > 0
                ):

                    print(
                        "🔴 Segmento cerrado "
                        "manualmente."
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

                self.force_cut_event.clear()

                with self.state_lock:

                    self.speech_active = False
                    self.current_segment_duration = 0.0

                continue

            # =================================================
            # SI NO HAY VAD
            # =================================================

            if not AUDIO_VAD_ENABLED:

                segment_buffer = np.concatenate(
                    (
                        segment_buffer,
                        audio_float,
                    )
                )

                if (
                    len(segment_buffer)
                    >= max_segment_samples
                ):

                    self._finish_segment(
                        segment_buffer
                    )

                    segment_buffer = np.array(
                        [],
                        dtype=np.float32,
                    )

                continue

            # =================================================
            # TODAVÍA NO HAY VOZ
            # =================================================

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

                if len(pre_roll_buffer) < 512:
                    continue

                speech = get_speech_timestamps(
                    pre_roll_buffer,
                    vad_options,
                    sampling_rate=AUDIO_SAMPLE_RATE,
                )

                if speech:

                    speech_active = True

                    segment_buffer = (
                        pre_roll_buffer.copy()
                    )

                    print(
                        "🗣️ Voz detectada. "
                        "Comenzando segmento..."
                    )

                with self.state_lock:

                    self.speech_active = speech_active

                    self.current_segment_duration = (
                        len(segment_buffer)
                        / AUDIO_SAMPLE_RATE
                    )

                continue

            # =================================================
            # ESTAMOS GRABANDO UN SEGMENTO
            # =================================================

            segment_buffer = np.concatenate(
                (
                    segment_buffer,
                    audio_float,
                )
            )

            current_duration = (
                len(segment_buffer)
                / AUDIO_SAMPLE_RATE
            )

            with self.state_lock:

                self.speech_active = True
                self.current_segment_duration = (
                    current_duration
                )

            # =================================================
            # LÍMITE DE SEGURIDAD
            # =================================================

            if (
                len(segment_buffer)
                >= max_segment_samples
            ):

                print(
                    "⚠️ Segmento alcanzó el "
                    f"límite de "
                    f"{AUDIO_MAX_SPEECH_SEGMENT_SECONDS} "
                    "segundos."
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

                with self.state_lock:

                    self.speech_active = False
                    self.current_segment_duration = 0.0

                continue

            # =================================================
            # ANALIZAR SILENCIO
            # =================================================

            analysis_window_seconds = max(
                4.0,
                (
                    VAD_MIN_SILENCE_MS
                    / 1000.0
                ) + 1.0,
            )

            analysis_window_samples = int(
                AUDIO_SAMPLE_RATE
                * analysis_window_seconds
            )

            analysis_audio = (
                segment_buffer[
                    -analysis_window_samples:
                ]
            )

            if len(analysis_audio) < 512:
                continue

            speech = get_speech_timestamps(
                analysis_audio,
                vad_options,
                sampling_rate=AUDIO_SAMPLE_RATE,
            )

            # =================================================
            # VOZ DETECTADA
            # =================================================

            if speech:

                last_speech_end = speech[-1]["end"]

                silence_samples = (
                    len(analysis_audio)
                    - last_speech_end
                )

                silence_ms = (
                    silence_samples
                    / AUDIO_SAMPLE_RATE
                    * 1000
                )

                # =================================================
                # FIN AUTOMÁTICO
                # =================================================

                if (
                    silence_ms
                    >= VAD_MIN_SILENCE_MS
                ):

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

                    with self.state_lock:

                        self.speech_active = False
                        self.current_segment_duration = 0.0

            # =================================================
            # NO HAY VOZ
            # =================================================

            else:

                silence_duration = (
                    len(analysis_audio)
                    / AUDIO_SAMPLE_RATE
                    * 1000
                )

                if (
                    silence_duration
                    >= VAD_MIN_SILENCE_MS
                ):

                    print(
                        "🔇 No se detectó voz. "
                        "Finalizando segmento."
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

                    with self.state_lock:

                        self.speech_active = False
                        self.current_segment_duration = 0.0

        # =====================================================
        # CERRAR SEGMENTO PENDIENTE
        # =====================================================

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

    def _finish_segment(
        self,
        audio,
    ):

        if audio is None:
            return

        if len(audio) == 0:
            return

        try:

            final_options = VadOptions(
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=(
                    VAD_MIN_SPEECH_DURATION_MS
                ),
                min_silence_duration_ms=(
                    VAD_MIN_SILENCE_MS
                ),
                speech_pad_ms=(
                    VAD_SPEECH_PAD_MS
                ),
            )

            speech = get_speech_timestamps(
                audio,
                final_options,
                sampling_rate=AUDIO_SAMPLE_RATE,
            )

        except Exception as error:

            print(
                f"⚠️ Error durante VAD final: "
                f"{error}"
            )

            speech = []

        # =====================================================
        # RECORTAR SILENCIO
        # =====================================================

        if speech:

            start = speech[0]["start"]

            end = speech[-1]["end"]

            audio = audio[start:end]

        minimum_samples = int(
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
    # AÑADIR A COLA
    # ========================================================

    def _put_segment(
        self,
        audio,
    ):

        while not self.stop_event.is_set():

            try:

                self.segment_queue.put(
                    audio,
                    timeout=0.5,
                )

                return

            except queue.Full:

                print(
                    "⚠️ Cola de segmentos llena. "
                    "Esperando espacio..."
                )

    # ========================================================
    # OBTENER SEGMENTO
    # ========================================================

    def get_segment(
        self,
        timeout=None,
    ):

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

def save_wav(
    audio_array,
    filename=None,
):
    """
    Guarda audio float32 [-1,1] como WAV int16.
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
# PRUEBA DIRECTA
# ============================================================

def main():

    print("=" * 80)
    print(
        "CAPTURADOR CONTINUO + SILERO VAD"
    )
    print("=" * 80)

    print(
        f"Fuente: {AUDIO_DEVICE}"
    )

    print(
        f"Sample rate: "
        f"{AUDIO_SAMPLE_RATE} Hz"
    )

    print(
        f"Chunk VAD: "
        f"{AUDIO_VAD_CHUNK_DURATION} s"
    )

    print(
        f"Umbral VAD: "
        f"{VAD_THRESHOLD}"
    )

    print(
        f"Silencio para terminar: "
        f"{VAD_MIN_SILENCE_MS} ms"
    )

    print(
        f"Cola máxima: "
        f"{SEGMENT_QUEUE_MAXSIZE}"
    )

    print("=" * 80)

    capturer = SpeechSegmentCapture()

    capturer.start()

    try:

        segment_count = 0

        while True:

            audio = capturer.get_segment(
                timeout=1
            )

            if audio is None:
                continue

            segment_count += 1

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"SEGMENTO #{segment_count}"
            )

            print(
                "=" * 60
            )

            duration = (
                len(audio)
                / AUDIO_SAMPLE_RATE
            )

            print(
                f"Duración: "
                f"{duration:.2f} segundos"
            )

            filename = save_wav(
                audio
            )

            print(
                f"Archivo WAV: "
                f"{filename}"
            )

    except KeyboardInterrupt:

        print(
            "\nCaptura detenida."
        )

    finally:

        capturer.stop()


if __name__ == "__main__":
    main()
