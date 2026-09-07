# Live Caption Linux

Sistema de **subtítulos / transcripción en tiempo real** para Linux: captura el audio de salida del sistema (PipeWire/PulseAudio), detecta voz con Silero VAD y transcribe con faster-whisper (GPU NVIDIA / CUDA).

## Características

- Captura continua desde el monitor de audio del sistema (no el micrófono)
- Detección de voz con Silero VAD y segmentación por pausas
- Transcripción con faster-whisper
- Soporte GPU NVIDIA (CUDA)
- Servidor web local para ver las transcripciones en el navegador
- Configuración centralizada en `config.py`

## Arquitectura

```text
PipeWire / PulseAudio
        ↓
audio_capture.py
        ↓
Silero VAD
        ↓
Segmento de voz
        ↓
transcriber.py (faster-whisper)
        ↓
main.py → server.py → navegador
```

## Estructura

- `src/main.py` — punto de entrada
- `src/audio_capture.py` — captura PipeWire monitor
- `src/transcriber.py` — audio → texto
- `models/` — modelos Whisper
- `tests/` — pruebas por módulo
- `docs/` — notas del proceso

## Entorno

```bash
python3 -m venv venv
source venv/bin/activate
pip install faster-whisper sounddevice numpy
```

> Prototipo en evolución. Pensado para Linux con PipeWire.
