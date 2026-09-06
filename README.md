# ClaseB - Subtitulador en tiempo real

Proyecto prototipo para clase de developer.

## Objetivo
Capturar el audio de salida del sistema (no el micrófono) y transcribirlo
a texto en tiempo real usando faster-whisper.

## Estructura
- src/main.py: punto de entrada del programa.
- src/audio_capture.py: captura el audio del sistema (PipeWire monitor).
- src/transcriber.py: convierte el audio capturado a texto.
- models/: modelos de Whisper descargados.
- tests/: pruebas individuales de cada módulo.
- docs/: notas y apuntes del proceso de aprendizaje.

## Cómo levantar el entorno (pendiente, próximo paso)
python3 -m venv venv
source venv/bin/activate
pip install faster-whisper sounddevice numpy


# Live Caption Linux

Sistema de transcripción de audio en tiempo real para Linux usando
faster-whisper, CUDA y Voice Activity Detection (VAD).

## Características

- Captura continua de audio desde PipeWire/PulseAudio.
- Detección de voz mediante Silero VAD.
- Segmentación automática por pausas.
- Transcripción mediante faster-whisper.
- Soporte para GPU NVIDIA mediante CUDA.
- Servidor web local para visualizar las transcripciones.
- Configuración centralizada mediante `config.py`.

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
transcriber.py
        ↓
faster-whisper
        ↓
main.py
        ↓
server.py
        ↓
Navegador
