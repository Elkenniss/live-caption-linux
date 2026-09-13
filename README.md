# Live Caption Linux

Sistema de **subtítulos / transcripción en tiempo real** para Linux: captura el audio de salida del sistema (PipeWire/PulseAudio), detecta voz con Silero VAD y transcribe con faster-whisper (GPU NVIDIA / CUDA).

> **Proyecto abierto a la comunidad.** Estoy empezando y quiero construir esto *con* otras personas. Si lo pruebas, tienes una idea, viste un bug o quieres mejorar algo: **abre un issue, deja un comentario o manda un PR**. No hace falta ser experto — las opiniones también cuentan.

## Cómo puedes ayudar

- Probarlo en tu distro y contar qué pasó
- Sugerir mejoras de UX, rendimiento o arquitectura
- Reportar bugs (con OS, GPU y pasos si puedes)
- Mejorar docs, tests o código
- Charlar ideas en [Discussions](https://github.com/Elkenniss/live-caption-linux/discussions) o por [issue](https://github.com/Elkenniss/live-caption-linux/issues/new/choose)

Guía rápida: [CONTRIBUTING.md](CONTRIBUTING.md)

> Nota: a veces voy un paso adelante en local antes de subir. Si algo no cuadra con lo último del README, pregunta en un issue — lo alineamos.

## Características

- Captura continua desde el monitor de audio del sistema (no el micrófono)
- Detección de voz con Silero VAD y segmentación por pausas
- Transcripción con faster-whisper
- Soporte GPU NVIDIA (CUDA)
- Servidor web local para ver las transcripciones en el navegador
- Controles en la UI: pausar, reanudar y cortar el segmento actual
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

> Prototipo en evolución. Pensado para Linux con PipeWire. Ideas para Windows u otras plataformas también son bienvenidas.

## Contacto del maintainer

- GitHub: [@Elkenniss](https://github.com/Elkenniss)
- Email: [donalitomontenegro@gmail.com](mailto:donalitomontenegro@gmail.com)
- Portafolio: https://elkenniss.github.io/portafolio/
