"""
CONFIGURACIÓN DEL PROYECTO
Todos los ajustes del proyecto están aquí para facilitar cambios.

Proyecto: Live Caption
Motor: faster-whisper
Plataforma: Linux
"""

# ============================================================
# CONFIGURACIÓN DE AUDIO
# ============================================================

# Dispositivo de audio que se utilizará para capturar el sonido.
#
# En PipeWire/PulseAudio normalmente se utiliza el "monitor"
# de la salida de audio para capturar lo que está reproduciendo
# el sistema.
#
# Para consultar dispositivos:
#   pactl list sources short
#

AUDIO_DEVICE = (
    "alsa_output.usb-GN_Audio_A_S_Jabra_EVOLVE_30_II_000FDF7EB6B209-00."
    "analog-stereo.monitor"
)

# Frecuencia de muestreo.
# Whisper/VAD trabajan normalmente a 16 kHz.
AUDIO_SAMPLE_RATE = 16000

# 1 = Mono
AUDIO_CHANNELS = 1


# ============================================================
# CAPTURA CONTINUA
# ============================================================

# Tamaño de los pequeños bloques que analizamos mientras
# escuchamos continuamente.
#
# IMPORTANTE:
# Esto NO significa que Whisper transcriba cada 0.25 segundos.
#
# Solo significa que el sistema revisa cada 0.25 segundos
# si existe voz o silencio.
#
# Menor:
#   + Detección más rápida
#   - Más comprobaciones
#
# Mayor:
#   + Menos comprobaciones
#   - Puede tardar un poco más en detectar cambios
#

AUDIO_VAD_CHUNK_DURATION = 0.25


# ============================================================
# VOICE ACTIVITY DETECTION (VAD)
# ============================================================

# Activa o desactiva la detección automática de voz.
#
# True:
#   El programa detecta cuándo empieza y termina la voz.
#
# False:
#   No se utilizará el modo de captura por voz.
#

AUDIO_VAD_ENABLED = True


# Umbral de detección de voz.
#
# Silero VAD produce una probabilidad de actividad vocal.
#
# 0.5 = valor equilibrado.
#
# Menor:
#   Más sensible.
#   Puede detectar más ruido/música como voz.
#
# Mayor:
#   Menos sensible.
#   Reduce falsos positivos, pero puede perder voz suave.
#

VAD_THRESHOLD = 0.5


# Duración mínima que debe tener una actividad vocal para
# considerarla como voz real.
#
# 250 ms = 0.25 segundos.
#

VAD_MIN_SPEECH_DURATION_MS = 250


# ============================================================
# ⭐ PARÁMETRO PRINCIPAL DEL LIVE CAPTION
# ============================================================

# Tiempo de SILENCIO necesario para considerar que la persona
# terminó de hablar y cerrar el segmento.
#
# 2000 ms = 2 segundos.
#
# Ejemplo:
#
#   Persona habla durante 8 segundos
#               ↓
#         hace una pausa
#               ↓
#          0.5 segundos
#          1.0 segundos
#          1.5 segundos
#          2.0 segundos
#               ↓
#       FINAL DEL SEGMENTO
#               ↓
#          Whisper procesa
#
# Puedes modificar libremente este valor.
#

VAD_MIN_SILENCE_MS = 2000


# Padding agregado alrededor del habla detectada.
#
# Sirve para evitar cortar demasiado cerca de la primera o
# última palabra.
#
# 200 ms = 0.2 segundos.
#

VAD_SPEECH_PAD_MS = 200


# Pequeña cantidad de audio anterior que se conserva cuando
# empieza a detectarse la voz.
#
# Esto ayuda a no cortar la primera sílaba/palabra.
#

AUDIO_PRE_ROLL_MS = 300


# ============================================================
# LÍMITE DE SEGURIDAD
# ============================================================

# Duración máxima de un segmento hablado.
#
# Normalmente el segmento termina antes gracias al silencio.
#
# Este valor evita que un segmento crezca indefinidamente si
# una persona habla durante muchísimo tiempo sin hacer pausas.
#
# 60 segundos = 1 minuto.
#

AUDIO_MAX_SPEECH_SEGMENT_SECONDS = 60


# ============================================================
# CONFIGURACIÓN DE WHISPER
# ============================================================

# ------------------------------------------------------------
# MODELO WHISPER
# ------------------------------------------------------------
#
# Modelos disponibles:
#
# tiny
# tiny.en
#
# base
# base.en
#
# small
# small.en
#
# medium
# medium.en
#
# large-v1
# large-v2
# large-v3
# large
#
# large-v3-turbo
# turbo
#
# distil-small.en
# distil-medium.en
# distil-large-v2
# distil-large-v3
# distil-large-v3.5
#
# .en = especializado en inglés
# Sin .en = multilingüe
#
# Para tu GTX 1660 SUPER de 6 GB:
#
# medium          -> Muy buen equilibrio
# large-v3        -> Mucho más pesado
# large-v3-turbo  -> Muy buena alternativa para probar después
#

WHISPER_MODEL = "medium"


# ------------------------------------------------------------
# IDIOMA
# ------------------------------------------------------------

# "es"   -> español
# "en"   -> inglés
# None   -> autodetección
#

WHISPER_LANGUAGE = "es"


# ------------------------------------------------------------
# DISPOSITIVO
# ------------------------------------------------------------

# "cuda" -> GPU NVIDIA
# "cpu"  -> CPU
#

WHISPER_DEVICE = "cuda"


# ------------------------------------------------------------
# TIPO DE CÁLCULO
# ------------------------------------------------------------

# GPU NVIDIA:
#
# float16
#   -> Muy buena opción para GPU
#
# int8_float16
#   -> Reduce memoria de GPU
#
#

WHISPER_COMPUTE_TYPE = "float16"


# ============================================================
# VAD DE WHISPER
# ============================================================
#
# Este VAD es una segunda capa de limpieza durante la
# transcripción.
#
# La captura principal ya utiliza VAD para decidir cuándo
# termina cada bloque.
#
# Puedes dejarlo activado.
#

WHISPER_VAD_FILTER = True


# ============================================================
# CONFIGURACIÓN DE TRANSCRIPCIÓN
# ============================================================

# Número de candidatos que Whisper considera al decodificar.
#
# 1  -> más rápido
# 5  -> equilibrio
# 10 -> más procesamiento
#

WHISPER_BEAM_SIZE = 5


# Temperatura de decodificación.
#
# 0.0 = comportamiento determinista.
# Recomendado para Live Caption.
#

WHISPER_TEMPERATURE = 0.0


# Utilizar el texto anterior como contexto.
#

WHISPER_CONDITION_ON_PREVIOUS_TEXT = True


# Generar timestamps por palabra.
#
# False = menor procesamiento.
#

WHISPER_WORD_TIMESTAMPS = False


# Umbral para determinar si Whisper considera que un segmento
# probablemente NO contiene voz.
#

WHISPER_NO_SPEECH_THRESHOLD = 0.6


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

# True:
#   Mostrar información detallada en consola.
#
# False:
#   Ejecución más limpia.
#

DEBUG = True


# ============================================================
# CONTROLES DEL LIVE CAPTION
# ============================================================

# Estado inicial de la captura.
AUDIO_CAPTURE_ENABLED = True

# Cada cuánto tiempo main.py consulta al servidor
# para comprobar si el usuario pulsó un botón.
CONTROL_POLL_INTERVAL = 0.1

# Número máximo de segmentos que pueden esperar
# para ser procesados por Whisper.
#
# Una cola de 30 es un margen bastante amplio para
# una conversación normal.
SEGMENT_QUEUE_MAXSIZE = 30
