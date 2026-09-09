# Signex — Manual completo (técnico y práctico)

> **Para qué sirve este documento.** Referencia exhaustiva de cómo está
> construido y cómo se usa el proyecto: arquitectura, cada módulo, los tres
> modelos con sus arquitecturas y números reales de entrenamiento, el pipeline
> de datos completo (capturar → entrenar → evaluar), la aplicación web, y el
> uso práctico día a día (atajos, flujos, pantallas, tiempos). Compilado
> directamente del código y la configuración vigentes.
>
> Estado descrito: 2026-08-15. Para una explicación sin jerga técnica ver
> [`docs/GUIA_SIMPLE.md`](GUIA_SIMPLE.md); para la visión de producto y su
> propósito social ver [`docs/IDEA.md`](IDEA.md); para el inventario funcional
> de la interfaz web ver [`docs/DESIGN_BRIEF.md`](DESIGN_BRIEF.md).

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Stack tecnológico](#2-stack-tecnológico)
3. [Arquitectura general](#3-arquitectura-general)
4. [El pipeline en tiempo real, frame a frame](#4-el-pipeline-en-tiempo-real-frame-a-frame)
5. [Normalización de landmarks — las fórmulas exactas](#5-normalización-de-landmarks--las-fórmulas-exactas)
6. [Los tres modelos de reconocimiento](#6-los-tres-modelos-de-reconocimiento)
7. [Segunda etapa: traducción glosa → frase (LLM)](#7-segunda-etapa-traducción-glosa--frase-llm)
8. [Voz de entrada y salida](#8-voz-de-entrada-y-salida)
9. [Capa de presentación (overlay, buffers, HUD)](#9-capa-de-presentación-overlay-buffers-hud)
10. [Registro de conversación exportable](#10-registro-de-conversación-exportable)
11. [`config.py` — referencia completa de parámetros](#11-configpy--referencia-completa-de-parámetros)
12. [Pipeline capturar → entrenar → evaluar](#12-pipeline-capturar--entrenar--evaluar)
13. [Estructura completa de archivos](#13-estructura-completa-de-archivos)
14. [La aplicación web](#14-la-aplicación-web)
15. [Uso práctico — atajos, flujos y tiempos](#15-uso-práctico--atajos-flujos-y-tiempos)
16. [Estado actual (números verificados)](#16-estado-actual-números-verificados)
17. [Limitaciones conocidas](#17-limitaciones-conocidas)
18. [Decisiones técnicas tomadas](#18-decisiones-técnicas-tomadas)
19. [Pendientes / roadmap](#19-pendientes--roadmap)
20. [Equipo y contexto](#20-equipo-y-contexto)

---

## 1. Resumen ejecutivo

Signex es una aplicación de visión por computadora que reconoce Lenguaje de
Señas Americano (ASL) en tiempo real a partir de una cámara común, sin
guantes ni sensores especiales, y lo convierte en texto y voz — y en sentido
inverso, transcribe la voz de la persona oyente a subtítulos. Existe en dos
superficies que comparten exactamente los mismos modelos y el mismo cálculo:
una **aplicación web** (`web/`, sin backend, el producto público) y una
**aplicación de escritorio** (`main.py`, en Python, la herramienta de trabajo
del equipo con la que se capturan y entrenan los modelos).

El sistema no clasifica imágenes: MediaPipe reduce cada mano a 21 puntos
(landmarks) con coordenadas (x, y, z), y tres redes neuronales entrenadas
**exclusivamente con datos propios** (nunca un dataset externo) clasifican esa
geometría — una para letras estáticas, una para números, y una temporal (TCN)
para señas dinámicas de una o dos manos. Un segundo modelo, un LLM (Claude),
convierte las glosas reconocidas en una frase gramaticalmente correcta.
Toda la inferencia corre localmente: el video nunca sale del dispositivo.

---

## 2. Stack tecnológico

| Capa | Tecnología | Rol |
|---|---|---|
| Lenguaje (escritorio) | Python 3.10+ | Toda la app de escritorio, captura y entrenamiento |
| Detección de manos/cuerpo | MediaPipe (`HandLandmarker`, `PoseLandmarker`) | Extrae landmarks de manos y hombros |
| Visión / cámara | OpenCV | Captura de cámara, overlay dibujado, ventana |
| Machine Learning | TensorFlow / Keras 3 | Definición, entrenamiento e inferencia de los 3 modelos |
| Utilidades ML | scikit-learn | Split estratificado, class weights, reportes de clasificación |
| Cómputo numérico | NumPy | Vectores y secuencias de landmarks |
| Voz de salida (TTS, escritorio) | SAPI5 vía `win32com` | Síntesis de voz offline en Windows |
| Voz de entrada (STT, escritorio) | Whisper (`faster-whisper`) | Transcribe habla a texto, en CPU |
| Traducción a lenguaje natural | Claude (API de Anthropic) | Convierte glosas ASL en una frase fluida |
| Frontend web | JavaScript vanilla, sin framework, sin build step | Todo el reconocimiento y la UI de `web/` |
| Detección web | `@mediapipe/tasks-vision` (vendorizado, versión fijada) | Misma tarea que MediaPipe Python, en el navegador |
| Voz web | Web Speech API (`SpeechSynthesis` + `SpeechRecognition`) | Sustituye SAPI5/Whisper en el navegador |
| Hosting web | Netlify (estático) | `netlify.toml` en la raíz, despliega `web/` en cada push |

---

## 3. Arquitectura general

```
                    ┌─────────────────────────────┐
                    │   Modelos entrenados (.h5)   │
                    │  letras · números · palabras │
                    └───────────────┬─────────────┘
                                    │  exportados a JSON de pesos
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
        ┌───────────────────────┐     ┌───────────────────────────┐
        │  App de escritorio     │     │  App web (web/)            │
        │  main.py (Python)      │     │  vanilla JS, sin backend    │
        │  · TensorFlow/Keras    │     │  · forward pass a mano      │
        │  · MediaPipe Python    │     │  · MediaPipe tasks-vision   │
        │  · SAPI5 / Whisper     │     │  · Web Speech API           │
        │  · herramienta de      │     │  · producto público         │
        │    captura+entrenam.   │     │    (Netlify)                │
        └───────────────────────┘     └───────────────────────────┘
```

Ambas superficies leen la **misma matemática** (normalización de landmarks,
construcción de features de palabras, remuestreo temporal). En Python esa
matemática vive en `src/utils.py`; en JavaScript está reimplementada en
`web/js/utils.js` y verificada contra la versión Python por un banco de
pruebas (`web/js/utils.test.html`, ver sección 14). Ningún dataset externo
alimenta los modelos: los datos son 100% captura propia por webcam
(`data/real_capture/`).

---

## 4. El pipeline en tiempo real, frame a frame

Descrito para la app de escritorio (`main.py` + `src/detector.py` +
`src/classifier.py`); la app web replica exactamente los mismos pasos en
JavaScript (sección 14).

1. **Captura de frame** (`Detector.get_frame`, `src/detector.py`) — OpenCV lee
   un frame de la cámara y lo **espeja horizontalmente** (`cv2.flip(frame,
   1)`) *antes* de pasarlo a MediaPipe. Esto importa: todo el dataset fue
   capturado sobre video espejado, así que cualquier consumidor de landmarks
   (incluida la web) debe espejar antes de detectar, no después.
2. **Detección** — `HandLandmarker` (modo `VIDEO`, hasta 2 manos) extrae 21
   landmarks `(x, y, z)` por mano detectada. En paralelo, si el modelo de pose
   está presente (`model/pose_landmarker_lite.task`), `PoseLandmarker`
   extrae los hombros (landmarks 11 y 12) — el ancla de cuerpo que solo usa el
   modelo de palabras.
3. **Empaquetado** (`_extract_landmarks`) — cada mano se guarda en dos formas
   en paralelo: por **orden de detección** (`landmarks_hand1`/`hand2`, lo que
   usa el pipeline de letras/números) y por **lateralidad** (`hands_by_side:
   {"Left", "Right"}`, lo que usa el pipeline de palabras, para que la misma
   seña caiga siempre en el mismo slot sin importar el orden en que MediaPipe
   detectó las manos). También se calcula `frame_aspect` = ancho/alto del
   frame — necesario para la normalización (sección 5).
4. **Clasificación** (`Classifier.classify`, `src/classifier.py`) — según el
   modo activo (`static_mode = "letters"` o `"numbers"`) corre el modelo
   estático correspondiente sobre `landmarks_hand1`. En paralelo, **siempre**
   actualiza el buffer de palabras (independientemente del modo visible) con
   el frame body-anchored de 130 valores, porque ese buffer también decide si
   la mano "está señando" (motion gate).
5. **Suavizado temporal** (`PredictionSmoother`, `src/utils.py`) — una
   predicción cruda por frame es ruidosa. El smoother mantiene una ventana
   deslizante de las últimas *N* predicciones y solo la confirma como
   "estable" si aparece un mínimo de veces dentro de esa ventana — elimina el
   parpadeo entre letras parecidas (P/Q, U/V) y evita que un frame ruidoso
   dispare una predicción falsa.
6. **Máquina de estados de palabras** (`WordCommitFSM`, `main.py`) — para el
   modo palabras, envuelve un `PredictionSmoother` propio más un cooldown por
   **tiempo real** (no por frames, ver sección 11): una vez confirmada una
   seña, queda "bloqueada" un rato antes de aceptar la siguiente, dando tiempo
   a que el usuario pase a la próxima seña sin que la misma se repita en
   ráfaga.
7. **Dibujo** (`_draw_hud`, `main.py`) — barra de confianza, panel de
   alternativas si la confianza es baja, subtítulo acumulado
   (`LetterBuffer`/`WordBuffer`), banner de traducción si corresponde, barra
   de transcripción del oyente (`SpeechBuffer`).
8. **Efectos secundarios** — si hubo un carácter/palabra nuevo: se envía a la
   cola de voz (`VoiceOutput`, hilo aparte) y al registro de conversación
   (`ConversationLog`).

Todo esto corre en el hilo principal salvo la voz (TTS), el reconocimiento de
voz (STT) y la traducción LLM, que corren en **hilos daemon** con colas, para
que nada de eso bloquee jamás el loop de cámara.

---

## 5. Normalización de landmarks — las fórmulas exactas

Fuente única de verdad: `src/utils.py`. Usada **idénticamente** en captura,
entrenamiento e inferencia — si alguna vez difiere entre esos tres puntos, el
modelo falla silenciosamente (ve datos con una forma distinta a la que
aprendió).

### 5.1. `normalize_landmarks` (letras, números — una mano, 63 valores)

Por cada mano, con `aspect` = ancho/alto del frame:

1. **Deshacer el estiramiento de MediaPipe.** MediaPipe normaliza x por el
   ancho del frame e y por el alto (z escala como x). La misma seña física
   produce vectores distintos en una webcam 16:9 que en un celular vertical
   9:16. Corrección: `y /= aspect` (x y z quedan intactos). Sin `aspect`, este
   paso se omite — solo es válido para casos legados.
2. **Centrar en la muñeca.** `centered = points - points[wrist]` (landmark 0
   como origen).
3. **Escalar por el tamaño de la mano.** `scale = ||centered[9]||` (distancia
   muñeca → base del dedo medio, landmark 9). Si `scale < 1e-6` se usa `1.0`
   para evitar división por cero. Resultado: `centered / scale`.

El resultado es invariante a posición, tamaño de mano y orientación de
cámara. Para dos manos (126 valores, no usado por letras/números pero
soportado por la función) se aplica el mismo proceso a cada mitad.

### 5.2. `build_word_features` (palabras — dos manos ancladas al cuerpo, 130 valores)

Por cada frame, con `left_hand`, `right_hand` (63 valores o `None` cada una),
`shoulder_l`, `shoulder_r` (o `None` si no se detectó pose), y `aspect`
**obligatorio** (lanza `ValueError` si falta — un forgotten call site falla
ruidosamente en vez de reintroducir el bug de estiramiento silenciosamente):

1. **Marco corporal** (si hay hombros): centro `cx,cy` = punto medio entre
   hombros (ya des-estirado en y); escala = distancia euclídea entre hombros
   (`hypot`, invariante a la inclinación de cabeza/torso). Sin hombros, no hay
   marco (`frame = None`).
2. **Por cada mano** (`_hand_block`, 65 valores = 63 forma + 2 posición):
   - **Forma:** los mismos 63 valores de `normalize_landmarks` (posición y
     tamaño invariantes) — la forma de la mano en sí, sin importar dónde está.
   - **Posición:** `(wrist_x - cx) / scale`, `(wrist_y/aspect - cy) / scale` —
     dónde está la muñeca **respecto al cuerpo**, no respecto a la pantalla.
     Sin marco corporal, la posición es `(0, 0)` — nunca la coordenada cruda
     de pantalla, porque eso reintroduciría justo el ruido que el ancla
     corporal existe para eliminar.
   - Una mano ausente es un bloque de puros ceros — un patrón distinguible
     que el modelo aprende a leer como "esa mano no está".
3. **Concatenación:** `[mano izquierda (65)] + [mano derecha (65)] = 130`.

### 5.3. Almacenamiento crudo y remuestreo

- `pack_word_raw` / `word_features_from_raw` (131 valores: 63+63 manos + 2+2
  hombros + 1 aspecto) — la captura guarda landmarks **crudos**, sin
  featurizar. La featurización (5.2) ocurre recién **al cargar** los datos
  para entrenar. Esto es deliberado: un cambio futuro de normalización (como
  la corrección de aspecto de 2026-07-12) se aplica retroactivamente a los
  datos ya capturados sin necesidad de recapturar todo el vocabulario.
- `resample_sequence(frames, n)` — remuestrea una secuencia de largo variable
  a exactamente `n` frames (índices por `linspace`), usado tanto para llevar
  cada take a `WORD_SEQ_LEN=32` como para el buffer en vivo.
- `mirror_sequence` / `mirror_word_sequence` — espejan horizontalmente
  (niegan las coordenadas x) para usarse como augmentation de entrenamiento;
  para palabras además intercambian los bloques de mano izquierda/derecha
  (una seña diestra espejada es la misma seña zurda).

---

## 6. Los tres modelos de reconocimiento

Los tres se entrenan **exclusivamente con landmarks capturados por webcam**
del propio equipo — cero datasets externos.

### 6.1. Letras (estáticas)

- **24 clases:** A–Y sin J ni Z (`model/labels_one_hand.json`): A B C D E F G
  H I K L M N O P Q R S T U V W X Y.
- **Entrada:** 63 valores (1 frame, 1 mano).
- **Arquitectura** (`training/train_letters.py::_build_model`, ~densa fully
  connected, con BatchNorm/Dropout/L2):

  ```
  Input(63)
    → Dense(256, relu, L2=1e-4) → BatchNorm → Dropout(0.35)
    → Dense(128, relu, L2=1e-4) → BatchNorm → Dropout(0.35)
    → Dense(64,  relu, L2=1e-4) → BatchNorm → Dropout(0.20)
    → Dense(32,  relu, L2=1e-4) → BatchNorm
    → Dense(24, softmax)
  ```

- **Pérdida:** cross-entropy con *label smoothing* 0.05 (evita que el modelo
  aprenda a estar "100% seguro" incluso cuando se equivoca). Optimizador Adam,
  lr `1e-3`.
- **Augmentation en entrenamiento** (todo el dataset se capturó en una sesión
  de cámara fija, así que el modelo nunca vio la seña desde otro ángulo — el
  augmentation compensa eso):
  - Rotación 3D rígida alrededor de la muñeca: roll ±25°, pitch/yaw ±12°
    (roll es más generoso porque x/y son ejes que MediaPipe estima
    directamente de píxeles; pitch/yaw tocan z, la profundidad, más ruidosa).
  - Ruido gaussiano (σ=0.012) — robustez al jitter de MediaPipe.
  - Jitter de escala ±8% — simula la mano a distintas distancias.
  - Espejado horizontal con probabilidad 0.5 — aprende ambas manos.
- **Class weights** balanceados; `EarlyStopping` (paciencia 35, sobre
  `val_accuracy`) + `ReduceLROnPlateau` (factor 0.7, paciencia 8). Batch 64,
  hasta 300 épocas.
- **Datos:** 3.750 muestras / 9 sesiones (cada CSV en
  `data/real_capture/letters/` se descubre automáticamente).
- **Resultado:** 98.9% de accuracy en validación (holdout 80/20 estratificado
  — número optimista, ver sección 17).

### 6.2. Números (estáticos)

- **10 clases:** 0–9 (`model/labels_numbers.json`).
- **Misma arquitectura, mismo pipeline** que letras (`train_numbers.py` es un
  espejo de `train_letters.py`), archivo separado (`model_numbers.h5`).
- **Por qué un modelo aparte y no una sola clase de 34 símbolos:** varios
  dígitos ASL colisionan visualmente con letras (2≈V, 6≈W, 9≈F). Un único
  conjunto cerrado tendría que arbitrar entre dos respuestas igualmente
  válidas; separarlos deja que el **modo activo** (elegido por la persona)
  desambigüe.
- **Datos:** 1.500 muestras / 4 sesiones.
- **Resultado:** 99.0% de accuracy en validación (holdout).

### 6.3. Palabras (dinámicas) — TCN temporal

- **19 clases:** 18 glosas + la clase negativa `nothing`
  (`model/labels_words.json`): drink, eat, finished, good, hello, help, let,
  me, more, need, no, nothing, please, sorry, thanks, that, want, yes, you.
- **Entrada:** secuencia de **32 frames × 130 valores** (ver sección 5.2).
- **Por qué TCN y no LSTM/GRU:** una seña real tiene orden — "la mano sube y
  después baja" no es lo mismo que al revés. Un resumen tipo media+desvío
  (el modelo anterior, descartado) es ciego a ese orden y colapsa cuando crece
  el vocabulario. Un TCN (convoluciones 1D dilatadas) lee la secuencia
  ordenada y además convierte limpio a JavaScript/TF.js — las redes
  recurrentes dan fricción al portar a la web.
- **Arquitectura** (`training/train_words.py::_build_model`):

  ```
  Input(32, 130)
    → Conv1D(64,  k=3, causal, dilation=1, relu) → BatchNorm
    → Conv1D(64,  k=3, causal, dilation=2, relu) → BatchNorm
    → Conv1D(128, k=3, causal, dilation=4, relu) → BatchNorm → Dropout(0.3)
    → Conv1D(128, k=3, causal, dilation=8, relu)
    → GlobalAveragePooling1D → Dropout(0.4)
    → Dense(128, relu)
    → Dense(19, softmax)
  ```

  Las dilataciones 1-2-4-8 amplían el campo receptivo capa a capa (cada capa
  "ve" más contexto temporal que la anterior) sin agrandar el kernel; el
  padding `causal` evita que una capa mire frames "futuros" respecto al
  frame que está procesando.
- **Pérdida:** `sparse_categorical_crossentropy`. Adam lr `1e-3`. Batch 16,
  hasta 200 épocas.
- **Augmentation offline** (aplicado una vez antes de entrenar, multiplica el
  dataset **x8**): original + espejado + *time-warp* (remapeo no lineal del
  tiempo, γ=0.7 y γ=1.4, sobre original y espejado — enseña la misma seña
  hecha más rápido/más lento) + *frame dropout* (10% de frames puestos en
  cero, simula a MediaPipe perdiendo la mano un instante) + *hand dropout*
  (una mano puesta en cero durante un tramo contiguo del 20-50% del take,
  simula una mano saliendo brevemente de cuadro).
  - Se **probó y se descartó** un noveno augmentation, *temporal crop*
    (recortar 10-25% del inicio o final y remuestrear) — midió una regresión
    real (CV bajó de 94.1% a 93.0%, recall de `nothing` de 0.76 a 0.74) y
    quedó en el código pero **sin aplicar**, documentado para no repetir el
    experimento a ciegas (`training/train_words.py::_temporal_crop`).
  - Jitter *online* por batch (no offline): escala global ±10%, ruido
    gaussiano σ=0.02.
- **Clase negativa `nothing`:** absorbe los frames ambiguos (reposo,
  transiciones, letras en movimiento). Sin ella, un softmax de conjunto
  cerrado etiqueta *todo* como alguna palabra y nunca se calla.
- **Validación:** `session_holdout` reserva **sesiones completas** (no takes
  individuales) para validación — nunca se valida contra el mismo día/luz con
  que se entrenó. Si hay menos de 2 sesiones, el set de validación queda
  vacío y el número reportado es solo de entrenamiento (optimista).
- **Datos (modelo vigente, recapturado 2026-07-18 tras el fix de aspecto):**
  668 takes / 11 sesiones.
- **Resultado:** 98.0% de accuracy en validación (holdout por sesión —
  optimista; ver sección 17 sobre la validación cruzada pendiente).

---

## 7. Segunda etapa: traducción glosa → frase (LLM)

`src/translator.py`, tecla **T** (solo modo palabras, solo escritorio).

El reconocedor entrega **glosas**: palabras clave en inglés, forma de cita
("WANT DRINK NOW"). ASL no conjuga verbos ni sigue el orden gramatical del
español, así que enseñarle al reconocedor formas conjugadas sería
lingüísticamente incorrecto y combinatoriamente inviable. En su lugar, la capa
de traducción — un LLM — recibe la secuencia de glosas y devuelve una frase
fluida en el idioma configurado (español por defecto).

- **Cliente perezoso y tolerante a fallos:** si el SDK `anthropic` no está
  instalado, o falta `ANTHROPIC_API_KEY`, o la llamada falla, cae a un
  **fallback offline** que simplemente une las glosas con espacios — la app
  nunca se rompe por esto.
- **No bloqueante:** `submit()` lanza un hilo daemon y retorna al instante;
  `poll()` se llama cada frame en el loop principal y devuelve el resultado
  cuando está listo (patrón cola + hilo, igual que TTS/STT).
- **Prompt de sistema** instruye al modelo: producir UNA frase natural y
  gramatical, agregando artículos/preposiciones/conjugación/tiempo que el
  español requiere, usando palabras de tiempo presentes en las glosas (now,
  later, yesterday, tomorrow, finished) para elegir el tiempo verbal, sin
  inventar contenido que no esté en las glosas, sin explicaciones ni comillas.
- **Modelo por defecto:** `claude-haiku-4-5` (rápido y económico, adecuado
  para esta tarea acotada); configurable en `config.py` sección 10. Los
  parámetros `thinking`/`output_config.effort` solo se envían si el modelo
  **no** es Haiku (Haiku no los acepta y respondería 400).
- **Hoy solo existe en escritorio** — es el próximo ítem planeado para la app
  web (ver sección 19).

---

## 8. Voz de entrada y salida

### 8.1. Salida de voz — TTS (`src/voice.py`, solo escritorio)

`VoiceOutput` encola texto y lo habla en un **hilo daemon** vía **SAPI5**
(Windows), usando `win32com.client.Dispatch("SAPI.SpVoice")` directamente en
vez de `pyttsx3`: `pyttsx3.runAndWait()` no reiniciaba correctamente el estado
interno de SAPI5 entre llamadas sucesivas dentro de un loop — solo hablaba la
primera vez. `win32com` directo evita ese bug, permaneciendo 100% offline.
`Speak()` es síncrono dentro del hilo (los ítems de la cola no se solapan).

### 8.2. Entrada de voz — STT (`src/speech_input.py`, solo escritorio)

`SpeechInput` es **push-to-talk** (tecla **P**), con una máquina de estados:
`LOADING → IDLE ⇄ LISTENING → TRANSCRIBING → IDLE` (o `UNAVAILABLE` si falta
`faster-whisper` o falla la carga del modelo).

- El modelo Whisper se carga en un hilo daemon al arrancar, sin bloquear la
  cámara. Tamaño configurable (`tiny` por defecto — más rápido, apto para
  demo; `base`/`small` para más precisión a costa de latencia).
- Micrófono vía `sounddevice.InputStream` a 16 kHz mono float32. Clips
  menores a 0.4s se descartan silenciosamente.
- Transcripción con `beam_size=1` (velocidad sobre precisión) y filtro VAD
  (descarta segmentos de silencio, mínimo 300ms) — cuantización `int8` en CPU
  para ~2x de velocidad con pérdida de precisión despreciable.
- Idioma autodetectado (`None`) o fijado (p. ej. `"es"`) en `config.py`.

En la **app web** ambos roles los cumple la **Web Speech API** del navegador
(`js/tts.js` / `js/stt.js`) — sin dependencias locales, con las voces
instaladas en el dispositivo del usuario.

---

## 9. Capa de presentación (overlay, buffers, HUD)

Todo en `src/overlay.py` salvo el HUD central, que vive directamente en
`main.py`.

- **`LetterBuffer`** — acumula letras confirmadas con un cooldown de
  `LETTER_COOLDOWN_FRAMES` frames entre caracteres aceptados (evita que una
  pose sostenida sature el buffer). Soporta en código los valores especiales
  `"del"` (borra el último carácter) y `"space"` (cierra la palabra actual y
  la registra en `pop_completed_words()` para el log de conversación) — **pero
  hoy ningún gesto del modelo de letras produce esos dos valores** (las 24
  clases entrenadas son A–Y; no hay clases `del`/`space` todavía) y `main.py`
  no tiene una tecla ligada a cerrar la palabra deletreada. En la práctica,
  en escritorio, el texto deletreado se sigue acumulando sin cierre
  automático — a diferencia de la web, que sí implementa un cierre por pausa
  o el botón `SPACE` (`CharBuffer` en `web/js/utils.js`). Muestra hasta los
  últimos 50 caracteres (`_MAX_VISIBLE_CHARS`); la web usa un límite propio
  más chico (28) por pantalla disponible.
- **`WordBuffer`** — arma la frase de señas reconocidas en modo palabras. Se
  autolimpia tras `WORD_SENTENCE_PAUSE_FRAMES` frames sin una seña nueva (no
  hace falta ninguna tecla). `get_words()` es lo que lee `translator.py` para
  la traducción LLM.
- **`SpeechBuffer`** — barra superior (navy) con las últimas transcripciones
  de voz del oyente (hasta 2 líneas, 55 caracteres cada una) y un indicador de
  estado (`cargando modelo...`, `[ REC ]`, `procesando...`, `no disponible`).
- **`WordCommitFSM`** (`main.py`) — envuelve el smoother de palabras
  (`PredictionSmoother(window=5, min_votes=3)`) más el cooldown temporal.
  `step()` devuelve `is_new=True` solo en el frame exacto en que se confirma
  una seña nueva — esa es la señal para hablarla y loguearla, evitando
  duplicar el efecto en cada frame que la palabra sigue en pantalla.
- **Barra de confianza** (`_draw_confidence_bar`) — verde ≥80%, amarillo
  ≥60%, rojo por debajo, dibujada bajo la predicción estática y bajo la
  palabra detectada.
- **Panel de alternativas** (`_draw_alternatives`) — se muestra en vez de un
  único resultado cuando la confianza cruda está por debajo de
  `LOW_CONFIDENCE_THRESHOLD` (0.70), listando el top-3 con sus porcentajes.
  Esto está **desacoplado** del umbral de aceptación (lo que se comete al
  buffer/voz): se puede mostrar una pista sin comprometerse a actuar sobre
  ella.
- **Avisos de honestidad** — si el modo activo es "numbers" o "words" pero el
  modelo correspondiente no está entrenado/cargado, el HUD muestra un aviso
  explícito ("Numbers model not trained yet" / "Word model not trained yet")
  en vez de dejar la pantalla en blanco o mostrar letras por error.

---

## 10. Registro de conversación exportable

`src/conversation_log.py`, tecla **E**.

`ConversationLog` acumula en memoria una lista de `(timestamp, source, text)`
— `source` es `"Senas"` (señas del signante: palabras completas o palabras
deletreadas cerradas) u `"Oyente"` (transcripciones de voz). No escribe nada a
disco hasta que se pide exportar — cero costo de I/O por frame.

`export()` escribe dos archivos en `logs/` con el mismo timestamp:

- `conversation_<AAAAMMDD_HHMMSS>.txt` — legible: `[hh:mm:ss] Fuente: texto`.
- `conversation_<AAAAMMDD_HHMMSS>.csv` — columnas `timestamp,source,text`
  (timestamp en ISO 8601).

La app web genera archivos **byte-compatibles** con este mismo formato
(`web/js/conversation.js`), así que un consumidor de los archivos no puede
distinguir cuál de las dos apps los generó.

---

## 11. `config.py` — referencia completa de parámetros

`config.py` es la **única fuente de verdad** de todos los parámetros del
escritorio — cambiar un valor ahí se propaga a toda la app sin tocar otro
archivo. Valores vigentes en el código (algunos comentarios explicativos
dentro de `config.py` quedaron desactualizados tras ajustes posteriores —
notablemente `WORD_COOLDOWN_SECONDS` y `WORD_CONFIDENCE_THRESHOLD`, cuyo
comentario todavía cita un valor viejo; la tabla de abajo usa el valor
**real** vigente en el código):

| Parámetro | Valor | Qué controla |
|---|---|---|
| `MODE` | `"words"` | Modo inicial (`L`/`W`/`N` lo cambian en vivo) |
| `LETTER_CONFIDENCE_THRESHOLD` | 0.80 | Confianza mínima para aceptar una letra |
| `NUMBER_CONFIDENCE_THRESHOLD` | 0.60 | Ídem para un número |
| `WORD_CONFIDENCE_THRESHOLD` | 0.75 | Ídem para una palabra |
| `LOW_CONFIDENCE_THRESHOLD` | 0.70 | Debajo de esto se muestra el panel top-3 |
| `ALT_MIN_CONFIDENCE` | 0.08 | Piso para que un candidato aparezca en el top-3 |
| `WORD_SEQ_LEN` | 32 | Frames a los que se remuestrea toda secuencia de palabra |
| `WORD_FEATURE_DIM` | 130 | Valores por frame del modelo de palabras |
| `WORD_BUFFER_FRAMES` | 45 (~1.5s @30fps) | Tamaño del buffer circular en vivo |
| `WORD_MIN_FRAMES` | 16 (~0.53s) | Mínimo de frames antes de correr el TCN |
| `WORD_MOTION_WINDOW` | 8 (~0.27s) | Ventana reciente para medir si la mano se mueve |
| `WORD_MIN_MOTION_STD` | 0.020 | Umbral de movimiento para activar el modelo de palabras |
| `WORD_NULL_LABEL` | `"nothing"` | Nombre de la clase negativa (debe existir en las labels) |
| `LETTER_SMOOTH_WINDOW` / `MIN_VOTES` | 7 / 5 | Ventana de votación para confirmar una letra |
| `WORD_SMOOTH_WINDOW` / `MIN_VOTES` | 5 / 3 | Ídem para una palabra |
| `WORD_COOLDOWN_SECONDS` | 2.0s | Bloqueo tras confirmar una palabra (tiempo real, no frames) |
| `WORD_SENTENCE_PAUSE_FRAMES` | 150 (~5s) | Inactividad antes de limpiar la frase de glosas |
| `MOTION_LETTERS` | `{"J", "Z"}` | Letras marcadas "requires motion" |
| `LETTER_COOLDOWN_FRAMES` | 20 (~0.67s) | Mínimo entre dos caracteres aceptados |
| `CAPTURE_TARGET_PER_WORD` | 34 | Objetivo acumulado de tomas por palabra, entre todas las sesiones |
| `CAPTURE_PER_SESSION_PER_WORD` | 7 | Tope de tomas por palabra en una sola sesión |
| `CAPTURE_OUTPUT_DIR` | `data/real_capture/words` | Destino de capturas de palabras |
| `CAMERA_INDEX` | 0 | Cámara del sistema a usar |
| `LEGACY_CAPTURE_ASPECT` | 16/9 | Aspecto asumido para CSVs anteriores a la columna `aspect` |
| `DETECTOR_MIN_DETECTION_CONFIDENCE` | 0.70 | Umbral para detectar una mano por primera vez |
| `DETECTOR_MIN_PRESENCE_CONFIDENCE` | 0.70 | Umbral para seguir considerando una mano presente |
| `DETECTOR_MIN_TRACKING_CONFIDENCE` | 0.50 | Umbral de tracking entre frames |
| `VOICE_ENABLED` | `True` | Interruptor maestro de TTS |
| `SPEECH_INPUT_ENABLED` | `True` | Interruptor maestro de STT |
| `SPEECH_WHISPER_MODEL` | `"tiny"` | Tamaño del modelo Whisper |
| `SPEECH_LANGUAGE` | `None` (autodetecta) | Idioma fijo para STT, si se desea |
| `TRANSLATION_ENABLED` | `True` | Interruptor maestro de la traducción LLM |
| `TRANSLATION_TARGET_LANGUAGE` | `"Spanish"` | Idioma de salida de la frase traducida |
| `TRANSLATION_MODEL` | `"claude-haiku-4-5"` | Modelo usado para traducir |
| `TRANSLATION_OFFLINE_FALLBACK` | `True` | Si `True`, sin API cae a unir las glosas tal cual |

---

## 12. Pipeline capturar → entrenar → evaluar

### 12.1. Capturar

- `capture/capture_letters.py` / `capture/capture_numbers.py` — graban poses
  estáticas a CSV timestampeado en `data/real_capture/letters/` o `numbers/`,
  con una columna `aspect` por fila (relación de aspecto real del frame).
  Re-ejecutar el script agrega un CSV nuevo — dataset multi-sesión sin tocar
  código. El vocabulario a capturar se edita en la lista `LABELS` al inicio
  del script.
- `capture/capture_words.py` — graba una secuencia `.npy` **cruda** (ver
  sección 5.3) por toma, más una fila en `manifest.csv`
  (`sample_id, gloss, subset, session_id`). El vocabulario se edita en la
  lista `WORDS` al inicio del script. Respeta la disciplina multi-sesión de
  `config.py` (tope por sesión, objetivo acumulado): al llegar al tope de una
  palabra en la sesión actual, el script avanza sola a la siguiente.
- `capture/rebuild_manifest.py` — utilidad para regenerar `manifest.csv` a
  partir de los archivos `.npy` presentes en `seq/`, para el caso de que el
  manifiesto quede desincronizado del contenido real de la carpeta.

**Disciplina multi-sesión (por qué importa más que la cantidad):** un modelo
entrenado con muchas tomas de una sola sesión memoriza esa sesión concreta
(alta validación, falla en vivo). Varias sesiones — distintos días, luz,
ropa, distancia — fuerzan al modelo a aprender lo que no cambia: la seña en
sí. Por eso la captura tiene un tope duro por sesión y un objetivo acumulado
entre sesiones (sección 11).

### 12.2. Entrenar

Cada entrenador **descubre sus datos solo**, sin rutas para editar:
`train_letters.py` / `train_numbers.py` toman todos los CSV de su carpeta;
`train_words.py` lee `manifest.csv` y, por cada fila, carga el `.npy` crudo,
lo remuestrea a `WORD_SEQ_LEN` y recién ahí aplica `word_features_from_raw`
por frame — construir features al cargar (no al capturar) es lo que permite
que un cambio de normalización futuro nunca invalide una captura ya hecha.

Cada corrida de entrenamiento escribe un registro comparable en
`runs/<modelo>_<timestamp>.json`: vocabulario, muestras por clase, número de
sesiones, hiperparámetros y accuracy de validación — así siempre se puede
saber qué datos produjeron qué modelo.

### 12.3. Evaluar

`training/evaluate.py` (palabras), `evaluate_letters.py`, `evaluate_numbers.py`
comparten `training/eval_common.py` (el reporte por clase + la validación
cruzada agrupada) y reutilizan la función `train_model(...)` de cada
entrenador — la evaluación nunca puede desviarse de cómo se entrena
realmente.

- **Sin `--cv`:** holdout rápido (el mismo split 80/20 o por sesión que usa
  el entrenador).
- **Con `--cv K`:** validación cruzada **agrupada** — por `session_id` en
  palabras, por archivo CSV (= una sesión) en letras/números — de forma que
  ninguna sesión se reparte entre entrenamiento y prueba dentro del mismo
  fold. Es el número honesto: el holdout normal es optimista porque valida
  contra datos de las mismas sesiones que entrenó.

---

## 13. Estructura completa de archivos

```
Sign_Language_Translator/
├── main.py                    # Punto de entrada de la app de escritorio
├── config.py                  # Única fuente de verdad de parámetros
├── requirements.txt           # Dependencias Python
├── netlify.toml                # Configuración de despliegue de web/
│
├── src/
│   ├── detector.py            # Cámara + MediaPipe (manos + pose)
│   ├── classifier.py          # Carga los 3 modelos y clasifica en vivo
│   ├── utils.py                # normalize_landmarks, build_word_features,
│   │                           #   resample_sequence, PredictionSmoother
│   ├── voice.py                # TTS offline (SAPI5/win32com, hilo daemon)
│   ├── overlay.py              # LetterBuffer + WordBuffer + SpeechBuffer
│   ├── conversation_log.py     # Registro exportable (TXT + CSV)
│   ├── speech_input.py         # STT (Whisper, push-to-talk)
│   └── translator.py           # Glosas → frase fluida (Claude API)
│
├── capture/
│   ├── capture_letters.py      # Captura de poses estáticas de letras
│   ├── capture_numbers.py      # Captura de poses estáticas de dígitos
│   ├── capture_words.py        # Captura de secuencias dinámicas
│   └── rebuild_manifest.py     # Regenera manifest.csv desde disco
│
├── training/
│   ├── train_letters.py        # Entrena el modelo de letras
│   ├── train_numbers.py        # Entrena el modelo de números
│   ├── train_words.py          # Entrena el modelo de palabras (TCN)
│   ├── run_log.py              # Escribe runs/<modelo>_<timestamp>.json
│   ├── eval_common.py          # Reporte + CV agrupada compartidos
│   ├── evaluate.py             # Evaluador de palabras
│   ├── evaluate_letters.py     # Evaluador de letras
│   └── evaluate_numbers.py     # Evaluador de números
│
├── tools/
│   ├── export_model_json.py    # Exporta pesos .h5 → JSON para la web
│   ├── make_fixtures.py        # Genera fixtures de paridad para el gate web
│   └── update_web.py           # Un comando: exporta pesos+fixtures+labels
│
├── model/                      # Modelos .h5, labels .json, .task de MediaPipe
│   └── legacy/                 # Modelo de palabras pre-corrección de aspecto
├── runs/                       # Un JSON por corrida de entrenamiento
├── logs/                       # Registros de conversación exportados (E)
│
├── data/real_capture/
│   ├── letters/                 # CSV de letras
│   ├── numbers/                 # CSV de números
│   ├── words/                   # seq/*.npy (crudo) + manifest.csv
│   └── words_legacy/            # Capturas pre-corrección de aspecto (archivadas)
│
├── docs/
│   ├── IDEA.md                  # Visión de producto, sin tecnicismos
│   ├── TECNICO.md               # Documento técnico exhaustivo (fuente interna)
│   ├── DESIGN_BRIEF.md          # Inventario funcional de la web (insumo de rediseño)
│   ├── GUIA_SIMPLE.md           # Este par de documentos: versión simplificada
│   └── MANUAL_COMPLETO.md       # ...y esta, la versión exhaustiva
│
└── web/                         # App web — ver sección 14
```

---

## 14. La aplicación web

Documentación fuente: [`web/README.md`](../web/README.md). Una sola página,
**sin backend, sin build step, sin framework** — HTML/CSS/JS plano — que
corre los tres modos enteramente en el navegador, en escritorio y en móvil.

### 14.1. Mapa de archivos

| Archivo | Rol |
|---|---|
| `index.html` | Header (toggle de modo, voz, settings, help), escenario de cámara con overlays, panel de conversación, diálogos |
| `css/style.css` | Sistema de diseño ("caption console": amarillo sobre casi-negro), responsive mobile-first |
| `js/config.js` | Umbrales/modelo COPIADOS de `config.py`; cadencias en **milisegundos** (ver 14.3) |
| `js/modes.js` | Descriptor por modo (kind, URL de modelo/labels, umbral, textos de UI) — lo único que cambia entre modos |
| `js/utils.js` | Puerto de `normalize_landmarks` + helpers de palabras + `TimeSmoother`/`CharBuffer`/`WordCommitFSM`/`WordBuffer` (versión temporizada) |
| `js/model.js` | `DenseModel` (letras/números) y `TCNModel` (palabras) — forward pass **a mano en JS puro**, sin TF.js |
| `js/pipeline.js` | cámara → canvas espejado → MediaPipe → features → modelo → smoother/FSM → UI |
| `js/tts.js` | Voz de salida (Web Speech), contraparte de `src/voice.py` |
| `js/stt.js` | Dictado del oyente (`SpeechRecognition`, oculto si no está soportado) |
| `js/conversation.js` | Store de conversación + export TXT/CSV (byte-compatible con `conversation_log.py`) |
| `js/ui.js` | Panel de chat, mensaje grande, diálogos, feedback de confirmación |
| `js/prefs.js` | Preferencias en `localStorage` (namespace `asl-web.`) |
| `js/utils.test.html` | **El gate**: fixtures reales + tests de comportamiento + tests de formato de export — todo debe estar en verde antes de desplegar |
| `model/` | Copias locales de pesos, labels y `.task` de MediaPipe |
| `vendor/mediapipe/` | `@mediapipe/tasks-vision` vendorizado, versión fijada |

### 14.2. Por qué no hay TensorFlow.js

El plan original era convertir los `.h5` a TF.js sin reentrenar. Su
conversor **no instala en Windows**, así que la solución fue exportar los
pesos a JSON (`tools/export_model_json.py`) e implementar los forward passes
a mano: `DenseModel` (Dense + BatchNorm) para letras/números, `TCNModel`
(Conv1D causal-dilatada + GlobalAveragePooling + Dense) para palabras — la
misma arquitectura de la sección 6, reimplementada en JS puro. Menos
dependencias y control total del costo por frame; el costo es mantener esas
implementaciones a mano en paralelo con Python.

### 14.3. Divergencia deliberada: milisegundos, no frames

El escritorio cuenta **frames** para ventanas de suavizado y cooldowns porque
su cámara corre a tasa fija (~30fps). La web va de ~15fps (celular, con pose
activo) a 60fps según el dispositivo — contar frames haría que un celular
necesitara sostener cada seña casi el triple de tiempo real. Por eso
`js/config.js` expresa cada umbral de cadencia en **milisegundos** (cada
valor es el gemelo de su constante en `config.py` a 30fps), y las clases
`TimeSmoother`/`CharBuffer`/`WordBuffer` reciben el timestamp del frame como
parámetro explícito (determinístico para tests). El comportamiento en
escritorio a 30fps es idéntico al de antes de esta divergencia.

### 14.4. Las cuatro reglas de paridad (romperlas = predicciones mal silenciosamente)

1. **Espejar antes de detectar.** El escritorio espeja el frame *antes* de
   pasarlo a MediaPipe; la web dibuja la cámara en un canvas espejado y
   detecta sobre ese canvas. Si las predicciones son basura pero la mano se
   detecta bien, sospechar esto primero.
2. **La matemática de `js/utils.js` debe igualar exactamente a
   `src/utils.py`** — nunca editar uno sin el otro, nunca confiar "a ojo":
   correr el gate.
3. **Los valores de `js/config.js` están copiados de `config.py`.** Tocar un
   umbral en un lugar exige tocarlo en el otro (cada constante referencia a
   su gemela).
4. **El runtime de MediaPipe y el `.task` son locales y de versión fijada.**
   Nunca cambiar a una copia de CDN ni subir la versión sin más: un detector
   de otra versión puede producir landmarks sutilmente distintos a los que
   vieron los datos de entrenamiento.

### 14.5. Tras reentrenar un modelo

```bash
python tools/update_web.py letters
python tools/update_web.py numbers
python tools/update_web.py words
```

Regenera pesos, fixtures de paridad y labels (y, para palabras, copia el
`.task` de pose la primera vez). Después, abrir
`web/js/utils.test.html` y confirmar que **todo** esté en verde antes de
desplegar — omitir este paso deja el sitio publicado sirviendo el modelo
viejo en silencio.

### 14.6. Despliegue

`netlify.toml` (raíz del repo) ya está configurado: conectar el repo en
Netlify despliega `web/` en cada push, con cache headers sensatos (modelo/JS
revalidan en cada visita; el runtime vendorizado se cachea para siempre).

---

## 15. Uso práctico — atajos, flujos y tiempos

### 15.1. App de escritorio — atajos de teclado

| Tecla | Acción |
|---|---|
| **Q** | Salir |
| **L** | Modo letras |
| **W** | Modo palabras |
| **N** | Modo números |
| **E** | Exportar el registro de conversación (TXT + CSV a `logs/`) |
| **T** | Traducir la frase señada a texto fluido (solo modo palabras) |
| **P** | Activar/desactivar el micrófono (si el STT está habilitado) |

Cambiar de modo (L/W/N) resetea los smoothers, el `WordCommitFSM`, y limpia
ambos buffers de subtítulo — pero **no** borra el registro de conversación ya
acumulado.

### 15.2. Elementos en pantalla — app web

(Inventario completo, `docs/DESIGN_BRIEF.md`.) Cabecera con marca, subtítulo
por modo, toggle Letters/Numbers/Words, botón de voz, Settings, Help. Sobre el
video: píldora de seguimiento (`Working…` / `No hand` / `Tracking` /
`Signing`), badge REC, selector de cámara (solo si hay 2+ cámaras). Sobre el
escenario: video espejado, capa de landmarks (21 puntos por mano + 2 hombros
en modo Words), mensajes bloqueantes de estado/error, readout de predicción
con barra de confianza, mensaje grande del oyente (8s), strip de subtítulo,
contador de FPS, panel de alternativas (top-3 bajo 70%). Acciones: `SPACE`
cierra la palabra deletreada (solo Letters/Numbers), `CLEAR` descarta el texto
en curso sin tocar la conversación ya registrada. Panel de conversación: `Rec`
/ `Export` / borrar (se arma al primer toque, ejecuta al segundo), lista de
mensajes con autor/hora/marca de grabado, barra de composición con dictado.

### 15.3. Flujos principales

- **Deletrear (Letters/Numbers).** Sostener la pose → se confirma tras un
  mínimo de votos estables (~0.23s) → el carácter entra al strip (cooldown
  ~0.67s entre caracteres) → repetir → cerrar la palabra (web: pausa 2s o
  `SPACE`; escritorio: sin mecanismo hoy, ver sección 9) → se pronuncia y
  entra a la conversación.
- **Señar una palabra (Words).** Firmar con naturalidad → el sistema solo
  clasifica mientras las manos se mueven (motion gate) → al confirmarse:
  pulso visual + voz + entrada en la conversación + la glosa se agrega a la
  frase del strip → tras ~5s sin señas nuevas, la frase se limpia sola.
- **Traducir (solo escritorio, tecla T).** Con al menos una glosa acumulada
  en el `WordBuffer`, `T` la envía al LLM; el resultado llega asíncronamente
  y se muestra como banner cian sobre el subtítulo, y se habla.
- **El oyente responde.** Escribe o dicta → el mensaje aparece en grande
  sobre el video (8s) → queda en la conversación del lado "Hearing"/"Oyente".
- **Exportar.** `E` (escritorio) o `Export` (web, requiere haber grabado con
  `Rec` primero) → descarga/escribe `conversation_<timestamp>.txt` +
  `.csv` en el mismo formato en ambas superficies.

### 15.4. Tiempos reales del sistema

| Cosa | Valor |
|---|---|
| Confianza mínima para aceptar | Letras 80% · Números 60% · Palabras 75% |
| Umbral para mostrar el top-3 | por debajo de 70% |
| Confirmación de una letra | ~0.23s de votos estables (7 frames, 5 votos) |
| Bloqueo entre dos letras iguales | ~0.67s (20 frames) |
| Cierre de palabra deletreada (web) | 2s sin letra nueva, o `SPACE` |
| Ventana de señal para una palabra | ~1.5s de movimiento acumulado (buffer 45 frames) |
| Mínimo de frames antes de clasificar una palabra | ~0.53s (16 frames) |
| Bloqueo tras aceptar una palabra | 2.0s (tiempo real, no frames) |
| Limpieza de la frase de glosas | ~5s de pausa (150 frames a 30fps) |
| Duración del mensaje grande del oyente (web) | 8s |
| FPS reales | 30-60 en escritorio; ~15-24 en teléfono (techo práctico de MediaPipe móvil) |

---

## 16. Estado actual (números verificados)

Verificado contra `model/`, `runs/` y los labels `.json` presentes en el
repositorio en esta revisión.

| Área | Estado |
|---|---|
| Detección de manos + pose (MediaPipe) | Completa |
| Clasificación en tiempo real (los 3 modelos) | Completa |
| Modelo de letras | Operativo — 24 clases, 3.750 muestras / 9 sesiones, 98.9% val accuracy |
| Modelo de números | Operativo — 10 clases, 1.500 muestras / 4 sesiones, 99.0% val accuracy |
| Modelo de palabras (TCN) | Operativo — 19 clases (18 + `nothing`), 668 takes / 11 sesiones, 98.0% val accuracy (`runs/words_20260718_233249.json`, la corrida más reciente registrada) |
| Validación cruzada leak-free (`--cv`) | Herramienta lista para los 3 modelos; **pendiente re-correrla** sobre el modelo de palabras recapturado — no hay registro de que se haya corrido después del 2026-07-18 |
| Captura (letras, números, palabras) | Completa, multi-sesión |
| Entrenamiento (los 3 modelos) | Completo, con registro por corrida en `runs/` |
| Traducción glosa→frase (LLM) | Completa en escritorio, con fallback offline; pendiente en web |
| Voz de salida / entrada | Completas (escritorio: SAPI5/Whisper; web: Web Speech) |
| Acumulación de letras + subtítulos | Completa (con la salvedad de la sección 9 sobre el cierre de palabra en escritorio) |
| Barra de confianza + top-3 alternativas | Completa |
| Registro de conversación exportable | Completo, formato compartido entre ambas apps |
| App web | Operativa — los 3 modos, voz, conversación bidireccional, grabación/exportación, gate de paridad en verde |

---

## 17. Limitaciones conocidas

- **J y Z** requieren movimiento; no se reconocen como poses estáticas y la
  interfaz las marca con un aviso.
- **Vocabulario de palabras acotado:** 18 señas + `nothing`. Más clases y más
  muestras por seña mejoran la fiabilidad.
- **El accuracy de validación no es el accuracy en vivo.** Los tres modelos
  reportan ≥98% en holdout, optimista cuando las muestras de validación
  vienen de las mismas sesiones que las de entrenamiento. El número honesto
  lo da la validación cruzada agrupada por sesión (`--cv`), pendiente de
  re-correr sobre el modelo de palabras recapturado (sección 16). El modelo
  anterior (pre-recaptura) daba 94.1% leak-free frente a 92.9% de holdout,
  con dos debilidades conocidas a re-verificar: recall de `nothing` 0.76 (a
  veces "dice" una palabra sin que haya seña) y confusión `need`/`want`
  (recall de `need` 0.82). Subir el umbral de confianza no lo corregía en ese
  modelo — el barrido de umbral daba accuracy parejo en todo el rango, señal
  de un problema de datos/separabilidad, no de calibración.
- **Calidad dependiente del entorno:** iluminación y posición de cámara
  influyen. En modo palabras los **hombros deben ser visibles** — las señas
  se anclan al cuerpo.
- **Plataforma de voz (escritorio):** SAPI5 es específico de Windows. La web
  no tiene esa limitación pero depende de qué voces tenga instaladas el
  dispositivo.
- **Rendimiento en móvil (web):** ~15-24fps en modo palabras — techo práctico
  de MediaPipe en teléfono, no un defecto corregible desde la app.
- **Cierre de palabra deletreada en escritorio:** sin mecanismo activo hoy
  (sección 9) — funcionalidad presente en el código de `LetterBuffer` pero
  sin gesto ni tecla que la dispare actualmente.

---

## 18. Decisiones técnicas tomadas (no cambiar sin discutir)

| Decisión | Elección | Razón |
|---|---|---|
| Lengua de señas | ASL | Mejor soporte de herramientas (MediaPipe) |
| Detección de manos/cuerpo | MediaPipe | Sin GPU, 1-2 manos + pose, versión JS para la web |
| Framework ML | Keras/TensorFlow | Portable en principio a TF.js (aunque en la práctica se optó por forward pass manual, sección 14.2) |
| Letras/números: arquitectura | Densa (fully connected) | La entrada es 1 frame de coordenadas, no píxeles |
| Palabras: arquitectura | TCN (Conv1D dilatadas) | Lee el orden del movimiento; escala mejor que media+std; sin fricción de LSTM/GRU al portar |
| Reconocer vs. traducir | El reconocedor da glosas; el LLM da la frase | ASL no conjuga verbos; conjugación/tiempo son tarea de la capa de traducción |
| Clase negativa `nothing` | Una clase explícita para "no hay seña" | Sin ella, el softmax cerrado etiqueta todo como alguna palabra |
| Fuente de datos | Captura propia por webcam, multi-sesión | Sin datasets externos; varias sesiones generalizan, una sola memoriza |
| Normalización | Corrección de aspecto + centrado en muñeca + escala por tamaño de mano | Invariante a posición, tamaño y orientación de cámara |
| Una sola `normalize_landmarks`/`build_word_features`/`resample_sequence` | En `src/utils.py` (y su puerto verificado en `web/js/utils.js`) | Deben ser idénticas en captura, entrenamiento e inferencia |
| Síntesis de voz (escritorio) | `win32com` (SAPI5) en vez de `pyttsx3` | `pyttsx3` no reiniciaba SAPI5 entre llamadas en un loop |
| Almacenamiento de capturas de palabras | Crudo (`pack_word_raw`), featurizado al cargar | Un cambio de normalización futuro no vuelve a invalidar capturas |
| Web: sin TensorFlow.js | Forward pass manual en JS puro | El conversor de TF.js no instala en Windows |
| Web: umbrales en milisegundos | Divergencia deliberada de `config.py` (que cuenta frames) | El fps varía 15-60 según el dispositivo; contar frames penalizaba el móvil |

---

## 19. Pendientes / roadmap

**Bajo esfuerzo**
- Re-correr `training/evaluate.py --cv 5` sobre el modelo de palabras
  recapturado (2026-07-18) para confirmar si `nothing` y la confusión
  `need`/`want` siguen siendo puntos débiles.

**Esfuerzo medio**
- Ampliar el vocabulario de palabras (más clases, más muestras por clase).
- Llevar la traducción glosa→frase (LLM) a la app web.

**Alto esfuerzo**
- Letras J y Z dinámicas — capturarlas como señas de movimiento e integrarlas
  al pipeline de palabras (mismo tratamiento que hello/thanks).

**Muy alto esfuerzo**
- Agregar palabras desde la app — UI de captura en vivo + reentrenamiento +
  recarga de modelo sin reiniciar.
- Feedback de corrección al modelo — guardar correcciones y reentrenar
  offline (sin online learning real, por complejidad).

---

## 20. Equipo y contexto

**Equipo:** Dilan Calvo, Adrián Durán y Nazareth Solís.
**Marco:** COTEPECOS — Especialidad de Desarrollo Web — 2026.
**Lengua de señas objetivo:** ASL (mejor soporte de herramientas); la salida
traducida se entrega en español.
