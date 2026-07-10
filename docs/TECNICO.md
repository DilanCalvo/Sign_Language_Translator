# Traductor de Lenguaje de Señas — Documento Técnico

> **Naturaleza de este documento.** Documento base (insumo) centrado
> exclusivamente en lo técnico: tecnologías, lenguaje, arquitectura, modelos y
> decisiones de ingeniería. Describe la **versión actual en Python** (lo que
> existe y funciona) y, al final, la **visión técnica de la versión Web** futura.
> Pensado para servir de fuente a documentos finales; por eso es exhaustivo.

---

## 1. Panorama técnico en una frase

Una aplicación de visión por computadora en **Python** que, a partir del video de
una webcam, extrae los puntos clave de la mano con **MediaPipe**, los clasifica
con **redes neuronales** (TensorFlow/Keras), estabiliza la predicción y la
convierte en texto y voz; en sentido inverso, transcribe voz a texto con
**Whisper**, y refina las señas reconocidas a lenguaje natural mediante un
**modelo de lenguaje (Claude)**.

---

## 2. Stack tecnológico

| Capa | Tecnología | Rol |
|------|-----------|-----|
| Lenguaje | **Python 3.10+** | Lenguaje principal de toda la aplicación |
| Detección de manos/cuerpo | **MediaPipe** | Extrae landmarks (puntos clave) de manos y pose |
| Visión / cámara | **OpenCV** | Captura de cámara, dibujo de overlay, manejo de ventana |
| Machine Learning | **TensorFlow / Keras** | Definición, entrenamiento e inferencia de los modelos |
| Utilidades ML | **scikit-learn** | Apoyo en preparación de datos / métricas |
| Cómputo numérico | **NumPy** | Manejo de vectores y secuencias de landmarks |
| Voz de salida (TTS) | **SAPI5 vía win32com** | Síntesis de voz offline en Windows |
| Voz de entrada (STT) | **Whisper (faster-whisper)** | Transcripción de habla a texto |
| Traducción a lenguaje natural | **Claude (API de Anthropic)** | Convierte glosas ASL en frases fluidas |

### Por qué cada elección

- **Python:** ecosistema maduro de visión y ML; rapidez de prototipado.
- **MediaPipe:** detecta manos (1–2) y pose sin GPU, en tiempo real, y tiene
  versión para JavaScript, lo que facilita la futura migración a web.
- **TensorFlow/Keras:** portable a la web con **TensorFlow.js sin reentrenar**, lo
  que alinea el prototipo con la versión Web futura.
- **win32com (SAPI5)** en lugar de pyttsx3: pyttsx3 no reiniciaba correctamente el
  motor SAPI5 entre llamadas dentro de un bucle (solo hablaba la primera vez);
  usar win32com directamente evita ese fallo.
- **Whisper (faster-whisper):** transcripción robusta, corre en CPU, modelos de
  distintos tamaños según el equilibrio velocidad/precisión deseado.
- **Claude (LLM):** la conversión de señas sueltas a frase natural es una tarea de
  lenguaje; se delega a un modelo de lenguaje en lugar de codificar reglas
  gramaticales a mano.

---

## 3. Idea técnica central: clasificar geometría, no píxeles

El sistema **no clasifica imágenes**. MediaPipe convierte cada mano en **21 puntos
clave** (landmarks) con coordenadas (x, y, z) = **63 valores numéricos**. Los
modelos aprenden la **geometría** de la mano (posición de las articulaciones), no
su apariencia.

Ventaja: el sistema es robusto a iluminación, tono de piel, fondo y distancia a
la cámara, porque todo eso desaparece al quedarnos solo con coordenadas. Además,
la entrada es muy ligera (decenas de números por frame en lugar de una imagen
completa), lo que permite tiempo real sin GPU.

**Normalización (clave).** Antes de clasificar, los landmarks se normalizan:
se **centran en la muñeca** y se **escalan** por el tamaño de la mano
(muñeca → nudillos). Así la representación es **invariante a la posición y al
tamaño** de la mano en la imagen. Esta normalización (`normalize_landmarks`) es
una **única fuente de verdad** usada idénticamente en captura, entrenamiento e
inferencia; si difirieran, el modelo fallaría.

---

## 4. Pipeline en tiempo real (por frame)

```
Cámara (OpenCV)
  → MediaPipe detecta la mano y extrae 21 landmarks (x, y, z) = 63 valores
  → normalize_landmarks (centrado en muñeca + escala invariante)
  → red neuronal clasifica la seña
  → suavizado temporal (PredictionSmoother) elimina parpadeo de un solo frame
  → predicción estable dibujada en pantalla (+ voz / subtítulos)
```

El **suavizado temporal** (`PredictionSmoother`) usa una ventana de votación:
una predicción solo se confirma si aparece un número mínimo de veces dentro de
las últimas N frames. Esto elimina el parpadeo entre clases parecidas y evita
que un frame ruidoso dispare una predicción falsa.

---

## 5. Los dos modelos de reconocimiento

El sistema entrena sus modelos **directamente con landmarks capturados por
webcam** (datos propios), no con datasets externos.

| Modelo | Entrada | Detecta | Arquitectura | Archivo |
|--------|---------|---------|--------------|---------|
| **Letras** | 63 valores (1 frame, una mano) | Letras estáticas A–Y | Red densa (fully connected) | `model/model_one_hand.h5` |
| **Palabras** | Secuencia `(32, 130)` (2 manos, anclada al cuerpo) | Señas dinámicas / palabras | **TCN** temporal (Conv1D dilatadas) | `model/model_words.h5` |

### 5.1. Modelo de letras (estáticas)

- **24 clases:** A–Y, **sin J ni Z** (estas requieren movimiento y no se pueden
  distinguir de un solo frame; la interfaz las marca como "requires motion").
- **Arquitectura densa:** la entrada es un único frame de coordenadas (no
  píxeles), por lo que una red totalmente conectada es la elección adecuada.
- Etiquetas en `model/labels_one_hand.json`.

### 5.2. Modelo de palabras (dinámicas) — TCN

Las señas reales **tienen movimiento y orden**: no basta con la forma de la mano,
importa *cómo* se mueve. Por eso el modelo de palabras es un **TCN (Temporal
Convolutional Network)**: convoluciones 1D dilatadas que leen la **secuencia
ordenada** de frames.

- **Entrada:** una secuencia de **32 frames** de **130 valores** cada uno.
- **130 valores por frame (body-anchored, dos manos):**
  - Mano izquierda: 63 (forma) + 2 (posición de la muñeca) = 65
  - Mano derecha: 63 (forma) + 2 (posición de la muñeca) = 65
  - Total = **130**
  - La "forma" son los 21 landmarks normalizados (invariante a posición/tamaño).
  - La "posición de muñeca" es (x, y) **respecto al centro de los hombros**,
    escalada por el ancho de hombros: el **ancla de cuerpo** (vía MediaPipe pose).
    Esto distingue señas con la misma forma pero en distinta ubicación (mano al
    pecho vs. a la frente) y soporta señas de dos manos.
  - Una mano ausente se representa con ceros (patrón que el modelo aprende como
    "esa mano no está").
- **Clase negativa `nothing`:** una clase explícita para "no hay seña". Es
  esencial: sin ella, un clasificador de conjunto cerrado (softmax) etiqueta
  *todo* como alguna palabra y "siempre dice algo". La clase negativa absorbe los
  frames ambiguos (reposo, transiciones, letras en movimiento).
- **Glosas en inglés** (forma de diccionario ASL); etiquetas en
  `model/labels_words.json`. El vocabulario lo define la lista `WORDS` en el
  script de captura.
- **Por qué TCN y no LSTM/GRU:** el TCN lee el orden del movimiento, escala mejor
  que un resumen `media+std` (que es ciego al orden) y se convierte limpiamente a
  TensorFlow.js para la web (los recurrentes dan fricción al portar).

Construcción de features: la función `build_word_features` (en `src/utils.py`) es
la **única fuente de verdad** y se usa idénticamente en captura, entrenamiento e
inferencia. El buffer en vivo se **remuestrea** a 32 frames con
`resample_sequence` para que toda secuencia tenga la misma forma.

---

## 6. Segunda etapa: traducción glosa → frase (capa LLM)

El reconocedor produce **glosas** (palabras clave en inglés, forma de cita: p. ej.
`WANT DRINK NOW`). Una segunda etapa (`src/translator.py`) envía esas glosas a
**Claude (API)** y obtiene una **frase fluida en español** ("Quiero tomar algo
ahora").

Principio de diseño **"reconocer, luego traducir"**:

- ASL **no conjuga verbos** ni sigue el orden gramatical del español. La
  conjugación y el tiempo verbal **viven en la capa de traducción, no en el
  reconocedor**: enseñarle al reconocedor formas conjugadas sería lingüísticamente
  incorrecto y combinatoriamente explosivo.
- Corre en un **hilo aparte** (no bloquea la cámara).
- **Degrada con elegancia:** sin API key o sin internet, junta las glosas tal
  cual (la app sigue funcionando offline).
- Modelo por defecto: un modelo rápido y económico (Claude Haiku), adecuado para
  esta tarea acotada. La llave se lee de la variable de entorno
  `ANTHROPIC_API_KEY`.

---

## 7. Entrada y salida de voz

### 7.1. Voz de salida — TTS (`src/voice.py`)

- Síntesis de voz **offline** con **SAPI5 vía win32com**, en un **hilo daemon**
  para no bloquear el bucle de cámara.
- Lee en voz alta las letras/palabras confirmadas.

### 7.2. Voz de entrada — STT (`src/speech_input.py`)

- Transcripción con **Whisper (faster-whisper)** en un **hilo daemon**.
- **Push-to-talk** con la tecla **P**.
- Tamaño de modelo configurable (`tiny` por defecto para demos; `base`/`small`
  para más precisión). El modelo se descarga una vez al primer uso.
- Idioma autodetectado o fijado (p. ej. `es`).

---

## 8. Capa de presentación / interacción

### 8.1. Overlay y buffers (`src/overlay.py`)

- **LetterBuffer:** acumula letras deletreadas (con cooldown, soporte de
  `del`/`space`) y las muestra como subtítulo.
- **WordBuffer:** arma la frase de señas reconocidas.
- **SpeechBuffer:** barra superior con la transcripción de la voz de la persona
  oyente.
- **Barra de confianza:** verde (>80%), amarillo (>60%), rojo por debajo.
- **Top-3 alternativas:** cuando la confianza es baja (<70%) se muestran las 3
  mejores candidatas con su porcentaje, en vez de una única adivinanza. Esto
  **desacopla** "lo que se acepta y se actúa" (umbral de 0.80) de "las pistas que
  se muestran" (<0.70). El sistema es honesto sobre su incertidumbre.

### 8.2. Registro de conversación (`src/conversation_log.py`)

- `ConversationLog` acumula lo conversado con marca de tiempo.
- Tecla **E** exporta a **TXT + CSV** en `logs/`.
- Captura señas (palabras), palabras deletreadas y transcripciones de voz.

### 8.3. Atajos de teclado

| Tecla | Acción |
|-------|--------|
| **Q** | Salir |
| **L** | Modo letras |
| **W** | Modo palabras |
| **N** | Modo números (reservado) |
| **T** | Traducir la frase señada a texto fluido (modo palabras) |
| **P** | Activar/desactivar micrófono (speech-to-text) |
| **E** | Exportar el registro de conversación (TXT + CSV) |

---

## 9. Configuración centralizada (`config.py`)

`config.py` es la **única fuente de verdad** de todos los parámetros: modo activo,
umbrales de confianza, ventanas de suavizado, detección de movimiento, parámetros
de captura, rutas de modelos, configuración de cámara, voz, speech-to-text y
traducción. Está fuertemente comentado: cambiar un valor ahí se propaga a todos
los módulos sin tocar otro archivo. Los parámetros nuevos van aquí, nunca
hardcodeados en los módulos.

El parámetro `MODE` selecciona qué se muestra: `"letters"`, `"words"` (default) o
`"numbers"` (reservado).

---

## 10. Flujo de datos: capturar → entrenar → usar

El sistema se basa en datos propios capturados por webcam (no datasets externos).

```
data/real_capture/
├── letters/   → CSV de poses estáticas
└── words/     → seq/*.npy (secuencias) + manifest.csv
```

1. **Capturar** (`capture/capture_letters.py`, `capture/capture_words.py`): graba
   muestras. Las palabras se guardan como una secuencia `.npy` por toma más un
   `manifest.csv`. Re-ejecutar **agrega** muestras (dataset multi-sesión).
2. **Entrenar** (`training/train_letters.py`, `training/train_words.py`): el
   entrenador de palabras lee el `manifest.csv` directamente (sin editar rutas) y
   produce el `.h5`.
3. **Usar** (`main.py`): carga los modelos y clasifica en tiempo real.

### Disciplina multi-sesión (decisión importante)

Capturar en **varias sesiones** (distintos días, iluminación, ropa, distancia) es
**más importante que la cantidad de tomas**. Un modelo entrenado con muchas tomas
de una sola sesión **memoriza esa sesión** (alta validación, falla en vivo);
varias sesiones lo hacen **generalizar**. Por eso la captura tiene un tope por
sesión y un objetivo acumulado entre sesiones.

---

## 11. Arquitectura del código (responsabilidad única por módulo)

```
Sign_Language_Translator/
├── main.py                  # Punto de entrada (traductor en tiempo real)
├── config.py                # Configuración central (única fuente de verdad)
├── requirements.txt         # Dependencias
│
├── src/
│   ├── detector.py          # Cámara + MediaPipe (landmarks de manos)
│   ├── classifier.py        # Carga modelos y clasifica
│   ├── utils.py             # normalize_landmarks + build_word_features +
│   │                        #   resample_sequence + PredictionSmoother
│   ├── voice.py             # Síntesis de voz (SAPI5/win32com, hilo daemon)
│   ├── overlay.py           # LetterBuffer + WordBuffer + SpeechBuffer
│   ├── conversation_log.py  # Registro exportable (TXT + CSV)
│   ├── speech_input.py      # Whisper push-to-talk (hilo daemon)
│   └── translator.py        # Glosas ASL → frase fluida (Claude API)
│
├── capture/
│   ├── capture_letters.py   # Captura poses estáticas de letras (CSV)
│   ├── capture_numbers.py   # Captura poses estáticas de dígitos 0–9 (CSV)
│   └── capture_words.py     # Captura secuencias dinámicas (.npy + manifest)
│
├── training/
│   ├── train_letters.py     # Entrena el modelo de letras
│   ├── train_numbers.py     # Entrena el modelo de números (auto-descubre CSVs)
│   └── train_words.py       # Entrena el modelo de palabras (TCN)
│
├── model/                   # Modelos .h5, labels .json, modelos MediaPipe
└── data/real_capture/       # letters/ + numbers/ (CSV) + words/ (seq/*.npy + manifest)
```

Convenciones: Python 3.10+ (PEP 8); **código, comentarios y docstrings en
inglés** (portafolio público); comentar el "por qué", no lo obvio; manejo de
errores solo en los límites del sistema (cámara, carga de modelos, lectura de
archivos); una sola fuente de verdad para la normalización y las features.

---

## 12. Decisiones técnicas tomadas (y su razón)

| Decisión | Elección | Razón |
|----------|----------|-------|
| Lengua de señas | ASL | Mejor soporte de herramientas (MediaPipe) |
| Detección de manos | MediaPipe | Sin GPU, 1–2 manos, versión JS para futuro web |
| Framework ML | Keras/TensorFlow | Portable a web con TensorFlow.js sin reentrenar |
| Letras: arquitectura | Densa (fully connected) | La entrada es 1 frame de coordenadas, no píxeles |
| Palabras: arquitectura | TCN (Conv1D dilatadas) | Lee el orden del movimiento; escala mejor que media+std; convierte limpio a TF.js. No LSTM/GRU (fricción al portar) |
| Reconocer vs. traducir | Reconocedor da glosas; LLM da la frase | ASL no conjuga; conjugación/tiempo son tarea del LLM |
| Clase negativa `nothing` | Una clase para "no hay seña" | Sin ella el softmax etiqueta todo → "siempre dice algo" |
| Fuente de datos | Captura propia por webcam, multi-sesión | Sin datasets externos; varias sesiones generalizan |
| Normalización | Centrar en muñeca + escalar por tamaño de mano | Invariante a posición y tamaño |
| Voz | win32com (SAPI5) | pyttsx3 no reiniciaba SAPI5 entre llamadas en un loop |

---

## 13. Estado técnico actual

| Área | Estado |
|------|--------|
| Detección de manos (MediaPipe) | Completa |
| Clasificación en tiempo real | Completa |
| Modelo de letras (A–Y estáticas) | Entrenado y operativo |
| Modelo de palabras (TCN) | Entrenado, operativo y validado — 92.9% val accuracy en entrenamiento (6 sesiones, 612 muestras); validación cruzada leak-free (`--cv 5`, 5-fold por sesión) confirma **94.1%** de generalización real entre sesiones |
| Captura de palabras (multi-sesión) | Completa |
| Entrenamiento de palabras (TCN) | Completo |
| Traducción glosa→frase (LLM) | Completa (con fallback offline) |
| Voz de salida (TTS) | Completa |
| Speech-to-text (Whisper) | Completo |
| Acumulación de letras + subtítulos | Completa |
| Barra de confianza + top-3 alternativas | Completo |
| Registro de conversación exportable | Completo |
| Números 0–9 | Pipeline completo (`capture_numbers.py` + `train_numbers.py`, validado end-to-end); falta capturar con cámara y entrenar |
| Letras J y Z (movimiento) | Pendiente (la interfaz las marca) |
| Agregar palabras desde la app | Pendiente |

---

## 14. Limitaciones técnicas actuales

- **J y Z** requieren movimiento; aún no se reconocen como poses estáticas.
- **Vocabulario de palabras acotado:** 16 palabras + `nothing`. Más clases y
  más muestras por seña mejoran la fiabilidad (todas las clases ya balanceadas
  a 34 muestras, `nothing` a 68).
- **Confusión en un cluster de señas y en la clase `nothing`:** la validación
  cruzada leak-free (`training/evaluate.py --cv 5`, 2026-07-09) da 94.1% global
  pero muestra dos puntos débiles persistentes: `nothing` tiene recall 0.76 (a
  veces "dice" una palabra cuando no hay seña) y `need` se confunde con `want`
  (`need` recall 0.82). Subir `WORD_CONFIDENCE_THRESHOLD` no lo corrige — el
  barrido de umbral da accuracy parejo en todo el rango, así que es un problema
  de datos/separabilidad, no de calibración. Fix probable: capturar más
  muestras variadas de esas clases.
- **Números (0–9):** infraestructura lista, falta capturar y entrenar el modelo.
- **Calidad dependiente del entorno:** iluminación y posición de cámara influyen;
  capturar muestras propias en el setup habitual mejora la precisión.
- **Plataforma de voz:** la síntesis offline (SAPI5) es específica de Windows.

---

## 15. Visión técnica de la versión Web (futuro)

La versión actual (Python, escritorio) se diseñó deliberadamente para **facilitar
la migración a la web**. La versión Web es la evolución planteada como producto
final: **alojada en un servidor y accesible desde el navegador**.

Ruta técnica prevista:

- **TensorFlow.js:** los modelos Keras se convierten a TF.js **sin reentrenar**.
  Por esto se eligió Keras y un TCN (las redes recurrentes habrían dado fricción
  al portar).
- **MediaPipe.js:** la detección de manos y pose tiene versión JavaScript, de modo
  que el mismo extractor de landmarks corre en el navegador.
- **Reimplementación idéntica de la normalización en JS:** `normalize_landmarks` y
  el armado de features deben replicarse exactamente en JavaScript para que las
  entradas coincidan con las del entrenamiento.
- **Web Speech API:** para voz de entrada/salida en el navegador (sustituyendo
  SAPI5 y Whisper local, que son dependencias de escritorio).
- **Alojamiento en servidor:** la app deja de instalarse; se sirve por la web,
  pudiendo correr la inferencia en el cliente (navegador) y/o apoyarse en
  servidor según convenga.
- **Mejor interfaz y funcionalidades específicas de web:** UI cuidada, uso desde
  móvil y posibles funciones nuevas que el entorno web habilita.

En síntesis: el prototipo Python **valida la viabilidad técnica**; la versión Web
**la lleva a producción accesible**, reutilizando los modelos y la lógica de
normalización ya validados.

---

## 16. Requisitos y ejecución (versión actual)

- **Requisitos:** Python 3.10+ y una webcam.
- **Instalación:** entorno virtual + `pip install -r requirements.txt`.
- **Ejecución:** `python main.py`. Los modelos preentrenados vienen incluidos, así
  que la app corre sin descargar datasets ni entrenar.
- **Opcional:** definir `ANTHROPIC_API_KEY` para habilitar la traducción a frase
  natural (sin ella, la app funciona y solo une las glosas).
