# Traductor de Lenguaje de Señas — Documento Técnico

> **Naturaleza de este documento.** Documento base (insumo) centrado
> exclusivamente en lo técnico: tecnologías, lenguaje, arquitectura, modelos y
> decisiones de ingeniería. Describe la **aplicación de escritorio en Python**
> (donde vive el pipeline de captura y entrenamiento) y, en la sección 15, la
> **versión Web**, que hoy ya está construida y operativa. Pensado para servir
> de fuente a documentos finales; por eso es exhaustivo.
>
> Última revisión de estado: 2026-07-19.

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

**Normalización (clave).** Antes de clasificar, los landmarks se normalizan en
tres pasos: primero se **deshace el estiramiento por-eje de MediaPipe** (x se
normaliza por el ancho del frame e y por el alto — z escala como x —, así que
la misma seña física produce vectores distintos en una webcam 16:9 que en un
teléfono vertical 9:16; con el aspect ratio del frame se corrige: `y /= aspecto`,
x y z intactos), luego se **centran en la muñeca** y se **escalan** por el
tamaño de la mano (muñeca → nudillos). Así la representación es **invariante a
la posición, al tamaño y a la orientación de la cámara**. Esta normalización
(`normalize_landmarks`) es una **única fuente de verdad** usada idénticamente en
captura, entrenamiento e inferencia; si difirieran, el modelo fallaría. Las
capturas nuevas guardan el aspecto por fila (columna `aspect` del CSV); las
anteriores a ese cambio usan `LEGACY_CAPTURE_ASPECT` (16:9, la webcam original).

> El pipeline de **palabras** también aplica la corrección (forma de mano Y
> ancla de cuerpo, `build_word_features` con `aspect` obligatorio) desde
> 2026-07-12. Los datos y el modelo anteriores a ese cambio eran features ya
> procesados sin corrección posible; quedaron archivados en
> `data/real_capture/words_legacy/` y `model/legacy/`, y el vocabulario se
> **recapturó completo** con el formato nuevo (11 sesiones, 668 takes): cada
> take guarda landmarks **crudos**
> (`(n_frames, 131)`: dos manos + hombros + aspecto por frame, layout en
> `src/utils.py::pack_word_raw`), y la featurización + el resampleo a
> `WORD_SEQ_LEN` ocurren al cargar (`train_words.py`) — así ningún cambio
> futuro de normalización o de longitud de secuencia vuelve a invalidar datos.

---

## 4. Pipeline en tiempo real (por frame)

```
Cámara (OpenCV)
  → MediaPipe detecta la mano y extrae 21 landmarks (x, y, z) = 63 valores
  → normalize_landmarks (corrección de aspecto + centrado en muñeca + escala invariante)
  → red neuronal clasifica la seña
  → suavizado temporal (PredictionSmoother) elimina parpadeo de un solo frame
  → predicción estable dibujada en pantalla (+ voz / subtítulos)
```

El **suavizado temporal** (`PredictionSmoother`) usa una ventana de votación:
una predicción solo se confirma si aparece un número mínimo de veces dentro de
las últimas N frames. Esto elimina el parpadeo entre clases parecidas y evita
que un frame ruidoso dispare una predicción falsa.

---

## 5. Los tres modelos de reconocimiento

El sistema entrena sus modelos **directamente con landmarks capturados por
webcam** (datos propios), no con datasets externos.

| Modelo | Entrada | Detecta | Arquitectura | Archivo |
|--------|---------|---------|--------------|---------|
| **Letras** | 63 valores (1 frame, una mano) | Letras estáticas A–Y | Red densa (fully connected) | `model/model_one_hand.h5` |
| **Números** | 63 valores (1 frame, una mano) | Dígitos 0–9 | Red densa (fully connected) | `model/model_numbers.h5` |
| **Palabras** | Secuencia `(32, 130)` (2 manos, anclada al cuerpo) | Señas dinámicas / palabras | **TCN** temporal (Conv1D dilatadas) | `model/model_words.h5` |

### 5.1. Modelo de letras (estáticas)

- **24 clases:** A–Y, **sin J ni Z** (estas requieren movimiento y no se pueden
  distinguir de un solo frame; la interfaz las marca como "requires motion").
- **Arquitectura densa:** la entrada es un único frame de coordenadas (no
  píxeles), por lo que una red totalmente conectada es la elección adecuada.
- Etiquetas en `model/labels_one_hand.json`.
- Entrenado con 3 750 muestras de 9 sesiones.

### 5.1b. Modelo de números (estáticos)

- **10 clases:** dígitos 0–9. Mismo formato de entrada, misma arquitectura y
  mismo pipeline de captura que las letras (`capture_numbers.py`,
  `train_numbers.py`); etiquetas en `model/labels_numbers.json`.
- **Modelo separado de las letras a propósito:** varios dígitos ASL colisionan
  con letras (2=V, 6=W, 9=F). En un único conjunto cerrado el clasificador
  tendría que elegir entre dos respuestas igualmente correctas; separarlos deja
  que el modo activo desambigüe.
- Entrenado con 1 500 muestras de 4 sesiones.

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

1. **Capturar** (`capture/capture_letters.py`, `capture/capture_numbers.py`,
   `capture/capture_words.py`): graba muestras. Las palabras se guardan como una
   secuencia `.npy` por toma más un `manifest.csv`. Re-ejecutar **agrega**
   muestras (dataset multi-sesión).
2. **Entrenar** (`training/train_letters.py`, `training/train_numbers.py`,
   `training/train_words.py`): cada entrenador descubre sus datos solo (las
   palabras leen el `manifest.csv`; letras y números toman todos los CSV de su
   carpeta — sin editar rutas) y produce el `.h5`. Cada corrida deja un registro
   comparable en `runs/<modelo>_<timestamp>.json` (datos, sesiones, accuracy,
   configuración, git SHA).
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
| Modelo de letras (A–Y estáticas) | Entrenado y operativo — 3 750 muestras / 9 sesiones, 98.9% val accuracy |
| Modelo de números (0–9) | Entrenado y operativo — 1 500 muestras / 4 sesiones, 99.0% val accuracy |
| Modelo de palabras (TCN) | Entrenado y operativo — vocabulario recapturado con corrección de aspecto: 18 glosas + `nothing`, 668 takes / 11 sesiones, 98.0% val accuracy |
| Captura (letras, números, palabras) | Completa y multi-sesión |
| Entrenamiento (los 3 modelos) | Completo, con registro por run en `runs/` |
| Evaluación leak-free (`--cv`, agrupada por sesión) | Herramienta completa para los 3 modelos; **pendiente re-correrla** sobre el modelo de palabras recapturado |
| Traducción glosa→frase (LLM) | Completa en escritorio (con fallback offline) |
| Voz de salida (TTS) | Completa |
| Speech-to-text (Whisper) | Completo |
| Acumulación de letras + subtítulos | Completa |
| Barra de confianza + top-3 alternativas | Completo |
| Registro de conversación exportable | Completo |
| **Versión Web** | **Operativa** — los 3 modos, voz, conversación bidireccional, grabación/exportación (ver sección 15) |
| Traducción glosa→frase en la Web | Pendiente (existe solo en escritorio) |
| Letras J y Z (movimiento) | Pendiente (la interfaz las marca) |
| Agregar palabras desde la app | Pendiente |

---

## 14. Limitaciones técnicas actuales

- **J y Z** requieren movimiento; aún no se reconocen como poses estáticas.
- **Vocabulario de palabras acotado:** 18 palabras + `nothing`. Más clases y
  más muestras por seña mejoran la fiabilidad (clases balanceadas a 34 takes,
  `nothing` a 68).
- **El accuracy de validación no es el accuracy en vivo.** Los tres modelos
  reportan >98% en holdout, pero ese número es optimista cuando las muestras de
  validación salen de las mismas sesiones que las de entrenamiento. El número
  honesto lo da la validación cruzada agrupada por sesión (`--cv`), que **está
  pendiente de re-correr** sobre el modelo de palabras recapturado. El modelo
  anterior (pre-recaptura) daba 94.1% leak-free frente a 92.9% de holdout, con
  dos debilidades conocidas que conviene volver a verificar: `nothing` (recall
  0.76 — a veces "dice" una palabra cuando no hay seña) y la confusión
  `need`/`want`. En ese modelo, subir `WORD_CONFIDENCE_THRESHOLD` no lo
  corregía: el barrido de umbral daba accuracy parejo en todo el rango, así que
  era un problema de datos/separabilidad, no de calibración.
- **Calidad dependiente del entorno:** iluminación y posición de cámara influyen;
  capturar muestras propias en el setup habitual mejora la precisión. En modo
  palabras los **hombros deben ser visibles** (las señas se anclan al cuerpo).
- **Plataforma de voz:** la síntesis offline (SAPI5) es específica de Windows.
  La versión web no tiene esa limitación (usa las voces del sistema vía Web
  Speech), pero depende de qué voces tenga instaladas el dispositivo.
- **Rendimiento en móvil (web):** ~15–24 fps en modo palabras. Es el techo
  práctico de MediaPipe en teléfono, no un defecto corregible desde la app.

---

## 15. Versión Web (construida y operativa)

Lo que en su momento fue el plan de migración **ya está implementado** en
`web/`: una sola página, **sin backend, sin build step y sin framework**, que
corre los tres modos (Letters / Numbers / Words) enteramente en el navegador,
en escritorio y en teléfono. La documentación detallada vive en
[`web/README.md`](../web/README.md).

Cómo se resolvió cada punto del plan original:

- **Inferencia en el navegador — sin TensorFlow.js.** El plan era convertir los
  `.h5` a TF.js, pero **su conversor no instala en Windows**. La solución fue
  exportar los pesos a JSON (`tools/export_model_json.py`) e implementar los
  forward passes a mano en JavaScript: `DenseModel` (Dense + BatchNorm, para
  letras y números) y `TCNModel` (Conv1D causal-dilatada + GlobalAvgPool +
  Dense, para palabras). Menos dependencias y control total del costo por
  frame; el precio es mantener esas implementaciones a mano.
- **MediaPipe.js:** se usa `@mediapipe/tasks-vision`, **vendorizado localmente y
  con versión fijada** (nunca desde CDN): un detector de otra versión podría
  producir landmarks sutilmente distintos a los de los datos de entrenamiento.
  La pose (ancla de cuerpo) se crea de forma perezosa, solo al entrar a modo
  palabras.
- **Normalización idéntica en JS:** `normalizeLandmarks` y `buildWordFeatures`
  están reimplementados en `web/js/utils.js` y **verificados por un banco de
  pruebas** (`web/js/utils.test.html`) que reproduce fixtures generados desde
  datos reales de captura. Es la única garantía de que web y escritorio ven lo
  mismo; correrlo es obligatorio antes de desplegar.
- **Voz:** salida por **Web Speech** (con selector de voces del dispositivo) y
  dictado del oyente por `SpeechRecognition` donde exista — sustituyen a SAPI5
  y a Whisper local, que son dependencias de escritorio.
- **Alojamiento:** despliegue estático (Netlify, ya configurado en
  `netlify.toml`). **La inferencia es 100% del lado del cliente**: el video
  nunca sale del dispositivo, lo que además resuelve la privacidad por diseño.
- **Funciones propias de la web:** panel de conversación bidireccional (el
  oyente escribe o dicta y su mensaje aparece en grande sobre la cámara),
  grabación y exportación del transcript en TXT/CSV **byte-compatible con los
  archivos del escritorio**, diálogos de ayuda y ajustes, selector de cámara.

**Divergencia deliberada respecto de `config.py`:** el escritorio cuenta
*frames* para las ventanas de suavizado porque su cámara corre a tasa fija; la
web va de ~15 a 60 fps según el dispositivo, así que allí los mismos umbrales
se expresan en **milisegundos** (cada valor equivale a su gemelo de `config.py`
a 30 fps). Sin esa divergencia, en un teléfono había que sostener cada seña
casi el triple de tiempo real.

**Lo que falta en la web:** la traducción glosa → frase con LLM (sección 6),
que hoy existe solo en la app de escritorio.

---

## 16. Requisitos y ejecución

### App de escritorio (Python)

- **Requisitos:** Python 3.10+ y una webcam.
- **Instalación:** entorno virtual + `pip install -r requirements.txt`.
- **Ejecución:** `python main.py`. Los modelos preentrenados vienen incluidos, así
  que la app corre sin descargar datasets ni entrenar.
- **Opcional:** definir `ANTHROPIC_API_KEY` para habilitar la traducción a frase
  natural (sin ella, la app funciona y solo une las glosas).

### App web

- **Requisitos:** un navegador reciente (Chrome, Edge o Safari) con cámara. No
  hay instalación ni dependencias que compilar.
- **Ejecución local:** `cd web && python -m http.server 8000`, luego abrir
  `http://localhost:8000`. La cámara funciona en `localhost` por la excepción
  de contexto seguro; en cualquier otro host exige HTTPS.
- **Banco de pruebas:** `http://localhost:8000/js/utils.test.html` debe estar
  todo en verde antes de desplegar.
- **Despliegue:** estático. `netlify.toml` (raíz del repo) ya publica `web/` en
  cada push.
- **Tras reentrenar un modelo:** `python tools/update_web.py <letters|numbers|words>`
  regenera pesos, fixtures y etiquetas para la web. Omitir este paso deja el
  sitio sirviendo el modelo viejo en silencio.
