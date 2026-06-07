# ROADMAP — Sign Language Translator

> Documento de ruta técnica y estratégica del proyecto. Pensado como contexto
> de trabajo del equipo (Dilan, Adrián, Nazareth — COTEPECOS, Desarrollo Web).
> El código y los docs públicos están en inglés; este archivo se mantiene en
> español por ser planeación interna, igual que `CLAUDE.md`.
>
> **Última actualización:** 2026-06-06

---

## Cambio de rumbo (2026-06-06) — leer primero

El spike de WLASL se **archivó**. El dataset llegó con ~50% de videos caídos
(URLs muertas) → ~10 clips por clase, y encima había una brecha de dominio
fuerte (señantes en estudio vs. webcam propia). Resultado: predicciones poco
fiables. La decisión: **volver a la app v1 (captura propia, una mano)** pero
trayendo las dos mejoras importantes que sí valían la pena del spike:

1. **Modelo de palabras temporal (TCN)** en lugar de `media+std` (que era ciego
   al orden y colapsaba al crecer el vocabulario). Ahora vive en la app
   principal: `training/train_words.py` entrena un TCN sobre secuencias
   `(32, 63)` capturadas por webcam — sin dataset externo, sin brecha de dominio.
2. **Capa de traducción LLM** (`src/translator.py`): glosas → frase fluida.

Lo que sigue **igual de válido** de este ROADMAP: el principio de separar
RECONOCER de TRADUCIR (§1), la arquitectura web objetivo (§3), la tabla de
portabilidad (§4) y la verdad de accesibilidad (§8). Lo que cambia es **de
dónde sale el reconocedor**: ya no de WLASL, sino de captura propia
multi-sesión, con una mano (63 valores). Los landmarks de cuerpo (manos+pose,
153) siguen siendo una **opción futura** si una mano se queda corta para señas
más complejas; el código del spike que los implementaba **se eliminó** (habría
que rehacerlo, no es difícil — es el mismo patrón con `PoseLandmarker`).

---

## 0. Objetivo principal (no perderlo de vista)

Permitir una **conversación fluida y en ambos sentidos** entre una persona
Sorda (que usa lengua de señas) y una persona oyente (que no la conoce), de
forma que **ambas entiendan**. No es "reconocer señas" — es comunicación.

Dos direcciones, siempre las dos:

- **A. Sordo → Oyente:** señas → texto en español → voz.
- **B. Oyente → Sordo:** voz → texto en español (legible para el Sordo).

**La versión final y presentable será esta misma, migrada a web y mejorada.**
Por eso cada decisión técnica se toma pensando en que el puerto a web sea
**mecánico, no un rewrite**.

---

## 1. Principio rector: separar RECONOCER de TRADUCIR

El error conceptual más caro sería pedirle al *reconocedor* que resuelva
gramática (conjugación, tiempos). No puede, y WLASL tampoco lo da. La
arquitectura correcta tiene **dos etapas separadas**:

```
ETAPA 1 — RECONOCIMIENTO                 ETAPA 2 — TRADUCCIÓN
(TCN sobre captura propia)               (resuelve conjugación / tiempos)

cámara → landmarks (manos+pose)     →    secuencia de glosas        →   español natural
→ modelo secuencial                      "YO TIENDA IR TERMINAR"    →   "Fui a la tienda."
→ glosa (forma de diccionario)           → LLM (API)                    → voz + subtítulo
```

**Por qué el ASL no se "conjuga" como el español:**
- El tiempo se marca **una vez** con un adverbio temporal (AYER, MAÑANA,
  TERMINAR/PASADO) y todo lo que sigue se entiende en ese tiempo.
- El aspecto (repetido, continuo) se expresa **modificando el movimiento**.
- Muchos verbos se "conjugan" por **dirección en el espacio** (YO-dar-TÚ vs
  TÚ-dar-YO), algo que ni siquiera está en las etiquetas de WLASL.

➡️ **La conjugación y los tiempos se resuelven en la Etapa 2 con un LLM**, no
capturando/reconociendo formas conjugadas. Esto es lo que se intentó antes de
otra manera y se quitó; ahora regresa, bien ubicado.

---

## 2. Estado actual (qué hay y qué se validó)

### App de escritorio — base activa (captura propia, una mano)

| Pieza | Estado |
|------|--------|
| Detección de manos (MediaPipe, 21 landmarks = 63) | ✅ `src/detector.py` |
| Letras A–Y (poses estáticas) | ✅ entrenado y operativo |
| Voz offline (SAPI5/win32com) | ✅ `src/voice.py` |
| Speech-to-text (Whisper, push-to-talk) | ✅ `src/speech_input.py` |
| Overlay + subtítulos + log exportable | ✅ `src/overlay.py`, `src/conversation_log.py` |
| **Modelo de palabras: TCN temporal** sobre secuencias `(32, 63)` | ✅ pipeline listo — **falta capturar y entrenar** |
| Clase negativa `nothing` (silencio cuando no se signa) | ✅ en captura + entrenamiento |
| **Capa de traducción LLM** (glosas → español) | ✅ `src/translator.py`, tecla T, con fallback offline |
| Números 0–9 | ⏳ infra lista, modelo pendiente (diferido por decisión) |

**El cambio clave vs. la v1 anterior:** el modelo de palabras pasó de `media+std`
(ciego al orden) a un **TCN** que lee el movimiento en el tiempo. Misma captura
propia por webcam, pero ahora el formato es **secuencia ordenada** (`.npy`), no
un resumen. Esto es lo que permite escalar el vocabulario.

**Pendiente inmediato (lo único que bloquea palabras):** capturar el vocabulario
**en varias sesiones** (otro día/luz/ropa) con `capture/capture_words.py` y
entrenar con `training/train_words.py`. Datos de una sola sesión memorizan la
sesión y fallan en vivo — es la lección que dejó el spike (val=100% engañoso).

### Spike WLASL — eliminado
Fue un experimento aislado (carpeta `wlasl_spike/`, 214 MB) que **se borró**. Su
único aporte de valor —la arquitectura TCN— ya vive en la app. Hallazgo técnico
que conviene recordar para un futuro puerto a landmarks de cuerpo: en MediaPipe
0.10.35 el `mp.solutions.holistic` legado ya no existe; hay que usar
**HandLandmarker + PoseLandmarker** por separado (mismos `.task` sirven en JS).

---

## 3. Arquitectura objetivo (web, final)

```
┌──────────────────────── DIRECCIÓN A: Sordo → Oyente ────────────────────────┐
│ cámara → MediaPipe.js (manos+pose) → normalize_body (JS)                     │
│   → TCN en TensorFlow.js → flujo de glosas → [pausa = fin de frase]          │
│   → backend /translate (LLM) → frase en español                             │
│   → Web Speech TTS (voz) + subtítulo en pantalla                            │
└─────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────── DIRECCIÓN B: Oyente → Sordo ────────────────────────┐
│ micrófono → speech-to-text (Web Speech / Whisper-WASM / backend)            │
│   → texto español → [opcional] LLM simplifica / reordena a glosa            │
│   → en pantalla para la persona Sorda (+ avatar en el futuro)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implicación clave:** la web **necesita un backend mínimo** porque una API key
de LLM no puede vivir en el cliente. Ese backend es un proxy `/translate`
(y, si se elige, también puede servir el STT pesado).

→ **Forma del sistema web:** *frontend estático (cámara + MediaPipe.js + TCN en
TF.js + Web Speech) + backend delgado (proxy LLM, quizá STT).*

---

## 4. Tabla de portabilidad: escritorio (spike) ↔ web (final)

| Componente | Escritorio (spike) | Web (final) | Esfuerzo de puerto |
|-----------|--------------------|-------------|--------------------|
| Landmarks | MediaPipe Tasks (Python) | `@mediapipe/tasks-vision` (JS), mismos `.task` | **bajo** |
| Normalización | `normalize_body.py` | reimplementar idéntica en JS | bajo (aritmética) |
| Modelo | Keras `.h5` | **TensorFlow.js** (`tensorflowjs_converter`) | bajo **si es TCN** |
| Voz (TTS) | SAPI5 / win32com | Web Speech `SpeechSynthesis` | medio (reimplementar) |
| STT | faster-whisper | Web Speech / Whisper-WASM / backend | medio–alto (decisión §6) |
| Traducción LLM | llamada API | fetch a backend proxy | bajo (mejora en web) |

**La única decisión que de verdad condiciona el puerto:** usar **TCN/1D-CNN** y
no **LSTM/GRU**. Un TCN convierte a TF.js limpio y rápido; los recurrentes dan
fricción (conversión, velocidad, estado manual). Para 100–200 señas el TCN
rinde igual o mejor. **Recomendación firme: TCN.**

---

## 5. Ruta por fases

| Fase | Qué | Dónde | Estado |
|------|-----|-------|--------|
| **0** | Arquitectura TCN + capa LLM en la app principal | Python (app) | ✅ hecho |
| **1** | **Capturar vocabulario multi-sesión** (~15 señas, crecer a ~100) → entrenar TCN → probar en vivo | captura + entrenamiento | ⏳ **siguiente paso** |
| **2** | Curar y ampliar vocabulario (frecuentes + visualmente distintas); ajustar umbrales | datos | pendiente |
| **3** | Llave de API real para la capa LLM; pulir la dirección B (oyente→sordo) | config + UX | pendiente |
| **4** | **Puerto web**: MediaPipe.js + `normalize_landmarks` JS + TCN en TF.js + Web Speech + backend proxy LLM | Web (final) | pendiente |
| **5** | Mejoras (ver §7); evaluar si conviene subir a landmarks de cuerpo (referencia: spike) | futuro | — |

**Criterio de éxito de la Fase 1 (decide si el enfoque escala):** capturar en
2–3 sesiones distintas y medir la precisión en vivo sobre señas frescas (no del
mismo día). Si las señas curadas y visualmente distintas se reconocen bien,
seguimos ampliando vocabulario; si no, ajustamos features/augmentation antes de
invertir en web.

---

## 6. Decisiones clave abiertas

1. **TCN vs LSTM** para el modelo secuencial → *recomendado TCN* (por el web).
2. **STT en web:**
   - Web Speech `SpeechRecognition`: cero peso, pero solo Chrome/Edge, manda
     audio a Google, requiere internet.
   - Whisper en navegador (transformers.js / whisper.cpp WASM): offline y
     privado, pero descarga pesada y más lento.
   - Backend con faster-whisper: mejor calidad, reutiliza código actual, exige
     servidor.
3. **Cuándo arrancar el backend mínimo** (lo necesita el LLM en web).
4. **Nivel de la Dirección B** (oyente→sordo): texto plano → texto simplificado
   por LLM → avatar de señas (futuro). Ver §8.

---

## 7. Posibilidades y mejoras futuras

### Reconocimiento — cómo leer el flujo de señas
| Enfoque | Qué es | Cuándo |
|--------|--------|--------|
| Aislado + gate de movimiento | una seña por pausa | **MVP** (lo planeado) |
| Ventana deslizante | clasificar continuo y suavizar | mejora de fluidez, sin datos nuevos |
| CSLR continuo (CTC seq2seq) | glosas sin segmentar | research; WLASL no trae datos continuos |

### Features (baratas, suben precisión en datos pequeños)
- **Velocidad** (deltas entre frames), **distancia entre manos**, **mano↔cara**.
- Augmentation de keypoints: espejo, *time-warp*, jitter, dropout de frames/joints.
- **Fine-tune / adaptación de dominio** con capturas propias de la webcam.

### Datos / vocabulario
- Curar señas **conversacionales frecuentes y visualmente distintas** (si dos se
  parecen mucho, cambiar una hasta tener más datos).
- Captura propia **multi-sesión** como fuente principal; crecer el vocabulario
  en bloques y medir en vivo antes de seguir ampliando.

### Dirección B (oyente → sordo), de menor a mayor esfuerzo
1. **Texto plano** (MVP). El LLM puede **simplificar** a lenguaje claro.
2. **Texto + glosa reordenada** (español → orden tipo-ASL) como apoyo visual.
3. **Avatar / video de señas** (síntesis): el ideal, pero es un proyecto en sí.

### UX / presentación web
- Vista de conversación a dos paneles (lado Sordo / lado oyente), subtítulos en
  vivo, indicador de turno, transcripción exportable (ya existe el concepto en
  v1: `conversation_log`).

---

## 8. Una verdad de accesibilidad (para tenerla nombrada)

Para muchas personas Sordas, **el español escrito es una segunda lengua** y su
nivel de lectura varía. Mostrar texto literal en la Dirección B no garantiza
comprensión plena. El MVP usa texto (y LLM que simplifica), pero el ideal
pedagógico es la salida en señas (avatar). Mencionarlo demuestra que el
proyecto entiende el objetivo de fondo, no solo el técnico.

---

## 9. Riesgos e incógnitas (que el spike debe medir)

- **Generalización entre sesiones**: el riesgo nº1. Datos de una sola sesión
  memorizan esa sesión → val alto, falla en vivo. Lo mitiga la captura
  multi-sesión (la mide la Fase 1, midiendo en vivo sobre señas frescas).
- **Señas visualmente similares** al crecer el vocabulario → curar señas
  distintas; subir a 2 manos/cuerpo si hace falta separar más.
- **Pausas como fronteras** de frase: simplificación que falla con signado
  fluido y co-articulado.
- **FPS y privacidad** de MediaPipe/STT/TTS en el navegador (puerto web).
- **Una mano (63) vs señas de dos manos / ubicadas en el cuerpo**: el límite del
  enfoque actual; subir a landmarks de cuerpo es la palanca si aparece.

---

## 10. Glosario de apoyo

- **Glosa:** etiqueta de una seña en su forma de diccionario (citation form).
  Ej.: "IR", "TIENDA". No es la frase en español, es el "token" del ASL.
- **WLASL:** *Word-Level ASL*, dataset de ~2000 glosas en ~21k videos de
  señantes reales. Subsets listos: WLASL100/300/1000/2000.
- **TCN:** *Temporal Convolutional Network*, red de convoluciones 1D sobre la
  secuencia temporal. Alternativa a LSTM/GRU; más rápida y web-portable.
- **CSLR:** *Continuous Sign Language Recognition*, reconocer señas en un flujo
  continuo sin segmentar previamente. Más difícil que el reconocimiento aislado.
- **Landmark:** punto clave (coordenada x,y,z) de mano o cuerpo que da MediaPipe.
- **Reconocimiento aislado:** clasificar un clip recortado que contiene **una**
  seña (lo que entrena WLASL). Se complementa con el gate de movimiento para
  segmentar en vivo.

---

## 11. Referencias

- Cómo capturar y entrenar: ver `README.md` (sección "Make it your own")
- Estado/decisiones internas: ver `CLAUDE.md`
- Dataset WLASL (referencia histórica, ya no se usa): https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
