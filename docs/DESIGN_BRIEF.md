# Signex — Brief funcional para rediseño

> **Para qué sirve este documento.** Describe **qué hace** la aplicación, qué
> elementos existen hoy en pantalla, qué datos maneja y qué restricciones reales
> la condicionan. **No propone ni juzga decisiones visuales** — el diseño es
> justamente lo que se quiere replantear a partir de esta información.
>
> Estado descrito: 2026-07-19. Los textos de interfaz citados están en inglés
> porque la UI del producto está en inglés.

---

## 1. Qué es la aplicación

Un traductor de **Lenguaje de Señas Americano (ASL)** en tiempo real. La cámara
ve las manos de una persona sorda, un modelo reconoce la seña y la app la
convierte en **texto en pantalla y voz**. En sentido inverso, la persona oyente
**escribe o dicta** y su mensaje aparece en grande sobre la cámara para que la
persona sorda lo lea sin salir del encuadre.

No es un lector de texto ni un diccionario: es una **herramienta de
conversación cara a cara entre dos personas que comparten un solo dispositivo**.

Todo el reconocimiento corre **localmente** (en el navegador o en la máquina):
no hay servidor, no hay cuentas, no hay registro, y **el video nunca sale del
dispositivo**.

---

## 2. Superficies del producto

| Superficie | Estado | Qué es |
|---|---|---|
| **App web** (`web/`) | Operativa y completa | El producto real y el objeto del rediseño. Una sola página, JavaScript sin framework ni build step, sin backend. Corre en escritorio y móvil (Chrome/Edge/Safari). |
| **App de escritorio** (`main.py`) | Operativa | Versión Python/OpenCV. Misma lógica de reconocimiento, HUD dibujado sobre el frame de video, control solo por teclado (`Q` salir, `L`/`W`/`N` modo, `E` exportar, `T` traducir, `P` micrófono). Es la herramienta de captura y desarrollo del equipo; secundaria como producto. |

La app web es la que se despliega públicamente (Netlify). Todo lo que sigue
describe la app web salvo que se indique lo contrario.

---

## 3. Quiénes la usan y en qué situación

Dos personas al mismo tiempo, frente a **una sola pantalla**:

- **Signer** (persona sorda o signante). Está **a 0,5–1,5 m de la cámara**,
  encuadrada de la cabeza a la cintura, con las manos en alto. **No puede
  tocar la pantalla mientras firma** y lee a esa distancia. En modo Words
  necesita que **los hombros sean visibles** (el modelo ancla las señas al
  cuerpo).
- **Oyente**. Está al lado o al otro lado del dispositivo, **sí toca la
  pantalla** (escribe, dicta, activa la grabación) y **sí puede oír** la voz
  sintetizada.

Escenario típico: un intercambio corto y no planificado — un mostrador, una
farmacia, un trámite, una consulta. Nadie instala nada ni configura nada antes:
se abre la página, se da permiso de cámara y se conversa.

---

## 4. Los tres modos de reconocimiento

Son excluyentes y se cambian con un toggle. Comparten cámara, detector y panel
de conversación; cambian el modelo, el vocabulario y las reglas de commit.

| Modo | Qué reconoce | Vocabulario actual | Cómo se comporta |
|---|---|---|---|
| **Letters** | Poses estáticas de una mano | 24 letras: **A–Y sin J ni Z** (requieren movimiento) | Se sostiene la pose hasta que "engancha". Las letras se **acumulan** formando una palabra. |
| **Numbers** | Poses estáticas de una mano | Dígitos **0–9** | Idéntico a Letters. Modelo aparte porque los dígitos ASL chocan con letras (2=V, 6=W, 9=F). |
| **Words** | Señas dinámicas, una o dos manos | **18 glosas**: drink, eat, finished, good, hello, help, let, me, more, need, no, please, sorry, thanks, that, want, yes, you | Se firma con naturalidad; la seña se reconoce como unidad completa y se agrega a una frase que se limpia sola tras una pausa larga. |

Notas de vocabulario relevantes para el diseño:

- Las palabras son **glosas en inglés en forma de diccionario** (`WANT DRINK`),
  no frases conjugadas. ASL no conjuga verbos.
- El modelo de palabras tiene además una clase negativa `nothing` que **nunca
  se muestra**: existe para que el sistema sepa callarse cuando no hay seña.
- **No existe una seña de "espacio" ni de "borrar"** en los modos estáticos:
  una palabra deletreada se cierra por **pausa** o con el botón `SPACE`.

---

## 5. Inventario de elementos en pantalla (estado actual)

Lo que hay hoy, agrupado por función. Es el inventario de **información y
controles que el rediseño debe seguir cubriendo** (no una prescripción de
dónde ubicarlos).

### 5.1 Cabecera

| Elemento | Función |
|---|---|
| Marca "Signex" | Identidad. |
| Subtítulo | Cambia con el modo: `"ASL letters, in your browser"` / `"ASL numbers…"` / `"ASL word signs…"`. |
| Toggle de modo | 3 botones: `Letters` / `Numbers` / `Words`. Se deshabilitan mientras carga un modelo. |
| Botón de voz | Activa/desactiva la voz sintetizada. Es un atajo del mismo ajuste que está en Settings. |
| Botón Settings | Abre el diálogo de ajustes. |
| Botón Help | Abre el diálogo de ayuda. |

### 5.2 Barra sobre el video (información *sobre la captura*)

| Elemento | Estados / contenido |
|---|---|
| **Píldora de seguimiento** | 4 estados vivos: `Working…` (arrancando/cargando), `No hand` (no ve manos), `Tracking` (ve manos, quieto), `Signing` (detecta movimiento de seña). Cambia varias veces por segundo. |
| **Badge REC** | Visible solo mientras la grabación está activa. |
| **Control de cámara** | Aparece **solo si hay 2 o más cámaras**. Con exactamente 2 se colapsa a un botón `Flip`; con más, abre un menú con los nombres de los dispositivos. |

### 5.3 El escenario de video

| Elemento | Función |
|---|---|
| Video en vivo | **Espejado** (el usuario se ve como en un espejo). Su relación de aspecto varía: 16:9 en escritorio, vertical en teléfono. |
| Capa de landmarks | Dibuja los 21 puntos de cada mano detectada y sus conexiones; en modo Words, además, **dos puntos en los hombros** (el ancla corporal). |
| **Status bloqueante** | Mensaje centrado *sobre* el video para estados en los que no hay nada que ver detrás: `starting…`, `loading hand detector…`, `requesting camera…`, `loading words…`, y errores (`Camera permission denied…`, `No camera found…`, `The detector failed repeatedly. Reload the page to retry.`, `detector hiccup — retrying…`). |
| **Readout de predicción** | La letra/número/palabra actual en grande (`–` si no hay), con **barra de confianza** y **porcentaje**. Distingue visualmente entre lo que el sistema **aceptó** y lo que solo **está sugiriendo** (una pista sin confirmar). |
| **Mensaje grande del oyente** | El texto que escribió/dictó el oyente, superpuesto en grande sobre el video para que el signer lo lea desde lejos. Se va solo a los **8 s** o al tocarlo. |
| **Strip de subtítulo** | El texto acumulado. En Letters/Numbers son los caracteres deletreados (se muestran los **últimos 28**); en Words, la frase de glosas en curso. |
| Contador de FPS | Diagnóstico permanente. |
| Panel de **alternativas** | Cuando la confianza está por debajo del 70 %, muestra el **top-3** de candidatos con sus porcentajes, para que el usuario entienda entre qué duda el modelo. |

### 5.4 Acciones sobre el reconocimiento

| Control | Función |
|---|---|
| `SPACE` | **Solo en Letters/Numbers.** Cierra manualmente la palabra deletreada: se pronuncia y entra a la conversación. |
| `CLEAR` | **Descarta** el texto en curso (strip + buffers). No toca la conversación ya registrada. |
| Pie de página | Aviso de privacidad, con texto propio por modo (`"Everything runs locally in your browser; no video leaves your device."`). |

### 5.5 Panel de conversación

Es el registro común de **ambos lados** y sobrevive a los cambios de modo.

| Elemento | Función |
|---|---|
| Botón `Rec` | Empieza/detiene la grabación del transcript. Es **consentimiento para guardar**, no un interruptor para poder hablar: la conversación funciona igual sin grabar. |
| Botón `Export` | Descarga el transcript grabado (TXT + CSV). Deshabilitado mientras no haya nada grabado. |
| Botón de borrar | Destructivo: **se arma al primer toque y ejecuta al segundo** (se desarma solo a los 2,5 s). |
| Lista de mensajes | Cada mensaje trae: **autor** (`Signs` o `Hearing`), **hora** `hh:mm:ss`, un **punto** si quedó incluido en la grabación, y el texto. Hace auto-scroll salvo que el usuario haya subido a leer historial. |
| Estado vacío | `"Both sides talk here. Signs appear as they are recognized; type (or dictate) to answer."` |
| Barra de composición | Campo de texto (`"Type to the signer…"`) + **botón de micrófono** (dictado; **se oculta por completo** si el navegador no soporta reconocimiento de voz) + botón de enviar. Enter también envía. |

### 5.6 Diálogos

**Help** — cinco bloques: preparar la escena (luz, fondo, encuadre, distancia),
cómo firmar en cada modo, cómo responde el oyente, cómo grabar, y consejos de
rendimiento/privacidad.

**Settings** — cuatro ajustes:

| Ajuste | Tipo | Notas |
|---|---|---|
| `Voice output` | Interruptor | Hablar en voz alta las señas reconocidas. |
| `Camera` | Selector | **Oculto si hay una sola cámara.** |
| `Voice` | Selector | Voces del sistema del dispositivo; las inglesas primero. Puede estar **vacío** en algunos dispositivos (`"No voices on this device"`). |
| `Letters mode speaks` | Radio | `Words` (habla la palabra al cerrarse) o `Letters` (habla cada letra). |

---

## 6. Flujos principales

**A. Deletreo (Letters/Numbers).**
Sostener la pose → el readout la muestra → al confirmarse se **acepta** (pulso
visual + vibración en Android) y el carácter entra al strip → repetir → una
**pausa de 2 s** o el botón `SPACE` cierra la palabra → se pronuncia y entra a
la conversación como mensaje del lado *Signs*.

**B. Seña de palabra (Words).**
Firmar con naturalidad → el sistema solo clasifica **mientras las manos se
mueven** → al confirmarse la seña: pulso visual, voz, entrada en la conversación
y la glosa se agrega a la frase del strip → tras **5 s sin señas** la frase se
limpia y empieza otra.

**C. El oyente responde.**
Escribe o dicta → envía → el texto aparece **en grande sobre el video** durante
8 s y queda en la conversación como mensaje del lado *Hearing*.

**D. Grabar y exportar.**
`Rec` → la conversación transcurre (los mensajes grabados quedan marcados) →
`Export` → se descargan **dos archivos**: `conversation_AAAAMMDD_HHMMSS.txt` y
`.csv`.

**E. Cambiar de modo.**
Toggle → se cierra la palabra deletreada en curso (no se pierde: entra a la
conversación) → se limpian buffers y strip → **la conversación se conserva**.

---

## 7. Datos y campos

**Entrada de conversación** (única estructura de datos del producto):

| Campo | Valores |
|---|---|
| `when` | Fecha/hora local del momento de la entrada. |
| `source` | Lado que habló: señas u oyente. (En los archivos exportados se escribe `Senas` / `Oyente` por compatibilidad con la app de escritorio; la UI muestra `Signs` / `Hearing`.) |
| `text` | Palabra deletreada, glosa reconocida o mensaje escrito/dictado. |
| `recorded` | Si la entrada quedó dentro de la grabación. |

**Formato exportado** — TXT legible (`[hh:mm:ss] Fuente: texto`) y CSV con
columnas `timestamp,source,text`. Ambos son **byte-compatibles con los archivos
de la app de escritorio**: un consumidor no puede distinguir cuál los generó.

**Preferencias persistentes** (localStorage, no hay cuenta ni sincronización):
voz activada, voz elegida, si Letters habla por palabra o por letra, y la cámara
seleccionada.

**Lo que NO se persiste:** la conversación. Al recargar la página se pierde todo
lo que no se haya exportado.

---

## 8. Comportamiento temporal (lo que condiciona el feedback)

Números reales del sistema, útiles porque **el diseño tiene que hacer legible
una espera**:

| Cosa | Valor |
|---|---|
| Confianza mínima para aceptar | Letters 80 %, Numbers 60 %, Words 75 % |
| Umbral para mostrar el top-3 | por debajo de 70 % |
| Confirmación de una letra | ~0,23 s de votos estables |
| Bloqueo entre dos letras iguales | ~0,67 s |
| Cierre automático de palabra deletreada | 2 s sin letra nueva |
| Ventana de señal para una palabra | 1,5 s de movimiento acumulado |
| Bloqueo tras aceptar una palabra | 1 s |
| Limpieza de la frase de glosas | 5 s de pausa |
| Duración del mensaje grande del oyente | 8 s |
| Fotogramas por segundo reales | 30–60 en escritorio; **15–24 en teléfono** (es el techo de MediaPipe en móvil, no un defecto corregible) |

Consecuencia práctica: **reconocer una seña tarda entre medio segundo y dos
segundos**, y en teléfono la app se siente más lenta pase lo que pase. Por eso
existen hoy la píldora de seguimiento (¿me está viendo?) y el pulso de
confirmación (¿entró?).

---

## 9. Estados que el diseño debe cubrir

Además del estado normal, la app tiene estos estados reales y frecuentes:

1. **Arranque** — descarga de ~17 MB (modelos + runtime) y permiso de cámara.
2. **Permiso denegado** / **sin cámara en el dispositivo**.
3. **Cambio de modo cargando** (el toggle se bloquea unos instantes).
4. **Error recuperable** del detector (aviso que se limpia solo).
5. **Error fatal** (solo queda recargar).
6. **No hay manos en cuadro** — el estado más común de todos.
7. **El modelo duda** (confianza baja → top-3).
8. **El dispositivo no tiene voces** instaladas → no hay salida de voz.
9. **El navegador no soporta dictado** → el oyente solo puede escribir.
10. **El detector de cuerpo falla** en Words → la seña sigue reconociéndose por
    forma de mano pero pierde el ancla corporal (degradación silenciosa).
11. **Una sola cámara** → los controles de cámara desaparecen por completo.
12. **El teléfono rota** → cambia la relación de aspecto del video en vivo.

---

## 10. Restricciones duras

Cosas que no son negociables por decisión técnica o de producto:

- **El video es el centro funcional**: el signer necesita verse para encuadrarse
  correctamente. Nada puede tapar sus manos ni su torso.
- **Dos roles simultáneos en una pantalla** con necesidades opuestas: uno lee de
  lejos y no puede tocar; el otro escribe de cerca.
- **Legibilidad a distancia**: todo lo que el signer deba leer se lee a
  0,5–1,5 m.
- **Nada crítico puede depender del audio** (el usuario principal es sordo). La
  voz es una salida *para el oyente*, no un canal de feedback.
- **Cero red en tiempo de uso**: sin CDN, sin fuentes remotas, sin llamadas
  externas. Todo activo debe estar servido localmente.
- **Sin backend, sin build step, sin framework**: HTML/CSS/JS plano.
- **Móvil y escritorio con el mismo código**; hoy el layout cambia alrededor de
  los 900 px de ancho.
- **Presupuesto de cómputo ajustado**: en teléfono cada elemento animado compite
  con el reconocimiento por el mismo hilo.

---

## 11. Fricciones observadas (hechos, no propuestas)

- El video y el panel de conversación **compiten por el espacio**, sobre todo en
  teléfono en vertical.
- La periferia del video concentra hoy **seis canales de información
  simultáneos**: píldora de seguimiento, badge REC, control de cámara, readout
  de confianza, FPS y strip de subtítulo.
- Existen **dos acciones distintas de "borrar"** (`CLEAR` descarta el texto en
  curso; el botón del panel borra la conversación entera) que el usuario puede
  confundir.
- La **barra de confianza y el top-3** son información de naturaleza técnica
  expuesta al usuario final; son útiles para corregir la postura de la mano,
  pero hablan el idioma del modelo, no el del usuario.
- El signer **no puede tocar la pantalla mientras firma**, pero `SPACE` y
  `CLEAR` son controles pensados para él.
- El **estado vacío inicial** debe enseñar a usar la app sin que nadie lea el
  Help, porque el uso típico es de un par de minutos y sin preparación.
- Nada indica de forma explícita **qué vocabulario conoce el sistema**: un
  usuario puede firmar durante un minuto una palabra que el modelo no tiene.

---

## 12. Qué no existe todavía

Relevante para no diseñar alrededor de funciones inexistentes — y para dejar
lugar a las que están planeadas:

- **Traducción glosa → frase con LLM en la web.** Hoy la web muestra glosas
  sueltas (`WANT DRINK`); la app de escritorio sí convierte esa secuencia en una
  frase fluida (tecla `T`). Es la siguiente función prevista para la web.
- **Letras J y Z** (requieren movimiento).
- **Agregar vocabulario desde la app** (hoy exige capturar y reentrenar con
  scripts).
- **Corregir al modelo** cuando se equivoca.
- **Historial persistente** entre sesiones o entre dispositivos.
- **Interfaz multilingüe**: la UI está solo en inglés, aunque el proyecto es de
  un equipo hispanohablante y el escenario de uso es en español.
- **Onboarding o calibración** de encuadre.

---

## 13. Resumen en una línea

Una página que **muestra a una persona firmando, convierte sus señas en texto y
voz, deja que la otra persona responda escribiendo, y guarda la conversación
completa si se lo piden** — funcionando entera dentro del navegador, sin
enviar nada a ningún lado.
