# Signex — Cómo funciona (explicación simple)

> **Para qué sirve este documento.** Explica el funcionamiento técnico del
> proyecto de forma accesible: no da por sentado conocimiento previo de
> programación, pero tampoco esconde los nombres y números técnicos reales —
> cada idea se explica primero en criollo y después, en un párrafo aparte, con
> su nombre técnico. Pensado para cualquier persona — un profesor, un
> familiar, alguien evaluando el proyecto — que quiera entender *cómo funciona
> por dentro* sin tener que leer código. Para el detalle técnico completo
> (arquitecturas exactas, código, parámetros) ver `docs/MANUAL_COMPLETO.md`;
> para la idea y el propósito del proyecto ver `docs/IDEA.md`.

---

## En una frase

Signex mira una mano a través de una cámara, entiende qué seña está haciendo,
y la convierte en texto y voz — y también hace el camino inverso: convierte lo
que dice una persona oyente en subtítulos para que la persona sorda los lea.

---

## El truco central: no mira una foto, mira un "esqueleto" de puntos

Uno podría imaginar que el programa mira la imagen de la mano como una foto y
trata de reconocerla, parecido a como se reconoce una cara. **No es así.**

Primero, una pieza llamada **MediaPipe** (una herramienta de Google) convierte
la mano en 21 puntos — las articulaciones: nudillos, puntas de los dedos,
muñeca — como si dibujara un esqueleto de palitos sobre la mano. La
inteligencia artificial que reconoce la seña **nunca ve la piel, la ropa ni el
fondo**: solo ve dónde están esos 21 puntos.

¿Por qué es una buena idea? Porque esos puntos son los mismos con cualquier
tono de piel, con buena o mala luz, con cualquier fondo, cerca o lejos de la
cámara. El sistema no tiene que "aprender a ignorar" todas esas variaciones —
directamente nunca las ve.

> **En términos técnicos:** esos 21 puntos se llaman *landmarks*, y cada uno
> tiene tres coordenadas (x, y, z) — es decir, cada mano se convierte en 63
> números. MediaPipe es una librería de Google que calcula esto en tiempo
> real sin necesitar una placa de video (GPU), lo que permite que la app
> corra fluida en una notebook común. Antes de clasificar, esos 63 números se
> "normalizan" (se centran en la muñeca y se escalan según el tamaño de la
> mano) para que la posición y el tamaño no afecten el resultado.

---

## Cómo "aprende" a reconocer una seña

No se usó ninguna base de datos bajada de internet. El equipo se grabó a sí
mismo haciendo cada seña, muchas veces, en días distintos — con distinta ropa,
distinta luz, sentados en distinto lugar. Esas grabaciones (ya convertidas en
"esqueletos de puntos") son lo que el sistema estudia para aprender a
distinguir una letra de otra, o una palabra de otra.

Grabar en **varios días distintos** es más importante que grabar muchas veces
el mismo día. Si solo se graba un día, el sistema puede memorizar detalles de
ese día en particular (esa luz, esa silla) en vez de aprender la seña en sí —
funciona perfecto en la prueba y falla apenas cambia algo. Grabando en días
distintos, el sistema se ve obligado a aprender lo que **no cambia**: la forma
real de la seña.

> **En términos técnicos:** cada modelo es una red neuronal (construida con
> TensorFlow/Keras) entrenada por *aprendizaje supervisado*: se le muestran
> miles de esos "esqueletos de puntos" ya etiquetados con la seña correcta, y
> el modelo ajusta sus parámetros internos hasta minimizar el error. Las
> grabaciones se guardan como archivos CSV (letras y números) o `.npy`
> (palabras) dentro de `data/real_capture/` — nunca se usó un dataset bajado
> de internet.

---

## Tres "cerebros" distintos, para tres trabajos distintos

El sistema tiene en realidad tres modelos de inteligencia artificial
entrenados por separado, y la persona elige cuál está activo:

| Modo | Qué reconoce | Por qué es un modelo aparte |
|---|---|---|
| **Letras** | El abecedario en señas (A a la Y, sin la J ni la Z) | Se reconoce con una sola "foto" de la mano quieta |
| **Números** | Los dígitos del 0 al 9 | Algunos números se ven **igual** que ciertas letras (el 2 se parece a la V, el 6 a la W). Si fuera el mismo modelo, no sabría si la persona quiso decir "2" o "V" — separarlos resuelve la ambigüedad |
| **Palabras** | Señas completas con movimiento (gracias, querer, ayuda...) | Estas señas **no se entienden con una sola foto** — hay que ver cómo se mueve la mano durante uno o dos segundos |

Letras y números se reconocen a partir de una sola pose (como sacar una foto).
Las palabras necesitan que el sistema mire una **secuencia** de instantes —
parecido a reconocer un paso de baile: una sola foto no alcanza, hace falta
ver el movimiento completo.

Para las palabras, el sistema además se fija **dónde está la mano respecto al
cuerpo** (usa los hombros como referencia). Esto es porque hay señas que se
hacen con la misma forma de mano pero en un lugar distinto — por ejemplo, cerca
del pecho o cerca de la frente pueden ser dos palabras diferentes. Sin esa
referencia al cuerpo, el sistema las confundiría.

> **En términos técnicos:** letras y números usan una red **densa**
> (*fully-connected*: capas `Dense` con `BatchNorm` y `Dropout`) porque la
> entrada es un único frame de 63 números. Palabras usa una arquitectura
> distinta, una **TCN** (*Temporal Convolutional Network*): convoluciones 1D
> que leen una secuencia de 32 instantes de 130 valores cada uno (63 de forma
> + 2 de posición, por cada mano), para captar cómo cambia la pose en el
> tiempo — no una sola foto.

---

## Cuando el sistema "duda"

A veces dos letras se parecen mucho entre sí y el sistema no está seguro de
cuál es. En vez de inventar una respuesta con falsa seguridad, **muestra qué
tan seguro está** (una barra de confianza, como un medidor) y, si duda
demasiado, muestra las 2 o 3 opciones entre las que está eligiendo — parecido
a un médico que dice "puede ser esto, o esto otro" en lugar de arriesgar un
diagnóstico a ciegas. Esta honestidad es una decisión de diseño deliberada: es
mejor decir "no estoy seguro" que inventar una respuesta.

> **En números concretos:** una letra se acepta como válida a partir del 80%
> de confianza, un número desde el 60%, y una palabra desde el 75%. Por
> debajo del 70% el sistema directamente no se compromete con una única
> respuesta y muestra el top-3 de candidatas en su lugar.

---

## De señas sueltas a una frase que suena natural

La lengua de señas no funciona como el español escrito: no conjuga los verbos
ni usa el mismo orden de palabras. Una persona puede señar el equivalente a
"YO QUERER AGUA AHORA". El sistema reconoce exactamente eso — las señas
sueltas, en su forma más simple.

Ahí entra un segundo paso, opcional: esas señas sueltas se le pasan a una
inteligencia artificial de lenguaje (Claude, de Anthropic) que arma con ellas
una oración bien formada — "Quiero agua ahora" o "Quiero tomar algo ahora".
Es como tener un traductor que toma palabras clave y arma con ellas una frase
completa. Si no hay conexión a internet este paso simplemente no ocurre y el
sistema muestra las señas sueltas tal cual — el resto de la app sigue
funcionando igual.

> **En términos técnicos:** ese segundo paso llama a la API de Claude
> (Anthropic) — el mismo tipo de modelo de lenguaje detrás de asistentes como
> ChatGPT o el propio Claude. Corre en un hilo aparte para no trabar la
> cámara mientras espera la respuesta, y si falla, no hay internet o no hay
> clave de API configurada, cae automáticamente al modo sin conexión (mostrar
> las señas sueltas) en vez de romper la app.

---

## Dos formas de usar la app

- **La aplicación web** — se abre en el navegador (computadora o celular), no
  hay que instalar nada. Es la pensada para que la use cualquier persona.
- **La aplicación de escritorio** — un programa que corre en la computadora
  del equipo. Es la herramienta de trabajo: con ella se graban las señas
  nuevas y se entrenan los modelos. Reconoce igual que la web, pero su
  pantalla es más simple porque no es el producto final, es la mesa de
  trabajo.

Las dos usan exactamente los mismos modelos entrenados y el mismo cálculo —
no son dos sistemas distintos, son dos formas de acceder al mismo "cerebro".

> **En términos técnicos:** la app de escritorio está escrita en Python
> (TensorFlow/Keras + OpenCV); la web es JavaScript puro, sin frameworks ni
> instalación — incluso reimplementa el cálculo de los modelos "a mano" en
> JavaScript en vez de usar TensorFlow.js, porque el conversor oficial de
> TensorFlow.js no se pudo instalar en Windows. Ambas versiones corren
> exactamente la misma fórmula de normalización, así que un modelo entrenado
> una sola vez sirve para las dos por igual.

Algo importante para la privacidad: **todo el reconocimiento pasa en el
dispositivo** de quien usa la app. El video de la cámara nunca se envía a
ningún servidor. Lo único que eventualmente sale a internet es el texto de las
señas ya reconocidas, y solo cuando se pide armar la frase natural (el paso
anterior) — nunca la imagen.

---

## Qué tan bien funciona hoy

- Reconoce **24 letras** del abecedario (todas menos J y Z, que necesitan
  movimiento) — entrenado con 3.750 grabaciones repartidas en 9 sesiones
  distintas.
- Reconoce los **10 dígitos** (0 al 9) — entrenado con 1.500 grabaciones en
  4 sesiones.
- Reconoce **18 palabras/señas completas** (por ejemplo: hola, gracias,
  querer, ayuda, comer, tomar, sí, no, perdón...) más una seña especial para
  "no estoy señando nada", que evita que el sistema hable cuando la persona no
  está haciendo ninguna seña en particular — entrenado con 668 grabaciones en
  11 sesiones distintas.
- En las pruebas internas, letras y números aciertan más del 98% de las veces;
  las palabras también rondan el 98%, aunque ese número es *optimista* porque
  todavía falta una prueba más estricta (probar con grabaciones de un tipo que
  el sistema nunca vio). Una versión anterior del modelo de palabras, medida
  con esa prueba más estricta, daba 94% — un número más realista de lo que
  pasa en el uso real del día a día.
- Se sabe que el sistema a veces confunde "necesitar" con "querer", y en
  ocasiones "dice" una palabra cuando en realidad la persona no estaba señando
  nada — son puntos débiles conocidos, no una sorpresa, y el equipo los tiene
  identificados para seguir mejorándolos.

---

## Qué falta

- Las letras **J y Z** (necesitan movimiento, así que en el fondo pertenecen
  más al modelo de palabras que al de letras).
- Ampliar el vocabulario de palabras — hoy son 18, la meta es que sean muchas
  más.
- Que se puedan agregar palabras nuevas **desde la propia app**, sin que el
  equipo tenga que grabar y reentrenar por fuera con scripts.
- Que la app web también arme frases naturales con IA (hoy ese paso solo
  existe en la versión de escritorio).
- Que el sistema pueda aprender de sus propios errores cuando alguien lo
  corrige.

---

## Equipo y contexto

Proyecto de **Dilan Calvo, Adrián Durán y Nazareth Solís** — COTEPECOS,
Especialidad de Desarrollo Web, 2026.
