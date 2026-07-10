# Traductor de Lenguaje de Señas — La Idea del Proyecto

> **Naturaleza de este documento.** Este es un documento base (insumo). Recopila
> de forma completa la idea central del proyecto, su justificación, su propósito
> y sus beneficios, sin entrar en detalles técnicos. Está pensado para servir de
> fuente de información a partir de la cual se redactarán documentos finales
> (presentación, póster, guion, material divulgativo). Por eso prioriza ser
> exhaustivo y claro antes que breve.

---

## 1. Resumen en una frase

Una aplicación que usa la cámara para **reconocer Lenguaje de Señas Americano
(ASL) en tiempo real y convertirlo en texto y voz**, eliminando la barrera de
comunicación entre personas sordas y personas oyentes sin que ninguna de las dos
necesite saber señas de antemano.

---

## 2. El problema que resolvemos

La comunicación entre una persona sorda que se expresa en lengua de señas y una
persona oyente que no la conoce está rota por defecto. Hoy esa brecha se cubre,
en el mejor de los casos, con uno de estos recursos, todos limitados:

- **Un intérprete humano.** Es la mejor solución en calidad, pero es caro,
  escaso, hay que agendarlo con antelación y no está disponible en lo cotidiano
  (una farmacia, una cita médica imprevista, un trámite, una conversación
  casual).
- **Escribir en papel o en el teléfono.** Es lento, interrumpe el ritmo natural
  de la conversación y asume que ambas partes leen y escriben con comodidad en
  el mismo idioma, lo cual no siempre es cierto: para muchas personas sordas la
  lengua de señas es su primera lengua y el español escrito es una segunda
  lengua.
- **Gestos improvisados.** Funcionan solo para lo más básico y se prestan a
  malentendidos.

El resultado es una **dependencia constante de terceros** para algo tan básico
como pedir información, ser atendido o mantener una conversación espontánea.
Esto se traduce en exclusión: en el acceso a servicios, en lo laboral, en lo
educativo y en lo social.

### Por qué importa

La comunicación no es un lujo, es la puerta de entrada a todo lo demás:
educación, salud, empleo y vida social. Cuando esa puerta depende de que haya un
intérprete disponible o de que la otra persona tenga paciencia para escribir, la
autonomía de la persona sorda queda condicionada. **El objetivo del proyecto es
devolver esa autonomía**: que la persona sorda pueda comunicarse en su lengua
natural y ser entendida de inmediato.

---

## 3. La idea central

La idea es construir un **traductor bidireccional y en tiempo real** que actúe
como puente entre los dos mundos:

- **De la persona sorda a la oyente:** la cámara observa las señas, el sistema
  las reconoce y las muestra como texto en pantalla y las reproduce en voz alta.
  La persona oyente "escucha" lo que se está señando.
- **De la persona oyente a la sorda:** el micrófono capta el habla, la
  transcribe a texto y la muestra en pantalla como subtítulos. La persona sorda
  "lee" lo que se está diciendo.

El principio rector es **que la tecnología desaparezca**: ninguna de las dos
personas debería tener que aprender nada nuevo ni cambiar su forma natural de
comunicarse. La persona sorda hace señas como siempre; la persona oyente habla
como siempre. El sistema traduce en medio, en silencio.

### Lo que NO es

- **No es un curso ni un diccionario de señas.** No busca enseñar lengua de
  señas; busca traducir en vivo.
- **No pretende reemplazar al intérprete humano** en contextos críticos
  (jurídicos, médicos complejos). Es una herramienta para la vida cotidiana,
  donde hoy directamente no hay nada.
- **No asume que la persona oyente sabe señas.** Justo lo contrario: existe
  precisamente porque no las sabe.

---

## 4. Cómo funciona (a nivel de idea, sin tecnicismos)

El usuario se sitúa frente a la cámara y hace una seña. El sistema:

1. **Observa la mano** y sigue su forma y su movimiento.
2. **Reconoce la seña** comparándola con lo que ha aprendido.
3. **Muestra el resultado** como texto en la pantalla.
4. **Lo dice en voz alta** para la persona oyente.
5. En sentido inverso, **escucha a la persona oyente** y lo muestra como
   subtítulos para la persona sorda.

Toda la conversación puede **guardarse y exportarse** como registro, de modo que
quede constancia de lo conversado (útil, por ejemplo, en una consulta o un
trámite).

### Dos formas de comunicar señas

El sistema contempla las dos maneras reales en que se usa la lengua de señas:

- **Deletreo (letra por letra).** Para nombres propios, direcciones o palabras
  que no tienen una seña propia. El usuario forma las letras con la mano y se van
  acumulando para construir la palabra.
- **Palabras y frases completas.** Las señas reales no son letras: son gestos con
  movimiento que representan una palabra o concepto entero ("querer", "ahora",
  "gracias"). El sistema reconoce esos gestos dinámicos y arma la frase.

### Del reconocimiento a una frase natural

Un detalle importante de la idea: la lengua de señas **no tiene la misma
gramática que el español**. No conjuga los verbos ni usa el mismo orden de
palabras. Una persona puede señar algo equivalente a "YO QUERER AGUA AHORA". El
sistema no se queda en esa versión telegráfica: tiene una capa que **convierte
esas señas sueltas en una frase fluida y bien escrita** en español ("Quiero un
poco de agua ahora"). Así el resultado suena natural para quien lo lee o lo
escucha.

---

## 5. ¿Para quién es? (beneficiarios)

- **Personas sordas e hipoacúsicas** que usan lengua de señas como medio
  principal de comunicación. Son el usuario central.
- **Personas oyentes que no saben señas** y necesitan comunicarse con una
  persona sorda: personal de atención al público, comercios, instituciones,
  personal de salud, docentes, y también familiares y amigos.
- **Entornos de atención e inclusión:** clínicas, oficinas de trámites,
  recepciones, aulas. Lugares donde el encuentro entre una persona sorda y una
  oyente ocurre todos los días y hoy no hay un puente disponible.

El valor está en que **beneficia a ambas partes a la vez**: no es una herramienta
solo "para sordos", es una herramienta de encuentro.

---

## 6. Beneficios y valor

- **Autonomía e independencia.** La persona sorda se comunica sin depender de que
  haya un intérprete o de que la otra parte sepa señas.
- **Inmediatez.** La traducción es en el momento, no agendada. Sirve para lo
  espontáneo, que es la mayor parte de la vida.
- **Comunicación natural en ambos sentidos.** Cada quien usa su forma de
  expresarse (señas / voz) sin esfuerzo adicional.
- **Sin curva de aprendizaje para la persona oyente.** No tiene que estudiar nada.
- **Accesibilidad de bajo costo.** Funciona con una cámara común; no requiere
  hardware especial ni guantes ni sensores.
- **Registro de la conversación.** Lo conversado puede quedar documentado y
  exportarse, lo que aporta seguridad y trazabilidad (importante en salud o
  trámites).
- **Dignidad.** Reduce la sensación de "ser una carga" o de tener que pedir ayuda
  para algo cotidiano; la comunicación se vuelve directa y de igual a igual.

---

## 7. Impacto social

El proyecto se enmarca en la **inclusión digital y la accesibilidad**. Su impacto
apunta a:

- **Reducir la exclusión comunicativa** de la comunidad sorda en lo cotidiano.
- **Acercar servicios** (salud, educación, atención pública) que hoy son difíciles
  de usar sin intérprete.
- **Sensibilizar:** al ser una herramienta que ambos lados usan, visibiliza la
  lengua de señas y normaliza su presencia.
- **Democratizar el acceso:** al apoyarse en una cámara común, la barrera de
  entrada económica es mínima.

Es, en esencia, un proyecto con **propósito social** apoyado en tecnología, no un
fin tecnológico en sí mismo.

---

## 8. El proyecto en dos etapas: versión actual y visión futura

Es clave entender que lo que existe hoy es una **primera versión (prototipo
funcional en Python)** y que la visión a la que apunta el proyecto es una
**versión Web mejorada**. Ambas comparten la misma idea; cambian el alcance, la
experiencia y la distribución.

### 8.1. Versión actual — Prototipo de escritorio (Python)

Es la versión que **ya funciona y se puede demostrar en vivo**. Su propósito es
**validar la idea**: probar que efectivamente se puede reconocer señas con una
cámara común y convertirlas en texto y voz en tiempo real.

Qué demuestra hoy:

- Reconocimiento de **letras estáticas** del abecedario.
- Reconocimiento de **palabras/señas con movimiento**.
- **Voz de salida** (lee en voz alta lo reconocido).
- **Subtítulos de entrada** (transcribe lo que dice la persona oyente).
- **Frases naturales** a partir de las señas reconocidas.
- **Registro exportable** de la conversación.
- Indicadores de **confianza** y alternativas cuando el sistema duda (es honesto
  sobre lo que reconoce, no inventa).

Carácter: es un prototipo de validación. Funciona, pero corre como un programa de
escritorio que hay que instalar, con un vocabulario acotado y una interfaz
funcional pensada para demostrar la capacidad, no para el usuario final.

### 8.2. Visión futura — Versión Web (la versión "mejorada")

Es **hacia dónde va el proyecto** y lo que se plantea como evolución en la
presentación. La idea es llevar todo lo aprendido en el prototipo a una
**aplicación web alojada en un servidor**, accesible desde el navegador. Las
mejoras que se buscan:

- **Acceso sin instalar nada.** Se entra desde un navegador (en computadora o
  teléfono); no hay que instalar programas ni configurar el entorno.
- **Disponible en línea, alojada en un servidor.** Cualquiera con el enlace y una
  cámara puede usarla, lo que multiplica el alcance.
- **Mejor diseño y experiencia de usuario.** Una interfaz cuidada, pensada para el
  usuario final y no solo para demostrar; más clara, accesible y agradable.
- **Funcionalidades distintas / ampliadas** propias del entorno web (por ejemplo,
  uso desde el móvil, una experiencia más pulida y posibles funciones nuevas que
  el formato web habilita).
- **Mayor alcance e inclusión real.** Al quitar la fricción de instalar, se acerca
  a un uso cotidiano y masivo, que es el verdadero objetivo de impacto.

En resumen: la **versión Python prueba que la idea es posible**; la **versión Web
la convierte en algo que la gente realmente puede usar en su día a día**.

> Nota para quien redacte los documentos finales: presentar el prototipo Python
> como "lo que ya logramos y podemos demostrar hoy" y la versión Web como "el
> siguiente paso y la visión del producto". Esa narrativa (logro presente +
> visión de futuro) es la columna de la presentación.

---

## 9. Diferenciadores (qué hace especial a esta idea)

- **Bidireccional:** no solo traduce señas a voz, también voz a texto. Cubre la
  conversación completa, no media conversación.
- **Solo necesita una cámara común:** sin guantes, sin sensores, sin hardware
  caro.
- **Traduce a lenguaje natural:** no entrega señas sueltas, entrega frases bien
  formadas, respetando que la gramática de las señas es distinta.
- **Honesto:** muestra cuán seguro está y ofrece alternativas cuando duda, en
  lugar de afirmar con falsa certeza.
- **Pensado para crecer:** la idea no termina en el prototipo; está diseñada para
  escalar hacia la web y llegar a más gente.

---

## 10. Contexto del proyecto

- **Equipo:** Dilan Calvo, Adrián Durán y Nazareth Solís.
- **Marco:** COTEPECOS — Especialidad de Desarrollo Web — 2026.
- **Naturaleza:** proyecto educativo con vocación de impacto social real.
- **Lengua de señas objetivo:** ASL (Lenguaje de Señas Americano), elegida por su
  mejor soporte de herramientas; la salida traducida se entrega en español.

---

## 11. Mensajes clave (para reutilizar en los documentos finales)

- "Comunicación sin barreras: que nadie tenga que aprender la lengua del otro
  para entenderse."
- "Una cámara común convierte señas en voz, y la voz en texto. En tiempo real."
- "Devolvemos autonomía: la persona sorda se comunica por sí misma, sin
  intermediarios."
- "Hoy lo demostramos en una app de escritorio; mañana, en una web al alcance de
  todos."
- "Tecnología con propósito: la herramienta desaparece, la conversación queda."
