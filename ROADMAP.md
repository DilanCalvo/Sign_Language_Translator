# Roadmap & Recomendaciones Futuras

Documento de planificación técnica para evolución del Sign Language Translator.
Analiza problemas actuales, soluciones posibles y decisiones arquitectónicas.

**Última actualización:** 2026-06-03  
**Estado actual:** MVP (conjugación básica, 8 verbos, 3 tiempos)

---

## 🎯 Visión General

### Fase 1: MVP (Actual) ✅
- Detección de letras ASL estática
- Acumulación de palabras en texto
- Conjugación básica por pronombre (presente)
- Voz offline (SAPI5)
- 8 verbos, 1 idioma (español)

### Fase 2: Pequeño (50–100 verbos, web)
- Soporte de 50+ verbos comunes
- Migración a web (tensorflowjs)
- Expansión a tiempos compuestos
- Múltiples marcadores temporales

### Fase 3: Mediano (500+ verbos, móvil, multiidioma)
- Arquitectura escalable (patrones de conjugación)
- Soporte móvil nativo
- 2–3 idiomas (español, inglés, ASL británico)
- Feedback de corrección

### Fase 4: Grande (Web completa, API, comunidad)
- Plataforma web con interfaz moderna
- API REST para terceros
- Community-driven vocabulary
- Fine-tuning de modelos

---

## 🚨 Cuello de Botella Identificado: Conjugación Manual

### El Problema
Cada vez que agregas un verbo, necesitas **18 líneas en config.py** (3 tiempos × 6 formas).

```
Escalabilidad:
- 8 verbos (MVP): 144 líneas ✅ Manejable
- 50 verbos: 900 líneas ⚠️ Incómodo
- 100 verbos: 1800 líneas ❌ Problema
- 500 verbos: 9000 líneas ❌❌ Inviable
```

### Análisis de Soluciones

| Solución | Ventajas | Desventajas | Esfuerzo | Recomendación |
|----------|----------|-------------|----------|---------------|
| **1. Librería Spacy** | Auto-genera todas las formas | 40MB, latencia 5-20ms, no existe en JS | 2h Python | ⭐⭐⭐ Para 500+ verbos |
| **2. API Externa** | Confiable, completo | Requiere internet, latencia 200-500ms | 1h | ❌ No para offline |
| **3. Patrones + Tabla** | Escalable, offline, JS-compatible | Más lógica inicial | 4-6h | **⭐⭐⭐⭐⭐ RECOMENDADO** |
| **4. ML Encoder-Decoder** | Aprende automáticamente | Complejo, ~90% precisión | 10h | ⭐⭐ Overkill |

### Decisión Recomendada: **Opción 3 (Patrones + Tabla)**

**Cuándo migrar:** Cuando tengas 40–50 verbos  
**Esfuerzo:** 4–6 horas  
**Resultado:**
```
Antes: 50 verbos × 18 formas = 900 líneas
Después:
  - 50 regulares: 50 líneas
  - 10 irregulares: 180 líneas
  - Patrones: 54 líneas
  Total: ~284 líneas (68% menos)
```

**Estructura:**
```python
# config.py - NUEVA ESTRUCTURA

VERBS_BASE = {
    "hablar": "ar",        # Regular -ar (genera automáticamente)
    "creer": "er",         # Regular -er
    "vivir": "ir",         # Regular -ir
    "ser": "irregular",    # Tabla aparte
    "tener": "irregular",
}

IRREGULAR_CONJUGATIONS = {
    "ser": {
        "present": {...},
        "past": {...},
        "future": {...},
    },
    "tener": {...},
}

CONJUGATION_PATTERNS = {
    "ar": {"present": {"1p_sg": "{root}o", ...}, ...},
    "er": {...},
    "ir": {...},
}
```

**Función generadora:**
```python
def conjugate_verb(infinitive: str, pronoun: str, tense: str) -> str:
    verb_type = VERBS_BASE.get(infinitive, "unknown")
    
    if verb_type == "irregular":
        return IRREGULAR_CONJUGATIONS[infinitive][tense][pronoun]
    
    pattern = CONJUGATION_PATTERNS[verb_type][tense]
    root = infinitive[:-2]
    return pattern[pronoun].format(root=root)
```

**Próximos pasos:**
- [ ] Cuando llegues a 40 verbos, planifica refactor a patrones
- [ ] Mantén test de regresión durante migración
- [ ] Duplica estructura en JS web

---

## 📋 Mejoras Recomendadas por Fase

### 🟢 Fase 2: Pequeño (Próximas 4–6 semanas)

#### P1: Expandir Vocabulario (Bajo esfuerzo, alto impacto)
```
Verbos a agregar (15–20):
- ver, dar, hablar, entender, saber
- venir, deber, decir, encontrar, esperar
- empezar, obtener, perder, pedir, seguir
- llevar, parecer, dejar, importar, conseguir

Tiempo: 2–3 horas
Resultado: vocabulario 2.5x
Bloques: Ninguno
```

**Cómo hacerlo:** Ver [HOW_TO_ADD_VERBS.md](HOW_TO_ADD_VERBS.md)

#### P2: Tiempos Compuestos (Presente Perfecto)
```
Agregar: "he creído", "has creído", "ha creído"
Marcadores: "he", "ha", "han"
Tiempo: 1–2 horas
Nuevos archivos: Ninguno (solo expandir config.py)
```

**Implementación:**
```python
VERB_CONJUGATIONS = {
    "creer": {
        "present": {...},
        "past": {...},
        "future": {...},
        "present_perfect": {  # NUEVO
            "1p_sg": "he creído",
            "3p_sg": "ha creído",
            ...
        },
    },
}

TIME_MARKERS = {
    "he": "present_perfect",
    "ha": "present_perfect",
    "han": "present_perfect",
}
```

#### P3: Web MVP (Migración inicial)
```
Tecnología: React/Vue + tensorflowjs + Web Speech API
Duración: 1–2 semanas
Equipo: 2 personas recomendado
Bloqueadores: Ninguno

Pasos:
1. Replicar grammar.js (conjugate_verb, detect_tense)
2. Replicar LetterBuffer en React
3. Integrar MediaPipe.js (hand landmarks)
4. Integrar Web Speech API (voz)
5. Deploy estático (GitHub Pages o Vercel)

Resultado: Aplicación web funcional, 60% features MVP
```

---

### 🟡 Fase 3: Mediano (6–12 semanas)

#### P1: Refactor a Patrones de Conjugación
```
Bloqueador: Cuando llegues a 40–50 verbos
Esfuerzo: 4–6 horas
Ganancia: 68% menos código, escalable a 500+ verbos

Pasos:
1. Crear CONJUGATION_PATTERNS en config.py
2. Refactorizar VERB_CONJUGATIONS
3. Actualizar conjugate_verb() con generador
4. Replicar en web (JavaScript)
5. Tests de regresión completos
```

#### P2: Subjuntivo Básico
```
Casos: "Espero que creo" → "espero que crea"
Marcadores: "espero que", "dudo que", "aunque"
Nuevas formas: 18 más (3 tiempos × 6 formas subjuntivo)
Esfuerzo: 2–3 horas
Impacto: Medio (menos frecuente que indicativo)
```

#### P3: Condicional
```
Casos: "Si tuviera dinero, viajaría"
Esfuerzo: 1–2 horas
Impacto: Bajo-Medio
```

#### P4: Multiidioma (Español + Inglés)
```
Estructura propuesta:
config.py:
├── LANGUAGE = "es"  # o "en"
├── VERBS_BASE_ES
├── IRREGULAR_CONJUGATIONS_ES
├── VERBS_BASE_EN
└── IRREGULAR_CONJUGATIONS_EN

grammar.py:
def conjugate_verb(infinitive, pronoun, tense, language="es"):
    config = get_language_config(language)
    # misma lógica, diferentes tablas

Esfuerzo: 3–4 horas
Resultado: App bilingüe
```

#### P5: Feedback y Corrección
```
Idea: Usuario corrige predicción, sistema aprende
Implementación:
1. Log de correcciones en JSON
2. Herramienta para revisar logs
3. Fine-tuning offline (entrenamiento modelo letra/palabra)
4. Recarga de modelo sin reiniciar app

Esfuerzo: 4–6 horas
Impacto: Mejora de precisión 5–10%
Complejidad: Media
```

---

### 🔴 Fase 4: Grande (12+ semanas)

#### P1: Migración Completa a Web
```
Stack recomendado:
- Frontend: React + TypeScript
- Backend: Node.js + FastAPI (ML)
- Database: PostgreSQL (usuarios, conversaciones)
- Deployment: Docker + Kubernetes o Vercel + AWS Lambda

Componentes:
- Landing page responsiva
- Editor de configuración (cambiar modelo, verbos, idioma)
- Historial de conversaciones
- Dashboard de estadísticas

Esfuerzo: 6–8 semanas
Equipo: 2–3 personas
Blocadores: Ninguno técnico
```

#### P2: API REST
```
Endpoints:
POST /api/translate
  - Input: landmarks JSON
  - Output: predicción + conjugación

GET /api/verbs?language=es&count=50
  - Retorna lista de verbos disponibles

POST /api/conjugate
  - Input: verbo, pronombre, tiempo, idioma
  - Output: forma conjugada

POST /api/feedback
  - Guardar corrección del usuario

Esfuerzo: 3–4 horas
Uso: Integración con terceros, apps móviles
```

#### P3: Aplicación Móvil Nativa
```
Tecnología: React Native
Comparte: grammar.js, modelos (tensorfloats)
Diferencias: UI móvil, cámara nativa, permisos

Esfuerzo: 4–6 semanas (con React Native experience)
Plataformas: iOS + Android
```

#### P4: Community Vocabulary
```
Sistema de crowd-sourcing:
1. Usuarios sugieren nuevos verbos
2. Sistema acepta conjugaciones propuestas
3. Validación comunitaria (upvotes)
4. Merge automático a config si pasa umbral

Esfuerzo: 2–3 semanas
Plataforma: GitHub Discussions o Discord bot

Beneficio: Vocabulario crece sin intervención del equipo
```

---

## 🏗️ Decisiones Arquitectónicas Tomadas

### D1: Tabla en config.py (No Base de Datos)
**Decisión:** Almacenar conjugaciones en Python/JSON, no en SQL.

**Razones:**
- ✅ MVP es offline, no necesita DB
- ✅ Config es versionable en Git
- ✅ Fácil sincronizar Python ↔ Web
- ✅ Cero setup (sin migración DB)

**Cuándo cambiar:** Cuando haya 10000+ registros o múltiples idiomas dinámicos.

---

### D2: Generar JSON Automáticamente
**Decisión:** Script `export_grammar.py` genera `model/grammar.json` desde config.py.

**Razones:**
- ✅ Una sola fuente de verdad
- ✅ Web siempre tiene datos actualizados
- ✅ Evita duplicación manual

**Proceso:**
```
Dev edita config.py
  ↓
Ejecuta: python scripts/export_grammar.py
  ↓
model/grammar.json se actualiza
  ↓
Deploy (web carga JSON actualizado)
```

---

### D3: Tiempos en Tabla, No en ML
**Decisión:** Conjugación por tabla de búsqueda, no ML/NLP.

**Razones:**
- ✅ Determinista (sin sorpresas)
- ✅ Rápido (1–5ms)
- ✅ Controlable (sabes por qué)
- ✅ Offline
- ❌ No escala bien (50+ verbos)

**Cuándo cambiar:** Ver sección "Cuello de Botella" arriba.

---

### D4: Pronunciación Local (SAPI5), No API
**Decisión:** Windows SAPI5 (win32com), no Google TTS ni Azure Speech.

**Razones:**
- ✅ Offline 100%
- ✅ Cero costo
- ✅ Bajo latencia (~200ms)
- ✅ Ya instalado en Windows
- ⚠️ Solo Windows, cero idiomas

**Cuándo cambiar:**
- Para web: Web Speech API (nativo, offline, gratis)
- Para Android: TTS nativo
- Para multiidioma: Google Cloud TTS (con internet)

---

### D5: Detección Manual de Tiempos (Override)
**Decisión:** Usuario puede presionar T/R/F para forzar tiempo.

**Razones:**
- ✅ Control total del usuario
- ✅ Evita ambigüedades ("él creyó" ¿cuándo?)
- ✅ Simple de implementar

**Alternativa futura:** Contexto semántico (analizar oración completa).

---

## 📊 Matriz de Prioridades

| Característica | Fase | Esfuerzo | Impacto | Bloqueadores | Prioridad |
|---|---|---|---|---|---|
| Agregar 15 verbos comunes | 2 | 2h | Alto | Ninguno | 🔴 P1 |
| Presente Perfecto | 2 | 2h | Medio | Ninguno | 🟡 P2 |
| Web MVP | 2 | 2 semanas | Muy Alto | Ninguno | 🔴 P1 |
| Refactor a Patrones | 3 | 6h | Alto | 40+ verbos | 🟡 P2 |
| Subjuntivo | 3 | 3h | Medio | Ninguno | 🟡 P2 |
| Multiidioma | 3 | 4h | Medio-Alto | Refactor | 🟡 P2 |
| Feedback Loop | 3 | 6h | Medio | Ingeniería datos | 🟡 P2 |
| Web Completa | 4 | 8 semanas | Muy Alto | Equipo | 🔴 P1 |
| API REST | 4 | 4h | Medio | Web MVP | 🟡 P2 |
| Móvil Nativo | 4 | 6 semanas | Alto | React Native exp. | 🟡 P2 |

---

## 🧪 Estrategia de Testing

### MVP (Actual)
```
✅ Unit tests: src/grammar.py
✅ Integration tests: LetterBuffer + conjugación
❌ UI tests: Manual (no automatizado)
```

### Fase 2
```
+ E2E tests: main.py con mocks
+ Visual regression: Capturas de pantalla
+ Performance: Benchmarks de latencia
```

### Fase 3
```
+ Web tests: Cypress/Playwright
+ Mobile tests: Detox
+ Load tests: Mil usuarios simultáneos
```

---

## 🔗 Referencias Internas

- [HOW_TO_ADD_VERBS.md](HOW_TO_ADD_VERBS.md) — Guía práctica
- [GRAMMAR_WEB_PLAN.md](GRAMMAR_WEB_PLAN.md) — Detalles web + mejoras
- [GRAMMAR_SHORT.md](GRAMMAR_SHORT.md) — Resumen 1 página
- [config.py](config.py) — Documentación inline
- [src/grammar.py](src/grammar.py) — Código comentado

---

## 📞 Contacto / Decisiones Futuras

Si antes de Fase 3 necesitas tomar decisión arquitectónica:

1. **¿Qué problema estoy resolviendo?** (escalabilidad, velocidad, feature nueva)
2. **¿Cuál es el MVP de eso?** (qué es lo mínimo)
3. **¿Cuál es el costo técnico?** (líneas de código, dependencias, complejidad)
4. **¿Hay alternativas?** (ver Análisis de Soluciones arriba)

Usa este documento como referencia.

---

**Última revisión:** 2026-06-03  
**Próxima revisión esperada:** Cuando fase 2 esté 80% completa  
**Mantenedor:** Dilan Calvo

