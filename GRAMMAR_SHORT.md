# Conjugación Verbal — Web & Mejoras Futuras (Resumen)

## ¿Qué se implementó?

Conjugación automática de verbos basada en pronombre previo. Sin ML — tabla de búsqueda determinista.

**Ejemplo:**
```
Usuario escribe:  yo creer ellos
Sistema muestra:  yo creo ellos
                     ↑ (conjugado automáticamente)
```

---

## Para Web (Futuro)

**Idea clave:** Misma tabla JSON (`model/grammar.json`), lógica idéntica en JavaScript.

```javascript
// 1. Cargar tabla
import grammar from "./model/grammar.json";

// 2. Conjugar verbo (igual que Python)
function conjugateVerb(verb, pronoun) {
  const form = grammar.pronouns[pronoun?.toLowerCase()];
  if (!form) return verb;
  return grammar.verbs[verb]?.[form] || verb;
}

// 3. Aplicar en overlay al mostrar texto
```

**Esfuerzo:** ~50 líneas de JavaScript (copiar lógica Python casi al pie).

---

## Mejoras Futuras (Prioridad)

| Mejora | Esfuerzo | Impacto | Qué Hacer |
|--------|----------|---------|-----------|
| **Agregar más verbos** (10–20) | Bajo | Alto | Editar config.py, correr export_grammar.py |
| **Acuerdo de adjetivos** ("cansado/a") | Medio | Medio | Nueva tabla ADJECTIVE_AGREEMENT |
| **Marcadores de tiempo** ("ayer creer" → "creí") | Medio-Alto | Alto | Detectar ayer/mañana, cambiar conjugación a pasado |
| **Pronombres clíticos** ("le da", "me da") | Alto | Medio | Requiere semántica, posible con regex simple |
| **Modo subjuntivo** ("espero que crean") | Muy Alto | Bajo | Nueva tabla por modo + lógica de detección |

**MVP + 2 semanas:** Agregar 15 más verbos comunes + test en web. Impacto inmediato: vocabulario 2x.

---

## Por qué funciona

- **Determinista:** Sin sorpresas. "yo" + "creer" siempre = "creo".
- **Escalable:** 100 verbos vs. 1000 — tablas crecen linealmente, sin reentrenamiento.
- **Portable:** Mismo JSON a Python, JavaScript, móvil, otros lenguajes.
- **Fallback seguro:** Si verbo no está en tabla, se muestra sin conjugar (usuario entiende igual).

---

## Checklist Próximo Sprint

- [ ] Web: Replicar `conjugateVerb()` en TypeScript
- [ ] Web: Cargar `model/grammar.json`
- [ ] Web: Integrar en componente overlay
- [ ] Test: "yo creo", "ellos pueden", "nosotros pensamos"
- [ ] Expandir: Agregar 10 verbos más comunes (ver, dar, hablar, etc.)

