# Cómo Agregar Nuevas Palabras (Verbos) al Sistema

## Respuesta Directa

**Sí, cada vez que agregues una palabra, debes poner el pasado, presente Y futuro** en `config.py`.

¿Por qué? Porque el sistema soporta 3 tiempos y si falta uno, esa palabra no conjugará para ese tiempo.

---

## Paso a Paso: Agregar "HABLAR"

### **Paso 1: Conjugar los 3 tiempos**

Primero, obtén las conjugaciones del verbo para 6 formas × 3 tiempos = **18 formas totales**.

Para "hablar":

| Forma | Presente | Pasado | Futuro |
|-------|----------|--------|--------|
| Yo | hablo | hablé | hablaré |
| Tú | hablas | hablaste | hablarás |
| Él/Ella | habla | habló | hablará |
| Nosotros | hablamos | hablamos | hablaremos |
| Vosotros | habláis | hablasteis | hablaréis |
| Ellos | hablan | hablaron | hablarán |

### **Paso 2: Copiar y Pegar en config.py**

Abre `config.py` y busca la sección `VERB_CONJUGATIONS`. Copia este bloque:

```python
"hablar": {
    "present": {
        "1p_sg": "hablo",
        "2p_sg": "hablas",
        "3p_sg": "habla",
        "1p_pl": "hablamos",
        "2p_pl": "habláis",
        "3p_pl": "hablan",
    },
    "past": {
        "1p_sg": "hablé",
        "2p_sg": "hablaste",
        "3p_sg": "habló",
        "1p_pl": "hablamos",
        "2p_pl": "hablasteis",
        "3p_pl": "hablaron",
    },
    "future": {
        "1p_sg": "hablaré",
        "2p_sg": "hablarás",
        "3p_sg": "hablará",
        "1p_pl": "hablaremos",
        "2p_pl": "hablaréis",
        "3p_pl": "hablarán",
    },
},
```

Y pégalo **antes de la llave de cierre** de `VERB_CONJUGATIONS`.

### **Paso 3: Exportar a JSON (para web)**

Ejecuta:

```bash
python scripts/export_grammar.py
```

Esto actualiza automáticamente `model/grammar.json` para que la web tenga los datos nuevos.

### **Paso 4: Test Manual (opcional)**

En `main.py`, cuando corras la app:

```
Usuario escribe: "YO HABLAR"
Presiona "R" (presente)
Resultado: "YO HABLO"  ✅

Usuario escribe: "YO HABLAR"
Presiona "T" (pasado)
Resultado: "YO HABLÉ"  ✅

Usuario escribe: "AYER YO HABLAR"
(Auto-detecta "ayer" = pasado)
Resultado: "AYER YO HABLÉ"  ✅
```

---

## Verbos Irregulares: Ejemplo "IR"

Algunos verbos cambian completamente en tiempo pasado. Ejemplo:

| Forma | Presente | Pasado | Futuro |
|-------|----------|--------|--------|
| Yo | voy | **fui** | iré |

Simplemente pones los valores correctos en la tabla:

```python
"ir": {
    "present": {
        "1p_sg": "voy",
        ...
    },
    "past": {
        "1p_sg": "fui",  # ← Diferente a "iba" (imperfecto)
        ...
    },
    ...
},
```

---

## Template: Copiar y Replicar

Para agregar un verbo fácilmente, copia esta plantilla y rellena:

```python
"INFINITIVO": {
    "present": {
        "1p_sg": "YO-FORMA",
        "2p_sg": "TU-FORMA",
        "3p_sg": "EL-FORMA",
        "1p_pl": "NOSOTROS-FORMA",
        "2p_pl": "VOSOTROS-FORMA",
        "3p_pl": "ELLOS-FORMA",
    },
    "past": {
        "1p_sg": "YO-PASADO",
        "2p_sg": "TU-PASADO",
        "3p_sg": "EL-PASADO",
        "1p_pl": "NOSOTROS-PASADO",
        "2p_pl": "VOSOTROS-PASADO",
        "3p_pl": "ELLOS-PASADO",
    },
    "future": {
        "1p_sg": "YO-FUTURO",
        "2p_sg": "TU-FUTURO",
        "3p_sg": "EL-FUTURO",
        "1p_pl": "NOSOTROS-FUTURO",
        "2p_pl": "VOSOTROS-FUTURO",
        "3p_pl": "ELLOS-FUTURO",
    },
},
```

---

## Checklist: Agregar un Verbo

- [ ] Obtuve conjugaciones para 3 tiempos (presente, pasado, futuro)
- [ ] Obtuve formas para 6 personas (1p_sg, 2p_sg, 3p_sg, 1p_pl, 2p_pl, 3p_pl)
- [ ] Copié el bloque en `config.py` dentro de `VERB_CONJUGATIONS`
- [ ] La estructura está correctamente indentada (Python es sensible a esto)
- [ ] Ejecuté `python scripts/export_grammar.py`
- [ ] Testeé manualmente en `main.py`

---

## Errores Comunes

### **"Syntax Error" en config.py**

**Causa:** Faltan comillas o comillas mal cerradas.

```python
# MALO:
"hablar": {
    "present": {
        "1p_sg": hablo,  # ← Falta comillas
    }
}

# BIEN:
"hablar": {
    "present": {
        "1p_sg": "hablo",  # ← Comillas OK
    }
}
```

### **Verbo no conjuga en cierto tiempo**

**Causa:** Ese tiempo no está en la tabla.

```python
# MALO:
"hablar": {
    "present": {...},
    "past": {...},
    # Falta "future"!
}

# BIEN:
"hablar": {
    "present": {...},
    "past": {...},
    "future": {...},  # ← Todos 3 tiempos
}
```

### **Olvidé ejecutar export_grammar.py**

La web no tendrá los datos nuevos. Solución: ejecuta:

```bash
python scripts/export_grammar.py
```

---

## Contexto: ¿Por Qué 3 Tiempos?

- **Presente:** "yo creo" (default)
- **Pasado:** "ayer yo creí" (usuario menciona "ayer")
- **Futuro:** "mañana yo creeré" (usuario menciona "mañana")

El usuario puede también presionar:
- **T** → forzar pasado
- **R** → forzar presente (default)
- **F** → forzar futuro

O el sistema auto-detecta marcadores en el texto ("ayer", "mañana", "ahora").

---

## Próximos Pasos (Post-MVP)

Cuando quieras soportar más tiempos (pretérito perfecto "he creído", condicional "creería", etc.), simplemente:

1. Agrega nueva clave a la tabla:
```python
"creer": {
    "present": {...},
    "past": {...},
    "future": {...},
    "present_perfect": {  # ← NUEVO
        "1p_sg": "he creído",
        ...
    },
}
```

2. Ejecuta `export_grammar.py`
3. En `main.py`, agrega tecla (ej. "Y" para "yesteday/present_perfect")

---

## Contacto / Ayuda

Si un verbo no funciona o tienes dudas, revisa:
1. `config.py` → `VERB_CONJUGATIONS` (sintaxis correcta?)
2. `model/grammar.json` → verifica que el verbo esté en el JSON
3. `test_integration.py` → corre los tests para detectar errores

