# Grammar & Conjugation — Web & Future Roadmap

## What We Did (MVP — Python)

**Automatic verb conjugation based on pronoun context.**

- **Location:** `src/grammar.py` (deterministic, no ML)
- **Tables:** `config.py` (`VERB_CONJUGATIONS`, `PRONOUN_TO_FORM`)
- **Integration:** `src/overlay.py` — `LetterBuffer` now conjugates verbs automatically
- **Export:** `model/grammar.json` — JSON parity with Python (for web)

**Example:**
```
User spells: Y O SPACE C R E E R SPACE E L L O S
Display:    "yo creo ellos"  (conjugated on-the-fly)
            ↑                ↑
          pronoun       verb → "creer" stays base form
                        (only conjugates when followed by another pronoun)
```

Actually, wait — the current logic conjugates verbs that *follow* pronouns:
```
"yo creer" → "yo creo"
"ellos creer" → "ellos creen"
```

This is the correct behavior for natural Spanish output.

---

## Web Implementation (tensorflowjs + React/Vue)

**Zero changes needed to grammar logic.** The Python JSON table is directly consumable by JavaScript:

### Step 1: Use the JSON file
```javascript
// Load the grammar table (pre-built by Python export_grammar.py)
import grammarTable from "./model/grammar.json";

const VERB_CONJUGATIONS = grammarTable.verbs;
const PRONOUN_TO_FORM = grammarTable.pronouns;
```

### Step 2: Replicate Python logic in TypeScript/JavaScript
```typescript
function conjugateVerb(verb: string, pronoun: string | null): string {
  if (!pronoun) return verb;

  const pronounNorm = pronoun.toLowerCase().trim();
  const form = PRONOUN_TO_FORM[pronounNorm];
  if (!form) return verb;

  if (!(verb in VERB_CONJUGATIONS)) return verb;

  return VERB_CONJUGATIONS[verb][form] || verb;
}

function isPronoun(word: string): boolean {
  return word.toLowerCase().trim() in PRONOUN_TO_FORM;
}
```

### Step 3: Apply in overlay (React example)
```typescript
const conjugateForDisplay = (text: string): string => {
  const words = text.split(" ");
  if (words.length < 2) return text;

  const conjugatedWords = words.map((word, i) => {
    if (i === 0) return word;
    const prevWord = words[i - 1];
    if (isPronoun(prevWord)) {
      return conjugateVerb(word, prevWord);
    }
    return word;
  });

  return conjugatedWords.join(" ");
};
```

**Cost:** ~50 lines of JavaScript. Deterministic, no ML, uses the same JSON table.

---

## Future Improvements (Roadmap)

### Phase 1: Expand Vocabulary (Low Effort)
- [ ] **Add 10–20 more verbs** to `VERB_CONJUGATIONS` in `config.py`
  - Common ASL verbs: `ver` (to see), `dar` (to give), `hablar` (to speak), `entender` (to understand), etc.
  - Run `python scripts/export_grammar.py` after each update → JSON auto-generated
  - Web auto-updates on next deploy

- [ ] **Add informal pronouns** (already partially done)
  - "nosotros/nosotras" → covered
  - Regional variations (e.g., "vos" in Argentina) → add to `PRONOUN_TO_FORM`

### Phase 2: Grammar Beyond Verbs (Medium Effort)
- [ ] **Adjective agreement**
  - "yo soy [ADJECTIVE]" → adjective also changes (e.g., "cansado" vs. "cansada")
  - New table: `ADJECTIVE_AGREEMENT` with gender/number variants
  - Trigger: detect "ser/estar" + next word, apply agreement

- [ ] **Noun-adjective agreement**
  - "el [ADJECTIVE] [NOUN]" → both adjective and noun are affected
  - Requires POS tagging or pattern detection

### Phase 3: Temporal Context (Medium-High Effort)
- [ ] **Tense markers**
  - Detect time expressions: "ayer" (yesterday), "mañana" (tomorrow), "ahora" (now)
  - Conjugate verbs to match tense: "ayer + creer" → "creí" (past)
  - Current system only does present tense (default)

- [ ] **Conditional/subjunctive**
  - "si yo pudiera" (if I could) → conditional forms
  - "espero que ellos crean" (I hope they believe) → subjunctive
  - New conjugation tables per mode

### Phase 4: Context-Aware Abbreviations (High Effort)
- [ ] **Word elision**
  - Spanish drops pronouns when clear from context:
    - "yo creo" can become "creo" (I believe)
    - Detect redundancy, suggest removal
  - Reverse: if a verb appears without subject, suggest inserting pronoun

- [ ] **Clitic pronouns**
  - "me da" vs. "le da" (indirect object pronouns)
  - Require semantic understanding

### Phase 5: Model-Assisted Grammar (Very High Effort)
- [ ] **Sequence tagging with NLP**
  - Train a small LSTM/BERT on Spanish sentences with POS tags
  - Input: [word1, word2, word3] → Output: [PRON, VERB, ADJ]
  - Replace rule-based logic with learned patterns

- [ ] **Feedback loop**
  - User corrects a conjugation → log it
  - Fine-tune the NLP model offline
  - Deploy updated model to web

---

## Architecture: Why This Approach Works

| Aspect | Choice | Why |
|--------|--------|-----|
| **Tables, not ML** | Deterministic lookup | Simple, debuggeable, no latency. Scales to 100s of verbs without retraining. |
| **JSON, not Python objects** | Human-readable format | Web, mobile, other languages can consume it. Easy version control. |
| **Export script** | `export_grammar.py` | One source of truth (Python config). Single command to update web. No manual duplication. |
| **Replicate logic, not data** | Python + JavaScript | Logic is simple enough (~20 lines each). Keeps logic close to the rules, easy to audit. |
| **Fallback: no conjugation** | If word not in table | Safe. "creer" is shown instead of crashing or showing garbage. User can still understand. |

---

## Implementation Checklist

### MVP (Done)
- [x] Grammar module (`src/grammar.py`)
- [x] Config tables (`config.py`)
- [x] LetterBuffer integration (`src/overlay.py`)
- [x] JSON export (`scripts/export_grammar.py`, `model/grammar.json`)
- [x] Tests (`test_grammar.py`)

### Next Steps (Phase 1 — Web)
- [ ] Replicate `conjugateVerb()` and `isPronoun()` in TypeScript/JavaScript
- [ ] Load `model/grammar.json` in web app
- [ ] Integrate into overlay component
- [ ] Test with real use cases (e.g., "yo creo", "ellos pueden")

### Phase 1 — Expand Verbs
- [ ] Add 10–20 more verbs to config
- [ ] Test conjugations
- [ ] Export and redeploy web

---

## Key Insight

**Conjugation in Spanish is regular enough (~70% of verbs) that a lookup table beats ML for this use case.**

The 30% of irregular verbs are the most common (ser, ir, tener, hacer, poder, etc.) — we've already included them. Adding more verbs takes seconds; training a model takes hours and needs data.

For verb tense/mood (present → past, subjunctive, etc.), we'd eventually need ML or regex patterns (Phase 3). But for pronoun-based conjugation of a known verb? Table lookup is perfect.

---

## Questions?

- **How often do I need to update the tables?** Only when you add new vocabulary. The export script is your friend.
- **What if I want more languages?** Duplicate the tables (e.g., `VERB_CONJUGATIONS_EN`, `PRONOUN_TO_FORM_EN`) and add a language picker to config.
- **Can I use this for other Romance languages?** Yes. French, Italian, Portuguese all have similar structures. Just add their tables.

