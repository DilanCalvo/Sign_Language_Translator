"""
Translation layer: ASL glosses -> fluent sentence (the second stage).

This is the half of the two-stage design that the recognizer deliberately does
NOT do. The recognizer outputs GLOSSES — English keywords in citation form, the
base form of each sign (e.g. ["want", "drink", "now"]). This module sends that
sequence to Claude and gets back a natural sentence in the target language
("Quiero tomar algo ahora.").

Why translation, and not the recognizer, handles conjugation and tense:
ASL does not conjugate verbs the way Spanish does — tense is set once with a
time sign/adverb and verbs inflect through space and movement, not endings.
Trying to make the recognizer output conjugated forms is linguistically wrong
and combinatorially explosive (one sign -> dozens of surface forms). So the
recognizer stays at the gloss level and the LLM produces fluent, conjugated
output. See ROADMAP.md and config.py section 10.

Design notes:
  - The API call runs in a background daemon thread so the camera loop never
    blocks while waiting on the network. submit() returns immediately; the main
    loop calls poll() each frame to pick up the result when it is ready.
  - Fully degradable: if the SDK is missing, there is no API key, or the call
    fails, it falls back to the raw glosses joined by spaces (when
    TRANSLATION_OFFLINE_FALLBACK is on) so the app keeps working offline.
  - The API key is read from the ANTHROPIC_API_KEY environment variable by the
    SDK; it is never hardcoded.
"""

import os
import queue
import threading


_SYSTEM_PROMPT = (
    "You translate American Sign Language (ASL) gloss sequences into fluent {lang}.\n"
    "The input is a sequence of ASL glosses: English keywords in citation (base) "
    "form, in signing order. ASL omits articles and copulas, does not conjugate "
    "verbs, and marks tense once with a time word rather than verb endings.\n"
    "Produce ONE natural, grammatical {lang} sentence that conveys the meaning: "
    "add the articles, prepositions, conjugation and tense that {lang} requires, "
    "and reorder words to natural {lang} syntax. If a time word is present "
    "(now, later, yesterday, tomorrow, finished/already), use it to choose the "
    "tense. Keep it faithful — do not invent content that is not in the glosses.\n"
    "Output ONLY the final sentence: no quotes, no gloss list, no explanation, "
    "no preamble."
)


class Translator:
    """Background ASL-gloss -> sentence translator backed by the Claude API."""

    def __init__(self, model, target_language, enabled=True, offline_fallback=True):
        self._model            = model
        self._lang             = target_language
        self._offline_fallback = offline_fallback
        self._client           = None
        self._results          = queue.Queue()
        self._busy             = False

        if not enabled:
            self._reason = "translation disabled in config"
            return

        # Build the client lazily and tolerate every failure mode (no SDK, no
        # key) — translation is an enhancement, never a hard dependency.
        try:
            import anthropic
        except ImportError:
            self._reason = "anthropic SDK not installed (pip install anthropic)"
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self._reason = "ANTHROPIC_API_KEY not set"
            return
        try:
            self._client = anthropic.Anthropic()
            self._reason = None
        except Exception as e:  # pragma: no cover - defensive
            self._reason = f"client init failed: {e}"

    @property
    def online(self) -> bool:
        """True if the API path is available (a key + SDK + client were set up)."""
        return self._client is not None

    @property
    def reason(self) -> str | None:
        """Why translation is offline (None when online). For startup messaging."""
        return self._reason

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, glosses) -> bool:
        """
        Start translating a gloss list in the background.

        Returns True if a translation was started (online and not already busy).
        When offline, immediately queues the fallback string instead so the
        caller still gets a result to display. Returns False only if a request
        is already in flight.
        """
        glosses = [g for g in glosses if g]
        if not glosses or self._busy:
            return False

        if not self.online:
            if self._offline_fallback:
                self._results.put(self._fallback(glosses))
            return False

        self._busy = True
        threading.Thread(target=self._worker, args=(glosses,), daemon=True).start()
        return True

    def poll(self):
        """Return a finished translation string if one is ready, else None."""
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------

    def _worker(self, glosses):
        try:
            text = self._call_api(glosses)
        except Exception as e:
            print(f"[translator] API error: {e}")
            text = self._fallback(glosses) if self._offline_fallback else None
        finally:
            self._busy = False
        if text:
            self._results.put(text)

    def _call_api(self, glosses) -> str:
        system = _SYSTEM_PROMPT.format(lang=self._lang)
        # A short, latency-sensitive task: keep thinking off and effort low so
        # the result comes back fast; instruct the model to emit only the
        # sentence (Opus may otherwise narrate its reasoning when thinking off).
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            system=system,
            messages=[{"role": "user", "content": " ".join(glosses)}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "").strip()
        return text or self._fallback(glosses)

    def _fallback(self, glosses) -> str:
        return " ".join(glosses)
