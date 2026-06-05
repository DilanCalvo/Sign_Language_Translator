"""
Speech-to-text input using faster-whisper (Whisper on CPU).

SpeechInput loads the Whisper model in a daemon thread at startup so the
camera loop is never delayed. The microphone is toggled with
start_recording() / stop_recording(); transcription also runs in a background
thread. Completed transcripts are retrieved non-blocking via get_transcript().

State machine:
    LOADING       model download / first-time initialization in progress
    IDLE          model ready, microphone closed
    LISTENING     microphone open, buffering audio frames
    TRANSCRIBING  microphone closed, Whisper processing the clip
    UNAVAILABLE   faster-whisper not installed or model load failed
"""

from __future__ import annotations

import queue
import threading
from enum import Enum, auto

import numpy as np


class SpeechState(Enum):
    LOADING      = auto()
    IDLE         = auto()
    LISTENING    = auto()
    TRANSCRIBING = auto()
    UNAVAILABLE  = auto()


class SpeechInput:
    """
    Push-to-talk speech recognizer backed by faster-whisper.

    Typical frame-loop usage:

        if key == ord('p'):
            if speech.state == SpeechState.IDLE:
                speech.start_recording()
            elif speech.state == SpeechState.LISTENING:
                speech.stop_recording()

        text = speech.get_transcript()   # non-blocking, call every frame
        if text:
            speech_buffer.add(text)
    """

    SAMPLE_RATE  = 16_000  # Hz — Whisper requires 16 kHz mono float32
    MIN_DURATION = 0.4     # seconds — clips shorter than this are discarded

    def __init__(self, model_size: str = "tiny",
                 language: str | None = None) -> None:
        self._model_size = model_size
        self._language   = language
        self._model      = None
        self._stream     = None

        self._state        = SpeechState.LOADING
        self._state_lock   = threading.Lock()
        self._audio_lock   = threading.Lock()
        self._audio_chunks: list[np.ndarray] = []
        self._transcript_q: queue.Queue[str]  = queue.Queue()
        self._model_ready  = threading.Event()

        threading.Thread(
            target=self._load_model, daemon=True, name="whisper-loader"
        ).start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> SpeechState:
        with self._state_lock:
            return self._state

    @property
    def is_ready(self) -> bool:
        return self._model_ready.is_set()

    def start_recording(self) -> bool:
        """
        Open the microphone and begin buffering audio.

        Returns True when recording has started, False if the component is
        not in IDLE state (model still loading, already recording, etc.).
        """
        with self._state_lock:
            if self._state != SpeechState.IDLE:
                return False
            self._state = SpeechState.LISTENING

        with self._audio_lock:
            self._audio_chunks = []

        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            return True
        except Exception as exc:
            print(f"[SpeechInput] Could not open microphone: {exc}")
            with self._state_lock:
                self._state = SpeechState.IDLE
            return False

    def stop_recording(self) -> None:
        """
        Close the microphone and queue a background transcription task.
        Clips shorter than MIN_DURATION are silently discarded.
        """
        with self._state_lock:
            if self._state != SpeechState.LISTENING:
                return
            self._state = SpeechState.TRANSCRIBING

        # _state_lock is released before stream.stop() so the audio callback
        # can finish cleanly — stream.stop() waits for in-flight callbacks.
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._audio_lock:
            chunks = list(self._audio_chunks)
            self._audio_chunks = []

        if not chunks:
            with self._state_lock:
                self._state = SpeechState.IDLE
            return

        audio = np.concatenate(chunks)
        if audio.size < self.SAMPLE_RATE * self.MIN_DURATION:
            with self._state_lock:
                self._state = SpeechState.IDLE
            return

        threading.Thread(
            target=self._transcribe, args=(audio,),
            daemon=True, name="whisper-transcribe"
        ).start()

    def get_transcript(self) -> str | None:
        """Non-blocking. Returns the most recent completed transcript or None."""
        try:
            return self._transcript_q.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        """Release the microphone if still open. Safe to call multiple times."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print(
                "[SpeechInput] faster-whisper not installed — speech input disabled.\n"
                "             Install with:  pip install faster-whisper"
            )
            with self._state_lock:
                self._state = SpeechState.UNAVAILABLE
            return

        try:
            # int8 quantization: ~2x faster on CPU with negligible accuracy loss.
            self._model = WhisperModel(
                self._model_size, device="cpu", compute_type="int8"
            )
            self._model_ready.set()
            with self._state_lock:
                self._state = SpeechState.IDLE
            print(f"[SpeechInput] Whisper '{self._model_size}' model ready.")
        except Exception as exc:
            print(f"[SpeechInput] Model load failed: {exc}")
            with self._state_lock:
                self._state = SpeechState.UNAVAILABLE

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status) -> None:
        # Called from sounddevice's internal audio thread.
        # Lock acquisition here is brief (single boolean read) so priority
        # inversion risk is negligible for a speech-recognition workload.
        with self._state_lock:
            recording = (self._state == SpeechState.LISTENING)
        if recording:
            # indata shape: (frames, channels). Flatten to 1-D for Whisper.
            with self._audio_lock:
                self._audio_chunks.append(indata[:, 0].copy())

    def _transcribe(self, audio: np.ndarray) -> None:
        try:
            segments, _ = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1,      # fastest; raise to 5 for higher accuracy
                vad_filter=True,  # skip silent segments automatically
                vad_parameters={"min_silence_duration_ms": 300},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text:
                self._transcript_q.put(text)
        except Exception as exc:
            print(f"[SpeechInput] Transcription error: {exc}")
        finally:
            with self._state_lock:
                self._state = SpeechState.IDLE
