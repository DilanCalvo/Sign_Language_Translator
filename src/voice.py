"""
Offline speech synthesis via Windows SAPI5.

VoiceOutput wraps the Windows Speech API in a background daemon thread
with a queue so speak() returns immediately and never blocks the camera loop.

pyttsx3 is a wrapper around the same SAPI5 API but its runAndWait() does
not reset SAPI5's internal state between calls in a loop, causing it to
speak only the first utterance. Using win32com directly avoids that bug
while remaining 100% offline (pywin32 is already a project dependency).
"""

import queue
import threading


class VoiceOutput:
    def __init__(self):
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text: str) -> None:
        """Enqueue text for speech. Non-blocking."""
        self._queue.put(str(text))

    def stop(self) -> None:
        """Signal the worker thread to exit cleanly."""
        self._queue.put(None)

    def _worker(self):
        import pythoncom
        import win32com.client

        # COM objects require CoInitialize in every thread that uses them.
        pythoncom.CoInitialize()
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            while True:
                text = self._queue.get()
                if text is None:
                    break
                # Speak() with default flags is synchronous: blocks until the
                # utterance finishes, so queue items are never overlapped.
                speaker.Speak(text)
        finally:
            pythoncom.CoUninitialize()
