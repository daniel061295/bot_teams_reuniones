import os
from pathlib import Path

from src.domain.entities.transcript import Transcript
from src.domain.ports.transcription_port import TranscriptionPort


class HybridTranscriber(TranscriptionPort):
    """
    Hybrid transcription: Groq API (primary) → faster-whisper local (fallback).

    Decision logic:
    1. GROQ_API_KEY not set → use faster-whisper directly.
    2. GROQ_API_KEY set    → try Groq first.
    3. Groq fails (quota/no internet/API error) → log + fallback to local.

    Both adapters are initialised lazily to avoid loading Whisper weights
    into memory when Groq is available and working.
    """

    def __init__(self) -> None:
        self._groq: TranscriptionPort | None = None
        self._local: TranscriptionPort | None = None

        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        self._groq_enabled = bool(groq_key)

        if self._groq_enabled:
            try:
                from src.infrastructure.transcription.groq_transcriber import GroqTranscriber
                self._groq = GroqTranscriber()
                print("[Hybrid] Groq API transcriber ready — will be used as primary.")
            except Exception as exc:  # noqa: BLE001
                print(f"[Hybrid] Could not initialise Groq transcriber ({exc}). Will use local only.")
                self._groq_enabled = False
        else:
            print("[Hybrid] GROQ_API_KEY not set — using faster-whisper local transcriber only.")

    def _get_local(self) -> TranscriptionPort:
        """Lazy-loads FasterWhisperTranscriber (heavy model weights)."""
        if self._local is None:
            from src.infrastructure.transcription.faster_whisper_transcriber import FasterWhisperTranscriber
            self._local = FasterWhisperTranscriber()
        return self._local

    def transcribe(self, audio_path: Path, language: str = "es", meeting_id: str = "") -> Transcript:
        """Transcribes using best available strategy."""
        if self._groq_enabled and self._groq is not None:
            try:
                return self._groq.transcribe(audio_path, language=language, meeting_id=meeting_id)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[Hybrid] Groq transcription failed: {type(exc).__name__}: {exc}\n"
                    "         → Falling back to local faster-whisper..."
                )
        return self._get_local().transcribe(audio_path, language=language, meeting_id=meeting_id)
