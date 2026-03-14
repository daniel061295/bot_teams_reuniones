from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities.transcript import Transcript


class TranscriptionPort(ABC):
    """
    Interface for transcribing audio files.
    """

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str = "es", meeting_id: str = "") -> Transcript:
        """
        Transcribes the given audio file and returns a Transcript object.
        """
        pass
