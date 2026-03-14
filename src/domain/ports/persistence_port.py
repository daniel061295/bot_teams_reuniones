from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities.meeting import Meeting
from src.domain.entities.transcript import Transcript
from src.domain.entities.summary import Summary


class PersistencePort(ABC):
    """
    Interface for saving meeting data.
    """

    @abstractmethod
    def save(self, meeting: Meeting, transcript: Transcript, summary: Summary) -> Path:
        """
        Saves the meeting minutes and returns the path to the saved file.
        """
        pass
