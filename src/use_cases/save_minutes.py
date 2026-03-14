import inject
from pathlib import Path

from src.domain.entities.meeting import Meeting
from src.domain.entities.transcript import Transcript
from src.domain.entities.summary import Summary
from src.domain.ports.persistence_port import PersistencePort


class SaveMinutesUseCase:
    """
    Orchestrates saving the meeting minutes.
    """

    @inject.autoparams()
    def __init__(self, persistence_port: PersistencePort):
        self._persistence_port = persistence_port

    def execute(self, meeting: Meeting, transcript: Transcript, summary: Summary) -> Path:
        """
        Saves the complete meeting minutes.
        """
        return self._persistence_port.save(meeting, transcript, summary)
