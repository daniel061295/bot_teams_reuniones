from abc import ABC, abstractmethod

from src.domain.entities.transcript import Transcript
from src.domain.entities.summary import Summary


class SummarizationPort(ABC):
    """
    Interface for summarizing a transcript.
    """

    @abstractmethod
    def summarize(self, transcript: Transcript) -> Summary:
        """
        Summarizes the given transcript and returns a Summary object.
        """
        pass
