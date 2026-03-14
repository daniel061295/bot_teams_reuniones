import inject

from src.domain.entities.transcript import Transcript
from src.domain.entities.summary import Summary
from src.domain.ports.summarization_port import SummarizationPort


class SummarizeMeetingUseCase:
    """
    Orchestrates the summarization of a meeting transcript.
    """

    @inject.autoparams()
    def __init__(self, summarization_port: SummarizationPort):
        self._summarization_port = summarization_port

    def execute(self, transcript: Transcript) -> Summary:
        """
        Generates a summary from the given transcript.
        """
        if not transcript.segments:
            raise ValueError("Transcript is empty, nothing to summarize.")

        return self._summarization_port.summarize(transcript)
