from dataclasses import dataclass
from typing import List


@dataclass
class TranscriptSegment:
    """
    Represents a specific segment of the transcribed audio.
    """
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    """
    Represents the full transcription of a meeting.
    """
    meeting_id: str
    segments: List[TranscriptSegment]
    language: str
    model_used: str

    @property
    def full_text(self) -> str:
        """
        Returns the full text of the transcription by joining all segments.
        """
        return " ".join([segment.text for segment in self.segments])
