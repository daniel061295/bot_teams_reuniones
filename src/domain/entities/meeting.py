from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class AudioConfig:
    """
    Configuration for audio capture.
    """
    sample_rate: int
    channels: int
    dtype: str
    loopback_device: Optional[str] = None
    microphone_device: Optional[str] = None


@dataclass
class Meeting:
    """
    Represents a meeting session.
    """
    id: str
    started_at: datetime
    audio_config: AudioConfig
    ended_at: Optional[datetime] = None
    audio_file_path: Optional[Path] = None

    @property
    def duration_seconds(self) -> float:
        """
        Calculates the duration of the meeting in seconds.
        """
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds()
