from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pathlib import Path

from src.domain.entities.meeting import AudioConfig


class AudioCapturePort(ABC):
    """
    Interface for capturing audio during a meeting.
    """

    @abstractmethod
    def start_recording(self, config: AudioConfig) -> None:
        """
        Starts the audio recording process.
        """
        pass

    @abstractmethod
    def stop_recording(self) -> Path:
        """
        Stops the recording process and returns the path to the saved audio file.
        """
        pass

    @property
    @abstractmethod
    def is_recording(self) -> bool:
        """
        Indicates whether a recording is currently in progress.
        """
        pass

    @abstractmethod
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        Returns a list of available audio devices.
        """
        pass
