import inject
from datetime import datetime
from pathlib import Path

from src.domain.entities.meeting import AudioConfig, Meeting
from src.domain.ports.audio_capture_port import AudioCapturePort


class RecordMeetingUseCase:
    """
    Orchestrates the recording of a meeting.
    """

    @inject.autoparams()
    def __init__(self, audio_capture_port: AudioCapturePort):
        self._audio_capture = audio_capture_port

    def start(self, meeting_id: str, audio_config: AudioConfig) -> Meeting:
        """
        Starts the meeting recording and creates a Meeting entity.
        """
        if self._audio_capture.is_recording:
            raise RuntimeError("A recording is already in progress.")

        start_time = datetime.now()
        self._audio_capture.start_recording(audio_config)

        return Meeting(
            id=meeting_id,
            started_at=start_time,
            audio_config=audio_config
        )

    def stop(self, meeting: Meeting) -> Meeting:
        """
        Stops the recording and updates the Meeting entity.
        """
        if not self._audio_capture.is_recording:
            raise RuntimeError("No recording is currently in progress.")

        audio_path = self._audio_capture.stop_recording()
        meeting.ended_at = datetime.now()
        meeting.audio_file_path = audio_path

        return meeting
