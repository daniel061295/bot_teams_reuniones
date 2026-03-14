import inject

from src.domain.entities.meeting import Meeting
from src.domain.entities.transcript import Transcript
from src.domain.ports.transcription_port import TranscriptionPort


class TranscribeMeetingUseCase:
    """
    Orchestrates the transcription of a recorded meeting.
    """

    @inject.autoparams()
    def __init__(self, transcription_port: TranscriptionPort):
        self._transcription_port = transcription_port

    def execute(self, meeting: Meeting, language: str = "es") -> Transcript:
        """
        Transcribes the audio file from the meeting.
        """
        if meeting.audio_file_path is None or not meeting.audio_file_path.exists():
            raise ValueError("Meeting does not have a valid audio file to transcribe.")

        return self._transcription_port.transcribe(
            audio_path=meeting.audio_file_path,
            language=language,
            meeting_id=meeting.id
        )
