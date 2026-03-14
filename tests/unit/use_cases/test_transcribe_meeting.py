import inject
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.domain.entities.meeting import AudioConfig, Meeting
from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.ports.transcription_port import TranscriptionPort
from src.use_cases.transcribe_meeting import TranscribeMeetingUseCase

@pytest.fixture
def mock_transcription():
    mock = MagicMock(spec=TranscriptionPort)
    return mock

@pytest.fixture(autouse=True)
def inject_config(mock_transcription):
    def configure(binder):
        binder.bind(TranscriptionPort, mock_transcription)
    inject.clear_and_configure(configure)
    yield
    inject.clear()

def test_transcribe_meeting_success(mock_transcription):
    use_case = TranscribeMeetingUseCase()
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    # Need to mock Path.exists to return True
    audio_path = MagicMock(spec=Path)
    audio_path.exists.return_value = True
    
    meeting = Meeting(id="test-123", started_at=datetime.now(), audio_config=config, audio_file_path=audio_path)
    
    expected_transcript = Transcript(
        meeting_id="test-123",
        segments=[TranscriptSegment(0, 1, "test")],
        language="es",
        model_used="whisper"
    )
    mock_transcription.transcribe.return_value = expected_transcript
    
    transcript = use_case.execute(meeting, "en")
    
    mock_transcription.transcribe.assert_called_once_with(
        audio_path=audio_path,
        language="en",
        meeting_id="test-123"
    )
    assert transcript == expected_transcript

def test_transcribe_meeting_no_audio(mock_transcription):
    use_case = TranscribeMeetingUseCase()
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    meeting = Meeting(id="test-123", started_at=datetime.now(), audio_config=config, audio_file_path=None)
    
    with pytest.raises(ValueError):
        use_case.execute(meeting)
        
def test_transcribe_meeting_audio_not_exists(mock_transcription):
    use_case = TranscribeMeetingUseCase()
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    audio_path = MagicMock(spec=Path)
    audio_path.exists.return_value = False
    meeting = Meeting(id="test-123", started_at=datetime.now(), audio_config=config, audio_file_path=audio_path)
    
    with pytest.raises(ValueError):
        use_case.execute(meeting)
