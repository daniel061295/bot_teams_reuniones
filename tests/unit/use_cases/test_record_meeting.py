import inject
from datetime import datetime
from unittest.mock import MagicMock
from pathlib import Path
import pytest

from src.domain.entities.meeting import AudioConfig, Meeting
from src.domain.ports.audio_capture_port import AudioCapturePort
from src.use_cases.record_meeting import RecordMeetingUseCase

@pytest.fixture
def mock_audio_capture():
    mock = MagicMock(spec=AudioCapturePort)
    # Configure default behavior
    mock.is_recording = False
    mock.stop_recording.return_value = Path("test_audio.wav")
    return mock

@pytest.fixture(autouse=True)
def inject_config(mock_audio_capture):
    def configure(binder):
        binder.bind(AudioCapturePort, mock_audio_capture)
    inject.clear_and_configure(configure)
    yield
    inject.clear()

def test_start_recording(mock_audio_capture):
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    use_case = RecordMeetingUseCase()
    
    meeting = use_case.start("test-123", config)
    
    mock_audio_capture.start_recording.assert_called_once_with(config)
    assert meeting.id == "test-123"
    assert meeting.audio_config == config
    assert isinstance(meeting.started_at, datetime)

def test_start_recording_already_in_progress(mock_audio_capture):
    mock_audio_capture.is_recording = True
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    use_case = RecordMeetingUseCase()
    
    with pytest.raises(RuntimeError):
        use_case.start("test-123", config)

def test_stop_recording(mock_audio_capture):
    mock_audio_capture.is_recording = True
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    use_case = RecordMeetingUseCase()
    meeting = Meeting(id="test-123", started_at=datetime.now(), audio_config=config)
    
    updated_meeting = use_case.stop(meeting)
    
    mock_audio_capture.stop_recording.assert_called_once()
    assert updated_meeting.ended_at is not None
    assert updated_meeting.audio_file_path == Path("test_audio.wav")

def test_stop_recording_not_in_progress(mock_audio_capture):
    use_case = RecordMeetingUseCase()
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    meeting = Meeting(id="test-123", started_at=datetime.now(), audio_config=config)
    
    with pytest.raises(RuntimeError):
        use_case.stop(meeting)
