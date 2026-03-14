import inject
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.domain.entities.meeting import AudioConfig, Meeting
from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.entities.summary import Summary
from src.domain.ports.persistence_port import PersistencePort
from src.use_cases.save_minutes import SaveMinutesUseCase

@pytest.fixture
def mock_persistence():
    mock = MagicMock(spec=PersistencePort)
    return mock

@pytest.fixture(autouse=True)
def inject_config(mock_persistence):
    def configure(binder):
        binder.bind(PersistencePort, mock_persistence)
    inject.clear_and_configure(configure)
    yield
    inject.clear()

def test_save_minutes_success(mock_persistence):
    use_case = SaveMinutesUseCase()
    
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    meeting = Meeting(id="test-123", started_at=datetime.now(), audio_config=config)
    transcript = Transcript(
        meeting_id="test-123",
        segments=[TranscriptSegment(0, 1, "text")],
        language="es",
        model_used="whisper"
    )
    summary = Summary(
        meeting_id="test-123",
        executive_summary="",
        technical_decisions=[],
        blockers=[],
        action_items=[],
        raw_response="",
        generated_at=datetime.now()
    )
    
    expected_path = Path("minuta.md")
    mock_persistence.save.return_value = expected_path
    
    path = use_case.execute(meeting, transcript, summary)
    
    mock_persistence.save.assert_called_once_with(meeting, transcript, summary)
    assert path == expected_path
