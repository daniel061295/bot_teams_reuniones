import inject
from datetime import datetime
from unittest.mock import MagicMock
import pytest

from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.entities.summary import Summary
from src.domain.ports.summarization_port import SummarizationPort
from src.use_cases.summarize_meeting import SummarizeMeetingUseCase

@pytest.fixture
def mock_summarization():
    mock = MagicMock(spec=SummarizationPort)
    return mock

@pytest.fixture(autouse=True)
def inject_config(mock_summarization):
    def configure(binder):
        binder.bind(SummarizationPort, mock_summarization)
    inject.clear_and_configure(configure)
    yield
    inject.clear()

def test_summarize_meeting_success(mock_summarization):
    use_case = SummarizeMeetingUseCase()
    
    transcript = Transcript(
        meeting_id="test-123",
        segments=[TranscriptSegment(0, 1, "test audio content")],
        language="es",
        model_used="whisper"
    )
    
    expected_summary = Summary(
        meeting_id="test-123",
        executive_summary="Executive",
        technical_decisions=["Tech"],
        blockers=["Block"],
        action_items=["Action"],
        raw_response="",
        generated_at=datetime.now()
    )
    mock_summarization.summarize.return_value = expected_summary
    
    summary = use_case.execute(transcript)
    
    mock_summarization.summarize.assert_called_once_with(transcript)
    assert summary == expected_summary

def test_summarize_meeting_empty_transcript(mock_summarization):
    use_case = SummarizeMeetingUseCase()
    
    transcript = Transcript(
        meeting_id="test-123",
        segments=[],
        language="es",
        model_used="whisper"
    )
    
    with pytest.raises(ValueError):
        use_case.execute(transcript)
