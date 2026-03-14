import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.summarization.gemini_summarizer import GeminiSummarizer
from src.domain.entities.transcript import Transcript, TranscriptSegment

@patch("src.infrastructure.summarization.gemini_summarizer.genai")
@patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
def test_gemini_summarizer_success(mock_genai):
    # Mock the return values from the package
    mock_model = MagicMock()
    
    # First call: probe (ping)
    # Second call: actual summary
    mock_response_probe = MagicMock()
    mock_response_probe.text = "ok"
    
    mock_response_summary = MagicMock()
    mock_response_summary.text = "Mocked LLM Response"
    
    # side_effect returns probe response first, then actual response
    mock_model.generate_content.side_effect = [mock_response_probe, mock_response_summary]
    mock_genai.GenerativeModel.return_value = mock_model
    
    summarizer = GeminiSummarizer()
    
    transcript = Transcript(
        meeting_id="m123",
        segments=[TranscriptSegment(start=0, end=1, text="Test text.")],
        language="es",
        model_used="whisper"
    )
    
    summary = summarizer.summarize(transcript)
    
    # 2 calls: one for _get_model() probe and one for actual summarize()
    assert mock_model.generate_content.call_count == 2
    assert summary.meeting_id == "m123"
    assert "Mocked LLM Response" in summary.raw_response

@patch.dict("os.environ", clear=True)
def test_gemini_summarizer_no_api_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
        GeminiSummarizer()
