import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.infrastructure.transcription.hybrid_transcriber import HybridTranscriber
from src.domain.entities.transcript import Transcript, TranscriptSegment


def _make_transcript(model: str = "groq-turbo") -> Transcript:
    return Transcript(
        meeting_id="m1",
        segments=[TranscriptSegment(0, 1, "Hola")],
        language="es",
        model_used=model,
    )


@patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"})
def test_hybrid_uses_groq_when_key_present():
    expected = _make_transcript(model="groq-whisper-large-v3-turbo")
    transcriber = HybridTranscriber.__new__(HybridTranscriber)
    transcriber._groq_enabled = True
    transcriber._local = None
    mock_groq = MagicMock()
    mock_groq.transcribe.return_value = expected
    transcriber._groq = mock_groq
    result = transcriber.transcribe(Path("audio.wav"), language="es", meeting_id="m1")
    mock_groq.transcribe.assert_called_once_with(Path("audio.wav"), language="es", meeting_id="m1")
    assert result == expected


@patch.dict("os.environ", {"GROQ_API_KEY": "fake-key"})
def test_hybrid_falls_back_to_local_on_groq_error():
    local_result = _make_transcript(model="faster-whisper-medium-cpu")
    transcriber = HybridTranscriber.__new__(HybridTranscriber)
    transcriber._groq_enabled = True
    mock_groq = MagicMock()
    mock_groq.transcribe.side_effect = Exception("quota exceeded")
    transcriber._groq = mock_groq
    mock_local = MagicMock()
    mock_local.transcribe.return_value = local_result
    transcriber._local = mock_local
    result = transcriber.transcribe(Path("audio.wav"), language="es", meeting_id="m1")
    mock_groq.transcribe.assert_called_once()
    mock_local.transcribe.assert_called_once()
    assert result == local_result


@patch.dict("os.environ", {}, clear=True)
def test_hybrid_uses_local_when_no_api_key():
    local_result = _make_transcript(model="faster-whisper-medium-cpu")
    transcriber = HybridTranscriber.__new__(HybridTranscriber)
    transcriber._groq_enabled = False
    transcriber._groq = None
    mock_local = MagicMock()
    mock_local.transcribe.return_value = local_result
    transcriber._local = mock_local
    result = transcriber.transcribe(Path("audio.wav"), language="es", meeting_id="m1")
    mock_local.transcribe.assert_called_once()
    assert result == local_result


def test_hybrid_get_local_lazy_loads():
    transcriber = HybridTranscriber.__new__(HybridTranscriber)
    transcriber._groq_enabled = False
    transcriber._groq = None
    transcriber._local = None
    mock_fw_instance = MagicMock()
    with patch(
        "src.infrastructure.transcription.faster_whisper_transcriber.FasterWhisperTranscriber",
        return_value=mock_fw_instance,
        create=True,
    ):
        transcriber._local = mock_fw_instance
        result = transcriber._get_local()
    assert result is mock_fw_instance
    assert transcriber._get_local() is mock_fw_instance
