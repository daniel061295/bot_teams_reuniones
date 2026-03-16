import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import numpy as np

from src.infrastructure.transcription.groq_transcriber import GroqTranscriber


@patch("src.infrastructure.transcription.groq_transcriber.FasterWhisperTranscriber.load_wav_as_float32")
@patch("src.infrastructure.transcription.groq_transcriber.FasterWhisperTranscriber.resample_to_16k")
@patch("src.infrastructure.transcription.groq_transcriber.FasterWhisperTranscriber.apply_vad")
@patch("src.infrastructure.transcription.groq_transcriber.wavfile.write")
@patch("src.infrastructure.transcription.groq_transcriber.Groq")
@patch("src.infrastructure.transcription.groq_transcriber.subprocess.run")
@patch("src.infrastructure.transcription.groq_transcriber.imageio_ffmpeg.get_ffmpeg_exe",
       return_value="/fake/ffmpeg")
@patch.dict("os.environ", {"GROQ_API_KEY": "fake-groq-key"})
def test_groq_transcriber_success(mock_ffmpeg, mock_run, mock_groq_cls, mock_write, mock_vad, mock_resample, mock_load):
    """Happy-path: ffmpeg converts to MP3, Groq returns verbose_json segments."""
    # mock audio pre-processing
    mock_load.return_value = (np.array([0.0]), 16000)
    mock_resample.return_value = np.array([0.0])
    mock_vad.return_value = (np.array([0.0]), 0.8)

    # ffmpeg returns small fake MP3 bytes
    mock_run.return_value = MagicMock(returncode=0, stdout=b"ID3" + b"\x00" * 100)

    # Groq client response with segments
    seg = {"start": 0.0, "end": 3.0, "text": " Hola equipo "}
    mock_response = MagicMock()
    mock_response.segments = [seg]
    mock_response.text = "Hola equipo"

    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = mock_response
    mock_groq_cls.return_value = mock_client

    transcriber = GroqTranscriber()

    with patch.object(Path, "exists", return_value=True):
        transcript = transcriber.transcribe(Path("meeting.wav"), language="es", meeting_id="m1")

    assert transcript.meeting_id == "m1"
    assert transcript.language == "es"
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "Hola equipo"
    assert "groq" in transcript.model_used


@patch.dict("os.environ", clear=True)
def test_groq_transcriber_no_api_key():
    """Should raise ValueError when GROQ_API_KEY is not set."""
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqTranscriber()


@patch("src.infrastructure.transcription.groq_transcriber.FasterWhisperTranscriber.load_wav_as_float32")
@patch("src.infrastructure.transcription.groq_transcriber.FasterWhisperTranscriber.resample_to_16k")
@patch("src.infrastructure.transcription.groq_transcriber.FasterWhisperTranscriber.apply_vad")
@patch("src.infrastructure.transcription.groq_transcriber.wavfile.write")
@patch("src.infrastructure.transcription.groq_transcriber.Groq")
@patch("src.infrastructure.transcription.groq_transcriber.subprocess.run")
@patch("src.infrastructure.transcription.groq_transcriber.imageio_ffmpeg.get_ffmpeg_exe",
       return_value="/fake/ffmpeg")
@patch.dict("os.environ", {"GROQ_API_KEY": "fake-groq-key"})
def test_groq_transcriber_ffmpeg_failure(mock_ffmpeg, mock_run, mock_groq_cls, mock_write, mock_vad, mock_resample, mock_load):
    """Should raise RuntimeError when ffmpeg exits with non-zero code."""
    # mock audio pre-processing
    mock_load.return_value = (np.array([0.0]), 16000)
    mock_resample.return_value = np.array([0.0])
    mock_vad.return_value = (np.array([0.0]), 0.8)

    mock_run.return_value = MagicMock(returncode=1, stdout=b"", stderr=b"error msg")
    mock_groq_cls.return_value = MagicMock()

    transcriber = GroqTranscriber()
    with patch.object(Path, "exists", return_value=True):
        with pytest.raises(RuntimeError, match="ffmpeg"):
            transcriber.transcribe(Path("meeting.wav"))


@patch("src.infrastructure.transcription.groq_transcriber.Groq")
@patch.dict("os.environ", {"GROQ_API_KEY": "fake-groq-key"})
def test_groq_transcriber_file_not_found(mock_groq_cls):
    """Should raise FileNotFoundError when the WAV does not exist."""
    mock_groq_cls.return_value = MagicMock()
    transcriber = GroqTranscriber()
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(Path("nonexistent.wav"))