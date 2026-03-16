import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

from src.infrastructure.transcription.faster_whisper_transcriber import (
    FasterWhisperTranscriber,
    WHISPER_SAMPLE_RATE,
)


@patch("src.infrastructure.transcription.faster_whisper_transcriber.WhisperModel")
@patch("src.infrastructure.transcription.faster_whisper_transcriber.torch")
@patch("src.infrastructure.transcription.faster_whisper_transcriber.wavfile")
def test_faster_whisper_transcriber_success(mock_wavfile, mock_torch, mock_wm):
    mock_torch.cuda.is_available.return_value = False
    mock_seg = MagicMock()
    mock_seg.start = 0.0
    mock_seg.end = 2.5
    mock_seg.text = " Hello "
    mock_model_instance = MagicMock()
    mock_model_instance.transcribe.return_value = (iter([mock_seg]), MagicMock())
    mock_wm.return_value = mock_model_instance
    mock_wavfile.read.return_value = (
        WHISPER_SAMPLE_RATE,
        np.ones(WHISPER_SAMPLE_RATE, dtype=np.int16) * 1000,
    )
    transcriber = FasterWhisperTranscriber()
    with patch.object(Path, "exists", return_value=True):
        transcript = transcriber.transcribe(Path("fake.wav"), language="es", meeting_id="m1")
    assert transcript.meeting_id == "m1"
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "Hello"
    assert "faster-whisper" in transcript.model_used


@patch("src.infrastructure.transcription.faster_whisper_transcriber.WhisperModel")
@patch("src.infrastructure.transcription.faster_whisper_transcriber.torch")
def test_faster_whisper_file_not_found(mock_torch, mock_wm):
    mock_torch.cuda.is_available.return_value = False
    transcriber = FasterWhisperTranscriber()
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(Path("missing.wav"))


def test_faster_whisper_vad_logic():
    silence = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)
    signal = np.ones(WHISPER_SAMPLE_RATE, dtype=np.float32) * 0.1
    audio = np.concatenate([silence, signal])
    voiced, ratio = FasterWhisperTranscriber.apply_vad(audio, energy_threshold=0.01)
    assert 0.4 < ratio < 0.7
    assert len(voiced) < len(audio)


def test_faster_whisper_stereo_to_mono():
    stereo = np.array([[0.5, -0.5], [0.5, -0.5]], dtype=np.float32)
    with patch("src.infrastructure.transcription.faster_whisper_transcriber.wavfile.read",
               return_value=(WHISPER_SAMPLE_RATE, stereo)):
        mono, _ = FasterWhisperTranscriber.load_wav_as_float32(Path("stereo.wav"))
    assert mono.ndim == 1
    assert np.allclose(mono, 0.0)


def test_faster_whisper_resample():
    audio = np.ones(44100, dtype=np.float32)
    resampled = FasterWhisperTranscriber.resample_to_16k(audio, src_rate=44100)
    assert len(resampled) == WHISPER_SAMPLE_RATE
