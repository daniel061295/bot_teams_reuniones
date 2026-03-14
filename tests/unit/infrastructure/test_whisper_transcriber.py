import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

from src.infrastructure.transcription.whisper_transcriber import WhisperTranscriber, WHISPER_SAMPLE_RATE

@patch("src.infrastructure.transcription.whisper_transcriber.whisper")
@patch("src.infrastructure.transcription.whisper_transcriber.torch")
@patch("src.infrastructure.transcription.whisper_transcriber.wavfile")
def test_whisper_transcriber_success(mock_wavfile, mock_torch, mock_whisper):
    # Configure mocks
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    
    # Mock result format of whisper transcribe
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello"}
        ]
    }
    mock_whisper.load_model.return_value = mock_model
    
    # Mock wavfile readout
    # 1 second of "loud" noise to pass VAD
    mock_wavfile.read.return_value = (WHISPER_SAMPLE_RATE, np.ones((WHISPER_SAMPLE_RATE,), dtype=np.int16) * 1000)
    
    transcriber = WhisperTranscriber()
    
    # Execute
    audio_path = Path("fake.wav")
    with patch.object(Path, 'exists', return_value=True):
        transcript = transcriber.transcribe(audio_path, language="es", meeting_id="m123")
    
    # Verifications
    mock_model.transcribe.assert_called_once()
    actual_args, actual_kwargs = mock_model.transcribe.call_args
    assert isinstance(actual_args[0], np.ndarray)  # Should pass the array, not path
    assert actual_kwargs["language"] == "es"
    assert actual_kwargs["temperature"] == 0
    
    assert transcript.meeting_id == "m123"
    assert transcript.language == "es"
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "Hello"

@patch("src.infrastructure.transcription.whisper_transcriber.whisper")
@patch("src.infrastructure.transcription.whisper_transcriber.torch")
def test_whisper_transcriber_file_not_found(mock_torch, mock_whisper):
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    
    transcriber = WhisperTranscriber()
    audio_path = Path("missing.wav")
    
    with patch.object(Path, 'exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(audio_path, language="en", meeting_id="m123")

def test_whisper_transcriber_vad_logic():
    """Test the internal _apply_vad method directly."""
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    
    # 1 second of silence + 1 second of signal
    silence = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)
    signal = np.ones(WHISPER_SAMPLE_RATE, dtype=np.float32) * 0.1
    audio = np.concatenate([silence, signal])
    
    voiced_audio, ratio = transcriber._apply_vad(audio, energy_threshold=0.01)
    
    # Should retain roughly half (the signal part plus some padding)
    assert 0.4 < ratio < 0.7
    assert len(voiced_audio) < len(audio)
    assert len(voiced_audio) > 0

def test_whisper_transcriber_stereo_to_mono():
    """Test that stereo audio is correctly converted to mono."""
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    
    # Stereo array: Left is 0.5, Right is -0.5 -> Mean is 0
    stereo = np.array([[0.5, -0.5], [0.5, -0.5]], dtype=np.float32)
    
    # Mocking path exists since it's used in load_wav but it calls read
    with patch("src.infrastructure.transcription.whisper_transcriber.wavfile.read") as mock_read:
        mock_read.return_value = (WHISPER_SAMPLE_RATE, stereo)
        mono = transcriber._load_wav_as_float32(Path("stereo.wav"))
        
        assert mono.ndim == 1
        assert np.allclose(mono, 0.0)
