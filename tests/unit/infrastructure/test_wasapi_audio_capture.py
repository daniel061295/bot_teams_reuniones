import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

from src.infrastructure.audio.wasapi_audio_capture import WasapiAudioCapture
from src.domain.entities.meeting import AudioConfig

@patch("src.infrastructure.audio.wasapi_audio_capture.pyaudio")
@patch("src.infrastructure.audio.wasapi_audio_capture.wavfile")
def test_wasapi_audio_capture_success(mock_wavfile, mock_pyaudio):
    # Setup PyAudio mock structure
    mock_pa_instance = MagicMock()
    mock_pyaudio.PyAudio.return_value = mock_pa_instance
    
    # Mock loopback search
    mock_pa_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 1}
    mock_pa_instance.get_device_info_by_index.return_value = {
        "index": 1, "name": "Device 1", "defaultSampleRate": 44100, "maxInputChannels": 2
    }
    mock_pa_instance.get_wasapi_loopback_analogue_by_dict.return_value = {
        "index": 2, "name": "Device 1 (Loopback)", "defaultSampleRate": 44100, "maxInputChannels": 2
    }
    mock_pa_instance.get_device_count.return_value = 2
    
    capture = WasapiAudioCapture()
    
    # 1. Test device listing
    devices = capture.list_devices()
    assert len(devices) == 2
    assert "Loopback" in [d["name"] for d in devices if d.get("is_default_output")][0]
    
    # 2. Test start recording
    mock_stream = MagicMock()
    mock_pa_instance.open.return_value = mock_stream
    
    config = AudioConfig(sample_rate=16000, channels=1, dtype="int16", microphone_device="Device 1")
    capture.start_recording(config)
    
    assert capture.is_recording
    assert mock_pa_instance.open.call_count == 2 # 1 for loopback, 1 for mic
    
    # Simulate internal callback data
    capture._loopback_frames = [b"\x00\x00"] * 10
    capture._mic_frames = [b"\x00\x00"] * 10
    
    # 3. Test stop recording
    path = capture.stop_recording()
    
    assert not capture.is_recording
    assert mock_pa_instance.terminate.called
    mock_wavfile.write.assert_called_once()
    assert path.suffix == ".wav"

@patch("src.infrastructure.audio.wasapi_audio_capture.pyaudio")
def test_wasapi_audio_capture_start_twice(mock_pyaudio):
    mock_pa_instance = MagicMock()
    mock_pyaudio.PyAudio.return_value = mock_pa_instance
    mock_pa_instance.get_host_api_info_by_type.return_value = {"defaultOutputDevice": 1}
    mock_pa_instance.get_wasapi_loopback_analogue_by_dict.return_value = {"index": 2, "defaultSampleRate": 44100, "maxInputChannels": 2, "name": "L"}

    capture = WasapiAudioCapture()
    config = AudioConfig(sample_rate=16000, channels=1, dtype="int16")
    
    capture.start_recording(config)
    with pytest.raises(RuntimeError, match="Recording already in progress"):
        capture.start_recording(config)

def test_wasapi_audio_capture_stop_not_recording():
    capture = WasapiAudioCapture()
    with pytest.raises(RuntimeError, match="No recording to stop"):
        capture.stop_recording()

def test_wasapi_audio_capture_mixing_logic():
    """Test the internal _mix_streams logic."""
    # 2 frames of value 1000 and -1000
    lb_frames = [np.array([1000, 1000], dtype=np.int16).tobytes()]
    mic_frames = [np.array([2000, 2000], dtype=np.int16).tobytes()]
    
    mixed = WasapiAudioCapture._mix_streams(lb_frames, mic_frames)
    
    # (1000 + 2000) // 2 = 1500
    assert mixed[0] == 1500
    assert mixed.dtype == np.int16
