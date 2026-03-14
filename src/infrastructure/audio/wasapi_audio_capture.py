import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyaudiowpatch as pyaudio
import scipy.io.wavfile as wavfile

from src.domain.entities.meeting import AudioConfig
from src.domain.ports.audio_capture_port import AudioCapturePort

# Internal sample rate used during streaming (chosen for WASAPI loopback compatibility)
_CAPTURE_RATE = 44100
_CAPTURE_CHANNELS = 2
_CHUNK_FRAMES = 1024
_FORMAT = pyaudio.paInt16


class WasapiAudioCapture(AudioCapturePort):
    """
    Implementation of AudioCapturePort using pyaudiowpatch for Windows.

    Captures two audio streams simultaneously and mixes them into a single WAV:

    * **WASAPI Loopback** - the default playback output device (everything you
      hear in your headphones: Teams participants, videos, etc.).
    * **Microphone** - your local input device, so your own voice is included.

    Both streams are resampled / converted to the same format before mixing.
    """

    def __init__(self) -> None:
        self._is_recording: bool = False
        self._loopback_frames: List[bytes] = []
        self._mic_frames: List[bytes] = []
        self._lock = threading.Lock()
        self._loopback_stream: Optional[Any] = None
        self._mic_stream: Optional[Any] = None
        self._pa: Optional[pyaudio.PyAudio] = None
        self._config: Optional[AudioConfig] = None

    # ------------------------------------------------------------------
    # AudioCapturePort interface
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """Whether a recording session is currently active."""
        with self._lock:
            return self._is_recording

    def list_devices(self) -> List[Dict[str, Any]]:
        """
        Returns a list of audio devices available via WASAPI.

        The default playback device (used for WASAPI loopback) is annotated
        with ``<-- [DEFAULT OUTPUT - WASAPI Loopback]``.
        """
        pa = pyaudio.PyAudio()
        devices: List[Dict[str, Any]] = []
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_output_idx: int = wasapi_info["defaultOutputDevice"]

            for i in range(pa.get_device_count()):
                dev = dict(pa.get_device_info_by_index(i))
                if i == default_output_idx:
                    dev["name"] = f"{dev['name']}  <-- [DEFAULT OUTPUT - WASAPI Loopback]"
                    dev["is_default_output"] = True
                else:
                    dev["is_default_output"] = False
                # Normalise keys to match the sounddevice convention used by main.py
                dev["max_input_channels"] = dev.get("maxInputChannels", 0)
                dev["max_output_channels"] = dev.get("maxOutputChannels", 0)
                devices.append(dev)
        finally:
            pa.terminate()
        return devices

    def start_recording(self, config: AudioConfig) -> None:
        """
        Starts simultaneous WASAPI loopback and microphone recording.

        :param config: Audio capture configuration.
        :raises RuntimeError: If a recording is already in progress.
        :raises RuntimeError: If no WASAPI loopback device can be found.
        """
        with self._lock:
            if self._is_recording:
                raise RuntimeError("Recording already in progress.")

            self._config = config
            self._loopback_frames = []
            self._mic_frames = []
            self._is_recording = True

        self._pa = pyaudio.PyAudio()

        # -- Loopback device -----------------------------------------------
        try:
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as exc:
            raise RuntimeError("WASAPI host API not available on this system.") from exc

        default_output_idx: int = wasapi_info["defaultOutputDevice"]
        loopback_device = self._pa.get_device_info_by_index(default_output_idx)

        # pyaudiowpatch requires the loopback variant of the output device
        loopback_device = self._pa.get_wasapi_loopback_analogue_by_dict(loopback_device)
        if loopback_device is None:
            raise RuntimeError(
                "Could not obtain WASAPI loopback device for the default output. "
                "Ensure WASAPI is enabled in Windows Sound settings."
            )

        loopback_rate = int(loopback_device["defaultSampleRate"])
        loopback_channels: int = loopback_device["maxInputChannels"]

        print(f"Loopback device: {loopback_device['name']} @ {loopback_rate} Hz")

        self._loopback_stream = self._pa.open(
            format=_FORMAT,
            channels=loopback_channels,
            rate=loopback_rate,
            input=True,
            input_device_index=int(loopback_device["index"]),
            frames_per_buffer=_CHUNK_FRAMES,
            stream_callback=self._loopback_callback,
        )

        # -- Microphone device -----------------------------------------------
        mic_device_index: Optional[int] = None
        if config.microphone_device:
            for i in range(self._pa.get_device_count()):
                dev = self._pa.get_device_info_by_index(i)
                if config.microphone_device.lower() in dev["name"].lower():
                    mic_device_index = i
                    break

        try:
            mic_device = self._pa.get_device_info_by_index(
                mic_device_index if mic_device_index is not None
                else self._pa.get_default_input_device_info()["index"]
            )
            mic_rate = int(mic_device["defaultSampleRate"])
            print(f"Microphone  : {mic_device['name']} @ {mic_rate} Hz")

            self._mic_stream = self._pa.open(
                format=_FORMAT,
                channels=1,
                rate=mic_rate,
                input=True,
                input_device_index=int(mic_device["index"]),
                frames_per_buffer=_CHUNK_FRAMES,
                stream_callback=self._mic_callback,
            )
            self._mic_stream.start_stream()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not open microphone - {exc}. Only loopback will be recorded.")
            self._mic_stream = None

        self._loopback_stream.start_stream()

    def stop_recording(self) -> Path:
        """
        Stops all streams, mixes loopback + microphone, and saves a WAV file.

        :return: Path to the saved WAV file.
        :raises RuntimeError: If no recording is in progress.
        """
        with self._lock:
            if not self._is_recording:
                raise RuntimeError("No recording to stop.")
            self._is_recording = False

        for stream in (self._loopback_stream, self._mic_stream):
            if stream is not None:
                stream.stop_stream()
                stream.close()
        self._loopback_stream = None
        self._mic_stream = None

        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

        # -- Mix streams ---------------------------------------------------
        mixed = self._mix_streams(self._loopback_frames, self._mic_frames)

        self._loopback_frames = []
        self._mic_frames = []

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = output_dir / f"audio_capture_{timestamp}.wav"

        sample_rate = self._config.sample_rate if self._config else _CAPTURE_RATE
        wavfile.write(str(file_path), sample_rate, mixed)
        return file_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _loopback_callback(self, in_data: bytes, frame_count: int, time_info: Any, status: int) -> tuple:
        """PyAudio stream callback for the loopback channel."""
        if self._is_recording:
            self._loopback_frames.append(in_data)
        return (None, pyaudio.paContinue)

    def _mic_callback(self, in_data: bytes, frame_count: int, time_info: Any, status: int) -> tuple:
        """PyAudio stream callback for the microphone channel."""
        if self._is_recording:
            self._mic_frames.append(in_data)
        return (None, pyaudio.paContinue)

    @staticmethod
    def _mix_streams(
        loopback_frames: List[bytes],
        mic_frames: List[bytes],
    ) -> np.ndarray:
        """
        Converts raw byte buffers to int16 arrays and averages them.

        The shorter stream is zero-padded to match the longer one before mixing.

        :param loopback_frames: Raw PCM bytes from the loopback stream.
        :param mic_frames: Raw PCM bytes from the microphone stream.
        :return: Mixed mono int16 numpy array.
        """
        def to_mono_int16(frames: List[bytes]) -> np.ndarray:
            if not frames:
                return np.array([], dtype=np.int16)
            raw = np.frombuffer(b"".join(frames), dtype=np.int16)
            # If stereo, average channels
            if raw.size % 2 == 0:
                raw = raw.reshape(-1, 2).mean(axis=1).astype(np.int16)
            return raw

        lb = to_mono_int16(loopback_frames)
        mic = to_mono_int16(mic_frames)

        if lb.size == 0:
            return mic if mic.size > 0 else np.zeros(1, dtype=np.int16)
        if mic.size == 0:
            return lb

        # Pad shorter stream to same length
        max_len = max(lb.size, mic.size)
        lb = np.pad(lb, (0, max_len - lb.size))
        mic = np.pad(mic, (0, max_len - mic.size))

        # Mix: average both channels scaled to avoid clipping
        mixed = ((lb.astype(np.int32) + mic.astype(np.int32)) // 2).astype(np.int16)
        return mixed
