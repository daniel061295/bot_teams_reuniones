import os
from pathlib import Path
import numpy as np
import whisper
import torch
from scipy.io import wavfile

from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.ports.transcription_port import TranscriptionPort

# Whisper expects audio at 16000 Hz as a float32 numpy array
WHISPER_SAMPLE_RATE = 16000


class WhisperTranscriber(TranscriptionPort):
    """
    Implementation of TranscriptionPort using OpenAI's Whisper model locally.
    Loads WAV files directly via scipy to avoid requiring FFmpeg on the host system.
    """

    def __init__(self):
        # Determine if CUDA (GPU) is available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # We read the preferred model from the environment variable, default to 'small'
        model_name = os.getenv("WHISPER_MODEL", "small")
        self.model_name = model_name

        print(f"Loading Whisper model '{model_name}' on {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
        print("Model loaded.")

    def _load_wav_as_float32(self, audio_path: Path) -> np.ndarray:
        """
        Loads a WAV file using scipy and converts it to a mono float32 array at
        WHISPER_SAMPLE_RATE (16000 Hz), bypassing the need for FFmpeg.

        :param audio_path: Path to the WAV file.
        :return: Mono float32 numpy array normalised to [-1.0, 1.0].
        """
        sample_rate, data = wavfile.read(str(audio_path))

        # Convert to float32 in [-1, 1]
        if data.dtype == np.int16:
            audio: np.ndarray = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        else:
            audio = data.astype(np.float32)

        # Stereo -> mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16 kHz if needed
        if sample_rate != WHISPER_SAMPLE_RATE:
            import math
            ratio = WHISPER_SAMPLE_RATE / sample_rate
            new_length = math.ceil(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        return audio

    def _apply_vad(
        self,
        audio: np.ndarray,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        frame_ms: int = 30,
        energy_threshold: float = 0.003,
        padding_frames: int = 8,
    ) -> tuple[np.ndarray, float]:
        """
        Voice Activity Detection (VAD) based on RMS energy.

        Splits the audio into fixed-length frames and removes frames whose
        Root Mean Square energy is below ``energy_threshold``. Neighbouring
        voiced frames are merged and padded so that sentence boundaries are
        not clipped abruptly.

        :param audio: Mono float32 array at ``sample_rate``.
        :param sample_rate: Sample rate of the audio (default 16000 Hz).
        :param frame_ms: Frame length in milliseconds (default 30 ms).
        :param energy_threshold: Minimum RMS energy to consider a frame as voiced.
        :param padding_frames: Extra frames kept before and after each voiced region.
        :return: Tuple of (filtered audio array, ratio of voiced to total audio).
        :raises ValueError: If the audio array is empty.
        """
        if len(audio) == 0:
            raise ValueError("Audio array is empty; nothing to filter.")

        frame_size = int(sample_rate * frame_ms / 1000)
        # Split into fixed-size frames (drop the last incomplete one)
        n_frames = len(audio) // frame_size
        frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)

        # Compute RMS energy per frame
        rms: np.ndarray = np.sqrt(np.mean(frames ** 2, axis=1))
        is_voice = rms >= energy_threshold

        # Pad voiced regions so we don't clip word boundaries
        padded = is_voice.copy()
        for i in range(n_frames):
            if is_voice[i]:
                start = max(0, i - padding_frames)
                end = min(n_frames, i + padding_frames + 1)
                padded[start:end] = True

        voiced_ratio = float(padded.sum()) / n_frames
        print(
            f"VAD: {padded.sum()}/{n_frames} frames voiced "
            f"({voiced_ratio * 100:.1f}% of audio retained)"
        )

        # Reconstruct audio keeping only voiced frames
        voiced_audio: np.ndarray = frames[padded].flatten()
        return voiced_audio, voiced_ratio

    def transcribe(self, audio_path: Path, language: str = "es", meeting_id: str = "") -> Transcript:
        """
        Transcribes the given audio file using Whisper.

        :param audio_path: Path to a WAV audio file.
        :param language: BCP-47 language code (e.g. "es", "en").
        :param meeting_id: Identifier of the parent meeting.
        :return: Populated :class:`Transcript` entity.
        :raises FileNotFoundError: If the audio file does not exist.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Starting transcription of {audio_path}...")

        # Load audio without FFmpeg
        audio_array = self._load_wav_as_float32(audio_path)

        # Voice Activity Detection: filter out silence before sending to Whisper.
        # This greatly reduces hallucinations on quiet/silent segments.
        audio_array, voiced_ratio = self._apply_vad(audio_array)
        if voiced_ratio < 0.05:
            print("WARNING: Less than 5% of the audio contains speech. Transcript may be empty or inaccurate.")

        # Whisper transcription - accepts numpy float32 arrays directly
        # Anti-hallucination parameters:
        # - temperature=0: deterministic decoding (no random sampling)
        # - no_speech_threshold=0.6: skip segments likely to be silence/noise
        # - condition_on_previous_text=False: avoid confabulation based on prior text
        result = self.model.transcribe(
            audio_array,
            language=language,
            fp16=torch.cuda.is_available(),
            temperature=0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )

        segments = []
        for seg in result.get("segments", []):
            segment = TranscriptSegment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
            )
            segments.append(segment)

        return Transcript(
            meeting_id=meeting_id,
            segments=segments,
            language=language,
            model_used=f"whisper-{self.model_name}-{self.device}",
        )
