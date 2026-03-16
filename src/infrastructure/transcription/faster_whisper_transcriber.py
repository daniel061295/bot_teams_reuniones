import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from faster_whisper import WhisperModel

from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.ports.transcription_port import TranscriptionPort

# Whisper expects audio at 16000 Hz as a float32 numpy array
WHISPER_SAMPLE_RATE = 16000


class FasterWhisperTranscriber(TranscriptionPort):
    """
    Local transcription adapter using faster-whisper (CTranslate2 backend).

    Drop-in replacement for the original WhisperTranscriber: same models,
    same accuracy, but 4x faster on CPU and up to 8x faster on GPU thanks
    to the optimised CTranslate2 runtime.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # int8 on CPU gives the best speed/accuracy trade-off
        compute_type = "float16" if self.device == "cuda" else "int8"

        model_name = os.getenv("WHISPER_MODEL", "medium")
        self.model_name = model_name

        print(f"[FasterWhisper] Loading model '{model_name}' on {self.device} ({compute_type})...")
        self.model = WhisperModel(model_name, device=self.device, compute_type=compute_type)
        print("[FasterWhisper] Model loaded.")

    @staticmethod
    def load_wav_as_float32(audio_path: Path) -> tuple[np.ndarray, int]:
        """Reads a WAV file and returns (mono float32 array, original_sample_rate)."""
        sample_rate, data = wavfile.read(str(audio_path))

        if data.dtype == np.int16:
            audio: np.ndarray = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        else:
            audio = data.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio, sample_rate

    @staticmethod
    def resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
        """Linear resampling from src_rate to 16 kHz."""
        if src_rate == WHISPER_SAMPLE_RATE:
            return audio
        import math
        ratio = WHISPER_SAMPLE_RATE / src_rate
        new_length = math.ceil(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    @staticmethod
    def apply_vad(
        audio: np.ndarray,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        frame_ms: int = 30,
        energy_threshold: float = 0.003,
        padding_frames: int = 8,
    ) -> tuple[np.ndarray, float]:
        """
        Voice Activity Detection based on RMS energy.
        :return: (voiced_audio array, voiced_ratio 0-1)
        """
        if len(audio) == 0:
            raise ValueError("Audio array is empty; nothing to filter.")

        frame_size = int(sample_rate * frame_ms / 1000)
        n_frames = len(audio) // frame_size
        frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)

        rms: np.ndarray = np.sqrt(np.mean(frames ** 2, axis=1))
        is_voice = rms >= energy_threshold

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

        return frames[padded].flatten(), voiced_ratio

    def transcribe(self, audio_path: Path, language: str = "es", meeting_id: str = "") -> Transcript:
        """Transcribes a WAV file locally using faster-whisper."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"[FasterWhisper] Transcribing {audio_path}...")

        audio, src_rate = self.load_wav_as_float32(audio_path)
        audio = self.resample_to_16k(audio, src_rate)

        audio, voiced_ratio = self.apply_vad(audio)
        if voiced_ratio < 0.05:
            print("WARNING: Less than 5% of the audio contains speech.")

        raw_segments, _info = self.model.transcribe(
            audio,
            language=language,
            temperature=0,
            vad_filter=False,
            condition_on_previous_text=False,
        )

        segments = [
            TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
            for seg in raw_segments
        ]

        print(f"[FasterWhisper] Done — {len(segments)} segments.")
        return Transcript(
            meeting_id=meeting_id,
            segments=segments,
            language=language,
            model_used=f"faster-whisper-{self.model_name}-{self.device}",
        )
