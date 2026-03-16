import io
import os
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from groq import Groq, APIConnectionError, APIStatusError, RateLimitError
from scipy.io import wavfile

from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.ports.transcription_port import TranscriptionPort
from src.infrastructure.transcription.faster_whisper_transcriber import FasterWhisperTranscriber

# Groq enforces a 25 MB per-request limit for audio files.
# We convert to MP3 at 128 kbps mono to stay well below the limit and maintain quality.
_GROQ_MODEL = "whisper-large-v3-turbo"  # Fastest + highest quality, free tier
_MAX_BYTES = 24 * 1024 * 1024           # 24 MB safety margin


class GroqTranscriber(TranscriptionPort):
    """
    Cloud transcription adapter using the Groq Whisper API.

    Uses ``whisper-large-v3-turbo`` (the best Whisper model on Groq's free tier).
    Audio is pre-processed (VAD + 16kHz resample) and converted to MP3 via ffmpeg 
    so it stays under Groq's 25 MB limit and provides excellent transcription quality.

    Raises the original Groq exceptions on failure so the HybridTranscriber
    can reliably detect quota/connection issues and fall back locally.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self._client = Groq(api_key=api_key)

    # ------------------------------------------------------------------
    # TranscriptionPort interface
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: Path, language: str = "es", meeting_id: str = "") -> Transcript:
        """
        Transcribes a WAV file via the Groq Whisper API after optimizing it.

        :raises FileNotFoundError: If the audio file does not exist.
        :raises groq.RateLimitError: When the daily free quota is exhausted.
        :raises groq.APIConnectionError: When there is no internet connection.
        :raises groq.APIStatusError: For other API-level errors.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print("[Groq] Pre-processing audio (VAD & Resampling)...")
        audio, src_rate = FasterWhisperTranscriber.load_wav_as_float32(audio_path)
        audio = FasterWhisperTranscriber.resample_to_16k(audio, src_rate)
        audio, voiced_ratio = FasterWhisperTranscriber.apply_vad(audio)
        
        if voiced_ratio < 0.05:
            print("WARNING: Less than 5% of the audio contains speech.")
            
        print(f"[Groq] Transcribing via Groq API (model: {_GROQ_MODEL})...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = Path(tmp_wav.name)
            wavfile.write(str(tmp_wav_path), 16000, audio)
            
        try:
            mp3_bytes = self._to_mp3(tmp_wav_path)
            print(f"[Groq] Audio size after VAD+MP3 conversion: {len(mp3_bytes) / 1024 / 1024:.2f} MB")
        finally:
            if tmp_wav_path.exists():
                try:
                    tmp_wav_path.unlink()
                except Exception:
                    pass

        audio_file = ("audio.mp3", io.BytesIO(mp3_bytes), "audio/mpeg")

        response = self._client.audio.transcriptions.create(
            file=audio_file,
            model=_GROQ_MODEL,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            temperature=0, # Use temperature 0 for more deterministic and accurate transcripts
        )

        segments = []
        raw_segments = getattr(response, "segments", None) or []
        for seg in raw_segments:
            # Each segment is a dict-like object with start, end, text
            start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
            end = seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0)
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            segments.append(TranscriptSegment(start=float(start), end=float(end), text=text.strip()))

        # If verbose_json returned no segments, fall back to the flat text
        if not segments and hasattr(response, "text") and response.text:
            segments = [TranscriptSegment(start=0.0, end=0.0, text=response.text.strip())]

        print(f"[Groq] Done — {len(segments)} segments received.")
        return Transcript(
            meeting_id=meeting_id,
            segments=segments,
            language=language,
            model_used=f"groq-{_GROQ_MODEL}",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mp3(wav_path: Path) -> bytes:
        """
        Converts the WAV file to MP3 (128 kbps, mono, 16 kHz) using ffmpeg.
        Returns the raw MP3 bytes ready to be sent to the Groq API.

        :raises RuntimeError: If ffmpeg conversion fails.
        """
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe,
            "-y",                   # overwrite without asking
            "-i", str(wav_path),    # input file
            "-f", "mp3",            # output format
            "-ab", "128k",          # 128 kbps bitrate for better quality
            "-ac", "1",             # mono
            "-ar", "16000",         # 16 kHz (Whisper native)
            "-",                    # write to stdout
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg conversion failed (exit {result.returncode}): "
                f"{result.stderr.decode(errors='replace')[:500]}"
            )

        if len(result.stdout) > _MAX_BYTES:
            raise RuntimeError(
                f"Converted MP3 is {len(result.stdout) / 1024 / 1024:.1f} MB, "
                f"which exceeds Groq's 25 MB limit.  "
                "Consider splitting long recordings."
            )

        return result.stdout