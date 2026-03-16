import io
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
from groq import Groq, APIConnectionError, APIStatusError, RateLimitError

from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.ports.transcription_port import TranscriptionPort

_GROQ_MODEL = "whisper-large-v3-turbo"
_MAX_BYTES = 24 * 1024 * 1024  # 24 MB safety margin


class GroqTranscriber(TranscriptionPort):
    """
    Cloud transcription adapter using the Groq Whisper API.
    Converts WAV -> MP3 via ffmpeg before sending to stay under Groq's 25 MB limit.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self._client = Groq(api_key=api_key)

    def transcribe(self, audio_path: Path, language: str = "es", meeting_id: str = "") -> Transcript:
        """
        Transcribes a WAV file via the Groq Whisper API.
        Raises groq exceptions on failure so HybridTranscriber can detect them.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"[Groq] Transcribing {audio_path} via Groq API (model: {_GROQ_MODEL})...")

        mp3_bytes = self._to_mp3(audio_path)
        print(f"[Groq] Audio size after MP3 conversion: {len(mp3_bytes) / 1024 / 1024:.2f} MB")

        audio_file = ("audio.mp3", io.BytesIO(mp3_bytes), "audio/mpeg")
        response = self._client.audio.transcriptions.create(
            file=audio_file,
            model=_GROQ_MODEL,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        segments = []
        raw_segments = getattr(response, "segments", None) or []
        for seg in raw_segments:
            start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
            end = seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0)
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            segments.append(TranscriptSegment(start=float(start), end=float(end), text=text.strip()))

        if not segments and hasattr(response, "text") and response.text:
            segments = [TranscriptSegment(start=0.0, end=0.0, text=response.text.strip())]

        print(f"[Groq] Done — {len(segments)} segments received.")
        return Transcript(
            meeting_id=meeting_id,
            segments=segments,
            language=language,
            model_used=f"groq-{_GROQ_MODEL}",
        )

    @staticmethod
    def _to_mp3(wav_path: Path) -> bytes:
        """Converts WAV to MP3 at 64 kbps mono 16 kHz via ffmpeg."""
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(wav_path),
            "-f", "mp3", "-ab", "64k", "-ac", "1", "-ar", "16000",
            "-",
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
                "exceeds Groq's 25 MB limit. Consider splitting long recordings."
            )
        return result.stdout
