import inject

from src.domain.ports.audio_capture_port import AudioCapturePort
from src.domain.ports.transcription_port import TranscriptionPort
from src.domain.ports.summarization_port import SummarizationPort
from src.domain.ports.persistence_port import PersistencePort

from src.infrastructure.audio.wasapi_audio_capture import WasapiAudioCapture
from src.infrastructure.transcription.hybrid_transcriber import HybridTranscriber
from src.infrastructure.summarization.gemini_summarizer import GeminiSummarizer
from src.infrastructure.persistence.markdown_persistence import MarkdownPersistence


def configure_dependency_injection():
    """
    Configures the dependency injection container.
    Maps ports (interfaces) to their concrete infrastructure adapters.
    """
    def configure(binder):
        binder.bind_to_provider(AudioCapturePort, WasapiAudioCapture)
        # HybridTranscriber tries Groq first, falls back to faster-whisper locally
        binder.bind_to_constructor(TranscriptionPort, HybridTranscriber)
        binder.bind_to_constructor(SummarizationPort, GeminiSummarizer)
        binder.bind_to_provider(PersistencePort, MarkdownPersistence)

    inject.configure(configure)
