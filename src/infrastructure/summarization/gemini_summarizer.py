import os
from datetime import datetime
import google.generativeai as genai

from src.domain.entities.transcript import Transcript
from src.domain.entities.summary import Summary
from src.domain.ports.summarization_port import SummarizationPort


class GeminiSummarizer(SummarizationPort):
    """
    Implementation of SummarizationPort using Google's Gemini Flash.
    """

    # Models to try in priority order (first successful one is used)
    _MODEL_CANDIDATES = [
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        genai.configure(api_key=api_key)
        self._model_name: str | None = None
        self.model: genai.GenerativeModel | None = None

    def _get_model(self) -> genai.GenerativeModel:
        """
        Lazily initialises the Gemini model the first time it is needed.
        Tries each candidate in order and returns the first one that responds.

        :return: A ready :class:`genai.GenerativeModel` instance.
        :raises RuntimeError: If none of the candidate models are reachable.
        """
        if self.model is not None:
            return self.model

        last_error: Exception | None = None
        for model_name in self._MODEL_CANDIDATES:
            try:
                candidate = genai.GenerativeModel(model_name)
                # Lightweight probe - 1 token just to validate the model endpoint
                candidate.generate_content("ping", generation_config={"max_output_tokens": 1})
                print(f"Using Gemini model: {model_name}")
                self.model = candidate
                self._model_name = model_name
                return self.model
            except Exception as exc:  # noqa: BLE001
                print(f"Model '{model_name}' unavailable ({exc}). Trying next fallback...")
                last_error = exc

        raise RuntimeError(
            f"No Gemini model could be loaded. Last error: {last_error}"
        )

    def summarize(self, transcript: Transcript) -> Summary:
        """
        Summarizes the given transcript using the best available Gemini model.

        :param transcript: The :class:`Transcript` to summarize.
        :return: A :class:`Summary` entity with the raw Gemini response.
        """
        prompt = (
            "Actua como Solution Architect. Analiza esta transcripcion de una reunion de desarrollo en Windows. "
            "Extrae de forma estructurada en Markdown exactamente con estas secciones (y nada mas):\n\n"
            "Resumen Ejecutivo: (Maximo 3 lineas).\n"
            "Decisiones Tecnicas: (Foco en arquitectura, cambios de base de datos o migraciones).\n"
            "Bloqueantes: (Cualquier impedimento mencionado).\n"
            "Action Items: (Tareas con responsables y fechas si se mencionan).\n\n"
            f"Transcripcion:\n{transcript.full_text}"
        )

        response = self._get_model().generate_content(prompt)
        response_text = response.text

        return Summary(
            meeting_id=transcript.meeting_id,
            executive_summary="Ver raw_response para detalles.",
            technical_decisions=["Ver raw_response para detalles."],
            blockers=["Ver raw_response para detalles."],
            action_items=["Ver raw_response para detalles."],
            raw_response=response_text,
            generated_at=datetime.now()
        )
