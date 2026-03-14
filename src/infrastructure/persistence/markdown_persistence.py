from pathlib import Path

from src.domain.entities.meeting import Meeting
from src.domain.entities.transcript import Transcript
from src.domain.entities.summary import Summary
from src.domain.ports.persistence_port import PersistencePort


class MarkdownPersistence(PersistencePort):
    """
    Implementation of PersistencePort to save meeting minutes in Markdown format.
    """

    def __init__(self, output_dir: str = "minutas"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, meeting: Meeting, transcript: Transcript, summary: Summary) -> Path:
        """
        Saves the meeting minutes and returns the path to the saved file.
        """
        # Format: minuta_reunion_YYYY-MM-DD_HH-mm.md
        # Windows safe format (avoid ':')
        timestamp_str = meeting.started_at.strftime("%Y-%m-%d_%H-%M")
        filename = f"minuta_reunion_{timestamp_str}.md"
        filepath = self.output_dir / filename

        duration_mins = meeting.duration_seconds / 60

        content = f"""# Minuta de Reunion

**ID:** {meeting.id}
**Fecha:** {meeting.started_at.strftime('%Y-%m-%d')}
**Hora de Inicio:** {meeting.started_at.strftime('%H:%M:%S')}
**Duracion:** {duration_mins:.2f} minutos
**Modelo de Transcripcion:** {transcript.model_used}

---

## Estructura Generada por IA (Gemini)

{summary.raw_response}

---

## Transcripcion Bruta

```text
{transcript.full_text}
```
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath
