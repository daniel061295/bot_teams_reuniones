import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime

from src.infrastructure.persistence.markdown_persistence import MarkdownPersistence
from src.domain.entities.meeting import AudioConfig, Meeting
from src.domain.entities.transcript import Transcript, TranscriptSegment
from src.domain.entities.summary import Summary

def test_markdown_persistence_save(tmp_path):
    # Use tmp_path to mock the output directory
    persistence = MarkdownPersistence(output_dir=str(tmp_path))
    
    # Create test data
    start_time = datetime(2026, 3, 10, 10, 0, 0)
    end_time = datetime(2026, 3, 10, 11, 0, 0) # 60 minutes
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    meeting = Meeting(id="m123", started_at=start_time, ended_at=end_time, audio_config=config)
    
    transcript = Transcript(
        meeting_id="m123",
        segments=[TranscriptSegment(0, 1, "Hello World")],
        language="es",
        model_used="whisper"
    )
    
    summary = Summary(
        meeting_id="m123",
        executive_summary="Exec",
        technical_decisions=["Dec 1"],
        blockers=["Block 1"],
        action_items=["Action 1"],
        raw_response="This is a summary response from the LLM.",
        generated_at=datetime.now()
    )
    
    # Save
    filepath = persistence.save(meeting, transcript, summary)
    
    # Verify file exists
    assert filepath.exists()
    assert filepath.name == "minuta_reunion_2026-03-10_10-00.md"
    
    # Read and verify content
    content = filepath.read_text(encoding="utf-8")
    assert "m123" in content
    assert "60.00 minutos" in content
    assert "whisper" in content
    assert "This is a summary response from the LLM." in content
    assert "Hello World" in content
