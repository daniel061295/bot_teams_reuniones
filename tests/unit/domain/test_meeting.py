from datetime import datetime, timedelta
from pathlib import Path
from src.domain.entities.meeting import AudioConfig, Meeting

def test_meeting_duration_completed():
    """Test duration calculation for a completed meeting."""
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=45, seconds=30)
    
    meeting = Meeting(
        id="test-123",
        started_at=start_time,
        audio_config=config,
        ended_at=end_time,
        audio_file_path=Path("test.wav")
    )
    
    assert meeting.duration_seconds == 2730.0

def test_meeting_duration_in_progress():
    """Test duration calculation for a meeting that hasn't ended."""
    config = AudioConfig(sample_rate=44100, channels=2, dtype="int16")
    start_time = datetime.now()
    
    meeting = Meeting(
        id="test-123",
        started_at=start_time,
        audio_config=config
    )
    
    assert meeting.duration_seconds == 0.0
