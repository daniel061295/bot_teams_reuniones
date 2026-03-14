from src.domain.entities.transcript import Transcript, TranscriptSegment

def test_transcript_full_text():
    """Test that full_text joins all segments correctly."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="Hello team."),
        TranscriptSegment(start=2.5, end=5.0, text="Let's start the meeting."),
        TranscriptSegment(start=5.5, end=7.0, text="Great."),
    ]
    
    transcript = Transcript(
        meeting_id="test-123",
        segments=segments,
        language="en",
        model_used="whisper-small"
    )
    
    expected_text = "Hello team. Let's start the meeting. Great."
    assert transcript.full_text == expected_text

def test_transcript_empty_segments():
    """Test full_text with empty segments."""
    transcript = Transcript(
        meeting_id="test-123",
        segments=[],
        language="en",
        model_used="whisper-small"
    )
    
    assert transcript.full_text == ""
