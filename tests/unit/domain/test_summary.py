from datetime import datetime
from src.domain.entities.summary import Summary

def test_summary_creation():
    """Test that summary dataclass is created correctly."""
    generated_at = datetime.now()
    summary = Summary(
        meeting_id="test-123",
        executive_summary="Short summary.",
        technical_decisions=["Decided to use Clean Architecture."],
        blockers=["Waiting for API keys."],
        action_items=["Alice to setup repo.", "Bob to write docs."],
        raw_response="...",
        generated_at=generated_at
    )
    
    assert summary.meeting_id == "test-123"
    assert len(summary.technical_decisions) == 1
    assert len(summary.action_items) == 2
    assert summary.generated_at == generated_at
