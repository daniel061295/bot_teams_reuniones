from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class Summary:
    """
    Represents the structured summary of a meeting.
    """
    meeting_id: str
    executive_summary: str
    technical_decisions: List[str]
    blockers: List[str]
    action_items: List[str]
    raw_response: str
    generated_at: datetime
