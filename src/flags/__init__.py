"""Package exports for `src.flags` (Section 12.2)."""

from src.flags.conflict_detection import detect_conflicts
from src.flags.uncertainty import uncertainty_flags

__all__ = ["detect_conflicts", "uncertainty_flags"]
