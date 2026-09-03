"""Domain action space for payment recovery operations."""

from enum import Enum, unique


@unique
class Action(str, Enum):
    """The bounded recovery action space defined by SPEC.md."""

    RETRY_NOW = "RETRY_NOW"
    RETRY_LATER = "RETRY_LATER"
    SEND_LINK = "SEND_LINK"
    NUDGE = "NUDGE"
    ESCALATE = "ESCALATE"
    STOP = "STOP"

    @property
    def is_retry(self) -> bool:
        """Returns True if the action is a gateway retry."""
        return self in (Action.RETRY_NOW, Action.RETRY_LATER)

    @property
    def is_terminal_stop(self) -> bool:
        """Returns True if the action permanently stops recovery."""
        return self == Action.STOP

    @property
    def is_escalation(self) -> bool:
        """Returns True if the action escalates to human operations."""
        return self == Action.ESCALATE

    def __repr__(self) -> str:
        return f"Action.{self.name}"
