"""Temporal stabilization and cooldown for candidate-panel swipe actions."""


class CandidateSwipeGate:
    """Accept one stable horizontal swipe at a time.

    GestureMotion already applies a distance threshold and dominant-axis
    check. This gate briefly settles that event to reject immediate opposing
    jitter, then applies a cooldown before another candidate can be selected.
    """

    DIRECTIONS = {"SWIPE_LEFT", "SWIPE_RIGHT"}

    def __init__(self, settle_seconds=0.12, cooldown_seconds=0.6):
        self.settle_seconds = settle_seconds
        self.cooldown_seconds = cooldown_seconds
        self.pending_direction = None
        self.pending_since = None
        self.last_action = None

    def reset(self):
        self.pending_direction = None
        self.pending_since = None

    def update(self, swipe, active=True, now=None):
        """Return a committed direction or None; ``now`` enables deterministic tests."""
        if now is None:
            import time
            now = time.monotonic()

        if not active:
            self.reset()
            return None

        if swipe in self.DIRECTIONS:
            if (self.last_action is not None
                    and now - self.last_action < self.cooldown_seconds):
                return None
            if self.pending_direction is None:
                self.pending_direction = swipe
                self.pending_since = now
            elif swipe != self.pending_direction:
                # Opposing motion inside the settle interval is ambiguous.
                self.reset()
                return None

        if (self.pending_direction is not None
                and now - self.pending_since >= self.settle_seconds):
            direction = self.pending_direction
            self.last_action = now
            self.reset()
            return direction
        return None
