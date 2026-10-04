"""Exclusive input ownership while a recommendation carousel is visible."""

from gestures.candidate_swipe_gate import CandidateSwipeGate


MODE_NORMAL = "NORMAL"
MODE_RECOMMENDATION = "RECOMMENDATION"


class RecommendationInteractionMode:
    """Consumes hand input for the recommendation panel while it is active."""

    def __init__(self, renderer, swipe_gate=None):
        self.renderer = renderer
        self.swipe_gate = swipe_gate or CandidateSwipeGate()
        self.mode = MODE_NORMAL

    @property
    def active(self):
        return self.mode == MODE_RECOMMENDATION

    def update(self, swipe=None, now=None):
        """Sync mode and consume this frame's hand input when the panel owns it."""
        next_mode = (MODE_RECOMMENDATION if self.renderer.has_candidate_panel
                     else MODE_NORMAL)
        if next_mode != self.mode:
            self.swipe_gate.reset()
            self.mode = next_mode
            # Consume both transition frames: motion used to open/confirm a
            # recommendation must not also become normal scene interaction.
            self.swipe_gate.update(None, active=self.active, now=now)
            return True

        if self.active:
            direction = self.swipe_gate.update(swipe, active=True, now=now)
            if direction is not None:
                self.renderer.navigate_candidate(direction)
            return True

        self.swipe_gate.update(None, active=False, now=now)
        return False

    def dismiss(self):
        """Clear panel state and return input ownership to normal mode."""
        self.renderer.set_candidate_recommendation(None)
        self.swipe_gate.reset()
        self.mode = MODE_NORMAL
