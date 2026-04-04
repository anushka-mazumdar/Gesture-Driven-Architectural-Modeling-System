import numpy as np
import math


class Snapping:

    def __init__(self, snap_distance=60.0):

        self.snap_distance   = snap_distance
        self.snap_candidate  = None   # object we might snap to
        self.snap_target     = None   # object being moved toward snap
        self.snapped_pairs   = []     # list of (obj_a, obj_b) locked pairs


    def update(self, moving_obj, all_objects):
        """
        Call every frame while an object is being moved.
        Returns the candidate object if snap is possible, else None.
        """

        if moving_obj is None:
            self.snap_candidate = None
            return None

        best      = None
        best_dist = float('inf')

        for obj in all_objects:

            if obj is moving_obj:
                continue

            dist = self._centre_dist(moving_obj, obj)

            if dist < self.snap_distance and dist < best_dist:
                best_dist = dist
                best      = obj

        self.snap_candidate = best

        # visual preview — highlight candidate
        for obj in all_objects:
            if obj is not moving_obj and obj is not best:
                if not obj.selected:
                    obj.selected = False

        if best:
            best.selected = True   # yellow highlight as snap preview

        return best


    def confirm_snap(self, moving_obj):
        """
        Lock moving_obj against snap_candidate.
        Call when pinch is released near a candidate.
        """

        if moving_obj is None or self.snap_candidate is None:
            return False

        target  = self.snap_candidate
        face_pos = self._nearest_face_position(moving_obj, target)

        moving_obj.position = face_pos.copy()

        self.snapped_pairs.append((moving_obj, target))
        self.snap_candidate = None

        target.selected = False

        return True


    def detach(self, obj):
        """Remove obj from all snap pairs."""

        self.snapped_pairs = [
            (a, b) for a, b in self.snapped_pairs
            if a is not obj and b is not obj
        ]


    def _centre_dist(self, obj_a, obj_b):

        ba = obj_a.get_bounds()
        bb = obj_b.get_bounds()

        if ba is None or bb is None:
            return float('inf')

        ca = obj_a.position + ba['center']
        cb = obj_b.position + bb['center']

        return float(np.linalg.norm(ca - cb))


    def _nearest_face_position(self, moving_obj, target):
        """
        Return position that places moving_obj flush against
        the nearest face of target.
        """

        bm = moving_obj.get_bounds()
        bt = target.get_bounds()

        if bm is None or bt is None:
            return moving_obj.position.copy()

        tm = moving_obj.position + bm['center']
        tt = target.position     + bt['center']

        diff = tm - tt

        # find dominant axis
        axis = int(np.argmax(np.abs(diff)))

        half_moving = (bm['max'][axis] - bm['min'][axis]) / 2.0
        half_target = (bt['max'][axis] - bt['min'][axis]) / 2.0

        sign    = np.sign(diff[axis])
        snap_pos = moving_obj.position.copy()

        snap_pos[axis] = (
            target.position[axis]
            + bt['center'][axis]
            + sign * (half_target + half_moving)
            - bm['center'][axis]
        )

        return snap_pos