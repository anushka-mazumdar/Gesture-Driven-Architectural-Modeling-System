import numpy as np
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SnapPreview:
    """A display-only proposal between two world-space mesh anchors."""

    moving_object: object
    target_object: object
    moving_anchor_id: str
    target_anchor_id: str
    kind: str
    moving_position: tuple
    target_position: tuple
    distance: float

    def to_dict(self):
        return {
            "moving_anchor_id": self.moving_anchor_id,
            "target_anchor_id": self.target_anchor_id,
            "kind": self.kind,
            "moving_position": list(self.moving_position),
            "target_position": list(self.target_position),
            "distance": self.distance,
        }


import numpy as np
import math


class SnapGroup:
    """
    A rigid cluster of snapped shapes.
    One shape is the 'anchor'; all others store a frozen offset relative to it.
    Moving/rotating/scaling the anchor propagates to every member.
    """

    def __init__(self, anchor):
        self.anchor  = anchor
        self.members = [anchor]                # anchor is always index-0
        self.offsets = {id(anchor): np.zeros(3, dtype=np.float32)}

    def add(self, obj, offset):
        """offset = obj.position - anchor.position at snap time."""
        if obj not in self.members:
            self.members.append(obj)
            self.offsets[id(obj)] = np.array(offset, dtype=np.float32)

    def remove(self, obj):
        if obj is self.anchor and len(self.members) > 1:
            # promote next member as new anchor, rebase offsets
            new_anchor = self.members[1]
            base_off   = self.offsets[id(new_anchor)].copy()
            new_offsets = {}
            for m in self.members:
                new_offsets[id(m)] = self.offsets[id(m)] - base_off
            self.offsets = new_offsets
            self.anchor  = new_anchor

        self.members  = [m for m in self.members if m is not obj]
        self.offsets.pop(id(obj), None)

    def sync_to_anchor(self):
        """Re-position every non-anchor member from anchor's current position."""
        for m in self.members:
            if m is self.anchor:
                continue
            m.position = self.anchor.position + self.offsets[id(m)]

    def centroid(self):
        positions = [m.position for m in self.members]
        return np.mean(positions, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.members)


class Snapping:

    SNAP_DIST    = 60.0   # centre-to-centre proximity to trigger snap
    DETACH_SPEED = 22.0   # px/frame hand velocity to break a snap
    ANCHOR_PREVIEW_DISTANCE = 40.0  # inclusive world-unit radius

    def __init__(self, snap_distance=60.0, enabled=False):

        self.snap_distance  = snap_distance
        # The application has no active Enable Snapping control yet. Keep
        # anchor metadata passive unless a setting explicitly enables this.
        self.enabled        = bool(enabled)
        self.snap_candidate = None   # (moving_obj, target_obj, snap_pos)
        self._groups        = []     # list[SnapGroup]
        self.snap_preview   = None

    def preview_nearest_anchors(self, moving_obj, all_objects):
        """Select the nearest compatible anchor pair without changing objects.

        Vertices, edges, and faces match only like-for-like. Center anchors are
        descriptive metadata rather than connection points, so they are not
        considered. Equal-distance pairs use scene order then stable anchor IDs.
        """
        if not self.enabled or moving_obj is None:
            self.snap_preview = None
            return None

        grouped = self.get_group(moving_obj)
        excluded = list(grouped.members) if grouped else [moving_obj]
        moving_anchors = tuple(getattr(moving_obj, "snap_anchors", ()) or ())
        if not moving_anchors:
            self.snap_preview = None
            return None

        best_key = None
        best_preview = None
        for target_index, target in enumerate(all_objects or ()):
            if any(target is member for member in excluded):
                continue
            target_anchors = tuple(getattr(target, "snap_anchors", ()) or ())
            for moving_anchor in moving_anchors:
                if moving_anchor.kind not in ("vertex", "edge", "face"):
                    continue
                moving_position = np.asarray(moving_anchor.world_position(moving_obj), dtype=float)
                if moving_position.shape != (3,) or not np.all(np.isfinite(moving_position)):
                    continue
                for target_anchor in target_anchors:
                    if target_anchor.kind != moving_anchor.kind:
                        continue
                    target_position = np.asarray(target_anchor.world_position(target), dtype=float)
                    if target_position.shape != (3,) or not np.all(np.isfinite(target_position)):
                        continue
                    distance = float(np.linalg.norm(moving_position - target_position))
                    if not math.isfinite(distance) or distance > self.ANCHOR_PREVIEW_DISTANCE:
                        continue
                    key = (distance, target_index, moving_anchor.anchor_id,
                           target_anchor.anchor_id)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_preview = SnapPreview(
                            moving_obj, target, moving_anchor.anchor_id,
                            target_anchor.anchor_id, moving_anchor.kind,
                            tuple(float(value) for value in moving_position),
                            tuple(float(value) for value in target_position), distance,
                        )
        self.snap_preview = best_preview
        return best_preview

    def clear_preview(self):
        self.snap_preview = None

    

    def update(self, moving_obj, all_objects):
        """
        Call every frame while an object is being dragged.
        Finds the nearest snappable face-pair and stores a candidate.
        Returns the candidate target object, or None.
        """
        if not self.enabled:
            self._clear_candidate(all_objects)
            return None

        if moving_obj is None:
            self._clear_candidate(all_objects)
            return None

        # objects already grouped with moving_obj — skip them
        moving_group   = self.get_group(moving_obj)
        grouped_with   = set(moving_group.members) if moving_group else {moving_obj}

        best_target    = None
        best_dist      = float('inf')
        best_snap_pos  = None

        for obj in all_objects:
            if obj in grouped_with:
                continue

            dist = self._centre_dist(moving_obj, obj)
            if dist < self.snap_distance and dist < best_dist:
                snap_pos = self._face_snap_position(moving_obj, obj)
                if snap_pos is not None:
                    best_dist     = dist
                    best_target   = obj
                    best_snap_pos = snap_pos

        for obj in all_objects:
            if obj is not best_target:
                obj.highlighted = False
        if best_target:
            best_target.highlighted = True

        self.snap_candidate = (moving_obj, best_target, best_snap_pos) if best_target else None

        if best_target and best_dist < self.snap_distance * 0.4:
            self.confirm_snap(moving_obj)
            return best_target

        return best_target

    def confirm_snap(self, moving_obj):
        """
        Lock moving_obj (and its group) against the candidate target.
        Call when manipulation ends while a candidate exists.
        """
        if not self.enabled:
            if self.snap_candidate:
                _, target, _ = self.snap_candidate
                if target is not None:
                    target.highlighted = False
                self.snap_candidate = None
            return False
        if moving_obj is None or self.snap_candidate is None:
            return False

        mover, target, snap_pos = self.snap_candidate

        if mover is not moving_obj:
            return False

        delta = snap_pos - moving_obj.position
        moving_group = self.get_group(moving_obj)

        if moving_group:
            for m in moving_group.members:
                m.position = m.position + delta
        else:
            moving_obj.position = snap_pos.copy()

        target_group  = self.get_group(target)
        moving_group2 = self.get_group(moving_obj)   

        if target_group and moving_group2:
            for m in list(moving_group2.members):
                offset = m.position - target_group.anchor.position
                target_group.add(m, offset)
            self._groups = [g for g in self._groups if g is not moving_group2]

        elif target_group:
            offset = moving_obj.position - target_group.anchor.position
            target_group.add(moving_obj, offset)

        elif moving_group2:
            offset = target.position - moving_obj.position
            moving_group2.add(target, offset)

        else:
            g = SnapGroup(target)
            offset = moving_obj.position - target.position
            g.add(moving_obj, offset)
            self._groups.append(g)

        target.selected = False
        target.highlighted = False
        self.snap_candidate = None
        return True

    def detach(self, obj):
        """
        Pull obj out of its SnapGroup (called on fast peace-sign yank).
        If the group shrinks to 1, dissolve it entirely.
        """
        if not self.enabled:
            return
        group = self.get_group(obj)
        if group is None:
            return

        group.remove(obj)

        if len(group) <= 1:
            self._groups = [g for g in self._groups if g is not group]

    def get_group(self, obj):
        """Return the SnapGroup obj belongs to, or None."""
        for g in self._groups:
            if obj in g.members:
                return g
        return None

    def has_candidate(self):
        return self.snap_candidate is not None

    def cancel_snap(self, all_objects=None):
        if all_objects is not None:
            self._clear_candidate(all_objects)
        else:
            self.snap_candidate = None

    def propagate_move(self, obj, dx, dy):
        """
        After moving obj by (dx, dy), push the same delta to all
        other members of its group.
        """
        if not self.enabled:
            return
        group = self.get_group(obj)
        if group is None:
            return
        for m in group.members:
            if m is obj:
                continue
            m.position[0] += dx
            m.position[1] += dy

    def propagate_depth(self, obj, dz):
        if not self.enabled:
            return
        group = self.get_group(obj)
        if group is None:
            return
        for m in group.members:
            if m is obj:
                continue
            m.position[2] += dz

    def propagate_rotate(self, obj, delta_rx, delta_ry):
        """Rotate all group members by the same delta (simple rigid rotation)."""
        if not self.enabled:
            return
        group = self.get_group(obj)
        if group is None:
            return
        for m in group.members:
            if m is obj:
                continue
            m.rotation[0] += delta_rx
            m.rotation[1] += delta_ry

    def propagate_scale(self, obj, new_scale):
        """
        Scale all group members uniformly to the same absolute scale value.
        (Keeps proportions locked — they scaled together.)
        """
        if not self.enabled:
            return
        group = self.get_group(obj)
        if group is None:
            return
        for m in group.members:
            if m is obj:
                continue
            m.scale = np.array([new_scale, new_scale, new_scale], dtype=np.float32)

    

    def _centre_dist(self, obj_a, obj_b):
        ba = obj_a.get_bounds()
        bb = obj_b.get_bounds()
        if ba is None or bb is None:
            return float('inf')
        ca = obj_a.position + ba['center']
        cb = obj_b.position + bb['center']
        return float(np.linalg.norm(ca - cb))

    def _face_snap_position(self, moving_obj, target):
        """
        Return the position moving_obj should jump to so its nearest face
        sits flush against target's nearest face (face-to-face, like blocks).
        Returns None if bounds unavailable.
        """
        bm = moving_obj.get_bounds()
        bt = target.get_bounds()
        if bm is None or bt is None:
            return None

        cm = moving_obj.position + bm['center']
        ct = target.position     + bt['center']
        diff = cm - ct

        # dominant axis = the axis shapes are most separated along
        axis = int(np.argmax(np.abs(diff)))

        half_m = (bm['max'][axis] - bm['min'][axis]) / 2.0
        half_t = (bt['max'][axis] - bt['min'][axis]) / 2.0
        sign   = np.sign(diff[axis])

        snap_pos = moving_obj.position.copy().astype(np.float32)
        snap_pos[axis] = (
            target.position[axis]
            + bt['center'][axis]
            + sign * (half_t + half_m)
            - bm['center'][axis]
        )

        return snap_pos

    def _clear_candidate(self, all_objects):
        if self.snap_candidate:
            _, target, _ = self.snap_candidate
            if target:
                target.highlighted = False
        self.snap_candidate = None
