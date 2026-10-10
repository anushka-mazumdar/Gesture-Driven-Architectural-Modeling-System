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


class SnapAssembly:
    """Logical group of separately-rendered objects joined by snap commits."""

    def __init__(self, anchor):
        self.anchor  = anchor
        self.members = [anchor]                # anchor is always index-0
        self.offsets = {id(anchor): np.zeros(3, dtype=np.float32)}
        self._last_states = {id(anchor): self._state(anchor)}

    @staticmethod
    def _state(obj):
        rotation = np.asarray(obj.rotation, dtype=float).copy()
        return {
            "position": np.asarray(obj.position, dtype=float).copy(),
            "rotation": rotation,
            "rotation_matrix": Snapping._rotation_matrix(rotation),
            "scale": np.asarray(obj.scale, dtype=float).copy(),
        }

    def remember_transforms(self):
        """Refresh baselines after an assembly-wide transform."""
        self._last_states = {id(member): self._state(member)
                             for member in self.members}

    def add(self, obj, offset):
        """offset = obj.position - anchor.position at snap time."""
        if obj not in self.members:
            self.members.append(obj)
            self.offsets[id(obj)] = np.array(offset, dtype=np.float32)
            self._last_states[id(obj)] = self._state(obj)

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
        self._last_states.pop(id(obj), None)
        self.remember_transforms()

    def sync_to_anchor(self):
        """Reposition members from the anchor's stored positional offsets."""
        for m in self.members:
            if m is self.anchor:
                continue
            m.position = self.anchor.position + self.offsets[id(m)]
        self.remember_transforms()

    def centroid(self):
        positions = [m.position for m in self.members]
        return np.mean(positions, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.members)


# Backwards-compatible name used by existing snapping and anchor code.
SnapGroup = SnapAssembly


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

    @property
    def assemblies(self):
        """Current logical assemblies; members remain individual scene objects."""
        return tuple(self._groups)

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

    def commit_preview(self, moving_obj, all_objects=None):
        """Commit the current preview with a deterministic rigid transform.

        The active anchor is aligned to the target anchor. Face normals are
        made opposing; edge directions are aligned using their stable endpoint
        ordering. Vertices require translation only. Geometry and scale stay
        untouched, and the connected objects are registered in one assembly.
        """
        preview = self.snap_preview
        if (not self.enabled or preview is None or moving_obj is None
                or preview.moving_object is not moving_obj
                or preview.target_object is moving_obj):
            return False
        target_obj = preview.target_object
        if all_objects is not None and not any(target_obj is item for item in all_objects):
            return False
        group = self.get_group(moving_obj)
        if group and any(target_obj is member for member in group.members):
            return False

        moving_anchor = self._find_anchor(
            moving_obj, preview.moving_anchor_id, preview.kind
        )
        target_anchor = self._find_anchor(
            target_obj, preview.target_anchor_id, preview.kind
        )
        if moving_anchor is None or target_anchor is None:
            return False

        moving_group = self.get_group(moving_obj)
        group_before = ({id(member): SnapAssembly._state(member)
                         for member in moving_group.members}
                        if moving_group is not None else {})
        pivot_before = np.asarray(moving_obj.position, dtype=float).copy()
        rotation_delta = np.eye(3, dtype=float)
        if preview.kind == "face":
            moving_normal = moving_anchor.world_normal(moving_obj)
            target_normal = target_anchor.world_normal(target_obj)
            if moving_normal is not None and target_normal is not None:
                rotation_delta = self._rotation_between(
                    np.asarray(moving_normal, dtype=float),
                    -np.asarray(target_normal, dtype=float),
                )
        elif preview.kind == "edge":
            moving_direction = self._edge_world_direction(moving_obj, moving_anchor)
            target_direction = self._edge_world_direction(target_obj, target_anchor)
            if moving_direction is not None and target_direction is not None:
                rotation_delta = self._rotation_between(moving_direction, target_direction)

        if not np.allclose(rotation_delta, np.eye(3), atol=1e-12):
            current_rotation = self._rotation_matrix(moving_obj.rotation)
            self._write_rotation(moving_obj, rotation_delta @ current_rotation)

        # Recompute after rotation so translation places the active anchor
        # exactly at the target's current transformed anchor position.
        moving_position = np.asarray(moving_anchor.world_position(moving_obj), dtype=float)
        target_position = np.asarray(target_anchor.world_position(target_obj), dtype=float)
        if (not np.all(np.isfinite(moving_position))
                or not np.all(np.isfinite(target_position))):
            return False
        translated = np.asarray(moving_obj.position, dtype=float) + (target_position - moving_position)
        translation_delta = translated - np.asarray(moving_obj.position, dtype=float)
        self._write_position(moving_obj, translated)
        if moving_group is not None:
            for member in moving_group.members:
                if member is moving_obj:
                    continue
                state = group_before[id(member)]
                member_position = (pivot_before
                                   + rotation_delta @ (state["position"] - pivot_before)
                                   + translation_delta)
                self._write_position(member, member_position)
                self._write_rotation(
                    member, rotation_delta @ self._rotation_matrix(state["rotation"])
                )
            moving_group.remember_transforms()
        self._join_assembly(moving_obj, target_obj)
        self.clear_preview()
        return True

    def _join_assembly(self, moving_obj, target_obj):
        """Join two snapped objects, merging existing assemblies when needed."""
        moving_group = self.get_group(moving_obj)
        target_group = self.get_group(target_obj)
        if moving_group is not None and moving_group is target_group:
            return moving_group

        if target_group is not None:
            destination = target_group
        elif moving_group is not None:
            destination = moving_group
        else:
            destination = SnapAssembly(target_obj)
            self._groups.append(destination)

        source = moving_group if destination is target_group else target_group
        if source is not None:
            for member in list(source.members):
                if member not in destination.members:
                    destination.add(member, member.position - destination.anchor.position)
            self._groups = [group for group in self._groups if group is not source]

        for member in (moving_obj, target_obj):
            if member not in destination.members:
                destination.add(member, member.position - destination.anchor.position)
        destination.remember_transforms()
        return destination

    @staticmethod
    def _find_anchor(obj, anchor_id, kind):
        for anchor in getattr(obj, "snap_anchors", ()) or ():
            if anchor.anchor_id == anchor_id and anchor.kind == kind:
                return anchor
        return None

    @classmethod
    def _edge_world_direction(cls, obj, edge_anchor):
        if len(edge_anchor.source_vertices) != 2:
            return None
        endpoints = []
        for source_key in edge_anchor.source_vertices:
            vertex = next((anchor for anchor in getattr(obj, "snap_anchors", ()) or ()
                           if anchor.kind == "vertex"
                           and anchor.source_vertices == (source_key,)), None)
            if vertex is None:
                return None
            endpoints.append(np.asarray(vertex.world_position(obj), dtype=float))
        direction = endpoints[1] - endpoints[0]
        length = float(np.linalg.norm(direction))
        return direction / length if length > 1e-12 else None

    @staticmethod
    def _rotation_between(source, target):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        source_len = float(np.linalg.norm(source))
        target_len = float(np.linalg.norm(target))
        if source_len <= 1e-12 or target_len <= 1e-12:
            return np.eye(3, dtype=float)
        source /= source_len
        target /= target_len
        cross = np.cross(source, target)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if sine <= 1e-12:
            if cosine >= 0:
                return np.eye(3, dtype=float)
            # For an antiparallel pair, choose the least-aligned basis axis so
            # the 180-degree rotation is stable across runs/platforms.
            basis = np.eye(3)[int(np.argmin(np.abs(source)))]
            axis = np.cross(source, basis)
            axis /= np.linalg.norm(axis)
            return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=float)
        axis = cross / sine
        skew = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)

    @staticmethod
    def _rotation_matrix(rotation_degrees):
        angles = np.radians(np.asarray(rotation_degrees, dtype=float).reshape(3))
        sx, sy, sz = np.sin(angles)
        cx, cy, cz = np.cos(angles)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
        return rz @ ry @ rx

    @staticmethod
    def _write_rotation(obj, matrix):
        # Extract a Three.js Euler XYZ rotation (Rz * Ry * Rx) in degrees.
        sy = max(-1.0, min(1.0, -float(matrix[2, 0])))
        y = math.asin(sy)
        cy = math.cos(y)
        if abs(cy) > 1e-8:
            x = math.atan2(matrix[2, 1], matrix[2, 2])
            z = math.atan2(matrix[1, 0], matrix[0, 0])
        else:
            x = math.atan2(-matrix[1, 2], matrix[1, 1])
            z = 0.0
        values = np.degrees((x, y, z))
        try:
            obj.rotation[:] = values
        except (TypeError, AttributeError):
            obj.rotation = values

    @staticmethod
    def _write_position(obj, position):
        try:
            obj.position[:] = position
        except (TypeError, AttributeError):
            obj.position = position

    

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

        self._join_assembly(moving_obj, target)

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
        group = self.get_group(obj)
        if group is None:
            return
        dx, dy = float(dx), float(dy)
        if dx == 0.0 and dy == 0.0:
            return
        for m in group.members:
            if m is obj:
                continue
            m.position[0] += dx
            m.position[1] += dy
            state = group._last_states.get(id(m))
            if state is not None:
                state["position"][:2] = m.position[:2]
        state = group._last_states.get(id(obj))
        if state is not None:
            state["position"][:] = obj.position

    def propagate_depth(self, obj, dz):
        group = self.get_group(obj)
        if group is None:
            return
        dz = float(dz)
        if dz == 0.0:
            return
        for m in group.members:
            if m is obj:
                continue
            m.position[2] += dz
            state = group._last_states.get(id(m))
            if state is not None:
                state["position"][2] = m.position[2]
        state = group._last_states.get(id(obj))
        if state is not None:
            state["position"][:] = obj.position

    def propagate_rotate(self, obj, delta_rx, delta_ry):
        """Rotate every member around the manipulated object's current pivot."""
        group = self.get_group(obj)
        if group is None:
            return
        previous = group._last_states.get(id(obj))
        if previous is None:
            group.remember_transforms()
            return
        current_euler = np.asarray(obj.rotation, dtype=float)
        if np.array_equal(current_euler, previous["rotation"]):
            return
        old_rotation = previous["rotation_matrix"]
        new_rotation = self._rotation_matrix(obj.rotation)
        rotation_delta = new_rotation @ old_rotation.T
        old_pivot = previous["position"]
        new_pivot = np.asarray(obj.position, dtype=float)
        for member in group.members:
            if member is obj:
                continue
            state = group._last_states.get(id(member))
            if state is None:
                continue
            translated_position = state["position"] + (new_pivot - old_pivot)
            rotated_position = new_pivot + rotation_delta @ (translated_position - new_pivot)
            self._write_position(member, rotated_position)
            member_rotation = rotation_delta @ state["rotation_matrix"]
            self._write_rotation(member, member_rotation)
            state["position"][:] = member.position
            state["rotation"][:] = member.rotation
            state["rotation_matrix"][:] = member_rotation
        previous["position"][:] = obj.position
        previous["rotation"][:] = obj.rotation
        previous["rotation_matrix"][:] = new_rotation

    def propagate_scale(self, obj, new_scale):
        """Scale assembly geometry and member offsets about the active object."""
        group = self.get_group(obj)
        if group is None:
            return
        previous = group._last_states.get(id(obj))
        if previous is None:
            group.remember_transforms()
            return
        old_scale = float(previous["scale"][0])
        if not math.isfinite(old_scale) or old_scale <= 1e-12:
            group.remember_transforms()
            return
        ratio = float(new_scale) / old_scale
        if not math.isfinite(ratio) or ratio <= 0:
            group.remember_transforms()
            return
        if math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
            previous["position"][:] = obj.position
            previous["scale"][:] = obj.scale
            return

        pivot = np.asarray(obj.position, dtype=float)
        for member in group.members:
            if member is obj:
                continue
            state = group._last_states.get(id(member))
            if state is None:
                continue
            self._write_position(member, pivot + ratio * (state["position"] - previous["position"]))
            try:
                member.scale[:] = state["scale"] * ratio
            except (TypeError, AttributeError):
                member.scale = state["scale"] * ratio
            state["position"][:] = member.position
            state["scale"][:] *= ratio
        previous["position"][:] = obj.position
        previous["scale"][:] = obj.scale

    

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
