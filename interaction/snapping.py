import numpy as np
import math
from dataclasses import dataclass
from scipy.spatial import cKDTree


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
    moving_signature: tuple = None
    target_signature: tuple = None
    target_kind: str = None

    def to_dict(self):
        return {
            "moving_anchor_id": self.moving_anchor_id,
            "target_anchor_id": self.target_anchor_id,
            "kind": self.kind,
            "target_kind": self.target_kind or self.kind,
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
        # Stable transforms in the assembly anchor's local coordinate space.
        # These are captured when membership changes, not every frame, so
        # repeated rotations never accumulate member-to-member drift.
        self._relative_transforms = {id(anchor): self._relative_state(anchor)}

    def _relative_state(self, obj):
        anchor_state = self._state(self.anchor)
        member_state = self._state(obj)
        anchor_rotation = anchor_state["rotation_matrix"]
        anchor_scale = anchor_state["scale"]
        safe_scale = np.where(np.abs(anchor_scale) > 1e-12, anchor_scale, 1.0)
        return {
            "position": anchor_rotation.T @ (member_state["position"] - anchor_state["position"]) / safe_scale,
            "rotation": anchor_rotation.T @ member_state["rotation_matrix"],
            "scale": member_state["scale"] / safe_scale,
        }

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
            self._relative_transforms[id(obj)] = self._relative_state(obj)

    def remove(self, obj):
        promoted_anchor = False
        if obj is self.anchor and len(self.members) > 1:
            # promote next member as new anchor, rebase offsets
            new_anchor = self.members[1]
            base_off   = self.offsets[id(new_anchor)].copy()
            new_offsets = {}
            for m in self.members:
                new_offsets[id(m)] = self.offsets[id(m)] - base_off
            self.offsets = new_offsets
            self.anchor  = new_anchor
            promoted_anchor = True

        self.members  = [m for m in self.members if m is not obj]
        self.offsets.pop(id(obj), None)
        self._last_states.pop(id(obj), None)
        self._relative_transforms.pop(id(obj), None)
        if promoted_anchor and self.members:
            # Membership removal promoted a new anchor; express every
            # remaining member relative to that new stable pivot.
            self._relative_transforms = {
                id(member): self._relative_state(member) for member in self.members
            }
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
        # World-space anchors and the target grid survive frame-to-frame
        # manipulation. Signatures cheaply detect transforms/anchor changes.
        self._world_anchor_cache = {}
        self._spatial_cache_key = None
        self._spatial_cache = {}
        self.preview_stats = {"cache_hits": 0, "cache_rebuilds": 0,
                              "spatial_rebuilds": 0, "candidates_checked": 0}

    @staticmethod
    def _anchor_signature(obj, anchors):
        def vector(value, default):
            try:
                result = np.asarray(value, dtype=float).reshape(3)
            except (TypeError, ValueError):
                result = np.asarray(default, dtype=float)
            return tuple(float(x) for x in result)
        source = getattr(obj, "snap_anchors", ())
        geometry = (id(source), len(anchors),
                    id(source) if isinstance(source, tuple) else hash(anchors))
        return (vector(getattr(obj, "position", (0, 0, 0)), (0, 0, 0)),
                vector(getattr(obj, "rotation", (0, 0, 0)), (0, 0, 0)),
                vector(getattr(obj, "scale", (1, 1, 1)), (1, 1, 1)),
                id(anchors), geometry)

    def _world_anchors(self, obj):
        anchors = tuple(getattr(obj, "snap_anchors", ()) or ())
        signature = self._anchor_signature(obj, anchors)
        cached = self._world_anchor_cache.get(id(obj))
        if cached is not None and cached[0] is obj and cached[1] == signature:
            self.preview_stats["cache_hits"] += 1
            return cached[2], signature

        position = np.asarray(signature[0], dtype=float)
        rotation = self._rotation_matrix(signature[1])
        scale = np.asarray(signature[2], dtype=float)
        records = {}
        for anchor in anchors:
            local = np.asarray(anchor.local_position, dtype=float)
            world = rotation @ (local * scale) + position
            normal = None
            if anchor.local_normal is not None:
                local_normal = np.asarray(anchor.local_normal, dtype=float)
                unscaled = np.divide(local_normal, scale, out=np.zeros(3),
                                     where=np.abs(scale) > 1e-12)
                normal_value = rotation @ unscaled
                length = float(np.linalg.norm(normal_value))
                if length > 1e-12:
                    normal = normal_value / length
            records[(anchor.kind, anchor.anchor_id)] = (anchor, world, normal)
        self._world_anchor_cache[id(obj)] = (obj, signature, records)
        self.preview_stats["cache_rebuilds"] += 1
        return records, signature

    def _target_spatial_index(self, targets):
        signatures = []
        for target_index, target in targets:
            _, signature = self._world_anchors(target)
            signatures.append((id(target), signature))
        key = tuple((index, id(target)) for index, target in targets), tuple(signatures)
        if key == self._spatial_cache_key:
            return self._spatial_cache

        entries = []
        for (target_index, target), (_, signature) in zip(targets, signatures):
            records, _ = self._world_anchors(target)
            has_endpoints = any(anchor_id.startswith("endpoint:")
                                for (anchor_kind, anchor_id) in records
                                if anchor_kind == "vertex")
            for (kind, anchor_id), record in records.items():
                if kind not in ("vertex", "edge", "face"):
                    continue
                if (kind == "vertex" and has_endpoints
                        and not anchor_id.startswith("endpoint:")):
                    continue
                anchor, point, _ = record
                if point.shape == (3,) and np.all(np.isfinite(point)):
                    cell = tuple(np.floor(point / self.ANCHOR_PREVIEW_DISTANCE).astype(np.int64))
                    entries.append((cell, kind, target_index, target, anchor, point))
        grid = {}
        for entry in entries:
            grid.setdefault(entry[1], []).append(entry)
        for kind, kind_entries in grid.items():
            grid[kind] = (cKDTree(np.asarray([item[5] for item in kind_entries],
                                             dtype=float)), kind_entries)
        self._spatial_cache_key = key
        self._spatial_cache = grid
        self.preview_stats["spatial_rebuilds"] += 1
        return self._spatial_cache

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

        eligible_targets = [(i, obj) for i, obj in enumerate(all_objects or ())
                            if not any(obj is member for member in excluded)]
        grid = self._target_spatial_index(eligible_targets)
        moving_records, moving_signature = self._world_anchors(moving_obj)
        moving_has_endpoints = any(anchor_id.startswith("endpoint:")
                                   for kind, anchor_id in moving_records
                                   if kind == "vertex")
        best_key = None
        best_preview = None
        for kind in ("vertex", "edge", "face"):
            indexed = grid.get(kind)
            if indexed is None:
                continue
            tree, entries = indexed
            if moving_has_endpoints and kind != "vertex":
                continue
            movers = [(anchor, moving_records[(kind, anchor.anchor_id)][1])
                      for anchor in moving_anchors
                      if anchor.kind == kind
                      and (kind != "vertex" or not moving_has_endpoints
                           or anchor.anchor_id.startswith("endpoint:"))
                      and (kind, anchor.anchor_id) in moving_records
                      and np.all(np.isfinite(moving_records[(kind, anchor.anchor_id)][1]))]
            if not movers:
                continue
            target_kinds = (("vertex", "edge", "face") if moving_has_endpoints
                            and kind == "vertex" else (kind,))
            points = np.asarray([item[1] for item in movers], dtype=float)
            for compatible_kind in target_kinds:
                compatible = grid.get(compatible_kind)
                if compatible is None:
                    continue
                target_tree, target_entries = compatible
                distances, indices = target_tree.query(
                    points, k=1,
                    distance_upper_bound=np.nextafter(self.ANCHOR_PREVIEW_DISTANCE, math.inf)
                )
                distances = np.asarray(distances).reshape(-1)
                indices = np.asarray(indices).reshape(-1)
                valid = np.isfinite(distances) & (indices < len(target_entries))
                if not np.any(valid):
                    continue
                nearest = float(np.min(distances[valid]))
                # Only moving anchors tied for the global best in this target
                # kind can affect the deterministic winner.
                for mover_index in np.flatnonzero(valid & (distances <= nearest + 1e-12)):
                    moving_anchor, moving_position = movers[int(mover_index)]
                    tied_indices = target_tree.query_ball_point(
                        moving_position, float(distances[mover_index]) + 1e-12
                    )
                    for candidate_index in tied_indices:
                        _, target_kind, target_index, target, target_anchor, target_position = target_entries[candidate_index]
                        self.preview_stats["candidates_checked"] += 1
                        distance = float(np.linalg.norm(moving_position - target_position))
                        if not math.isfinite(distance) or distance > self.ANCHOR_PREVIEW_DISTANCE:
                            continue
                        moving_endpoint = moving_anchor.anchor_id.startswith("endpoint:")
                        target_endpoint = target_anchor.anchor_id.startswith("endpoint:")
                        endpoint_priority = (0 if moving_endpoint and target_endpoint
                                             else 1 if moving_endpoint or target_endpoint
                                             else 2)
                        key = (endpoint_priority, distance, target_index, moving_anchor.anchor_id,
                               target_anchor.anchor_id)
                        if best_key is None or key < best_key:
                            best_key = key
                            _, target_signature = self._world_anchors(target)
                            best_preview = SnapPreview(
                                moving_obj, target, moving_anchor.anchor_id,
                                target_anchor.anchor_id, moving_anchor.kind,
                                tuple(float(value) for value in moving_position),
                                tuple(float(value) for value in target_position), distance,
                                moving_signature, target_signature, target_kind,
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
        if preview.moving_signature is not None:
            _, current_moving_signature = self._world_anchors(moving_obj)
            _, current_target_signature = self._world_anchors(target_obj)
            if (current_moving_signature != preview.moving_signature
                    or current_target_signature != preview.target_signature):
                self.clear_preview()
                return False
        if all_objects is not None and not any(target_obj is item for item in all_objects):
            return False
        group = self.get_group(moving_obj)
        if group and any(target_obj is member for member in group.members):
            return False

        moving_anchor = self._find_anchor(
            moving_obj, preview.moving_anchor_id, preview.kind
        )
        target_anchor = self._find_anchor(
            target_obj, preview.target_anchor_id, preview.target_kind or preview.kind
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
            moving_normal = self._world_anchors(moving_obj)[0][
                (moving_anchor.kind, moving_anchor.anchor_id)][2]
            target_normal = self._world_anchors(target_obj)[0][
                (target_anchor.kind, target_anchor.anchor_id)][2]
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
        elif preview.kind == "vertex":
            # For elongated meshes, orient the selected end toward the target
            # before translating it onto the target surface point. This keeps
            # rods/beams straight and prevents side-on or embedded connections.
            alignment = self._longitudinal_connection_rotation(
                moving_obj, moving_anchor, target_obj, target_anchor
            )
            if alignment is not None:
                rotation_delta = alignment

        if not np.allclose(rotation_delta, np.eye(3), atol=1e-12):
            current_rotation = self._rotation_matrix(moving_obj.rotation)
            self._write_rotation(moving_obj, rotation_delta @ current_rotation)

        # Recompute after rotation so translation places the active anchor
        # exactly at the target's current transformed anchor position.
        moving_position = self._world_anchors(moving_obj)[0][
            (moving_anchor.kind, moving_anchor.anchor_id)][1].copy()
        target_position = self._world_anchors(target_obj)[0][
            (target_anchor.kind, target_anchor.anchor_id)][1].copy()
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

    def _longitudinal_connection_rotation(self, moving_obj, moving_anchor,
                                          target_obj, target_anchor):
        """Aim an elongated object's selected end toward a vertex connection.

        The dominant local mesh axis is used, so the rule applies to rods,
        beams, and other genuinely elongated candidates without depending on
        their candidate IDs or assumed local axis. Compact objects are left
        unchanged. Only rotation is computed here; scale and vertices remain
        untouched and the active anchor is translated onto the target later.
        """
        try:
            vertices = np.asarray(moving_obj.vertices, dtype=float).reshape((-1, 3))
        except (AttributeError, TypeError, ValueError):
            return None
        if len(vertices) < 3 or not np.all(np.isfinite(vertices)):
            return None
        center = np.mean(vertices, axis=0)
        centered = vertices - center
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        ordered_values = np.sort(np.maximum(eigenvalues, 0.0))
        transverse = float(ordered_values[-2]) if len(ordered_values) > 1 else 0.0
        semantic_endpoint = moving_anchor.anchor_id.startswith("endpoint:")
        if (transverse <= 1e-12
                or (not semantic_endpoint
                    and math.sqrt(float(ordered_values[-1]) / transverse) < 2.0)):
            return None
        projected = centered @ axis
        longitudinal = float(np.max(projected) - np.min(projected))
        residual = centered - np.outer(projected, axis)
        radial = float(np.max(np.linalg.norm(residual, axis=1)))
        if longitudinal <= 1e-9 or radial <= 1e-9 or longitudinal / radial < 2.0:
            return None

        scale = np.asarray(moving_obj.scale, dtype=float).reshape(3)
        rotation = self._rotation_matrix(moving_obj.rotation)
        world_axis = rotation @ (axis * scale)
        world_axis_length = float(np.linalg.norm(world_axis))
        if world_axis_length <= 1e-12:
            return None
        world_axis /= world_axis_length

        anchor_side = float(np.dot(
            np.asarray(moving_anchor.local_position, dtype=float) - center, axis
        ))
        if abs(anchor_side) <= 1e-9:
            return None
        source_direction = world_axis * (1.0 if anchor_side > 0 else -1.0)

        object_position = np.asarray(moving_obj.position, dtype=float).reshape(3)
        center_world = rotation @ (center * scale) + object_position
        target_record = self._world_anchors(target_obj)[0].get(
            (target_anchor.kind, target_anchor.anchor_id)
        )
        if target_record is None:
            return None
        desired_direction = target_record[1] - center_world
        if float(np.linalg.norm(desired_direction)) <= 1e-9:
            return None
        return self._rotation_between(source_direction, desired_direction)

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
        Legacy gesture entry point. Keep its Enable Snapping gate for the
        existing peace-sign interaction, then delegate to the public API.
        """
        if not self.enabled:
            return False
        return self.detach_object(obj)

    def detach_object(self, obj_or_id):
        """Detach one assembly member without changing its world transform.

        Accepts the object reference or its Python ``id(obj)``. Returns True
        only when an assembly membership was actually removed. Explicit API
        calls are independent of the snapping toggle; that toggle controls
        new snapping, not management of an existing assembly.
        """
        obj = obj_or_id
        if isinstance(obj_or_id, int):
            obj = next((member for group in self._groups
                        for member in group.members
                        if id(member) == obj_or_id), None)
        group = self.get_group(obj) if obj is not None else None
        if group is None:
            return False

        group.remove(obj)
        if len(group) <= 1:
            self._groups = [g for g in self._groups if g is not group]
        return True

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
        """Rotate an assembly from a stable anchor-relative transform."""
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
        if obj is group.anchor:
            # Rebuild members from their snap-time local transforms instead
            # of incrementally rotating the last frame's floating-point state.
            anchor_scale = np.asarray(obj.scale, dtype=float)
            for member in group.members:
                if member is obj:
                    continue
                relative = group._relative_transforms.get(id(member))
                if relative is None:
                    continue
                self._write_position(
                    member,
                    new_pivot + new_rotation @ (relative["position"] * anchor_scale),
                )
                self._write_rotation(
                    member, new_rotation @ relative["rotation"]
                )
                try:
                    member.scale[:] = relative["scale"] * anchor_scale
                except (TypeError, AttributeError):
                    member.scale = relative["scale"] * anchor_scale
                state = group._last_states[id(member)]
                state.update(group._state(member))
            previous["position"][:] = obj.position
            previous["rotation"][:] = obj.rotation
            previous["rotation_matrix"][:] = new_rotation
            return
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
