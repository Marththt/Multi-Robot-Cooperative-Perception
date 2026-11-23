"""OptiTrack: every rigid body Motive streams, not only a fixed list, so the boxes in the room are found by
themselves (vf/localization/mocap_pose.py itself is not changed).

AllBodiesHub(robot_rbs) is MocapHub that also keeps every other rigid body it receives; tracked() lists those with
a fresh pose (tracked now). boxes(arena, exclude) = the tracked rigid bodies inside the arena that are not robots:
a box taken away disappears within STALE_AFTER (0.2 s), a box put in appears at once.
"""

from vf.localization.mocap_pose import MocapHub


class AllBodiesHub(MocapHub):
    """MocapHub([robot rigid bodies]) that also records every other rigid body of the stream. wait_ready() waits
    for the robots only."""

    def _on_frame(self, data_frame):
        with self._lock:
            for body in data_frame.rigid_bodies:
                self._data.setdefault(int(body.id_num), None)
        super()._on_frame(data_frame)

    def tracked(self):
        with self._lock:
            ids = list(self._data)
        return sorted(rb for rb in ids if self.get_pose(rb) is not None)

    def boxes(self, arena, exclude=(), only=None):
        """[{'id', 'x', 'y', 'yaw'}] of the tracked rigid bodies inside arena (x0, x1, y0, y1), robots (exclude)
        left out; only: restrict to these ids (None = all)."""
        x0, x1, y0, y1 = arena
        out = []
        for rb in self.tracked():
            if rb in exclude or (only is not None and rb not in only):
                continue
            q = self.get_pose(rb)
            if q is not None and x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                out.append(dict(id=rb, x=float(q[0]), y=float(q[1]), yaw=float(q[2])))
        return out
