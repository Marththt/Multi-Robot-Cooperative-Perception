"""Pose provider (x, y, yaw) from OptiTrack -- the position source
explicitly required by the professor's document (September 2026,
Section 3.4: "common world coordinate frame... via OptiTrack") as the
shared reference across robots, instead of the on-board self-localization
of localization.py.

REWRITTEN in this session: the previous version guessed a NatNet interface
(client.subscribe(name, callback)). Now that you sent your previous
OptiTrack project (live_pp.py, two_robots.py, goto_point.py, check_grid.py,
mocap_pose.py), there is no need to guess any more: mocap_pose.py is
already a planar pose source that is DONE AND TESTED against Motive via
natnet 0.3.0, with all the tricky details already solved (Y-up/Z-up
convention, quaternion order, discarding stale data, a SINGLE NatNet
connection shared across several rigid bodies to avoid port conflicts).
This module does NOT duplicate that work: it wraps it in the same
interface as localization.OdometryLocalizer
(pose/position/current_zone/close), so patrol.py can choose between
on-board odometry and OptiTrack with a single parameter (pose_provider in
Patrol.__init__) without the rest of the pipeline (ObjectDetector,
logging) knowing which one is in use.

What to pass as pose_source
------------------------------------
Any object from your mocap_pose.py with a method
`.get_pose() -> (x, y, yaw) or None`:
  - mocap_pose.MocapPose(rb_id, ...)                one robot per
    process (the normal case for patrol.py: one process per robot,
    see the patrol.py docstring)
  - mocap_pose.MocapHub(rb_ids, ...).view(rb_id)     several rigid bodies
    in the SAME process (e.g. if in future patrol.py handled several
    robots in a single script, as live_pp.py does)

Why pose()/position() here can return None
------------------------------------------------------------
Unlike OdometryLocalizer (which always returns a triple, at worst (0,0,0)
before the first odometry sample), get_pose() of mocap_pose.py EXPLICITLY
returns None when data is missing or stale (reflective markers occluded,
tracking lost mid-run -- see STALE_AFTER in mocap_pose.py). Faking a
(0,0,0) pose in that case would be FALSE data passed off as real, worse
than signalling "I don't know": None propagates correctly, without any
changes, through perception.ObjectDetector (pose_fn can already return
None, see its call to estimate_object_position) and through
current_zone()/position() below (None -> unknown_zone, not a crash). The
ONLY place that needed updating to handle this is the status print in
patrol.py Patrol.run(), which assumed an always-valid triple -- see the
fix there.
"""

import logging
from typing import List, Optional, Tuple

from vf.localization.localization import ZoneRegion

log = logging.getLogger(__name__)


class OptiTrackPose:
    """Same public interface as localization.OdometryLocalizer:
    pose(), position(), current_zone(), close() -- see the module docstring
    for why pose()/position() here can return None (OdometryLocalizer never
    does)."""

    def __init__(self, pose_source, regions: List[ZoneRegion],
                 unknown_zone: str = "unknown",
                 close_source: bool = False):
        self.pose_source = pose_source
        self.regions = regions
        self.unknown_zone = unknown_zone
        # close_source: True ONLY if this instance is the sole owner of
        # pose_source (e.g. a MocapPose dedicated to ONE robot/process).
        # With a view of a MocapHub SHARED by several robots in the same
        # process, leave it False: whoever owns the hub closes it once for
        # everyone (see live_pp.py); closing it here would break the other
        # robots still using it.
        self.close_source = close_source
        self._last_known_pose = None   # only for a more useful log

    def pose(self) -> Optional[Tuple[float, float, float]]:
        """None if tracking is missing or stale (see the module docstring)
        -- NEVER an invented (0,0,0)."""
        pose = self.pose_source.get_pose()
        if pose is None:
            if self._last_known_pose is not None:
                log.debug("OptiTrack: tracking lost (last known "
                          "pose: %s)", self._last_known_pose)
            else:
                log.debug("OptiTrack: no data received yet")
            return None
        self._last_known_pose = pose
        return pose

    def position(self) -> Optional[Tuple[float, float]]:
        pose = self.pose()
        if pose is None:
            return None
        return pose[0], pose[1]

    def current_zone(self) -> str:
        pos = self.position()
        if pos is None:
            return self.unknown_zone
        x, y = pos
        for region in self.regions:
            if (region.x_min <= x < region.x_max
                    and region.y_min <= y < region.y_max):
                return region.name
        return self.unknown_zone

    def close(self):
        if not self.close_source:
            return
        try:
            self.pose_source.close()
        except Exception as e:
            log.debug("closing the OptiTrack source failed: %s", e)
