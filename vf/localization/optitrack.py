"""Pose provider (x, y, yaw) from OptiTrack: the common world frame shared by all robots.

It wraps a pose source from mocap_pose.py (MocapPose for one rigid body per process, or
MocapHub(...).view(rb_id) for several rigid bodies in one process) in the same interface as
localization.OdometryLocalizer (pose / position / current_zone / close), so the acquisition scripts
can switch between odometry and OptiTrack with one option.

pose() and position() return None when the tracking data is missing or stale (markers occluded,
tracking lost; see STALE_AFTER in mocap_pose.py): a fake (0, 0, 0) would be wrong data, while None
propagates as "position unknown" (unknown zone, no world position for that detection).
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
        # everyone; closing it here would break the other
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
