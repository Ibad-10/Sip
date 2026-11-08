"""Explore a new home and save the map (sensors only: lidar, head depth camera, object detector).

  uv run python -m sip.tasks.explore --seconds 400            # writes runs/map.npz
"""
import argparse
import os

import mujoco
import numpy as np

from sip.nav.explore import MapBuilder
from sip.nav.follow import PathFollower, wrap
from sip.nav.grid import astar
from sip.nav.lidar import scan
from sip.perception.objects import CAM, aim_head, pixel_to_world, render_head
from sip.tasks.water import TICK_STEPS


def explore(m, d, io, detector=None, seconds: float = 400.0, log=print, on_tick=None) -> MapBuilder:
    mb = MapBuilder()
    r = mujoco.Renderer(m, 120, 160)
    t, k, follower, tried, goal, stuck_t, last_p = 0.0, 0, None, [], None, 0.0, None
    io.stow(lift=0.5)
    while t < seconds:
        p = io.base_pose()
        if k % 10 == 0:
            mb.add_scan(p, scan(m, d, p, n_rays=180, max_range=4.0), 4.0)
        if k % 8 == 0:
            # head sweeps left/right while driving; depth adds obstacles above lidar height
            look = p.yaw + 0.9 * np.sin(t * 0.8)
            aim_head(m, d, io, np.array([p.x + 1.5 * np.cos(look), p.y + 1.5 * np.sin(look), 0.7]), snap=False)
            rgb, depth, seg = render_head(m, d, r, rgb=detector is not None and getattr(detector, "needs_rgb", False))
            vs, us = np.mgrid[0:depth.shape[0]:4, 0:depth.shape[1]:4]
            z = depth[vs, us].ravel()
            ok = (z > 0.2) & (z < 3.0)
            pts = pixel_to_world(m, d, CAM, us.ravel()[ok], vs.ravel()[ok], z[ok], depth.shape[1], depth.shape[0])
            mb.add_points(pts)
            if detector is not None:
                for det in detector.detect(m, d, rgb, depth, seg):
                    if det.label in ("stove", "sink", "bed", "sofa", "tv") and det.label not in mb.rooms:
                        mb.see_object(det.label, det.xyz)
        if follower is None or follower.done or (last_p and t - stuck_t > 6.0):
            if follower is not None and goal is not None:
                tried.append(goal)
            goal = mb.next_goal(p, tried=tried)
            if goal is None:
                log(f"exploration done: no reachable frontier left, {mb.coverage() * 100:.0f}% of the area seen")
                break
            g = mb.grid.inflate(0.3)
            g.occ |= ~mb.known
            for x in np.arange(p.x - 0.15, p.x + 0.16, mb.grid.res):    # free the robot's own cells
                for y in np.arange(p.y - 0.15, p.y + 0.16, mb.grid.res):
                    g.mark(x, y, False)
            path = astar(g, (p.x, p.y), goal)
            if path is None:
                tried.append(goal)
                follower = None
                continue
            follower = PathFollower(path, lookahead=0.25, v_max=0.3)
            stuck_t, last_p = t, (p.x, p.y)
            log(f"exploring: heading to the edge of the unknown at ({goal[0]:.1f}, {goal[1]:.1f})")
        if last_p and np.hypot(p.x - last_p[0], p.y - last_p[1]) > 0.1:
            stuck_t, last_p = t, (p.x, p.y)
        io.drive(*follower.step(p))
        for _ in range(TICK_STEPS):
            mujoco.mj_step(m, d)
        t += TICK_STEPS * m.opt.timestep
        k += 1
        if on_tick:
            on_tick(t)
    io.drive(0, 0)
    for room, xy in mb.rooms.items():
        log(f"named a room from what it saw: {room} near ({xy[0]:.1f}, {xy[1]:.1f})")
    return mb


def main(argv=None):
    from sip.app import Sim
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=400)
    ap.add_argument("--out", default="runs/map.npz")
    a = ap.parse_args(argv)
    sim = Sim()
    mb = explore(sim.m, sim.d, sim.io, sim.task.det, a.seconds)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    mb.save(a.out)
    print("saved", a.out, "coverage", round(mb.coverage(), 3), "rooms", mb.rooms)


if __name__ == "__main__":
    main()
