"""Explain view for judges, in Rerun: decision graph, cause-and-effect log, map with old/new
route and memory, camera with detections and mouth, safety and Gemini verifier, budget strip.
Every run is saved to an .rrd for replay."""
import numpy as np
import rerun as rr

STATES = ["IDLE", "LOCATE", "SEARCH", "NAV_TO_OBJECT", "VERIFY", "GRASP", "NAV_TO_PERSON",
          "APPROACH", "HOLD", "RETREAT", "RETURN", "ASK", "HALT"]
EDGES = [("IDLE", "LOCATE"), ("LOCATE", "SEARCH"), ("LOCATE", "NAV_TO_OBJECT"), ("SEARCH", "NAV_TO_OBJECT"),
         ("NAV_TO_OBJECT", "VERIFY"), ("VERIFY", "GRASP"), ("VERIFY", "SEARCH"), ("GRASP", "NAV_TO_PERSON"),
         ("NAV_TO_PERSON", "APPROACH"), ("APPROACH", "HOLD"), ("HOLD", "RETREAT"), ("RETREAT", "RETURN"),
         ("RETURN", "IDLE"), ("SEARCH", "ASK"), ("APPROACH", "ASK"), ("ASK", "LOCATE")]
_POS = {s: (np.cos(2 * np.pi * i / 11) * 300, np.sin(2 * np.pi * i / 11) * 300) for i, s in enumerate(STATES[:11])}
_POS["ASK"], _POS["HALT"] = (0.0, 0.0), (0.0, -120.0)
_KIND_LEVEL = {"safety": "WARN", "alert": "ERROR", "say": "INFO", "plan": "INFO", "state": "DEBUG",
               "perceive": "INFO", "dose": "INFO"}
GREY, LIT = (90, 90, 90, 255), (255, 170, 0, 255)


class Explainer:
    def __init__(self, app_id: str = "sip", spawn: bool = True, save_path: str | None = None):
        self.rec = rr.RecordingStream(app_id)
        if spawn:
            self.rec.spawn()
        if save_path:
            self.rec.save(save_path)
        self.rec.log("decisions/edges", rr.GraphEdges(EDGES, graph_type="directed"), static=True)
        self.rec.log("map/house", rr.TextDocument("Top-down house map (metres)"), static=True)

    def _t(self, t: float) -> None:
        self.rec.set_time("sim", duration=float(t))

    def log_frame(self, t, rgb, detections, mouth, mouth_px=None, dist_to_mouth=None) -> None:
        self._t(t)
        if rgb is not None:
            self.rec.log("camera/rgb", rr.Image(rgb))
        if detections:
            self.rec.log("camera/rgb/detections", rr.Boxes2D(
                array=np.array([d.box for d in detections], float), array_format=rr.Box2DFormat.XYXY,
                labels=[f"{d.label} {d.conf:.2f}" for d in detections]))
        if mouth_px is not None and mouth.visible:
            self.rec.log("camera/rgb/mouth", rr.Points2D([mouth_px], radii=4, colors=[(255, 60, 60)],
                                                         labels=[f"mouth {dist_to_mouth:.2f} m"]))
        if dist_to_mouth is not None:
            self.rec.log("safety/dist_to_mouth_m", rr.Scalars(dist_to_mouth))

    def log_map(self, t, grid, pose, path_old, path_new, memory) -> None:
        self._t(t)
        occ = np.argwhere(grid.occ)
        if len(occ):
            pts = np.array([grid.center(i, j) for i, j in occ[::2]])
            self.rec.log("map/obstacles", rr.Points2D(pts, radii=0.03, colors=[(120, 120, 120)]))
        self.rec.log("map/robot", rr.Points2D([[pose.x, pose.y]], radii=0.18, colors=[(30, 144, 255)],
                                              labels=["robot"]))
        if path_old:
            self.rec.log("map/route_old", rr.LineStrips2D([path_old], colors=[(200, 80, 80)], radii=0.02))
        if path_new:
            self.rec.log("map/route_new", rr.LineStrips2D([path_new], colors=[(60, 200, 90)], radii=0.04))
        for label in ("glass", "organiser"):
            w = memory.where(label)
            if w is not None:
                c = memory.confidence(label, t)
                self.rec.log(f"map/memory/{label}", rr.Points2D(
                    [w[0][:2]], radii=0.08, colors=[(255, 215, 0, int(60 + 195 * c))],
                    labels=[f"{label} ({c:.0%})"]))

    def log_state(self, t, state: str, reason: str) -> None:
        self._t(t)
        self.rec.log("decisions/nodes", rr.GraphNodes(
            STATES, labels=STATES, positions=[_POS[s] for s in STATES],
            colors=[LIT if s == state else GREY for s in STATES], radii=[30 if s == state else 18 for s in STATES]))
        self.rec.log("decisions/now", rr.TextDocument(f"**{state}**\n\n{reason}", media_type="text/markdown"))

    def log_safety(self, t, decision, gemini: dict | None, verified) -> None:
        self._t(t)
        lines = [f"Safety: **{decision.action}** ({decision.reason}), max {decision.max_speed_mps:.2f} m/s"]
        if gemini:
            lines.append(f"Gemini suggests: {gemini.get('label')} - {gemini.get('plan', '')}")
        if verified:
            ok, checks = verified
            lines.append(("Verified: " if ok else "Rejected: ") + ", ".join(checks))
        self.rec.log("safety/state", rr.TextDocument("\n\n".join(lines), media_type="text/markdown"))

    def log_budget(self, t, loop_hz: float, latencies: dict, mem_mb: float, online: bool) -> None:
        self._t(t)
        self.rec.log("budget/loop_hz", rr.Scalars(loop_hz))
        self.rec.log("budget/mem_mb", rr.Scalars(mem_mb))
        for k, v in latencies.items():
            self.rec.log(f"budget/latency_ms/{k}", rr.Scalars(v))
        self.rec.log("budget/network", rr.TextDocument("online" if online else "OFFLINE - all on device"))

    def log_event(self, event) -> None:
        self._t(event.t)
        self.rec.log("events", rr.TextLog(f"[{event.kind}] {event.text}", level=_KIND_LEVEL.get(event.kind, "INFO")))

    def close(self) -> None:
        self.rec.flush()
        self.rec.disconnect()
