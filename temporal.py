"""Temporal decision layer (pure Python - no ML dependencies).
Combines per-frame eye-closed / yawning flags into OK / CAUTION / ALERT using:
  * continuous eye closure  >= closure_alert_s    (a blink is ~0.1-0.4 s, so it is ignored)
  * PERCLOS (fraction of time eyes closed in the last window_s seconds) >= perclos_thr
  * yawns (sustained >= yawn_min_s) in the last yawn_window_s >= yawn_alert_count
"""
from collections import deque
from dataclasses import dataclass


@dataclass
class MonitorConfig:
    window_s: float = 10.0
    perclos_thr: float = 0.40
    closure_alert_s: float = 1.5
    yawn_min_s: float = 1.0
    yawn_window_s: float = 60.0
    yawn_alert_count: int = 2


class DrowsinessMonitor:
    def __init__(self, cfg=None):
        self.cfg = cfg or MonitorConfig()
        self.reset()

    def reset(self):
        self.eye, self.yawns = deque(), deque()
        self._closed_since = self._yawn_since = None
        self._yawn_counted = False

    def update(self, t, eye_closed, yawning):
        c = self.cfg
        self.eye.append((t, 1 if eye_closed else 0))
        while self.eye and t - self.eye[0][0] > c.window_s:
            self.eye.popleft()
        perclos = sum(v for _, v in self.eye) / len(self.eye)
        span = t - self.eye[0][0]

        if eye_closed:
            self._closed_since = self._closed_since if self._closed_since is not None else t
            closure = t - self._closed_since
        else:
            self._closed_since, closure = None, 0.0

        if yawning:
            if self._yawn_since is None:
                self._yawn_since, self._yawn_counted = t, False
            if not self._yawn_counted and t - self._yawn_since >= c.yawn_min_s:
                self.yawns.append(t)
                self._yawn_counted = True
        else:
            self._yawn_since, self._yawn_counted = None, False
        while self.yawns and t - self.yawns[0] > c.yawn_window_s:
            self.yawns.popleft()

        reasons = []
        if closure >= c.closure_alert_s:
            reasons.append("prolonged eye closure")
        if perclos >= c.perclos_thr and span >= c.window_s * 0.5:
            reasons.append("high PERCLOS")
        if len(self.yawns) >= c.yawn_alert_count:
            reasons.append("repeated yawning")

        if reasons:
            state = "ALERT"
        elif len(self.yawns) == 1 or (perclos >= c.perclos_thr / 2 and span >= c.window_s * 0.5):
            state = "CAUTION"
        else:
            state = "OK"
        return dict(state=state, perclos=perclos, closure_s=closure,
                    yawns=len(self.yawns), reasons=reasons)
