"""Time-window guard for ops (spec §11): 02:30–04:30 access, stop or checkpoint by 04:25; injectable clock for tests."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from typing import Callable


@dataclass
class Window:
    start: dtime = dtime(2, 30)
    deadline: dtime = dtime(4, 25)
    end: dtime = dtime(4, 30)
    clock: Callable[[], datetime] = datetime.now

    def _at(self, now, t):
        return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)

    def inside(self) -> bool:
        now = self.clock()
        return self._at(now, self.start) <= now < self._at(now, self.end)

    def seconds_left(self) -> float:
        """Seconds to the 04:25 deadline; 0 or less outside the usable part of the window."""
        now = self.clock()
        if now < self._at(now, self.start):
            return 0.0
        return (self._at(now, self.deadline) - now).total_seconds()

    def may_start(self, estimate_s: float, checkpoint_s: float | None = None) -> bool:
        """Refuse a job that cannot finish before the deadline, or reach its first checkpoint if it has one."""
        if not self.inside():
            return False
        left = self.seconds_left()
        if left <= 0:
            return False
        need = estimate_s if checkpoint_s is None else min(estimate_s, checkpoint_s)
        return need <= left

    def must_stop(self, margin_s: float = 0.0) -> bool:
        return self.seconds_left() <= margin_s

    def next_start(self) -> datetime:
        now = self.clock()
        s = self._at(now, self.start)
        return s if now < s else s + timedelta(days=1)
