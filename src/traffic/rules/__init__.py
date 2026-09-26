"""Rule checker: turns trajectories + scene map + signal states into raw events."""
from __future__ import annotations

from .base import Context, RawEvent
from .collisions import accident, near_miss
from .direction import solid_line_crossing, turns, wrong_way
from .occupancy import congestion, road_obstacle, stopped_vehicle
from .pedestrians import failure_to_yield, jaywalking
from .signal_rules import red_light, stop_line

# rule name in params["rules"] -> function(ctx) -> list[RawEvent]
SIMPLE_RULES = {
    "stopped_vehicle": stopped_vehicle,
    "congestion": congestion,
    "road_obstacle": road_obstacle,
    "wrong_way": wrong_way,
    "turns": turns,
    "solid_line_crossing": solid_line_crossing,
    "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield,
    "red_light": red_light,
    "stop_line": stop_line,
    "accident": accident,
}


def run_rules(ctx: Context) -> list[RawEvent]:
    events: list[RawEvent] = []
    for name, fn in SIMPLE_RULES.items():
        if ctx.rule(name).get("enabled", True):
            events += fn(ctx)
    if ctx.rule("near_miss").get("enabled", True):
        events += near_miss(ctx, [e for e in events if e.label == "accident"])
    return events


__all__ = ["Context", "RawEvent", "run_rules"]
