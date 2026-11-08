"""Countdown to shutdown and the cheapest response for each ongoing pump failure.

Implements the rules in SPEC.md "Plans":
- Trucks: all trucks in a plan are called now and arrive together after response_min, then one
  trip each every cycle_min. Each trip removes truck_capacity_bbl and costs cost_per_trip.
  Every trip must arrive before the repair ends and can't remove more than is in the tank.
- Shut-in: instant, can start late, lasts until the repair ends. Cost = lost revenue x hours.
- Do nothing: the tank trips at shutdown and ALL wells lose revenue until the repair ends.
- Shared trucks: all ongoing failures in a system share its whole fleet at "now".

Known simplification: a truck is assigned to at most one facility per plan (no chaining a
truck from one tank to the next inside the repair window). See TODO in plan_system().
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Dataset
from .detect import Event

EPS_BBL = 0.01
WINDOW_LAST = 6  # samples used for the live well rate


@dataclass
class Situation:
    """Everything the planner needs about one failing facility at "now"."""
    facility_id: str
    system_id: str
    v0: float  # bbl in tank now
    v_shut: float  # bbl at shutdown
    capacity: float
    inflow_bpd: float
    wells: list[dict]  # [{id, water_bpd, rev_per_hour}]
    repair_min: float  # minutes until the repair ends
    response_min: float
    cycle_min: float
    tripped: bool


@dataclass
class Plan:
    trucks: int = 0
    trips_per_truck: int = 0
    shut_in: list[str] = field(default_factory=list)
    shut_in_start_min: int = 0
    trips: bool = False  # does the tank still reach shutdown?
    trip_at_min: float | None = None
    cost: float = 0.0
    truck_cost: float = 0.0
    shut_in_cost: float = 0.0
    trip_cost: float = 0.0

    def summary(self) -> str:
        parts = []
        if self.trucks:
            parts.append(f"{self.trucks} truck{'s' * (self.trucks > 1)}, {self.trips_per_truck} trip{'s' * (self.trips_per_truck > 1)}")
        if self.shut_in:
            parts.append(f"shut in {', '.join(self.shut_in)} at +{self.shut_in_start_min} min")
        if self.trips:
            parts.append(f"accept shut-in at +{self.trip_at_min:.0f} min" if parts else "do nothing (accept the automatic shut-in)")
        if not parts:
            parts.append("no action needed (tank won't reach shutdown before the repair)")
        return " + ".join(parts) + f" (~${self.cost:,.0f})"

    def actions(self, facility_id: str) -> list[dict]:
        acts = []
        if self.trucks:
            acts.append({"type": "dispatch_truck", "facility_id": facility_id, "trucks": self.trucks,
                         "trips_per_truck": self.trips_per_truck})
        if self.shut_in:
            acts.append({"type": "shut_in_wells", "wells": self.shut_in, "start_after_min": self.shut_in_start_min})
        return acts or [{"type": "none"}]


# --- situation ------------------------------------------------------------

def situation(ds: Dataset, ev: Event) -> Situation:
    f = ds.facilities[ev.facility_id]
    cap = float(f["tank"]["capacity_bbl"])
    level = ds.at(f["tank"]["id"], "level_pct")
    repair_h = float(ds.ops["expected_repair_hours"]["pump_failure"])
    repair_min = (ev.onset_unix + repair_h * 3600 - ds.now) / 60
    fo = ds.facility_ops(ev.facility_id)
    tripped = level >= ds.shutdown_pct - 0.05
    wells = []
    for w in f["wells"]:
        st = ds.series(w["id"], "status").iloc[-WINDOW_LAST:]
        rate = ds.series(w["id"], "water_bpd").iloc[-WINDOW_LAST:]
        # live rate when producing; baseline when shut in (it comes back at that rate)
        water = float(rate[st == 1].mean()) if (st == 1).any() else float(w["baseline_water_bpd"])
        wells.append({"id": w["id"], "water_bpd": water, "rev_per_hour": ds.well_revenue_per_hour(w)})
    return Situation(
        facility_id=ev.facility_id, system_id=ev.system_id, v0=level / 100 * cap,
        v_shut=ds.shutdown_pct / 100 * cap, capacity=cap, inflow_bpd=sum(w["water_bpd"] for w in wells),
        wells=wells, repair_min=repair_min, response_min=float(fo["response_min"]), cycle_min=float(fo["cycle_min"]),
        tripped=tripped,
    )


def hours_to_limit(s: Situation) -> dict | None:
    """p50 plus a simple +/-3% inflow band for p10/p90."""
    if s.tripped:
        return {"p50": 0.0, "p10": 0.0, "p90": 0.0}
    if s.inflow_bpd <= 0:
        return None
    room = s.v_shut - s.v0
    h = lambda q: round(room / (q / 24), 2)
    return {"p50": h(s.inflow_bpd), "p10": h(s.inflow_bpd * 1.03), "p90": h(s.inflow_bpd * 0.97)}


# --- simulation -------------------------------------------------------------

def _arrivals(s: Situation, trips: int) -> list[float]:
    return [s.response_min + j * s.cycle_min for j in range(trips)]


def simulate(s: Situation, ds: Dataset, trucks: int, trips: int, shut: list[dict], start: float) -> Plan | None:
    """Cost of one candidate plan, or None if it breaks a rule."""
    cap_trip = float(ds.ops["emergency_trucking"]["truck_capacity_bbl"])
    price = float(ds.ops["emergency_trucking"]["cost_per_trip"])
    arr = _arrivals(s, trips) if trucks else []
    if any(a >= s.repair_min for a in arr):
        return None
    i_full = s.inflow_bpd / 1440  # bbl per minute
    w_cut = sum(w["water_bpd"] for w in shut) / 1440

    def vol(t: float, removed: float) -> float:
        cut = w_cut * max(0.0, t - start) if shut else 0.0
        return s.v0 + i_full * t - cut - removed

    # walk segment by segment; the level only rises between arrivals, so check right before each one
    removed, trip_at = 0.0, None
    for t in arr + [s.repair_min]:
        v = vol(t, removed)
        if v >= s.v_shut - EPS_BBL:
            # it crossed inside this segment (level was below shutdown at the previous check)
            trip_at = _crossing(s, vol, removed, t)
            break
        if t < s.repair_min:
            if v < trucks * cap_trip:  # can't remove more than is in the tank
                return None
            removed += trucks * cap_trip

    n_trips_used = len([a for a in arr if trip_at is None or a < trip_at])
    truck_cost = trucks * n_trips_used * price
    if trip_at is not None and n_trips_used < len(arr):
        return None  # trips after the trip are wasted: a cheaper plan with fewer trips exists
    shut_cost = 0.0
    if shut:
        end = trip_at if trip_at is not None else s.repair_min
        shut_cost = sum(w["rev_per_hour"] for w in shut) * max(0.0, end - start) / 60
    trip_cost = 0.0
    if trip_at is not None:
        trip_cost = sum(w["rev_per_hour"] for w in s.wells) * (s.repair_min - trip_at) / 60
    return Plan(trucks=trucks if arr else 0, trips_per_truck=trips if arr else 0, shut_in=[w["id"] for w in shut],
                shut_in_start_min=int(start) if shut else 0, trips=trip_at is not None, trip_at_min=trip_at,
                cost=truck_cost + shut_cost + trip_cost, truck_cost=truck_cost, shut_in_cost=shut_cost,
                trip_cost=trip_cost)


def _crossing(s: Situation, vol, removed: float, t_hi: float) -> float:
    """Binary search for the minute the level reaches shutdown, below t_hi."""
    lo, hi = 0.0, t_hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if vol(mid, removed) >= s.v_shut - EPS_BBL:
            hi = mid
        else:
            lo = mid
    return hi


def _latest_start(s: Situation, ds: Dataset, trucks: int, trips: int, shut: list[dict]) -> float | None:
    """Latest whole minute to start this shut-in so the tank never trips (None if impossible/unneeded)."""
    cap_trip = float(ds.ops["emergency_trucking"]["truck_capacity_bbl"])
    arr = _arrivals(s, trips) if trucks else []
    w_cut = sum(w["water_bpd"] for w in shut) / 1440
    if w_cut <= 0:
        return None
    i_full = s.inflow_bpd / 1440
    latest, removed, needed = math.inf, 0.0, False
    for t in arr + [s.repair_min]:
        excess = s.v0 + i_full * t - removed - (s.v_shut - 2 * EPS_BBL)
        if excess > 0:
            needed = True
            latest = min(latest, t - excess / w_cut)
        if t < s.repair_min:
            removed += trucks * cap_trip
    if not needed:
        return None
    return math.floor(latest) if latest >= 0 else None


# --- search -----------------------------------------------------------------

def best_plans(s: Situation, ds: Dataset, max_trucks: int) -> dict[str, Plan]:
    """Cheapest plan overall plus the best of each kind, using at most `max_trucks` trucks."""
    if s.tripped:
        p = Plan(trips=True, trip_at_min=0.0,
                 trip_cost=sum(w["rev_per_hour"] for w in s.wells) * s.repair_min / 60)
        p.cost = p.trip_cost
        return {"best": p, "do_nothing": p}

    cands: list[Plan] = []
    max_trips = max(0, math.ceil((s.repair_min - s.response_min) / s.cycle_min)) if s.cycle_min else 0
    subsets = [list(c) for r in range(1, len(s.wells) + 1) for c in itertools.combinations(s.wells, r)]
    for trucks in range(0, max_trucks + 1):
        for trips in ([0] if trucks == 0 else range(1, max_trips + 1)):
            base = simulate(s, ds, trucks, trips, [], 0)
            if base is not None:
                cands.append(base)
            for sub in subsets:
                start = _latest_start(s, ds, trucks, trips, sub)
                if start is None:
                    continue
                p = simulate(s, ds, trucks, trips, sub, start)
                if p is not None and not p.trips:
                    cands.append(p)
    out = {}
    out["best"] = min(cands, key=lambda p: (p.cost, p.trucks * p.trips_per_truck))
    out["do_nothing"] = next(p for p in cands if p.trucks == 0 and not p.shut_in)
    trucks_only = [p for p in cands if p.trucks and not p.shut_in]
    shut_only = [p for p in cands if p.shut_in and not p.trucks]
    if trucks_only:
        out["trucks_only"] = min(trucks_only, key=lambda p: p.cost)
    if shut_only:
        out["shut_in_only"] = min(shut_only, key=lambda p: p.cost)
    return out


def plan_system(ds: Dataset, events: list[Event]) -> dict[str, dict]:
    """Plan every ongoing pump failure together, per system, sharing that system's fleet.

    Returns {event_id: {"situation", "hours_to_limit", "plans", "allocated_trucks"}}.
    TODO: let a truck that finishes at one tank go on to another inside the repair window.
    """
    out: dict[str, dict] = {}
    by_sys: dict[str, list[Event]] = {}
    for e in events:
        if e.category == "pump_failure" and e.status == "ongoing":
            by_sys.setdefault(e.system_id, []).append(e)

    for sys_id, evs in by_sys.items():
        fleet = ds.fleet_size(sys_id)
        sits = {e.event_id: situation(ds, e) for e in evs}
        cache: dict[tuple[str, int], dict[str, Plan]] = {}

        def plans_for(eid: str, k: int) -> dict[str, Plan]:
            if (eid, k) not in cache:
                cache[(eid, k)] = best_plans(sits[eid], ds, k)
            return cache[(eid, k)]

        best_alloc, best_cost = None, math.inf
        for alloc in itertools.product(range(fleet + 1), repeat=len(evs)):
            if sum(alloc) > fleet:
                continue
            cost = sum(plans_for(e.event_id, k)["best"].cost for e, k in zip(evs, alloc))
            if cost < best_cost - 1e-6:
                best_alloc, best_cost = alloc, cost
        for e, k in zip(evs, best_alloc):
            out[e.event_id] = {"situation": sits[e.event_id], "hours_to_limit": hours_to_limit(sits[e.event_id]),
                               "plans": plans_for(e.event_id, k), "allocated_trucks": k, "fleet": fleet,
                               "shares_with": [o.event_id for o in evs if o is not e]}
    return out


def analyze(ds: Dataset, events: list[Event]) -> list[Event]:
    """Fill countdowns, plans and costs into the events, then sort them like the alerts view."""
    planned = plan_system(ds, events)
    for e in events:
        if e.event_id in planned:
            p = planned[e.event_id]
            best: Plan = p["plans"]["best"]
            e.hours_to_limit = p["hours_to_limit"]
            e.recommended_actions = best.actions(e.facility_id)
            e.est_cost = round(best.cost, 2)
            e.plan_summary = best.summary()
    return sort_alerts(ds, events)


def sort_alerts(ds: Dataset, events: list[Event]) -> list[Event]:
    """SPEC alerts view: ongoing pump failures by smallest hours_to_limit (ties: higher lost revenue/hour),
    then other ongoing problems, then resolved ones."""
    def key(e: Event):
        if e.status != "ongoing":
            return (2, 0, 0, -e.onset_unix)
        if e.category == "pump_failure":
            htl = (e.hours_to_limit or {}).get("p50")
            return (0, htl if htl is not None else math.inf, -ds.facility_revenue_per_hour(e.facility_id), 0)
        return (1, 0, 0, e.onset_unix)
    return sorted(events, key=key)


def projection(s: Situation, ds: Dataset, plan: Plan, step_min: float = 5) -> pd.DataFrame:
    """Tank level (%) from now until the repair ends, if this plan is followed. For the what-if chart."""
    cap_trip = float(ds.ops["emergency_trucking"]["truck_capacity_bbl"])
    arr = _arrivals(s, plan.trips_per_truck) if plan.trucks else []
    shut_rate = sum(w["water_bpd"] for w in s.wells if w["id"] in plan.shut_in) / 1440
    i_full = s.inflow_bpd / 1440
    t, v, rows, ai, tripped = 0.0, s.v0, [], 0, s.tripped
    while t <= s.repair_min + 1e-9:
        rows.append({"minutes": t, "level_pct": v / s.capacity * 100})
        dt = min(step_min, s.repair_min - t) if t < s.repair_min else step_min
        if dt <= 0:
            break
        rate = 0.0 if tripped else i_full - (shut_rate if plan.shut_in and t >= plan.shut_in_start_min else 0.0)
        v += rate * dt
        t += dt
        while ai < len(arr) and arr[ai] <= t + 1e-9:
            if not tripped:
                v -= plan.trucks * cap_trip
            ai += 1
        if v >= s.v_shut:
            v, tripped = s.v_shut, True
    return pd.DataFrame(rows)
