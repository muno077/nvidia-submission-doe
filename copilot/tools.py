"""Tools the Nemotron agents can call. Each returns JSON text.

They wrap the deterministic code (detect / planner) so numbers stay exact; the agents decide
which tool to call, interpret the results and explain them. Inputs are forgiving (an event id,
facility id or pump id all work; "none" means no wells) because models mix these up.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from .data import Dataset, iso
from .detect import Event
from .planner import Plan, plan_system, simulate, situation


def _plan_json(p: Plan) -> dict:
    return {"summary": p.summary(), "cost": round(p.cost, 2), "trucks": p.trucks, "trips_per_truck": p.trips_per_truck,
            "shut_in_wells": p.shut_in, "shut_in_start_min": p.shut_in_start_min, "tank_still_trips": p.trips,
            "trip_at_min": None if p.trip_at_min is None else round(p.trip_at_min)}


def _wells(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        x = [w for w in x.replace(";", ",").split(",")]
    return [w.strip() for w in x if w and w.strip().lower() not in ("none", "null", "[]", "")]


class Toolbox:
    """Shared state for one copilot run: events, the optimizer's plans, and plans the Planner submits."""

    def __init__(self, ds: Dataset, events: list[Event]):
        self.ds, self.events = ds, events
        self.failures = [e for e in events if e.category == "pump_failure" and e.status == "ongoing"]
        self.planned = plan_system(ds, events)
        self.submitted: dict[str, dict] = {}  # event_id -> plan the Planner submitted

    def resolve(self, ref: str) -> Event | None:
        """Find an ongoing pump failure by event id, facility id or pump id."""
        ref = (ref or "").strip()
        for e in self.failures:
            if ref in (e.event_id, e.facility_id, e.asset_id):
                return e
        return None

    def options(self, e: Event) -> dict:
        p = self.planned[e.event_id]
        s = p["situation"]
        return {
            "event_id": e.event_id, "facility_id": e.facility_id, "system_id": e.system_id,
            "level_pct": round(s.v0 / s.capacity * 100, 1), "hours_to_limit": p["hours_to_limit"],
            "repair_hours_left": round(s.repair_min / 60, 2), "fleet_size": p["fleet"],
            "trucks_allocated_here": p["allocated_trucks"], "shares_fleet_with": p["shares_with"],
            "options": {k: _plan_json(v) for k, v in p["plans"].items()},
        }

    def check(self, e: Event, trucks: int, trips: int, wells: list[str], start: int) -> dict:
        fleet = self.ds.fleet_size(e.system_id)
        if trucks > fleet:
            return {"valid": False, "reason": f"{trucks} trucks > fleet of {fleet}"}
        s = situation(self.ds, e)
        unknown = [w for w in wells if w not in {x["id"] for x in s.wells}]
        if unknown:
            return {"valid": False, "reason": f"wells {unknown} are not at {e.facility_id}"}
        p = simulate(s, self.ds, trucks, trips if trucks else 0, [w for w in s.wells if w["id"] in wells], start)
        if p is None:
            return {"valid": False, "reason": "breaks a rule: a trip arrives after the repair ends, removes more than "
                                              "the tank holds, or arrives after the tank trips"}
        return {"valid": True, **_plan_json(p)}

    def audit(self) -> list[str]:
        """Rule checks on everything submitted: missing events, invalid plans, fleet overuse per system."""
        problems = []
        for e in self.failures:
            sub = self.submitted.get(e.event_id)
            if sub is None:
                problems.append(f"{e.event_id} ({e.facility_id}) has no submitted plan.")
            elif not sub["check"]["valid"]:
                problems.append(f"{e.event_id} ({e.facility_id}) plan is invalid: {sub['check']['reason']}")
        for sys_id in {e.system_id for e in self.failures}:
            used = sum(s["trucks"] for eid, s in self.submitted.items() if s["system_id"] == sys_id)
            if used > self.ds.fleet_size(sys_id):
                problems.append(f"System {sys_id} plans use {used} trucks but the fleet has {self.ds.fleet_size(sys_id)}.")
        return problems

    # --- tools ---------------------------------------------------------------
    def tools(self) -> dict[str, list]:
        box = self

        @tool
        def list_alerts() -> str:
            """List every problem still in progress at "now", in alert order, with countdowns and recommended actions."""
            rows = [{"event_id": e.event_id, "category": e.category, "facility_id": e.facility_id,
                     "asset_id": e.asset_id, "onset": iso(e.onset_unix), "confidence": e.confidence,
                     "level_pct": e.level_pct, "hours_to_limit": e.hours_to_limit, "plan": e.plan_summary,
                     "evidence": e.evidence} for e in box.events if e.status == "ongoing"]
            return json.dumps({"now": iso(box.ds.now), "alerts": rows})

        @tool
        def get_sensor_summary(asset_id: str, metric: str, hours: float = 6.0) -> str:
            """Recent readings for one sensor: latest, min, max, mean and ~12 evenly spaced points.
            Metrics: facility inflow_bpd; well water_bpd, status; tank level_pct; pump status, flow_bpd,
            discharge_psi; pipeline receive_flow_bpd, pressure_psi."""
            ds = box.ds
            if not ds.has_series(asset_id, metric):
                return json.dumps({"error": f"no series {asset_id}/{metric}"})
            s = ds.series(asset_id, metric).loc[ds.now - int(hours * 3600):]
            step = max(1, len(s) // 12)
            pts = [{"t": iso(t), "v": round(float(v), 2)} for t, v in s.iloc[::step].items()]
            return json.dumps({"asset_id": asset_id, "metric": metric, "latest": round(float(s.iloc[-1]), 2),
                               "min": round(float(s.min()), 2), "max": round(float(s.max()), 2),
                               "mean": round(float(s.mean()), 2), "points": pts})

        @tool
        def get_facility_info(facility_id: str) -> str:
            """Network facts for a facility (or any of its asset ids): tank, pump, pipeline, wells (water rate and
            revenue/hour), truck response and cycle times, the system's fleet size and healthy pressures."""
            ds = box.ds
            fid = ds.asset_to_facility.get(facility_id)  # facility ids map to themselves
            if fid is None and box.resolve(facility_id):
                fid = box.resolve(facility_id).facility_id
            if fid not in ds.facilities:
                return json.dumps({"error": f"unknown facility {facility_id}"})
            f = ds.facilities[fid]
            disp = ds.disposal_psi(fid)
            return json.dumps({
                "facility_id": fid, "system_id": f["system_id"], "tank": f["tank"], "pump": f["pump"],
                "pipeline": f["pipeline"],
                "wells": [{**w, "revenue_per_hour": round(ds.well_revenue_per_hour(w), 2)} for w in f["wells"]],
                "trucking": ds.facility_ops(fid), "fleet_size": ds.fleet_size(f["system_id"]),
                "healthy_pipeline_psi_running": disp + 50, "healthy_discharge_psi_running": disp + 150,
                "tank_limits_pct": ds.network["defaults"]["tank_limits_pct"],
            })

        @tool
        def compare_plans(event_or_facility_id: str) -> str:
            """For an ongoing pump failure (event id, facility id or pump id): countdown, repair time left, trucks the
            system-wide optimizer allocated to it, and the cheapest plan of each kind."""
            e = box.resolve(event_or_facility_id)
            if e is None:
                return json.dumps({"error": f"no ongoing pump failure matches {event_or_facility_id}",
                                   "ongoing_pump_failures": [[x.event_id, x.facility_id] for x in box.failures]})
            return json.dumps(box.options(e))

        @tool
        def submit_plan(event_id: str, trucks: int = 0, trips_per_truck: int = 0, shut_in_wells: list[str] | str | None = None,
                        shut_in_start_min: int = 0, reason: str = "") -> str:
            """Submit your chosen plan for one ongoing pump failure. Call once per pump failure (again to replace it).
            Use trucks=0 for no trucks and an empty list for no shut-in. Returns the rule check and cost."""
            e = box.resolve(event_id)
            if e is None:
                return json.dumps({"accepted": False, "error": f"no ongoing pump failure matches {event_id}",
                                   "ongoing_pump_failures": [[x.event_id, x.facility_id] for x in box.failures]})
            wells = _wells(shut_in_wells)
            chk = box.check(e, int(trucks or 0), int(trips_per_truck or 0), wells, int(shut_in_start_min or 0))
            box.submitted[e.event_id] = {"event_id": e.event_id, "facility_id": e.facility_id, "system_id": e.system_id,
                                         "trucks": int(trucks or 0),
                                         "trips_per_truck": int(trips_per_truck or 0), "shut_in_wells": wells,
                                         "shut_in_start_min": int(shut_in_start_min or 0), "reason": reason, "check": chk}
            best = box.planned[e.event_id]["plans"]["best"]
            return json.dumps({"accepted": True, "event_id": e.event_id, "check": chk,
                               "optimizer_best_cost": round(best.cost, 2)})

        @tool
        def validate_plan(event_or_facility_id: str, trucks: int = 0, trips_per_truck: int = 0,
                          shut_in_wells: list[str] | str | None = None, shut_in_start_min: int = 0) -> str:
            """Check one plan against the SPEC rules (trips before repair ends, tank contents, fleet size) and
            simulate it. Returns whether it is valid, whether the tank still trips, and its cost."""
            e = box.resolve(event_or_facility_id)
            if e is None:
                return json.dumps({"valid": False, "reason": f"no ongoing pump failure matches {event_or_facility_id}",
                                   "ongoing_pump_failures": [[x.event_id, x.facility_id] for x in box.failures]})
            return json.dumps(box.check(e, int(trucks or 0), int(trips_per_truck or 0), _wells(shut_in_wells),
                                        int(shut_in_start_min or 0)))

        return {
            "investigator": [list_alerts, get_sensor_summary, get_facility_info],
            "planner": [compare_plans, get_facility_info, submit_plan],
            "critic": [validate_plan, compare_plans],
        }
