"""Run model-written Python safely-ish, and turn the agent's rules into events.

Every piece of agent code runs in a separate Python process:
- isolated interpreter (-I), its own temp folder, a time limit
- a stripped environment: the NVIDIA_API_KEY and everything else in os.environ is NOT passed in
- input goes in as a pickle we wrote; output comes back as JSON/text, never as a pickle, so a bad
  script can't plant code in the parent process
It is not a real security sandbox (it can still read files and use the network), which is fine
for a hackathon on your own machine and worth saying out loud if asked.

The agent's detector is a set of small functions, one per problem category:

    def pipeline_leak(f):
        "Pipeline pressure well below disposal + 50 psi while the pump runs."
        running = f.pump_status == 1
        deficit = f.disposal_psi + 50 - f.pipeline_pressure_psi
        return deficit.where(running).rolling(6, min_periods=6).median() > 10

`f` is one facility's time series (see FRAME_DOC). The function returns True / False per timestamp,
or NaN where it can't tell (e.g. the pump is off). The framework, not the agent, turns that into
events: carry the last known value across NaN gaps, drop blips, merge runs that nearly touch.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset, iso
from .detect import Event, _runs

CATEGORIES = ["pump_failure", "pipeline_leak", "pump_degradation"]
CATEGORY_ASSET = {"pump_failure": "pump", "pipeline_leak": "pipeline", "pump_degradation": "pump"}
MIN_SAMPLES = 3  # a run must last 15 min to count
MERGE_GAP_S = 2 * 3600  # runs closer than this are one event

FRAME_DOC = """frames: dict facility_id -> DataFrame indexed by ts_unix (integer unix SECONDS, 5-minute steps;
  slice with f.loc[ts("2026-10-05T06:00:00Z"):], print times with iso(t)). Columns:
  tank_level_pct, facility_inflow_bpd, wells_water_bpd (sum), wells_producing (count),
  pump_status (0 stopped, 1 running, 2 fault), pump_flow_bpd, pump_discharge_psi,
  pipeline_receive_flow_bpd, pipeline_pressure_psi,
  constants per facility: capacity_bbl, max_flow_bpd, start_level_pct, stop_level_pct,
  disposal_psi, warning_pct, shutdown_pct.
facilities: list of {facility_id, system_id, pump_id, pipeline_id, tank_id, wells}.
labels: training events [{category, facility, asset, onset, onset_unix, end, end_unix}].
helpers: ts("2026-10-05T06:00:00Z") -> unix seconds; iso(unix) -> string."""

PRELUDE = '''
import json, math, sys, warnings
import pickle
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 20)
with open(sys.argv[1], "rb") as _fh:
    _d = pickle.load(_fh)
frames, facilities, labels = _d["frames"], _d["facilities"], _d["labels"]
def ts(s): return int(pd.Timestamp(s).timestamp())
def iso(u): return pd.Timestamp(int(u), unit="s").strftime("%Y-%m-%dT%H:%M:%SZ")
'''

DETECTOR_TAIL = '''
_out = {"masks": {}, "docs": {}, "errors": {}}
for _cat in ["pump_failure", "pipeline_leak", "pump_degradation"]:
    _fn = globals().get(_cat)
    if _fn is None:
        _out["errors"][_cat] = "function not defined"
        continue
    _out["docs"][_cat] = (_fn.__doc__ or "").strip()
    for _fid, _f in frames.items():
        try:
            _r = _fn(_f.copy())
            _r = pd.Series(_r, index=_f.index) if not isinstance(_r, pd.Series) else _r.reindex(_f.index)
            _v = pd.to_numeric(_r, errors="coerce").to_numpy(dtype=float)
            _codes = np.where(np.isnan(_v), -1, (_v != 0).astype(int))
            _out["masks"].setdefault(_fid, {})[_cat] = _codes.tolist()
        except Exception as _e:
            _out["errors"][_cat] = f"{type(_e).__name__}: {_e}"
            break
with open(sys.argv[2], "w") as _fh:
    json.dump(_out, _fh)
'''


COMPARE_TEMPLATE = '''
expr, cat, running_only, smooth = {expr}, {cat}, {running}, {smooth}
parts = []
for fid, f in frames.items():
    v = f.eval(expr).astype(float)
    if running_only:
        v = v.where(f.pump_status == 1)
    if smooth > 1:  # rolling median over the last `smooth` valid samples
        v = v.dropna().rolling(smooth, min_periods=smooth).median().reindex(f.index)
    inside = pd.Series(False, index=f.index); early = inside.copy(); other = inside.copy()
    for l in labels:
        if l["facility"] != fid:
            continue
        end = l["end_unix"] or int(f.index[-1]) + 300
        if l["category"] == cat:
            inside.loc[l["onset_unix"]: end - 1] = True
            early.loc[l["onset_unix"]: l["onset_unix"] + 2 * 3600 - 1] = True
        else:
            other.loc[l["onset_unix"]: end - 1] = True
    parts.append(pd.DataFrame({{"v": v, "inside": inside, "early": early, "skip": other & ~inside}}))
d = pd.concat(parts).dropna(subset=["v"])
d = d[~d.skip]
pin, pout, pearly = d.v[d.inside], d.v[~d.inside], d.v[d.early]
def q(s):
    return "n=%d p5=%.2f p25=%.2f median=%.2f p75=%.2f p95=%.2f" % ((len(s),) + tuple(s.quantile([.05, .25, .5, .75, .95]))) if len(s) else "n=0"
print("signal:", expr, "| problem:", cat, "| pump running only" if running_only else "", "| rolling median %d" % smooth if smooth > 1 else "")
print("HEALTHY        ", q(pout))
print("DURING PROBLEM ", q(pin))
print("FIRST 2 HOURS  ", q(pearly))
best = None
if len(pin) and len(pout):
    for t in np.unique(np.quantile(d.v, np.linspace(0.01, 0.99, 99))):
        for sign in (1, -1):
            hit = (sign * pin > sign * t).mean(); fa = (sign * pout > sign * t).mean()
            if best is None or hit - fa > best[0]:
                best = (hit - fa, t, sign, hit, fa)
    _, t, sign, hit, fa = best
    print("best single threshold: value %s %.2f -> catches %.0f%% of problem samples, %.2f%% of healthy samples"
          % (">" if sign > 0 else "<", t, 100 * hit, 100 * fa))
'''


def facility_frames(ds: Dataset, until: int | None = None) -> dict[str, pd.DataFrame]:
    """One tidy DataFrame per facility with friendly column names."""
    w = ds.wide if until is None else ds.wide.loc[: until - 1]
    out = {}
    for fid, f in ds.facilities.items():
        p, pl, t = f["pump"]["id"], f["pipeline"]["id"], f["tank"]["id"]
        well_ids = [x["id"] for x in f["wells"]]
        df = pd.DataFrame({
            "tank_level_pct": w[(t, "level_pct")],
            "facility_inflow_bpd": w[(fid, "inflow_bpd")],
            "wells_water_bpd": sum(w[(i, "water_bpd")] for i in well_ids),
            "wells_producing": sum(w[(i, "status")] for i in well_ids),
            "pump_status": w[(p, "status")],
            "pump_flow_bpd": w[(p, "flow_bpd")],
            "pump_discharge_psi": w[(p, "discharge_psi")],
            "pipeline_receive_flow_bpd": w[(pl, "receive_flow_bpd")],
            "pipeline_pressure_psi": w[(pl, "pressure_psi")],
        })
        df["capacity_bbl"] = f["tank"]["capacity_bbl"]
        df["max_flow_bpd"] = f["pump"]["max_flow_bpd"]
        df["start_level_pct"] = f["pump"]["start_level_pct"]
        df["stop_level_pct"] = f["pump"]["stop_level_pct"]
        df["disposal_psi"] = ds.disposal_psi(fid)
        df["warning_pct"] = ds.warning_pct
        df["shutdown_pct"] = ds.shutdown_pct
        out[fid] = df
    return out


def facility_meta(ds: Dataset) -> list[dict]:
    return [{"facility_id": fid, "system_id": f["system_id"], "pump_id": f["pump"]["id"],
             "pipeline_id": f["pipeline"]["id"], "tank_id": f["tank"]["id"], "wells": [w["id"] for w in f["wells"]]}
            for fid, f in ds.facilities.items()]


class Sandbox:
    """A temp folder holding one dataset (optionally cut at `until`) for agent code to run against."""

    def __init__(self, ds: Dataset, labels: list[dict] | None = None, until: int | None = None,
                 systems: set[str] | None = None):
        """`systems` limits what the agent can see (e.g. only the training systems)."""
        self.ds, self.until = ds, until
        self.dir = Path(tempfile.mkdtemp(prefix="discovery_"))
        keep = lambda fid: systems is None or ds.facilities[fid]["system_id"] in systems
        self.frames = {fid: f for fid, f in facility_frames(ds, until).items() if keep(fid)}
        meta = [m for m in facility_meta(ds) if keep(m["facility_id"])]
        self.data = self.dir / "data.pkl"
        with open(self.data, "wb") as fh:
            pickle.dump({"frames": self.frames, "facilities": meta, "labels": labels or []}, fh)

    def _run(self, script: str, timeout: int, out: Path | None = None) -> tuple[int, str]:
        path = self.dir / "script.py"
        path.write_text(script)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.dir), "LANG": "en_US.UTF-8"}  # no API keys
        try:
            r = subprocess.run([sys.executable, "-I", str(path), str(self.data), str(out or "")],
                               cwd=self.dir, env=env, capture_output=True, text=True, timeout=timeout)
            return r.returncode, (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
        except subprocess.TimeoutExpired:
            return -1, f"TIMEOUT after {timeout}s: make the code faster (vectorize, fewer loops)."

    def run_python(self, code: str, timeout: int = 60, max_chars: int = 4000) -> str:
        """Run exploration code; return what it printed (truncated)."""
        rc, text = self._run(PRELUDE + "\n" + code, timeout)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars; print less]"
        return text or "(no output: use print())"

    def compare_signal(self, expression: str, category: str, running_only: bool = True, smooth: int = 1) -> str:
        """Hypothesis test: how well does `expression` (a pandas expression over the frame columns) separate
        timestamps inside labelled `category` events from healthy ones? Runs in the sandbox like any agent code."""
        code = COMPARE_TEMPLATE.format(expr=repr(expression), cat=repr(category), running=running_only, smooth=int(smooth))
        return self.run_python(code, max_chars=2500)

    def run_detector(self, code: str, timeout: int = 120) -> dict:
        """Run the agent's rule functions on every facility; return masks, docstrings and errors."""
        out = self.dir / "result.json"
        out.unlink(missing_ok=True)
        rc, text = self._run(PRELUDE + "\n" + code + "\n" + DETECTOR_TAIL, timeout, out)
        if not out.exists():
            return {"masks": {}, "docs": {}, "errors": {"script": text[-1500:] or f"exit code {rc}"}}
        return json.loads(out.read_text())


def masks_to_events(ds: Dataset, result: dict, until: int | None = None) -> list[Event]:
    """The framework's half of the deal: per-timestamp verdicts -> events with onset/end."""
    events: list[Event] = []
    for fid, cats in result.get("masks", {}).items():
        f = ds.facilities[fid]
        index = ds.wide.index if until is None else ds.wide.loc[: until - 1].index
        for cat, codes in cats.items():
            if len(codes) != len(index):
                continue
            s = pd.Series(codes, index=index, dtype=float).replace(-1, np.nan).ffill().fillna(0).astype(bool)
            runs = []
            for start, end in _runs(s):
                last = end if end is not None else int(index[-1]) + 300
                if last - start >= MIN_SAMPLES * 300:
                    runs.append([start, end])
            merged: list[list] = []
            for start, end in runs:
                if merged and merged[-1][1] is not None and start - merged[-1][1] <= MERGE_GAP_S:
                    merged[-1][1] = end
                else:
                    merged.append([start, end])
            asset = f[CATEGORY_ASSET[cat]]["id"]
            doc = result.get("docs", {}).get(cat) or f"agent rule {cat}"
            for start, end in merged:
                hours = ((end or int(index[-1])) - start) / 3600
                events.append(Event(
                    event_id="", system_id=f["system_id"], facility_id=fid, asset_id=asset, category=cat,
                    status="ongoing" if end is None else "resolved", onset_unix=int(start),
                    end_unix=None if end is None else int(end),
                    confidence=round(min(0.99, 0.6 + 0.4 * min(1.0, hours / 6)), 2),
                    evidence=[f"Nemotron-discovered rule: {doc}", f"Rule has held since {iso(start)}."],
                ))
    events.sort(key=lambda e: e.onset_unix)
    for i, e in enumerate(events, 1):
        e.event_id = f"AG-{i:04d}"
        e.level_pct = round(ds.at(ds.facilities[e.facility_id]["tank"]["id"], "level_pct"), 1)
        if e.category != "pump_failure":
            e.recommended_actions = [{"type": "field_inspection", "asset_id": e.asset_id}]
    return events


def detect_with_agent_rules(ds: Dataset, code: str) -> tuple[list[Event], dict]:
    """Run a saved detector.py on any dataset (e.g. the unseen test/ network)."""
    box = Sandbox(ds)
    result = box.run_detector(code)
    return masks_to_events(ds, result), result.get("errors", {})
