"""Find problems from sensor readings only.

Signatures (worked out from history/full_week against its answer key, offline):
- pump_failure:     pump status == 2 (fault). Flow 0, tank fills.
- pipeline_leak:    while the pump runs, pipeline pressure at the disposal end sits below
                    disposal + 50 psi, and less water arrives than the pump sends.
- pump_degradation: while the pump runs, discharge pressure sits below disposal + 150 psi
                    and flow drops below max_flow_bpd.

Healthy noise is about +/-3 psi (sd), so a rolling median over 6 running samples (30 min of
pumping) with a 10 psi / 8 psi threshold separates cleanly. Tune thresholds with eval/run_eval.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import Dataset, iso

LEAK_PSI = 10.0  # pipeline pressure deficit that counts as a leak
DEGRADE_PSI = 8.0  # discharge pressure deficit that counts as wear
WINDOW = 6  # running samples in the rolling median (6 x 5 min)
CLEAR_SAMPLES = 6  # healthy running samples in a row before an event counts as over
RAMP_LOOKBACK_S = 24 * 3600  # how far back to look for where slow wear began
MERGE_GAP_S = 2 * 3600  # episodes closer than this are one problem


@dataclass
class Event:
    event_id: str
    system_id: str
    facility_id: str
    asset_id: str
    category: str  # pump_failure | pipeline_leak | pump_degradation
    status: str  # ongoing | resolved
    onset_unix: int
    end_unix: int | None
    confidence: float
    evidence: list[str] = field(default_factory=list)
    level_pct: float | None = None
    hours_to_limit: dict | None = None
    recommended_actions: list[dict] = field(default_factory=list)
    est_cost: float | None = None
    plan_summary: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["onset_ts"] = iso(self.onset_unix)
        d["end_ts"] = iso(self.end_unix)
        # events.json shape from SPEC.md
        order = ["event_id", "system_id", "facility_id", "asset_id", "category", "status", "onset_ts", "onset_unix",
                 "end_ts", "end_unix", "confidence", "evidence", "level_pct", "hours_to_limit",
                 "recommended_actions", "est_cost"]
        return {k: d[k] for k in order}


def _runs(mask: pd.Series) -> list[tuple[int, int | None]]:
    """(start_ts, end_ts or None if still true at the end) for each run of True."""
    out, start = [], None
    idx = mask.index
    vals = mask.to_numpy()
    for i, v in enumerate(vals):
        if v and start is None:
            start = int(idx[i])
        elif not v and start is not None:
            out.append((start, int(idx[i])))
            start = None
    if start is not None:
        out.append((start, None))
    return out


def _ramp_onset(d: pd.Series, first_flag: int, threshold: float) -> int:
    """Wear grows slowly, so the first flag lags the real start. Fit a hinge
    deficit = slope * max(0, t - t0) to the running samples around the flag and return t0."""
    win = d.loc[first_flag - RAMP_LOOKBACK_S: first_flag + 2 * 3600]
    if len(win) < WINDOW * 2:
        return first_flag
    t = win.index.to_numpy(dtype=float)
    y = win.to_numpy(dtype=float)
    best_t0, best_err = first_flag, np.inf
    for t0 in t[t <= first_flag]:
        x = np.clip(t - t0, 0, None)
        denom = (x * x).sum()
        if denom == 0:
            continue
        slope = (x * y).sum() / denom
        err = ((y - slope * x) ** 2).sum()
        if slope > 0 and err < best_err:
            best_t0, best_err = int(t0), err
    return best_t0


def _episodes(deficit: pd.Series, running: pd.Series, threshold: float, breaks: list[int] = (),
              onset_model: str = "step") -> list[dict]:
    """Group running-only samples where the rolling-median deficit exceeds `threshold`.

    `breaks` are timestamps that end any episode and restart the rolling window (e.g. the same pump
    failing: a failed pump gets repaired/replaced, so old wear doesn't carry over).
    An episode ends after CLEAR_SAMPLES healthy running samples in a row.
    onset_model: "step" (leak: problem appears at once) or "ramp" (wear: grows slowly).
    """
    d = deficit[running]
    if d.empty:
        return []
    segment = np.searchsorted(np.array(sorted(breaks), dtype=float), d.index.to_numpy(dtype=float), side="right")
    eps = []
    for seg_id in np.unique(segment):
        sd = d[segment == seg_id]
        med = sd.rolling(WINDOW, min_periods=WINDOW).median()
        cur, healthy = None, 0
        for ts, m in med.items():
            ts = int(ts)
            if m > threshold:
                healthy = 0
                if cur is None:
                    pos = sd.index.get_loc(ts)
                    window = sd.iloc[max(0, pos - WINDOW + 1): pos + 1]
                    first_bad = int(window[window > threshold].index[0])
                    onset = _ramp_onset(sd, first_bad, threshold) if onset_model == "ramp" else first_bad
                    cur = {"onset": onset, "end": None, "peak": 0.0, "last": 0.0}
                cur["peak"] = max(cur["peak"], float(m))
                cur["last"] = float(m)
            elif cur is not None:
                healthy += 1
                if healthy >= CLEAR_SAMPLES:
                    pos = sd.index.get_loc(ts)
                    cur["end"] = int(sd.index[pos - CLEAR_SAMPLES + 1])
                    eps.append(cur)
                    cur, healthy = None, 0
        if cur is not None:
            # a break (pump failure) closes the episode; otherwise it's still going
            nxt = sorted(breaks)[seg_id] if seg_id < len(breaks) else None
            cur["end"] = nxt
            eps.append(cur)
    return _merge(eps)


def _merge(eps: list[dict]) -> list[dict]:
    """Slow wear hovers near the threshold at first and can flap; join episodes that touch."""
    out: list[dict] = []
    for ep in sorted(eps, key=lambda e: e["onset"]):
        prev = out[-1] if out else None
        if prev and prev["end"] is not None and ep["onset"] <= prev["end"] + MERGE_GAP_S:
            prev["end"] = ep["end"]
            prev["peak"] = max(prev["peak"], ep["peak"])
            prev["last"] = ep["last"]
        else:
            out.append(ep)
    return out


def _confidence(margin: float, n_samples: int) -> float:
    """Bigger deficit and longer evidence -> more sure. Simple, explainable, tunable."""
    strength = 1 - np.exp(-max(margin, 0) / 10)
    duration = min(1.0, n_samples / 24)
    return round(float(0.5 + 0.45 * strength * (0.5 + 0.5 * duration)), 2)


def detect_events(ds: Dataset) -> list[Event]:
    events: list[Event] = []
    for fid, f in ds.facilities.items():
        sys_id = f["system_id"]
        pump, pipe = f["pump"]["id"], f["pipeline"]["id"]
        status = ds.series(pump, "status")
        running = status == 1
        disp = ds.disposal_psi(fid)

        # --- pump failure: status 2 ----------------------------------------
        failures = _runs(status == 2)
        for onset, end in failures:
            events.append(Event(
                event_id="", system_id=sys_id, facility_id=fid, asset_id=pump, category="pump_failure",
                status="ongoing" if end is None else "resolved", onset_unix=onset, end_unix=end, confidence=0.99,
                evidence=[f"{pump} reports status 2 (fault) since {iso(onset)}; flow is 0 and the tank is filling."],
            ))
        failure_onsets = [o for o, _ in failures]

        # --- pipeline leak: pressure at disposal end too low while pumping --
        p_def = (disp + 50) - ds.series(pipe, "pressure_psi")
        loss = (ds.series(pump, "flow_bpd") - ds.series(pipe, "receive_flow_bpd"))
        for ep in _episodes(p_def, running, LEAK_PSI, failure_onsets, "step"):
            n = int(((running) & (p_def.index >= ep["onset"]) & (p_def.index <= (ep["end"] or ds.now))).sum())
            recent_loss = loss[running & (loss.index >= ep["onset"])].tail(WINDOW).median()
            events.append(Event(
                event_id="", system_id=sys_id, facility_id=fid, asset_id=pipe, category="pipeline_leak",
                status="ongoing" if ep["end"] is None else "resolved", onset_unix=ep["onset"], end_unix=ep["end"],
                confidence=_confidence(ep["peak"] - LEAK_PSI, n),
                evidence=[
                    f"Pipeline pressure is {ep['last']:.0f} psi below normal ({disp + 50:.0f} psi expected while pumping).",
                    f"About {max(recent_loss, 0):.0f} bpd leaves the pump but never reaches the disposal site.",
                ],
            ))

        # --- pump degradation: discharge pressure + flow sag while pumping --
        d_def = (disp + 150) - ds.series(pump, "discharge_psi")
        ratio = ds.series(pump, "flow_bpd") / f["pump"]["max_flow_bpd"]
        for ep in _episodes(d_def, running, DEGRADE_PSI, failure_onsets, "ramp"):
            n = int(((running) & (d_def.index >= ep["onset"]) & (d_def.index <= (ep["end"] or ds.now))).sum())
            recent_ratio = ratio[running & (ratio.index >= ep["onset"])].tail(WINDOW).median()
            events.append(Event(
                event_id="", system_id=sys_id, facility_id=fid, asset_id=pump, category="pump_degradation",
                status="ongoing" if ep["end"] is None else "resolved", onset_unix=ep["onset"], end_unix=ep["end"],
                confidence=_confidence(ep["peak"] - DEGRADE_PSI, n),
                evidence=[
                    f"Discharge pressure is {ep['last']:.0f} psi below a healthy pump ({disp + 150:.0f} psi).",
                    f"Pump moves {recent_ratio:.0%} of its rated {f['pump']['max_flow_bpd']} bpd.",
                ],
            ))

    events.sort(key=lambda e: e.onset_unix)
    for i, e in enumerate(events, 1):
        e.event_id = f"EV-{i:04d}"
        e.level_pct = round(ds.at(ds.facilities[e.facility_id]["tank"]["id"], "level_pct"), 1)
        if e.category != "pump_failure":
            e.recommended_actions = [{"type": "field_inspection", "asset_id": e.asset_id}]
    return events
