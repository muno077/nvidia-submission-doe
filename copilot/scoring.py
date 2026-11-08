"""Compare detected events with labelled ones. Used by eval/run_eval.py and the discovery agent's feedback."""

from __future__ import annotations

import statistics

from .data import iso

MATCH_WINDOW_S = 4 * 3600  # a detection counts if its onset is within 4 h of the labelled onset
CATEGORIES = ["pump_failure", "pipeline_leak", "pump_degradation"]


def match(key: list[dict], found: list, window_s: int = MATCH_WINDOW_S):
    """Greedy one-to-one matching on (asset, category, nearest onset). Returns (pairs, missed, extra)."""
    pairs, missed, used = [], [], set()
    for k in key:
        best = None
        for e in found:
            if id(e) in used or e.asset_id != k["asset"] or e.category != k["category"]:
                continue
            dt = abs(e.onset_unix - k["onset_unix"])
            if dt <= window_s and (best is None or dt < abs(best.onset_unix - k["onset_unix"])):
                best = e
        if best is None:
            missed.append(k)
        else:
            used.add(id(best))
            pairs.append((k, best))
    extra = [e for e in found if id(e) not in used]
    return pairs, missed, extra


def _nearest(k: dict, found: list) -> str:
    """Why was this labelled event missed: late, early, or never fired? (Opposite fixes, so say which.)"""
    same = [e for e in found if e.asset_id == k["asset"] and e.category == k["category"]]
    if not same:
        return "never fired on this asset"
    e = min(same, key=lambda e: abs(e.onset_unix - k["onset_unix"]))
    dt = round((e.onset_unix - k["onset_unix"]) / 60)
    return f"nearest alarm on this asset started {abs(dt)} min {'late' if dt > 0 else 'early'} (must be within 240)"


def score(key: list[dict], found: list, examples: int = 4) -> dict:
    """Per-category recall / false alarms / onset error, overall F1, and a few concrete mistakes to study."""
    pairs, missed, extra = match(key, found)
    per = {}
    for c in CATEGORIES:
        errs = [(e.onset_unix - k["onset_unix"]) / 60 for k, e in pairs if k["category"] == c]
        per[c] = {"labelled": sum(k["category"] == c for k in key), "found": len(errs),
                  "false_alarms": sum(e.category == c for e in extra),
                  "median_onset_error_min": round(statistics.median(errs)) if errs else None}
    tp, fn, fp = len(pairs), len(missed), len(extra)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": round(f1, 3), "precision": round(precision, 3), "recall": round(recall, 3),
        "found": tp, "labelled": len(key), "false_alarms": fp, "per_category": per,
        "missed_examples": [{"category": k["category"], "asset": k["asset"], "onset": k["onset"], "end": k.get("end"),
                             "your_rule": _nearest(k, found)} for k in missed[:examples]],
        "false_alarm_examples": [{"category": e.category, "asset": e.asset_id, "onset": iso(e.onset_unix),
                                  "end": iso(e.end_unix)} for e in extra[:examples]],
    }
