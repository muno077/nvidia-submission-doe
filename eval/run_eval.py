"""Score the detector and planner against the practice answer keys (offline only).

    uv run python eval/run_eval.py                 # every folder under data/ with an answer key
    uv run python eval/run_eval.py data/history/snapshot_2026-10-08T1200

The app itself never reads answer_key.json. This script is the only place that does.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.data import find_datasets, iso, load_dataset  # noqa: E402
from copilot.detect import detect_events  # noqa: E402
from copilot.planner import analyze  # noqa: E402
from copilot.scoring import match  # noqa: E402


def cost_of(ref: str) -> float | None:
    m = re.search(r"~\$([\d,]+)", ref or "")
    return float(m.group(1).replace(",", "")) if m else (0.0 if (ref or "").startswith("none") else None)


def evaluate(folder: Path) -> dict:
    key = json.loads((folder / "answer_key.json").read_text())
    ds = load_dataset(folder)
    events = analyze(ds, detect_events(ds))
    pairs, missed, extra = match(key, events)

    print(f"\n=== {folder.name}  (now = {iso(ds.now)}) ===")
    print(f"detected {len(pairs)}/{len(key)} labelled events, {len(extra)} false alarms")
    for k, e in pairs:
        err = (e.onset_unix - k["onset_unix"]) / 60
        status_ok = "ok" if e.status == k["status"] else f"STATUS {e.status} vs {k['status']}"
        print(f"  {k['id']:<9} {k['category']:<17} {k['asset']:<15} onset err {err:+6.0f} min  {status_ok}")
    for k in missed:
        print(f"  MISSED  {k['id']:<9} {k['category']:<17} {k['asset']} onset {k['onset']}")
    for e in extra:
        print(f"  FALSE   {e.category:<17} {e.asset_id} onset {iso(e.onset_unix)} conf {e.confidence}")

    # ongoing pump failures: countdown, plan cost, and alerts order
    ongoing = [(k, e) for k, e in pairs if k["category"] == "pump_failure" and k["status"] == "ongoing"]
    if ongoing:
        print("  ongoing pump failures (ours vs answer key):")
        our_order = [e.event_id for e in events if e.category == "pump_failure" and e.status == "ongoing"]
        for k, e in ongoing:
            ref_cost = cost_of(k["reference_action"])
            ours = e.hours_to_limit["p50"] if e.hours_to_limit else None
            print(f"    {k['id']:<9} hours_to_limit {ours} vs {k.get('hours_to_limit_final')} | "
                  f"cost ${e.est_cost:,.0f} vs ${ref_cost:,.0f} | order {our_order.index(e.event_id) + 1} vs {k.get('order')}")
            print(f"              ours: {e.plan_summary}")
            print(f"              key:  {k['reference_action']}")
    return {"labelled": len(key), "found": len(pairs), "false": len(extra)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "data"
    folders = [Path(a) for a in sys.argv[1:]] or [p for p in find_datasets(root) if (p / "answer_key.json").exists()]
    totals = [evaluate(f) for f in folders]
    print(f"\nTOTAL: {sum(t['found'] for t in totals)}/{sum(t['labelled'] for t in totals)} detected, "
          f"{sum(t['false'] for t in totals)} false alarms")
