"""Load a dataset folder (network.json, ops.json, telemetry.csv).

Never loads answer_key.json: that file is for the offline eval only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STEP_S = 300  # telemetry is on a 5-minute grid


def iso(ts_unix: float | int | None) -> str | None:
    if ts_unix is None:
        return None
    return datetime.fromtimestamp(int(ts_unix), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Dataset:
    folder: Path
    network: dict
    ops: dict
    wide: pd.DataFrame  # index ts_unix, columns MultiIndex (asset_id, metric)
    systems: dict = field(default_factory=dict)  # system_id -> system dict
    facilities: dict = field(default_factory=dict)  # facility_id -> facility dict (with "system_id")
    asset_to_facility: dict = field(default_factory=dict)  # any asset id -> facility_id

    @property
    def start(self) -> int:
        return int(self.wide.index[0])

    @property
    def now(self) -> int:
        """The last timestamp in the telemetry is "now"."""
        return int(self.wide.index[-1])

    @property
    def shutdown_pct(self) -> float:
        return float(self.network["defaults"]["tank_limits_pct"]["shutdown"])

    @property
    def warning_pct(self) -> float:
        return float(self.network["defaults"]["tank_limits_pct"]["warning"])

    def series(self, asset_id: str, metric: str) -> pd.Series:
        return self.wide[(asset_id, metric)]

    def has_series(self, asset_id: str, metric: str) -> bool:
        return (asset_id, metric) in self.wide.columns

    def at(self, asset_id: str, metric: str, ts: int | None = None) -> float:
        s = self.series(asset_id, metric)
        return float(s.iloc[-1] if ts is None else s.loc[:ts].iloc[-1])

    def until(self, now: int) -> "Dataset":
        """A copy of this dataset cut off at `now` (time travel)."""
        cut = Dataset(self.folder, self.network, self.ops, self.wide.loc[:now])
        cut.systems, cut.facilities, cut.asset_to_facility = self.systems, self.facilities, self.asset_to_facility
        return cut

    # --- network helpers -------------------------------------------------
    def facility_ops(self, facility_id: str) -> dict:
        sys_id = self.facilities[facility_id]["system_id"]
        return self.ops["systems"][sys_id]["facilities"][facility_id]

    def fleet_size(self, system_id: str) -> int:
        return int(self.ops["systems"][system_id]["emergency_fleet_size"])

    def disposal_psi(self, facility_id: str) -> float:
        return float(self.systems[self.facilities[facility_id]["system_id"]]["disposal_site"]["pressure_psi"])

    def well_revenue_per_hour(self, well: dict) -> float:
        p = self.ops["prices"]
        return (well["oil_bpd"] * p["oil_per_bbl"] + well["gas_mcfd"] * p["gas_per_mcf"]) / 24.0

    def facility_revenue_per_hour(self, facility_id: str) -> float:
        return sum(self.well_revenue_per_hour(w) for w in self.facilities[facility_id]["wells"])


def load_dataset(folder: str | Path) -> Dataset:
    folder = Path(folder)
    network = json.loads((folder / "network.json").read_text())
    ops = json.loads((folder / "ops.json").read_text())
    long = pd.read_csv(folder / "telemetry.csv", usecols=["ts_unix", "asset_id", "metric", "value"])
    wide = long.pivot_table(index="ts_unix", columns=["asset_id", "metric"], values="value").sort_index()

    ds = Dataset(folder, network, ops, wide)
    for s in network["systems"]:
        ds.systems[s["id"]] = s
        for f in s["facilities"]:
            f = {**f, "system_id": s["id"]}
            ds.facilities[f["id"]] = f
            for aid in [f["id"], f["tank"]["id"], f["pump"]["id"], f["pipeline"]["id"], *(w["id"] for w in f["wells"])]:
                ds.asset_to_facility[aid] = f["id"]
    return ds


def find_datasets(root: str | Path) -> list[Path]:
    """Every folder under `root` that has the three dataset files."""
    root = Path(root)
    if not root.exists():
        return []
    hits = {p.parent for p in root.rglob("telemetry.csv")}
    return sorted(p for p in hits if (p / "network.json").exists() and (p / "ops.json").exists())
