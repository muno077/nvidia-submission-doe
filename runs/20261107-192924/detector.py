"""Monitor written by the Nemotron discovery agents (attempt 5)."""
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np


def pump_failure(f):
    "Pump is in fault status (2); returns NaN when the pump is stopped."
    result = (f.pump_status == 2).astype("boolean")
    result[f.pump_status == 0] = pd.NA
    return result


def pipeline_leak(f):
    "Pipeline receive flow significantly lower than pump flow while the pump runs, indicating a leak."
    running = f.pump_status == 1
    diff = (f.pump_flow_bpd - f.pipeline_receive_flow_bpd).where(running)
    # Flag when the pump flow exceeds pipeline receive flow by at least 5 % of max flow capacity
    threshold = f.max_flow_bpd * 0.05
    candidate = diff > threshold
    # Require at least 15 minutes (3 five‑min steps) of sustained drop to suppress isolated spikes
    leak = candidate.rolling(3, min_periods=3).max()
    leak = leak.where(running)
    return leak.astype("boolean")


def pump_degradation(f):
    "Pump flow gradually declines while running, indicating wear."
    running = f.pump_status == 1
    # 48‑step (4 hr) rolling median as recent baseline
    baseline = f.pump_flow_bpd.rolling(48, min_periods=48).median()
    # 6‑step (30 min) rolling median of current flow
    current = f.pump_flow_bpd.rolling(6, min_periods=6).median()
    # Degrade when current flow is at least 4 % below the recent baseline
    degrade = current < baseline - f.max_flow_bpd * 0.04
    # Require persistence of at least 12 steps (1 h) to suppress the short ~1 h false‑alarm events
    degrade = degrade.rolling(12, min_periods=12).max()
    degrade = degrade.where(running)
    return degrade.astype("boolean")

