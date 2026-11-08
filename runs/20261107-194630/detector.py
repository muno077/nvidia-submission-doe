"""Monitor written by the Nemotron discovery agents (attempt 3)."""
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

def pump_failure(f):
    "Alarm if pump_status==2 for 3+ steps, or pump flow near zero with inflow>0 while tank above start level for 3+ steps."
    max_flow = f['max_flow_bpd']
    # (a) explicit fault status
    cond_a = (f['pump_status'] == 2)
    persist_a = cond_a.rolling(3, min_periods=3).sum() == 3
    # (b) pump stopped (or not pumping) with near-zero flow while tank is filling AND tank level is above the start threshold (pump should be running)
    cond_b = (f['pump_flow_bpd'] < 0.05 * max_flow) & (f['facility_inflow_bpd'] > 0) & (f['tank_level_pct'] > f['start_level_pct'])
    persist_b = cond_b.rolling(3, min_periods=3).sum() == 3
    return persist_a | persist_b

def pipeline_leak(f):
    "Pipeline pressure <90% of disposal psi and flow imbalance >5% max_flow (both for 3+ steps)."
    # Scientist's exact signal: pressure < 0.90 * disposal_psi AND (pump_flow - pipeline_receive) > 0.05 * max_flow
    cond1 = f['pipeline_pressure_psi'] < (0.90 * f['disposal_psi'])
    cond2 = (f['pump_flow_bpd'] - f['pipeline_receive_flow_bpd']) > (0.05 * f['max_flow_bpd'])
    cond = cond1 & cond2
    # Framework ignores runs <15 min (3 steps) and merges gaps <2 h
    return cond

def pump_degradation(f):
    "48-step median flow ratio drops 2.5% below 96-step rolling baseline median for 12+ consecutive steps (pump running)."
    running = f['pump_status'] == 1
    ratio = (f['pump_flow_bpd'] / f['max_flow_bpd']).where(running)
    # 48-step (4 h) rolling median of flow ratio, need at least 24 running steps (2 h) in window
    med48 = ratio.rolling(48, min_periods=24).median()
    # 96-step (8 h) rolling baseline median, updated every step, need at least 48 running steps (4 h)
    baseline = ratio.rolling(96, min_periods=48).median()
    # Degradation if 48-step median is at least 2.5% below baseline
    cond = med48 < (baseline - 0.025)
    # Require condition to hold for 12 consecutive steps (1 h)
    persist = cond.rolling(12, min_periods=12).sum() == 12
    return persist.where(running)

