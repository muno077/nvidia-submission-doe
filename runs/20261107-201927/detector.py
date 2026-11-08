"""Monitor written by the Nemotron discovery agents (attempt 8)."""
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

def pump_failure(f):
    "Tank level above start_level_pct for 20 min with sustained rise >3%/hr, or rapid 30-min rise >5 pct, while pump off."
    pump_not_running = f.pump_status != 1
    level = f.tank_level_pct
    start = f.start_level_pct.iloc[0]
    
    # Condition 1: level above start_level_pct for 4 consecutive readings (20 min) with sustained rise
    above_start = level > start
    consecutive_above = above_start.rolling(4, min_periods=4).sum() == 4
    
    # 1-hour rise rate (12 periods)
    rise_rate_1h = level.diff(12)
    sustained_rise = (rise_rate_1h > 3).rolling(4, min_periods=4).sum() == 4
    
    cond1 = pump_not_running & consecutive_above & sustained_rise
    
    # Condition 2: 30-min increase >5 pct points while already above start_level_pct
    rise_30min = level.diff(6)
    cond2 = pump_not_running & above_start & (rise_30min > 5)
    
    return (cond1 | cond2).where(pump_not_running)

def pipeline_leak(f):
    "Pipeline pressure below facility 25th percentile (pump-running, first 24h) for 2 hours while pump runs."
    pump_running = f.pump_status == 1
    pressure = f.pipeline_pressure_psi
    
    # Use first 24 hours (288 periods) of pump-running data to compute healthy 25th percentile
    early_pump_running = pump_running.iloc[:288]
    early_pressure = pressure.iloc[:288]
    healthy_pressure = early_pressure[early_pump_running]
    if len(healthy_pressure) > 10:
        threshold = healthy_pressure.quantile(0.25)
    else:
        # Fallback: disposal_psi - 75
        threshold = f.disposal_psi.iloc[0] - 75
    
    below_thresh = pressure < threshold
    consecutive_below = below_thresh.rolling(24, min_periods=24).sum() == 24
    
    return consecutive_below.where(pump_running)

def pump_degradation(f):
    "Pump flow 24-hour average decline rate >0.5%/hr sustained for 4 hours while pump runs."
    pump_running = f.pump_status == 1
    flow = f.pump_flow_bpd
    
    # 24-hour decline rate: (flow_now - flow_24h_ago) / (24 * flow_24h_ago) per hour
    flow_24h_ago = flow.shift(288)
    decline_rate_per_hour = (flow - flow_24h_ago) / (24 * flow_24h_ago)
    
    # Decline >0.5%/hour (i.e., rate < -0.005)
    significant_decline = decline_rate_per_hour < -0.005
    
    # Sustained for 48 consecutive periods (4 hours)
    sustained_decline = significant_decline.rolling(48, min_periods=48).sum() == 48
    
    return sustained_decline.where(pump_running)

