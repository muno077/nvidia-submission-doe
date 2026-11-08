# Model Card: Industrial Oil/Water Production Network Monitor

### 1. What the Agents Learned from Docs and Data
The team ingested `SPEC.md`, `network.json`, and `examples.md` to encode physical operation baselines and problem signatures. Key learnings include:
- **Normal operation** follows a sawtooth cycle: tank level cycles between `start_level_pct` and `stop_level_pct`, pumped at `max_flow_bpd`, with wells sustaining continuous `oil_bpd`, `gas_mcfd`, `baseline_water_bpd`. Telemetry samples every 5 min (`ts_unix` +300, base `1791158400`).
- **Problem signatures** were mapped to reading changes: pump failure = pump stops, tank level rises to `shutdown` limit; pipeline leak = `pipeline_pressure_psi` drops below disposal setpoint, flow imbalance; pump degradation = gradual `pump_flow_bpd` decline relative to `max_flow_bpd`.
- All field names, ranges, and sampling intervals were derived directly from the domain brief, with no invented metrics.

### 2. Rules in Plain English & Thresholds
| Rule | Plain English Description | Thresholds |
|------|---------------------------|------------|
| **pump_failure** | Alarm if pump_status==2 for 3 consecutive steps, or pump_flow==0 for 3 steps while tank level rises. | 3-step (15 min) persistence; pump_status==2 or pump_flow_bpd==0 |
| **pipeline_leak** | Alarm if pipeline_pressure_psi < 90% of disposal site setpoint *and* pump_flow_bpd - pipeline_receive_flow_bpd > 5% of max_flow_bpd while pump runs. | Pressure < 0.90 × disposal_psi; flow delta > 5% max_flow; 3-step (15 min) persistence |
| **pump_degradation** | Alarm if pump_flow_bpd declines 2.5% below 8‑h rolling baseline for 1‑h persistence while pump runs. | Flow ratio < baseline - 0.025; 12-step (1 h) persistence |

### 3. Training vs Held‑Out Results vs Hand‑Built Baseline
| Metric | Training (S1–S3) | Held‑Out (S4–S5) | Hand‑Built Baseline (S4–S5) |
|---|---|---|---|
| **Overall F1** | 0.727 | 0.667 | 1.0 |
| **pump_failure** | Labelled: 6 / Found: 6 / False alarms: 0 / Onset error: 0 min | Labelled: 7 / Found: 7 / False alarms: 0 / Onset error: 0 min | – |
| **pipeline_leak** | Labelled: 3 / Found: 2 / False alarms: 2 / Onset error: 180 min | Labelled: 2 / Found: 0 / False alarms: 3 / Onset error: null | – |
| **pump_degradation** | Labelled: 3 / Found: 0 / False alarms: 0 / Onset error: null | Labelled: 2 / Found: 0 / False alarms: 0 / Onset error: null | – |

### 4. Limitations & Next Steps
- **Pipeline leak** suffers high false alarms on held‑out systems (3 vs 2 detected), suggesting setpoint or persistence miscalibration across facilities.
- **Pump degradation** remains undetected in both training and held‑out data (0/3 and 0/2 found), indicating the 2.5%/1h threshold may be too strict or the signal too subtle.
- **No cross‑system generalization** for leak detection: hand‑built baseline achieves F1=1.0 on held‑out, highlighting a gap between learned and rule‑based approaches.
- **Try next:** Sensitivity analysis on pressure/flow thresholds; experiment with longer persistence windows or alternative trend metrics (e.g., slope instead of rolling median); incorporate well‑baseline deviations; evaluate unsupervised anomaly scoring on held‑out telemetry.