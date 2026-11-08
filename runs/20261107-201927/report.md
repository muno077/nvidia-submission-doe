# Model Card: Industrial Oil/Water Production Network Monitor

**1. What the agents learned from the docs and data**  
From SPEC.md, network.json, examples.md: 5-min telemetry (ts_unix base 1791158400), normal sawtooth cycling (tank level between start/stop_level_pct, pump flow max_flow_bpd when running, pipeline pressure stable at disposal site), well baselines (oil_bpd, gas_mcfd, baseline_water_bpd per facility). Agents learned three failure signatures: pump failure (pump off + tank rise >3%/hr or >5 pct/30min), pipeline leak (pressure below 25th percentile for 2h), pump degradation (flow decline >0.5%/hr sustained 4h).

**2. Each rule in plain English with thresholds**  
- *pump_failure*: Pump off AND (a) tank_level_pct above start_level_pct for 4 readings (20 min) with 1-hr rise >3 pct, OR (b) 30-min (6-reading) rise >5 pct while above start.  
- *pipeline_leak*: pipeline_pressure_psi below 25th percentile of pump-running pressure (or disposal_psi - 75 fallback) for 24 consecutive readings (2 hours).  
- *pump_degradation*: pump_flow_bpd decline >0.5%/hr (rate < -0.005) sustained over 48 consecutive 5-min periods (4 hours) while pump runs.

**3. Training vs held-out results vs hand-built baseline**  
Training S1-S3: F1 0.636; pump_failure labelled 6/found 5/FA 1/onset 150min; pipeline_leak labelled 3/found 2/FA 2/onset 110min; pump_degradation labelled 3/found 0/onset null. Held-out S4-S5: F1 0.783; pump_failure labelled 7/found 7/FA 0/onset 145min; pipeline_leak labelled 2/found 2/FA 3/onset 58min; pump_degradation labelled 2/found 0/onset null. Hand-built baseline: F1 1.0 perfect, zero FA. Agents generalize better than training but trail baseline; pump_failure/pipeline_leak detect all held-out, degradation missed.

**4. Limitations and what to try next**  
- pump_degradation: zero detections; threshold too strict/signal noisy. Next: calibrate decline rate using healthy baselines, extend window, add smoothing.  
- pipeline_leak: 3 FA on held-out; 25th-percentile sensitive across facilities. Next: add hysteresis, facility-specific percentiles, or combine with tank fill-rate anomalies.  
- held-out F1 0.783 vs 1.0: refine rules. Next: ablation on window sizes, adaptive start/stop_level_pct per facility, or incorporate well baseline deviations.