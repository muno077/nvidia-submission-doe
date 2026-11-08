# Model Card: Industrial Oil/Water Production Network Monitor

**1. What the Agents Learned from Docs and Data**  
The team learned the physical dynamics of the S1–S5 network from `SPEC.md` and `network.json`: tanks cycle between `start_level_pct` and `stop_level_pct` via a sawtooth driven by pump flow and inflow; pumps deliver `max_flow_bpd` when healthy and 0 when failed; pipeline pressure stays steady at the disposal site baseline; telemetry samples every 5 min (`ts_unix` +300, base `1791158400`). From this, the agents derived three fault signatures: pump failure (pump stops, tank rises), pipeline leak (pressure drops, fill stalls), and pump degradation (gradual flow decline).

**2. Rules in Plain English with Thresholds**  
- **Pump Failure:** Alarms if `pump_status == 2` (stopped) for 3+ consecutive 5-min steps, OR if pump flow drops below 5% of `max_flow_bpd` while facility inflow > 0 and tank level is above `start_level_pct` for 3+ steps.  
- **Pipeline Leak:** Alarms if `pipeline_pressure_psi` falls below 90% of the disposal site’s `disposal_psi` AND pump flow minus pipeline receive flow exceeds 5% of `max_flow_bpd`, both conditions persisting for 3+ steps (framework merges gaps <2 h).  
- **Pump Degradation:** Alarms if the 48-step (4 h) rolling median of pump flow ratio drops at least 2.5% below the 96-step (8 h) rolling baseline median, and the condition holds for 12+ consecutive steps while the pump is running.

**3. Training vs Held-Out Results vs Hand-Built Baseline**  
| Dataset | F1 | Per-Rule Breakdown |
|---|---|---|
| Training (S1–S3) | 0.667 | pump_failure: 6 labelled / 6 found, 0 false alarms, 10 min median onset error; pipeline_leak: 0/3 found; pump_degradation: 0/3 found |
| Held-Out (S4–S5) | 0.778 | pump_failure: 7 labelled / 7 found, 0 false alarms, 10 min median onset error; pipeline_leak: 0/2 found; pump_degradation: 0/2 found |
| Hand-Built Baseline (S4–S5) | 1.0 | — |

**4. Limitations and What to Try Next**  
Pipeline leak and pump degradation remain undetected (0 found despite labels present), suggesting thresholds or window parameters may be too strict or miss subtle signatures. The hand-built baseline achieves F1 1.0 by using different logic. Next steps: calibrate thresholds using `answer_key.json`, expand features (e.g., tank fill-rate trends), and validate rolling-window sizes against the 5-min sampling interval. Consider semi-supervised learning on abundant normal telemetry.