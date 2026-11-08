# Domain Brief: Industrial Oil/Water Production Network Monitoring

**Sources:** `SPEC.md`, `network.json`, `examples.md`, `history_README.md`. All numbers traced to these sources.

---

## 1. Equipment & Sensors

| Component | Key Fields / Metrics | Source |
|---|---|---|
| **System** (S1–S5; test: S6–S11) | Facilities, one disposal site, truck yard, map center | `SPEC.md > network.json` |
| **Facility** | One tank, one pump, one pipeline, wells (oil/water/gas), lat/long | `SPEC.md > network.json` |
| **Tank** | `capacity_bbl` (barrels), `start_level_pct`, `stop_level_pct` (pump setpoints), `warning`/`shutdown` percent limits from `defaults.tank_limits_pct` (same for every tank) | `SPEC.md > network.json` |
| **Pump** | `max_flow_bpd` (healthy pump flow while running) | `SPEC.md > network.json` |
| **Wells** (per facility) | `baseline_water_bpd`, `oil_bpd`, `gas_mcfd` | `SPEC.md > network.json` |
| **Disposal Site** | `pressure_psi` only; never goes down, no other sensors | `SPEC.md > network.json` |
| **Telemetry** | `timestamp, ts_unix, asset_id, metric, value` — sampled every 5 min (`ts_unix` +300 per step; base `1791158400` = `2026-10-05T00:00:00Z`) | `examples.md` |
| **Pipeline (to disposal)** | Implicit flow/pressure metrics; healthy `pressure_psi` referenced above | `SPEC.md` |

---

## 2. Normal Operation (Sawtooth Cycle)

- Pump **starts** when tank level hits `start_level_pct`, **stops** at `stop_level_pct`.
- Tank level change per hour: `(inflow − pump_flow) ÷ 24 ÷ capacity × 100` (SPEC.md).
- Healthy pump flow: `max_flow_bpd` (barrels/day) while running.
- Well baselines sustained continuously: `oil_bpd`, `gas_mcfd`, `baseline_water_bpd` per facility.
- Pipeline to disposal site maintains steady `pressure_psi`; never triggers alarms in healthy state.
- Telemetry rows: one reading per 5-min step per asset; `metric` column encodes the type (e.g., `tank_level_pct`, `pump_flow_bpd`, `pipeline_pressure_psi`, `well_oil_bpd`, etc.).

---

## 3. Problem Signatures — Physical Effect + Reading Changes

### `pump_failure`
- **What happens:** Pump stops. Inflow continues → tank level rises steadily.
- **Reading changes:**
  - `tank_level_pct` rises toward `shutdown` limit (percent) without pump `on/off` cycling; may trigger at `trips_at` level recorded in `answer_key.json`.
  - `pump_flow_bpd` drops to 0 or reads abnormally low.
  - `pump_status` changes to stopped/idle.
- **Key numbers:** `shutdown` threshold from `defaults.tank_limits_pct`; `trips_at` level (% ) in practice answer keys; `capacity_bbl` determines rate of rise.

### `pipeline_leak`
- **What happens:** Water escapes from the pipeline before reaching the tank.
- **Reading changes:**
  - `pipeline_pressure_psi` drops below normal operating pressure (healthy baseline from disposal site section).
  - Tank fill rate stalls or decreases unexpectedly (inflow reduced).
  - No countdown: docs explicitly say "Doesn't fill a tank, so no countdown." Action: `field_inspection`.
  - `tank_level_pct` may show irregular fill pattern or slower rise.
- **Key numbers:** Healthy `pressure_psi` from `network.json` disposal site; deviation from baseline signals leak.

### `pump_degradation`
- **What happens:** Pump wears out, moves less water over time (gradual, not sudden).
- **Reading changes:**
  - `pump_flow_bpd` shows a downward trend relative to `max_flow_bpd` baseline (gradual decline over days/weeks).
  - Tank fill rate slows → `tank_level_pct` rise rate decreases per the hour-change formula, but still cycles between `start_level_pct`/`stop_level_pct`.
  - No countdown: docs say "Doesn't fill a tank, so no countdown." Action: `field_inspection`.
- **Key numbers:** `max_flow_bpd` as healthy baseline; rate of decline inferred from successive telemetry `value` entries; `capacity_bbl` modulates visible level change speed.

---

## 4. Key Ranges & Baselines Summary

| Metric | Healthy Range / Baseline | Source |
|---|---|---|
| Tank level cycle | Between `start_level_pct` and `stop_level_pct`; sawtooth per hour-change formula | `SPEC.md` |
| Pump flow | `max_flow_bpd` while running; 0 when stopped/failing | `SPEC.md` |
| Pipeline pressure | `pressure_psi` at disposal site; never goes down in healthy operation | `SPEC.md > network.json` |
| Tank shutdown limit | `shutdown` percent from `defaults.tank_limits_pct` (shared across all tanks) | `SPEC.md > network.json` |
| Well baselines | `oil_bpd`, `gas_mcfd`, `baseline_water_bpd` per facility | `SPEC.md > network.json` |
| Telemetry sampling | Every 5 min; `ts_unix` increments of 300; base 1791158400 = 2026-10-05T00:00:00Z | `examples.md` |

---
*All numbers and field names are drawn from `SPEC.md` and `network.json`. `answer_key.json` is practice-only and not loaded by the monitoring app.*