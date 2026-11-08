# Architecture: Nemotron Ops Copilot (ZeroDay "Beat the Shutdown")

One-liner: a multi-agent copilot that watches infrastructure telemetry, diagnoses what is failing,
predicts time-to-shutdown, and recommends the lowest-cost response. Oil-field produced water is the
first deployment; the same pattern covers any tanks/pumps/sensors network (wastewater, water utilities,
data center cooling).

## Core principle
**Code does the math. Nemotron does the reasoning.**
- Deterministic Python tools compute detections, forecasts, and costs. Numbers must be exact and
  reproducible because "cost optimization correctness" is judged.
- Nemotron agents decide *which tool to call next*, interpret evidence, handle concurrent failures,
  check each other's plans, and explain things in plain English. This is the "beyond a chatbot" part.

## Data rules (from SPEC.md / README)
- Load `network.json`, `ops.json`, `telemetry.csv` from files. **No hardcoding** of ids or values.
- **Never load `answer_key.json` in the app.** It is only for the offline eval harness.
- Must work on the hidden `test/` network (different systems, S6-S11).

## System diagram

```
telemetry.csv + network.json + ops.json
              |
        [Data loader]  (pandas, generic over any dataset folder)
              |
   +----------+-----------------------------------------------+
   |                 TOOLS (deterministic Python)              |
   |  get_series()  detect_anomalies()  forecast_time_to_limit() |
   |  plan_options()  validate_plan()  network_lookup()        |
   +----------+-----------------------------------------------+
              |  function calling
   +----------v-----------------------------------------------+
   |            AGENTS (LangGraph + Nemotron)                 |
   |                                                          |
   |  Orchestrator --> Triage agent (scan all facilities)     |
   |        |              |                                  |
   |        |              v                                  |
   |        |        Diagnosis agent (ReAct: hypothesize,     |
   |        |        pull series, confirm leak/degradation/   |
   |        |        pump failure, set confidence + evidence) |
   |        |              |                                  |
   |        |              v                                  |
   |        |        Response Planner (forecast + compare     |
   |        |        trucks / shut-in / do nothing; split the |
   |        |        shared fleet across simultaneous         |
   |        |        failures)                                |
   |        |              |                                  |
   |        |              v                                  |
   |        |        Critic agent (validate_plan; if a rule   |
   |        |        is broken, send back to Planner)         |
   |        v                                                 |
   |  Handoff writer (shift summary for the next operator)    |
   +----------+-----------------------------------------------+
              |
   events.json export  +  Operator UI (map, trends, alerts, what-if, handoff)
```

## Tools (what each does)

| Tool | Inputs | Returns |
|---|---|---|
| `get_series(asset_id, metric, start, end)` | asset, metric, window | time series for plots + agent reasoning |
| `detect_anomalies(facility_id)` | facility | candidate signals: pump status 2/0 while tank rising, pump flow below its own trend (degradation), pipeline `receive_flow` below pump `flow` (leak) |
| `forecast_time_to_limit(facility_id)` | facility | `hours_to_limit` p10/p50/p90 from level, inflow, pump flow, capacity, shutdown limit, expected repair hours |
| `plan_options(facility_id)` | facility | cost of each option using SPEC rules: truck trips (arrive at `response_min + n*cycle_min`, `cost_per_trip`, fleet cap), shut-in (oil/gas revenue prorated), do-nothing (auto shut-in cost) |
| `validate_plan(plan)` | plan | pass/fail + which rule broke (trips finish before repair ends, not more than tank contents, not above fleet size) |
| `network_lookup(id)` | any id | parent system, wells, tank, pump, pipeline, response times |

`detect_anomalies` signatures are not in the spec. Learn them from the plots of
`history/full_week` against its answer key (offline, by humans), then encode them as rules or simple
statistics. Agents never see the key.

## Agents (what each does)

1. **Orchestrator**: runs a tick ("now" = last timestamp), routes work, owns state.
2. **Triage**: scans every facility with `detect_anomalies`, returns a ranked list of suspects.
3. **Diagnosis (ReAct)**: for each suspect, reason, pull series, observe, repeat. Outputs category,
   onset time, confidence 0-1, and plain-English `evidence[]`.
4. **Response Planner**: calls `forecast_time_to_limit` + `plan_options`. For several pump failures in one
   system, allocates the single shared fleet by urgency (smallest hours-to-limit, then highest lost
   revenue per hour).
5. **Critic**: calls `validate_plan`; rejects and explains violations; Planner revises (max 2 loops).
6. **Handoff writer**: shift summary: what happened, what's open, what to watch next.

## Operator UI (judged on UX + aesthetics)
- Network map with facilities colored by status
- Sensor trend plots with the event onset marked
- Alert queue ordered by urgency, each with confidence, countdown, and recommended action
- **Agent trace panel**: shows each agent step and tool call (this is what NVIDIA judges want to see)
- What-if: replay and compare "do nothing" vs recommended plan cost
- Shift handoff view + `events.json` export
- Works on a phone

Suggested stack for a mixed-experience team: Python + **Streamlit** (fast) with pydeck/plotly, LangGraph
for the agent graph, Nemotron via the OpenAI-compatible API on build.nvidia.com.

## Eval harness (NVIDIA "agent simulation & eval" angle)
`eval/run_eval.py` replays each `history/snapshot_*` cut-off, runs the agents, and scores detections,
onset error, and plan cost against that folder's `answer_key.json`. This runs offline only and
never feeds the app. A scoreboard screenshot is great Devpost material.

## Stretch: generalize beyond oil (the wider-audience story)
Keep domain specifics in a config layer: asset types, metric names, units, cost model. Then show a second
dataset (e.g. wastewater lift stations) by swapping files, not code. Only attempt this if the core
works by Sunday morning.

## Suggested repo layout
```
hackutd-project/
  app/            Streamlit UI
  agents/         LangGraph graph + prompts
  tools/          deterministic tools above
  data_loader.py
  eval/           run_eval.py
  data/           copy of ZeroDay history/ (and test/ when released)
  ARCHITECTURE.md
```

## Build order (each step ends with something that runs and is committed)
1. **Tonight, before 11 PM:** Devpost project + team invited + tracks chosen + Submit pressed.
2. Data loader + `get_series` + a plot of one tank. Push.
3. `forecast_time_to_limit` + `plan_options`, checked by hand on one pump failure from the practice week.
4. `detect_anomalies` rules, tuned on `full_week`.
5. Wire ONE Nemotron agent end to end (Diagnosis) via LangGraph. Then add Planner, Critic, Handoff.
6. UI shell showing alerts + trace; make it pretty.
7. Eval harness + `events.json` export.
8. **Feature freeze ~3 hours before noon.** Demo video (about 2 min), Devpost writeup, make repo public.

## Team split (adjust to your skills)
- **You:** tools + detection + forecast/cost math (the correctness-critical core)
- **Teammate 2:** Streamlit UI, map, plots
- **Teammate 3:** agent prompts/LangGraph wiring with your tool signatures, then pitch, video, and Devpost writeup
- Agree on the tool function signatures first (table above) so everyone can work in parallel with fake data.
