# Beat the Shutdown: Nemotron Ops Copilot

A night-shift console for produced-water networks. It finds failing pumps, leaks and worn pumps from
sensor data alone, counts down to shutdown, picks the cheapest response (trucks, shutting in wells,
or accepting the shut-in), and runs a team of NVIDIA Nemotron agents that check the alerts, plan,
validate the plans and write the shift handoff.

Built for HackUTD 2026 (ZeroDay "Beat the Shutdown" challenge + NVIDIA Nemotron track).

## Run it

```bash
uv sync                                   # install dependencies (first time)
cp .env.example .env                      # then paste your NVIDIA_API_KEY into .env
uv run streamlit run app.py               # opens http://localhost:8501
```

Data: copy the challenge's `history/` folder (and `test/` when it's released) into `data/`.
The app finds every folder under `data/` that has `network.json`, `ops.json` and `telemetry.csv`,
or you can paste any folder path in the sidebar.

Check accuracy against the practice answer keys (never loaded by the app):

```bash
uv run python eval/run_eval.py
```

Current score: **72/72 labelled events detected, 0 false alarms** across full_week and the three
snapshots. Countdowns match the key within 0.02 h, and plan costs land within about $25 of the
reference plans.

## How it works

| Piece | File | What it does |
|---|---|---|
| Loader | `copilot/data.py` | Reads a dataset folder; time travel via `Dataset.until(ts)` |
| Detector | `copilot/detect.py` | Pump status 2 = failure; low pipeline pressure while pumping = leak; low discharge pressure while pumping = wear |
| Planner | `copilot/planner.py` | Countdown, plus an exhaustive search over trucks x trips x well shut-ins x start time, sharing each system's fleet |
| Agent tools | `copilot/tools.py` | `list_alerts`, `get_sensor_summary`, `get_facility_info`, `compare_plans`, `validate_plan` |
| Agents | `copilot/agents.py` | LangGraph: Investigator, then Planner, then Critic (loops back on REVISE), then Handoff writer |
| UI | `app.py` | Alerts, map, trends, what-if, copilot trace, events.json export |
| Eval | `eval/run_eval.py` | Scores against practice answer keys |
| Discovery lab | `copilot/discovery.py` | Nemotron agents read the docs, explore the data and **write their own detector** |
| Sandbox | `copilot/sandbox.py` | Runs model-written code in a separate, key-free process; turns rules into events |
| Doc search | `copilot/docs.py` | BM25 search over the challenge docs (agentic RAG) |

## Discovery lab

```bash
uv run python -m copilot.discovery      # Reader -> Scientist -> Engineer -> Reporter, saved to runs/<timestamp>/
uv run python eval/run_eval.py --detector runs/<timestamp>/detector.py
```

The agents train on systems S1–S3 only; their rules are graded on S4–S5, which they never see. In the app,
the "Discovery lab" tab shows the learning curve, the code Nemotron wrote, and its model card. The sidebar can
switch the whole console to the Nemotron-written rules (and run them on the unseen `test/` network at judging).

See [docs/TECHNICAL.md](docs/TECHNICAL.md) for the technical walkthrough, [ARCHITECTURE.md](ARCHITECTURE.md)
for the original design and [GIT_CHEATSHEET.md](GIT_CHEATSHEET.md) for Git basics.
