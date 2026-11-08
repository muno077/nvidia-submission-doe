# Technical walkthrough (for judges and interviews)

## The pitch in 30 seconds

Night-shift operators watch hundreds of sensors and have minutes to act when a pump fails. Our app finds every
problem from raw telemetry, counts down to shutdown, and computes the provably cheapest response under the
real constraints (shared truck fleets, repair windows, well economics). On top of that exact core, NVIDIA
Nemotron agents do the judgment work: they verify alerts, plan, check each other, and brief the next shift.
And in the **Discovery lab**, Nemotron goes further: given only the manuals and labelled history, it learns
the domain and **writes its own monitoring rules**, which we grade on equipment it has never seen.

## Architecture

```
                    dataset folder (network.json, ops.json, telemetry.csv)
                                         |
                                  copilot/data.py  (pivot to wide table, time travel)
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
  DETECTION (two interchangeable sources)                        DISCOVERY LAB
  - copilot/detect.py   hand-built rules                         copilot/discovery.py
  - runs/<id>/detector.py  rules WRITTEN BY NEMOTRON  <------    Reader -> Scientist -> Engineer -> Reporter
         |                                                       (docs.py BM25, sandbox.py, scoring.py)
  copilot/planner.py   countdown + exhaustive cost search
         |
  copilot/tools.py + agents.py   Nemotron copilot (LangGraph)
  Investigator -> Planner -> Critic (loop on REVISE) -> Handoff
         |
  app.py (Streamlit): alerts, map, trends, what-if, copilot trace, discovery lab, events.json
```

**Design principle: code does the math, the model does the judgment.** Anything that has one right answer
(is the pump in fault? which plan is cheapest?) is deterministic Python, so it is exact, instant, free and
testable. The LLM is used where judgment, language or open-ended investigation is needed.

## 1. Detection (hand-built baseline)

How we found the signatures: compared readings during the labelled events in `full_week/answer_key.json`
with readings outside them (only while the pump runs, because pipeline/discharge pressure carry no
information when the pump is off).

| Problem | Signal | Healthy | During problem |
|---|---|---|---|
| Pump failure | `pump status == 2` | 0 or 1 | 2 |
| Pipeline leak | (disposal + 50) - pipeline pressure | 0 +/- 3 psi | 13 to 100+ psi low |
| Pump wear | (disposal + 150) - discharge pressure | 0 +/- 3 psi | grows slowly to 60 psi |

Key choices:
- **Rolling median of 6 running samples (30 min of pumping)**, not a mean: a median ignores single noisy spikes.
  Healthy noise is ~3 psi standard deviation, so a 10 psi (leak) / 8 psi (wear) threshold is ~3 sigma.
- **Only evaluate while the pump runs**, and carry the state across pump-off gaps, so one leak doesn't become
  ten events.
- **Wear onset back-dating:** wear grows linearly, so it crosses any threshold hours after it starts. We fit
  a hinge model `deficit = slope * max(0, t - t0)` to the samples around the first alarm and report `t0`.
  That took median onset error from ~4 h late to under 3 h.
- **A pump failure ends the wear episode** on that pump (the pump gets repaired), matching the spec.

Result: **72/72 labelled events, 0 false alarms** across full_week and the 3 snapshots (`eval/run_eval.py`).

## 2. Planner (the cost optimizer)

Rules come straight from SPEC.md. For each ongoing pump failure, in minutes from "now":
- volume rises at the live inflow rate; trucks remove `truck_capacity_bbl` per trip at
  `response_min + k * cycle_min`; trips must arrive before the repair ends and can't remove more than is in the tank
- shut-in wells cut inflow by their water rate and cost their oil+gas revenue per hour until the repair ends
- doing nothing: the tank trips at shutdown and ALL wells lose revenue until the repair ends

**Search:** every combination of trucks (0 to fleet) x trips per truck x subset of wells to shut in.
For a shut-in, we compute the **latest start minute** that still keeps the tank below shutdown, in closed form:
cost grows with hours shut in, so the latest safe start is always the cheapest. Plans that still trip are
priced as well (sometimes a truck that only delays the trip is the cheapest).

**Shared fleet:** all ongoing failures in a system share its trucks. We enumerate every way to split the fleet
across the failures (fleet <= 3, failures <= 3, so at most 64 splits) and take the cheapest total.

**Validation:** on the snapshots, our countdowns match the answer key within 0.02 h, the alert order matches,
and plan costs are within ~$25 of the reference (differences come from sensor noise in the tank level at "now").

**Known simplification:** a truck isn't chained from one tank to another inside the repair window.

## 3. Nemotron copilot (LangGraph)

Four agents, each with its own ReAct loop (reason, call a tool, read the result, repeat):

| Agent | Tools | Job |
|---|---|---|
| Investigator | list_alerts, get_sensor_summary, get_facility_info | Re-check each alert against raw sensors |
| Planner | compare_plans, submit_plan | Choose and submit a structured plan per failure |
| Critic | validate_plan, compare_plans | Approve or send back with REVISE (max 2 loops) |
| Handoff writer | none | 30-second shift summary from structured facts |

Lessons from running on a 30B model (good interview material):
- **Free-text plans were unreliable**, so the Planner submits plans through a `submit_plan` tool call. The
  plan is structured data, rule-checked instantly, and the model gets the result back.
- **Defense in depth:** the Critic's APPROVED only counts if a deterministic audit (every failure has a valid
  plan, fleet not exceeded) also passes.
- **Forgiving tools:** models mix up event ids, facility ids and pump ids, and write "none" for empty lists.
  Tools accept any of them instead of failing.
- **Front-load context:** each agent gets the data it needs in its prompt (alerts, optimizer options), so tool
  calls are for verification, not for wandering.
- **Empty answers were a token budget problem:** Nemotron reasons before answering; with a 4k cap the reasoning
  ate the budget after several tool calls. Raising `max_tokens` fixed it.

## 4. Discovery lab: Nemotron writes its own monitor

The question: could an AI team build this monitoring system for a network it has never seen, starting from
the manuals and some labelled history? This is the part that is clearly "beyond a chatbot".

| Agent | Tools | Job |
|---|---|---|
| Reader | search_docs (BM25 over README/SPEC/examples) | Domain brief: equipment, normal behavior, problem physics |
| Scientist | compare_signal, run_python, search_docs | Tests candidate signals during vs outside labelled events, proposes rules |
| Engineer | code-as-action (replies with a python block) | Writes rule functions, gets scored, iterates on concrete mistakes |
| Reporter | none | Model card with rules, scores vs baseline, limits |

Graph: `Reader -> Scientist -> Engineer -> (if F1 < 0.95, reflect once: Scientist studies the misses) -> Reporter`.

**The contract (separation of concerns).** The agent writes three small functions
`pump_failure(f)`, `pipeline_leak(f)`, `pump_degradation(f)` that return True/False/NaN per timestamp for one
facility. The framework owns the fiddly parts (carry state over NaN, drop blips under 15 min, merge runs
under 2 h apart, build events). That keeps the model on the science (which signal, what threshold, what window)
and shrinks what it can get wrong.

**Honest evaluation: split by system, not by time.** The agents only ever see systems S1-S3 (data and labels).
The final rules are graded on S4-S5, different equipment the agents never saw, next to our hand-built
baseline. Every problem type appears on both sides. This mirrors judging, where the saved `detector.py`
runs on the brand-new `test/` network (S6-S11).

**The LLM writes the monitor; it doesn't run it.** The output is plain, readable Python. At runtime it costs
nothing, takes milliseconds, gives the same answer every time and can be audited line by line. Compare with
calling an LLM on every sensor reading.

**Sandbox.** Model-written code runs in a separate Python process (`python -I`), in a temp folder, with a time
limit and a stripped environment (the API key is not passed in). Results come back as JSON/text, never pickle,
so bad code can't inject objects into the app. It is not a hardened sandbox (it can read files), which is an
acceptable trade-off for a local hackathon demo; production would use a container.

**Agentic RAG without a vector DB.** The docs are ~40 chunks; BM25 keyword ranking is instant, deterministic
and needs no extra model. The agent decides when to search and what to ask.

**Prompt injection awareness.** The challenge README contains lines addressed to AI agents (formatting
demands unrelated to the task). The Reader is told the docs are reference data and to ignore instructions
inside them.

**What broke while building it (and the fix):**
1. *The Engineer wrote its code in its reply instead of calling the submit tool*, so nothing was scored. First fix:
   detect the code block and submit it for the model.
2. *The Engineer kept exploring the data instead of committing* (it re-did the Scientist's job). We tried forcing
   its first step with `tool_choice` (NVIDIA's API supports forcing a specific function).
3. *Forced or not, a 30B model writing a whole detector inside a JSON-escaped tool argument produced junk*
   (all code on one line, broken imports), even though the same model writes clean multi-line code in
   markdown. Final design: **code-as-action**. The Engineer simply replies with a ```python block; the
   harness extracts it, runs it in the sandbox, scores it and sends the score back as the next message,
   up to 5 attempts per round. Clear division of labor: the Scientist explores with tools, the Engineer only codes.
4. Lesson: match the interface to what the model is good at. Tool calls for small structured arguments
   (`submit_plan(event_id, trucks, ...)`), markdown code blocks for programs.
5. *A bigger Engineer model didn't help* (Ultra 550B scored about the same as Lightning 30B). The bottleneck was
   upstream: the Scientist compared raw readings with ad-hoc pandas and proposed a leak threshold ~105 psi
   too strict. Fix: give it an instrument, `compare_signal(formula, problem, smooth_window)`. It returns the
   signal's distribution when healthy, during the problem and in the **first 2 hours** of each problem (early
   detection is what's scored), plus the best single threshold. The agent still chooses which hypotheses to test.
6. *Vague feedback made the Engineer guess.* "Missed leak at S4" doesn't say whether the rule fired late or never;
   those need opposite fixes. Missed events now say "nearest alarm on this asset started 310 min late" or
   "never fired on this asset".
7. *Long sessions sometimes ended in an empty answer.* After a retry, the agent's tool calls and results are
   condensed into a short fresh request that only asks for the write-up.

### Results

Three complete runs on real Nemotron (each 16–31 minutes, 9 scored submissions). Held-out = systems S4–S5,
which the agents never saw. "found/labelled" per problem type.

| Run | Engineer model | Train F1 | Held-out F1 | Held-out failures | Held-out leaks | Held-out wear | Held-out false alarms |
|---|---|---|---|---|---|---|---|
| **#5 (selected)** | Lightning 30B | **0.727** | 0.667 | 7/7 | 0/2 | 0/2 | 3 |
| #6 | Ultra 550B | 0.667 | 0.778 | 7/7 | 0/2 | 0/2 | 0 |
| #7 (+ compare_signal) | Ultra 550B | 0.636 | 0.783 | 7/7 | 2/2 | 0/2 | 3 |
| Hand-built baseline | (human) | 1.0 | 1.0 | 7/7 | 2/2 | 2/2 | 0 |

How to talk about it honestly:
- **Every run independently discovered pump failures and caught all 7 on unseen equipment with the right
  onset.** Runs #5 and #7 also discovered the leak signals (flow lost between pump and disposal site, pipeline
  pressure below normal); #7 caught both held-out leaks.
- **No run cracked slow pump wear.** It hides in the noise for the first hours, and catching it within 4 hours
  needs a long smoothing window and a small threshold. Our own baseline needed a hinge-model trick to get it.
- **We select the demo run by training score (#5), not held-out score.** Picking the run that looks best on the
  test set would be leaking the test set into model selection.
- **Run-to-run variance is large** (held-out 0.67–0.78). That's the honest state of LLM-driven discovery with
  one week of data, and it's why the hand-built detector stays the default in the console, with the
  agent-written one a sidebar switch away.
- Grading run #5's detector on all four practice folders with `eval/run_eval.py --detector` gives 42/72 with
  17 false alarms (that includes its training systems, so it's not a held-out number).

## 5. Model choice

All models are served by build.nvidia.com through its OpenAI-compatible API, so LangChain's standard
`ChatOpenAI` client works by changing `base_url`.

**Model routing:** use the smallest model that does each job well.
- `nvidia/nemotron-3.5-lightning-30b-a3b` (mixture-of-experts, ~3B active parameters per token): the copilot
  agents, the Reader and the Scientist. These are many short tool-calling steps where speed matters.
- `nvidia/nemotron-3-ultra-550b-a55b`: the Engineer in the Discovery lab (`ENGINEER_MODEL`), the one step that
  writes a whole program, where quality matters more than speed.

Each is one environment variable (`NEMOTRON_MODEL`, `DISCOVERY_MODEL`, `ENGINEER_MODEL`). Not every model is
enabled for every API key (Super returned "not found for account" for ours), and Ultra had a temporary 502
outage during the event, which is why the app works with Lightning alone too.

## 6. Questions you may get

- **Why not let the LLM detect problems directly?** Cost, latency and trust: 15 facilities x 9 sensors every
  5 minutes is ~40k readings a day. Rules run in milliseconds and are auditable; the LLM's job is to write and
  explain them.
- **How do you know the discovered rules aren't overfit?** They're graded on systems the agents never saw, and
  the same file runs on the unseen test network at judging.
- **What stops the agent from cheating with the answer key?** It only gets S1-S3 labels in its sandbox; the
  held-out labels never enter its context. The app itself never loads any answer key.
- **Is the cost plan optimal?** Within the stated rules, yes for single-facility plans (exhaustive search with
  closed-form best start time); the shared-fleet split is exhaustive too. The one simplification is truck chaining.
- **What happens if Nemotron is down?** The console (detection, countdowns, plans, what-if) works without any
  LLM; only the copilot and discovery tabs need it.
- **Why LangGraph?** Explicit state and conditional edges (Critic -> Planner loop, Engineer -> Scientist reflection)
  are easy to read, test (we tested the graph with a scripted fake model) and show in the trace.

## 7. Limitations and next steps

- Truck chaining across tanks in one repair window.
- A leak repaired while the pump is off stays "ongoing" until the pump runs again.
- Discovery sees one labelled week; more history or a simulator would let the agents test more hypotheses.
- Real sandboxing (containers) before letting model-written code near production systems.
