"""Multi-agent copilot on NVIDIA Nemotron, wired with LangGraph.

    Investigator --> Planner --> Critic --(approved)--> Handoff writer
                        ^           |
                        +-(revise)--+   (at most MAX_REVISIONS times)

Each agent runs its own tool-calling (ReAct) loop: reason, call a tool, read the result, repeat.
Agents get the facts they need up front (alerts, optimizer options) and the Planner submits plans
through a tool, so plans are structured and every one is rule-checked in code before approval.
Every step is recorded in `trace` so the UI can show what the agents did.
"""

from __future__ import annotations

import json
import os
from typing import Callable, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .data import Dataset, iso
from .detect import Event
from .tools import Toolbox

load_dotenv()

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"  # not every model is enabled for every key
MAX_STEPS = 8  # tool calls per agent turn
MAX_REVISIONS = 2

ROLE = ("You work the night shift for a produced-water network: wells fill a tank, a pump empties it into a "
        "pipeline to a disposal site. If a pump fails the tank fills; at the shutdown limit the wells shut in and "
        "stop earning money until the repair ends. Only use facts from the data and tool results. Never invent numbers. ")

PROMPTS = {
    "investigator": ROLE + (
        "You are the Investigator. The current alerts are given to you. For each one, check the evidence with "
        "get_sensor_summary (pump: status, flow_bpd, discharge_psi; pipeline: pressure_psi, receive_flow_bpd; "
        "tank: level_pct) and compare with healthy values from get_facility_info. Check at most 6 sensors, soonest "
        "shutdown first. Then output one markdown bullet per alert: event id, facility, what is wrong, a verdict, and "
        "the reading behind it. Verdict is 'confirmed' if your check agrees, 'doubtful' only if a reading contradicts "
        "the alert, and 'not re-checked' if you ran out of checks (then rely on the alert's own evidence)."
    ),
    "planner": ROLE + (
        "You are the Response Planner. You get every ongoing pump failure with the optimizer's options; the 'best' "
        "option already shares the system's trucks across failures. For EACH pump failure call submit_plan exactly "
        "once with its event_id and the trucks, trips_per_truck, shut_in_wells and shut_in_start_min of the option "
        "you choose (normally 'best'; deviate only with a concrete reason). Use trucks=0 and shut_in_wells=[] when "
        "not used. After submitting all of them, reply with one line per event: event id, facility, action, cost, why."
    ),
    "critic": ROLE + (
        "You are the Critic. You get the submitted plans and an automatic rule audit. You may call validate_plan to "
        "double-check a plan. Check: every pump failure has a plan, each plan is valid, the tank doesn't trip unless "
        "that is the cheapest choice, and trucks per system stay within the fleet. Reply starting with APPROVED, "
        "or with REVISE followed by exactly what to fix (event id and the change)."
    ),
    "handoff": ROLE + (
        "You write the shift handoff. The next operator has 30 seconds. Use ONLY the data given. Format: one-line "
        "headline; 'Act now' (pump failures by countdown, with facility, time to shutdown and the plan); 'Inspect' "
        "(leaks and worn pumps, with facility and asset); 'Watch' (doubts from the investigator, or 'Nothing else'). "
        "Plain English, short bullets."
    ),
}


class State(TypedDict, total=False):
    findings: str
    plan: str
    critique: str
    approved: bool
    handoff: str
    revisions: int


def make_llm(model: str | None = None) -> ChatOpenAI:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not set. Put it in .env (see .env.example).")
    return ChatOpenAI(
        model=model or os.getenv("NEMOTRON_MODEL", DEFAULT_MODEL),
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=key,
        temperature=0.2,
        max_tokens=16384,  # Nemotron reasons before answering; a small cap can leave the answer empty
        timeout=300,  # writing a whole detector after long reasoning can take minutes
    )


def _text(msg: AIMessage) -> str:
    c = msg.content
    if isinstance(c, list):
        c = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return (c or "").strip()


def _write_up(llm: ChatOpenAI, msgs: list) -> str:
    """Condense a long tool session (each call + a slice of its result) and ask only for the final answer."""
    evidence = []
    for m in msgs[2:]:
        if isinstance(m, AIMessage):
            evidence += [f"CALL {c['name']}: {json.dumps(c['args'])[:600]}" for c in (m.tool_calls or [])]
        elif isinstance(m, ToolMessage):
            evidence.append(f"RESULT: {str(m.content)[:1200]}")
    prompt = (f"{msgs[1].content}\n\nYou already ran these steps:\n" + "\n".join(evidence)[-24000:] +
              "\n\nNow write your final answer from these results, in the format asked. Do not call tools.")
    return _text(llm.invoke([msgs[0], HumanMessage(prompt)]))


def run_agent(llm: ChatOpenAI, name: str, task: str, tools: list, emit: Callable[[dict], None],
              system: str | None = None, max_steps: int = MAX_STEPS, first_tool: str | None = None) -> str:
    """One agent's ReAct loop. `emit` receives each step for the trace.
    first_tool forces the first step to call that tool (e.g. make the Engineer commit a draft before exploring)."""
    by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools) if tools else llm
    msgs = [SystemMessage(system or PROMPTS[name]), HumanMessage(task)]
    for step in range(max_steps):
        if step == 0 and first_tool:
            forced = llm.bind_tools(tools, tool_choice={"type": "function", "function": {"name": first_tool}})
            ai: AIMessage = forced.invoke(msgs)
        else:
            ai = model.invoke(msgs)
        msgs.append(ai)
        if not getattr(ai, "tool_calls", None):
            out = _text(ai)
            if not out:  # reasoning models sometimes end with an empty message; ask once more
                msgs.append(HumanMessage("Write your final answer now as plain text, in the format asked."))
                out = _text(llm.invoke(msgs))
            if not out:  # still empty after a long session: fresh short call with the evidence condensed
                out = _write_up(llm, msgs)
            emit({"agent": name, "type": "output", "content": out})
            return out
        for call in ai.tool_calls:
            emit({"agent": name, "type": "tool_call", "name": call["name"], "content": json.dumps(call["args"])})
            try:
                result = by_name[call["name"]].invoke(call["args"])
            except Exception as exc:  # let the agent see the error and recover
                result = json.dumps({"error": str(exc)})
            emit({"agent": name, "type": "tool_result", "name": call["name"], "content": str(result)[:4000]})
            msgs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    msgs.append(HumanMessage("Step limit reached. Give your final answer now, without calling tools."))
    out = _text(llm.invoke(msgs))
    emit({"agent": name, "type": "output", "content": out})
    return out


def _alerts_brief(box: Toolbox) -> list[dict]:
    """Compact facts about every ongoing problem, with the plan that will be used."""
    rows = []
    for e in box.events:
        if e.status != "ongoing":
            continue
        row = {"event_id": e.event_id, "problem": e.category, "facility_id": e.facility_id, "asset_id": e.asset_id,
               "since": iso(e.onset_unix), "confidence": e.confidence, "evidence": e.evidence}
        if e.category == "pump_failure":
            sub = box.submitted.get(e.event_id)
            plan = sub["check"]["summary"] if sub and sub["check"]["valid"] else e.plan_summary
            row.update({"tank_level_pct": e.level_pct, "hours_to_shutdown": (e.hours_to_limit or {}).get("p50"),
                        "plan": plan})
        else:
            row["action"] = f"send an inspector to {e.asset_id}"
        rows.append(row)
    return rows


def build_graph(ds: Dataset, events: list[Event], emit: Callable[[dict], None], llm: ChatOpenAI | None = None):
    llm = llm or make_llm()
    box = Toolbox(ds, events)
    tools = box.tools()
    alerts_json = tools["investigator"][0].invoke({})  # list_alerts

    def investigator(state: State) -> State:
        task = f"Current alerts (from list_alerts):\n{alerts_json}"
        return {"findings": run_agent(llm, "investigator", task, tools["investigator"], emit)}

    def planner(state: State) -> State:
        if not box.failures:
            emit({"agent": "planner", "type": "output", "content": "No pump failures in progress; nothing to plan."})
            return {"plan": "No pump failures in progress."}
        options = [box.options(e) for e in box.failures]
        task = (f"Ongoing pump failures with the optimizer's options:\n{json.dumps(options)}\n\n"
                f"Investigator findings:\n{state['findings']}")
        if state.get("critique"):
            task += f"\n\nYour previous plan:\n{state['plan']}\n\nCritic feedback to fix:\n{state['critique']}"
        return {"plan": run_agent(llm, "planner", task, tools["planner"], emit)}

    def critic(state: State) -> State:
        revisions = state.get("revisions", 0) + 1
        if not box.failures:
            return {"critique": "APPROVED (no pump failures).", "approved": True, "revisions": revisions}
        audit = box.audit()
        task = (f"Submitted plans:\n{json.dumps(list(box.submitted.values()))}\n\n"
                f"Automatic rule audit: {json.dumps(audit) if audit else 'all checks passed'}\n\n"
                f"Planner's summary:\n{state['plan']}")
        out = run_agent(llm, "critic", task, tools["critic"], emit)
        approved = out.lstrip(" *#").upper().startswith("APPROVED") and not audit
        if audit and out.lstrip(" *#").upper().startswith("APPROVED"):
            out = "REVISE (the automatic audit failed):\n- " + "\n- ".join(audit)
        return {"critique": out, "approved": approved, "revisions": revisions}

    def route(state: State) -> str:
        return "handoff" if state["approved"] or state["revisions"] >= MAX_REVISIONS else "planner"

    def handoff(state: State) -> State:
        task = (f"Now: {iso(ds.now)}\nOngoing problems and the plans to run:\n{json.dumps(_alerts_brief(box))}\n\n"
                f"Investigator findings:\n{state['findings']}\n\nCritic verdict:\n{state['critique']}")
        return {"handoff": run_agent(llm, "handoff", task, [], emit)}

    g = StateGraph(State)
    for name, fn in [("investigator", investigator), ("planner", planner), ("critic", critic), ("handoff", handoff)]:
        g.add_node(name, fn)
    g.set_entry_point("investigator")
    g.add_edge("investigator", "planner")
    g.add_edge("planner", "critic")
    g.add_conditional_edges("critic", route, {"planner": "planner", "handoff": "handoff"})
    g.add_edge("handoff", END)
    return g.compile()


def run_copilot(ds: Dataset, events: list[Event], emit: Callable[[dict], None] = lambda _: None) -> State:
    return build_graph(ds, events, emit).invoke({"revisions": 0})
