"""Beat the Shutdown: operator console + Nemotron copilot.

    uv run streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from copilot.data import Dataset, find_datasets, iso, load_dataset
from copilot.detect import detect_events
from copilot.discovery import list_runs
from copilot.planner import analyze, plan_system, projection
from copilot.sandbox import detect_with_agent_rules

ROOT = Path(__file__).parent
CATEGORY_LABEL = {"pump_failure": "Pump failure", "pipeline_leak": "Pipeline leak", "pump_degradation": "Pump wearing out"}
CATEGORY_ICON = {"pump_failure": "🔴", "pipeline_leak": "🟠", "pump_degradation": "🟡"}

st.set_page_config(page_title="Beat the Shutdown", page_icon="🛢️", layout="wide")


# --- data -----------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading telemetry...")
def get_dataset(folder: str) -> Dataset:
    return load_dataset(folder)


@st.cache_resource(show_spinner="Detecting problems...")
def get_events(folder: str, now: int, detector: str | None = None):
    """detector: None for the hand-built rules, or a path to a Nemotron-written detector.py."""
    ds = get_dataset(folder).until(now)
    raw = detect_with_agent_rules(ds, Path(detector).read_text())[0] if detector else detect_events(ds)
    return ds, analyze(ds, raw)


def fmt_t(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%a %m-%d %H:%M UTC")


def fmt_hours(h: float | None) -> str:
    if h is None:
        return "-"
    return "TRIPPED" if h <= 0 else f"{int(h)}h {int(round(h % 1 * 60)):02d}m"


# --- sidebar: dataset + time travel -----------------------------------------------

with st.sidebar:
    st.header("Dataset")
    found = find_datasets(ROOT / "data")
    labels = [str(p.relative_to(ROOT)) for p in found]
    choice = st.selectbox("Folder", labels, index=next((i for i, l in enumerate(labels) if "mock_test" in l), 0)) if labels else None
    custom = st.text_input("...or another folder path (e.g. the test/ folder)")
    folder = custom.strip() or (str(ROOT / choice) if choice else "")
    if not folder or not (Path(folder) / "telemetry.csv").exists():
        st.error("Pick a folder with network.json, ops.json and telemetry.csv.")
        st.stop()

    full = get_dataset(folder)
    st.header("Time travel")
    st.caption("Replay the console as if an earlier moment were \"now\".")
    hours = list(range(full.start + 3600, full.now, 3600)) + [full.now]
    now = st.select_slider("Now", options=hours, value=full.now, format_func=fmt_t)

    st.header("Detector")
    runs = [r for r in list_runs() if (r / "detector.py").exists()]
    det_labels = ["Hand-built rules"] + [f"Nemotron-discovered · run {r.name}" for r in runs]
    det_choice = st.radio("Which rules find the problems?", det_labels, index=0)
    detector = None if det_choice == det_labels[0] else str(runs[det_labels.index(det_choice) - 1] / "detector.py")

ds, events = get_events(folder, int(now), detector)
ongoing = [e for e in events if e.status == "ongoing"]
failures = [e for e in ongoing if e.category == "pump_failure"]

# --- header -----------------------------------------------------------------------

st.title("🛢️ Beat the Shutdown")
st.caption(f"Night-shift console · {Path(folder).name} · now = {fmt_t(ds.now)}")

soonest = min((e.hours_to_limit["p50"] for e in failures if e.hours_to_limit), default=None)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Active problems", len(ongoing))
c2.metric("Pump failures", len(failures))
c3.metric("Soonest shutdown", fmt_hours(soonest))
c4.metric("Cost of recommended plans", f"${sum(e.est_cost or 0 for e in failures):,.0f}")

tab_alerts, tab_map, tab_trends, tab_whatif, tab_agent, tab_lab, tab_export = st.tabs(
    ["🚨 Alerts", "🗺️ Map", "📈 Trends", "🧪 What-if", "🤖 Nemotron copilot", "🧠 Discovery lab", "📤 Export"])

# --- alerts -----------------------------------------------------------------------

with tab_alerts:
    if not ongoing:
        st.success("All clear. No problems in progress.")
    for e in ongoing:
        with st.container(border=True):
            a, b, c = st.columns([3, 2, 4])
            a.markdown(f"### {CATEGORY_ICON[e.category]} {CATEGORY_LABEL[e.category]}")
            a.markdown(f"**{e.facility_id}** · `{e.asset_id}` · since {fmt_t(e.onset_unix)}")
            if e.category == "pump_failure":
                b.metric("Time to shutdown", fmt_hours((e.hours_to_limit or {}).get("p50")),
                         help="p10-p90: {p10}-{p90} h".format(**e.hours_to_limit) if e.hours_to_limit else None)
                b.caption(f"Tank at {e.level_pct}%")
                c.markdown(f"**Do this:** {e.plan_summary}")
            else:
                b.metric("Confidence", f"{e.confidence:.0%}")
                c.markdown(f"**Do this:** send an inspector to `{e.asset_id}`")
            with c.expander("Why the app thinks so"):
                for line in e.evidence:
                    st.markdown(f"- {line}")
    resolved = [e for e in events if e.status == "resolved"]
    if resolved:
        with st.expander(f"Resolved earlier ({len(resolved)})"):
            st.dataframe(pd.DataFrame([{"event": e.event_id, "problem": CATEGORY_LABEL[e.category], "asset": e.asset_id,
                                        "start": iso(e.onset_unix), "end": iso(e.end_unix)} for e in resolved]),
                         hide_index=True, width="stretch")

# --- map --------------------------------------------------------------------------

with tab_map:
    worst = {}
    for e in ongoing:
        rank = {"pump_failure": 2, "pipeline_leak": 1, "pump_degradation": 1}[e.category]
        worst[e.facility_id] = max(worst.get(e.facility_id, 0), rank)
    colors = {0: [34, 197, 94], 1: [245, 158, 11], 2: [239, 68, 68]}
    fac_rows, line_rows, disp_rows = [], [], []
    for sid, s in ds.systems.items():
        d = s["disposal_site"]
        disp_rows.append({"name": d["id"], "lon": d["lon"], "lat": d["lat"], "info": f"Disposal site {d['pressure_psi']} psi"})
        for f in s["facilities"]:
            level = ds.at(f["tank"]["id"], "level_pct")
            fac_rows.append({"name": f["id"], "lon": f["lon"], "lat": f["lat"], "color": colors[worst.get(f["id"], 0)],
                             "info": f"Tank {level:.0f}% · {len(f['wells'])} wells"})
            line_rows.append({"from": [f["lon"], f["lat"]], "to": [d["lon"], d["lat"]]})
    center = pd.DataFrame(fac_rows)[["lat", "lon"]].mean()
    st.pydeck_chart(pdk.Deck(
        map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
        initial_view_state=pdk.ViewState(latitude=center.lat, longitude=center.lon, zoom=8),
        layers=[
            pdk.Layer("LineLayer", line_rows, get_source_position="from", get_target_position="to",
                      get_color=[100, 116, 139], get_width=2),
            pdk.Layer("ScatterplotLayer", disp_rows, get_position=["lon", "lat"], get_fill_color=[59, 130, 246],
                      get_radius=900, radius_min_pixels=6, pickable=True),
            pdk.Layer("ScatterplotLayer", fac_rows, get_position=["lon", "lat"], get_fill_color="color",
                      get_radius=1200, radius_min_pixels=8, pickable=True),
        ],
        tooltip={"text": "{name}\n{info}"},
    ))
    st.caption("🟢 healthy · 🟠 leak or worn pump · 🔴 pump failure · 🔵 disposal site")

# --- trends -----------------------------------------------------------------------

with tab_trends:
    fac_ids = sorted(ds.facilities)
    default_fac = failures[0].facility_id if failures else (ongoing[0].facility_id if ongoing else fac_ids[0])
    fid = st.selectbox("Facility", fac_ids, index=fac_ids.index(default_fac))
    f = ds.facilities[fid]
    options = {f"{a} · {m}": (a, m) for (a, m) in ds.wide.columns if ds.asset_to_facility.get(a) == fid}
    default = [k for k in options if k.endswith("level_pct") or (k.endswith("flow_bpd") and f["pump"]["id"] in k)]
    picked = st.multiselect("Sensors", list(options), default=default)
    window_h = st.slider("Hours shown", 6, int((ds.now - ds.start) / 3600) + 1, 36)
    fig = go.Figure()
    for i, k in enumerate(picked):
        a, m = options[k]
        s = ds.series(a, m).loc[ds.now - window_h * 3600:]
        axis = "y2" if m == "level_pct" else "y"
        fig.add_trace(go.Scatter(x=pd.to_datetime(s.index, unit="s"), y=s.values, name=k, yaxis=axis, mode="lines"))
    if any(options[k][1] == "level_pct" for k in picked):
        for val, name in [(ds.warning_pct, "warning"), (ds.shutdown_pct, "shutdown")]:
            fig.add_hline(y=val, line_dash="dot", annotation_text=name, yref="y2")
    for e in events:
        if e.facility_id == fid and e.onset_unix >= ds.now - window_h * 3600:
            fig.add_vline(x=pd.to_datetime(e.onset_unix, unit="s"), line_dash="dash", line_color="red")
            fig.add_annotation(x=pd.to_datetime(e.onset_unix, unit="s"), y=1, yref="paper",
                               text=CATEGORY_LABEL[e.category], showarrow=False, font={"color": "red"})
    fig.update_layout(height=480, margin={"t": 30, "b": 10}, legend={"orientation": "h"},
                      yaxis2={"overlaying": "y", "side": "right", "title": "tank %", "range": [0, 100]})
    st.plotly_chart(fig, width="stretch")

# --- what-if ----------------------------------------------------------------------

with tab_whatif:
    if not failures:
        st.info("No pump failure in progress at this moment. Use Time travel to pick a moment with one.")
    else:
        planned = plan_system(ds, events)
        pick = st.selectbox("Pump failure", failures, format_func=lambda e: f"{e.event_id} · {e.facility_id}")
        p = planned[pick.event_id]
        s = p["situation"]
        st.markdown(f"Repair finishes in **{fmt_hours(s.repair_min / 60)}**. "
                    f"{p['allocated_trucks']} of the system's {p['fleet']} trucks allocated here"
                    + (f" (shared with {', '.join(p['shares_with'])})." if p["shares_with"] else "."))
        names = {"do_nothing": "Do nothing", "trucks_only": "Trucks only", "shut_in_only": "Shut in wells only",
                 "best": "Recommended"}
        st.dataframe(pd.DataFrame([{"option": names[k], "plan": v.summary(), "cost": f"${v.cost:,.0f}",
                                    "tank trips?": "yes" if v.trips else "no"} for k, v in p["plans"].items()]),
                     hide_index=True, width="stretch")
        fig = go.Figure()
        for k, v in p["plans"].items():
            proj = projection(s, ds, v)
            x = pd.to_datetime(ds.now + proj["minutes"] * 60, unit="s")
            fig.add_trace(go.Scatter(x=x, y=proj["level_pct"], name=names[k], mode="lines",
                                     line={"width": 4 if k == "best" else 2}))
        fig.add_hline(y=ds.shutdown_pct, line_dash="dot", line_color="red", annotation_text="shutdown")
        fig.update_layout(height=420, yaxis_title="tank level %", margin={"t": 30, "b": 10}, legend={"orientation": "h"})
        st.plotly_chart(fig, width="stretch")

# --- Nemotron copilot -------------------------------------------------------------

with tab_agent:
    st.markdown("Four Nemotron agents work the shift: **Investigator** checks every alert against the sensors, "
                "**Planner** picks responses, **Critic** validates each plan against the rules (and sends it back "
                "if wrong), **Handoff writer** briefs the next operator.")
    if not os.getenv("NVIDIA_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    if not os.getenv("NVIDIA_API_KEY"):
        st.warning("Add your NVIDIA_API_KEY to the .env file (see .env.example), then restart the app.")
    elif st.button("Run copilot", type="primary"):
        from copilot.agents import run_copilot
        trace = []
        with st.status("Agents working...", expanded=True) as status:
            def emit(step: dict):
                trace.append(step)
                if step["type"] == "tool_call":
                    st.write(f"**{step['agent']}** → `{step['name']}({step['content']})`")
                elif step["type"] == "output":
                    st.write(f"**{step['agent']}** finished.")
            try:
                result = run_copilot(ds, events, emit)
                status.update(label="Done", state="complete", expanded=False)
                st.session_state["copilot"] = {"now": ds.now, "folder": folder, "result": result, "trace": trace}
            except Exception as exc:
                status.update(label="Copilot failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
    run = st.session_state.get("copilot")
    if run and run["now"] == ds.now and run["folder"] == folder:
        st.subheader("Shift handoff")
        st.markdown(run["result"].get("handoff", ""))
        with st.expander("Agent trace"):
            for step in run["trace"]:
                label = f"{step['agent']} · {step['type']}" + (f" · {step['name']}" if step.get("name") else "")
                st.markdown(f"**{label}**")
                st.code(step["content"][:2000], language="json" if step["type"] != "output" else "markdown")

# --- Discovery lab ----------------------------------------------------------------

with tab_lab:
    st.markdown(
        "Nemotron builds the monitor itself, with no hand-written rules. **Reader** studies the docs (search), "
        "**Scientist** runs Python on the training systems (S1–S3) to find each problem's signature, **Engineer** "
        "writes rule functions and iterates on the score, **Reporter** writes the model card. The rules are then "
        "graded on systems the agents never saw (S4–S5), next to the hand-built baseline.")
    if os.getenv("NVIDIA_API_KEY") and st.button("Run discovery (takes several minutes)", type="primary"):
        from copilot.discovery import run_discovery
        with st.status("Agents learning the network...", expanded=True) as status:
            def show(step: dict):
                if step["type"] == "tool_call":
                    st.write(f"**{step['agent']}** → `{step['name']}`")
                elif step["type"] == "score":
                    d = json.loads(step["content"])
                    st.write(f"📊 **{step['name']}**: F1 {d.get('f1', d.get('held_out_f1'))}")
                elif step["type"] == "output":
                    st.write(f"**{step['agent']}** finished.")
            try:
                saved = run_discovery(show)
                status.update(label=f"Saved run {saved.name}", state="complete", expanded=False)
                st.rerun()
            except Exception as exc:
                status.update(label="Discovery failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")

    all_runs = list_runs()
    if not all_runs:
        st.info("No discovery runs yet.")
    else:
        def train_f1(p):
            return (json.loads((p / "summary.json").read_text()).get("train") or {}).get("f1", 0)
        # default to the run with the best TRAINING score (choosing on held-out would leak the test set)
        best_i = max(range(len(all_runs)), key=lambda i: train_f1(all_runs[i]))
        run_dir = st.selectbox("Run", all_runs, index=best_i,
                               format_func=lambda p: f"{p.name} · train F1 {train_f1(p)}")
        summ = json.loads((run_dir / "summary.json").read_text())
        train, held, base = summ.get("train") or {}, summ.get("held_out") or {}, summ.get("baseline") or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Train F1 (S1–S3)", train.get("f1", "-"))
        m2.metric("Held-out F1 (unseen S4–S5)", held.get("f1", "-"),
                  delta=round(held["f1"] - base["held"]["f1"], 3) if held and base else None,
                  help="Delta is versus the hand-built baseline on the same held-out systems.")
        m3.metric("Baseline held-out F1", base.get("held", {}).get("f1", "-"))
        m4.metric("Attempts / time", f"{len(summ['attempts'])} · {summ['seconds'] // 60} min")
        st.caption(f"Models: {summ.get('model')} (Reader, Scientist, Reporter) · "
                   f"{summ.get('engineer_model', summ.get('model'))} (Engineer)")

        if summ["attempts"]:
            fig = go.Figure()
            xs = [a["n"] for a in summ["attempts"]]
            fig.add_trace(go.Scatter(x=xs, y=[a["train_f1"] for a in summ["attempts"]], mode="lines+markers",
                                     name="overall F1"))
            for cat in CATEGORY_LABEL:
                ys = [a["per_category"][cat]["found"] / max(1, a["per_category"][cat]["labelled"]) for a in summ["attempts"]]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"{CATEGORY_LABEL[cat]} recall",
                                         line={"dash": "dot"}))
            fig.update_layout(height=320, title="Learning curve (training systems)", xaxis_title="submission",
                              yaxis_range=[0, 1.05], margin={"t": 40, "b": 10}, legend={"orientation": "h"})
            st.plotly_chart(fig, width="stretch")

        if held:
            st.dataframe(pd.DataFrame([{
                "problem": CATEGORY_LABEL[c],
                "agent: found / labelled": f"{held['per_category'][c]['found']} / {held['per_category'][c]['labelled']}",
                "agent: false alarms": held["per_category"][c]["false_alarms"],
                "agent: onset error (min)": held["per_category"][c]["median_onset_error_min"],
                "baseline: found / labelled": f"{base['held']['per_category'][c]['found']} / {base['held']['per_category'][c]['labelled']}",
            } for c in CATEGORY_LABEL]), hide_index=True, width="stretch")
            st.caption("Held-out systems only: equipment the agents never saw.")

        left, right = st.columns(2)
        with left:
            st.subheader("Rules Nemotron wrote")
            if (run_dir / "detector.py").exists():
                st.code((run_dir / "detector.py").read_text(), language="python")
        with right:
            st.subheader("Model card")
            st.markdown((run_dir / "report.md").read_text())
        with st.expander("Domain brief (Reader)"):
            st.markdown((run_dir / "brief.md").read_text())
        with st.expander("Full agent trace"):
            for step in json.loads((run_dir / "trace.json").read_text()):
                label = f"[{step['t']}s] {step['agent']} · {step['type']}" + (f" · {step['name']}" if step.get("name") else "")
                st.markdown(f"**{label}**")
                body, lang = step["content"], None
                if step["type"] == "tool_call" and step.get("name") in ("run_python", "submit_detector"):
                    body, lang = json.loads(body).get("code", body), "python"
                st.code(body[:1500], language=lang)
        st.caption("Use these rules in the console: sidebar → Detector → Nemotron-discovered.")

# --- export -----------------------------------------------------------------------

with tab_export:
    payload = json.dumps([e.to_json() for e in events], indent=2)
    st.download_button("Download events.json", payload, file_name="events.json", mime="application/json")
    st.code(payload[:5000], language="json")
