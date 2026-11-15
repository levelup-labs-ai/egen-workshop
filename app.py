"""Bench workshop app: browse the reference requests, run the agent, see what it did, run the dataset.

Run: streamlit run app.py
"""

import html
import inspect
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st

from workshop_helpers import tools
from workshop_helpers.evals import CASES, found_good_fit, make_grounded_judge, no_ineligible_pick
from workshop_helpers import evals, experiments
from workshop_helpers.experiments import DATASET_NAME, ensure_dataset, list_experiments, load_outputs, run_experiment
from workshop_helpers.setup import setup_clients

PROMPT_V1 = """You are Bench, Fernhill's staffing assistant. Delivery managers and account leads ask you to find consultants for client work.
Today is Monday, November 16, 2026.

Use the tools to find consultants who match the request. Recommend up to 3 people, with a one-line reason for each.
Write a short message to the manager summarizing your recommendation.
"""

GROUNDED_JUDGE_PROMPT = """You check a staffing assistant's reply for factual errors about consultants.
You get the request, the assistant's message, and the raw tool results the assistant saw.

Label "fail" only if the message states a fact that the tool results contradict or don't contain, such as:
- a name attached to the wrong consultant id
- a skill, certification, industry, location, or seniority the consultant doesn't have in the tool results
- availability that conflicts with the assignments or PTO in the tool results (e.g. "fully available" when they have PTO in the window)
Ignore opinions and adjectives ("strong", "great fit", "experienced"). Ignore people who are only mentioned as not available.
Label "pass" otherwise.

Return JSON: {"label": "pass" or "fail", "reason": "<one sentence; quote the wrong claim if any>"}
"""

GREEN, RED, AMBER, GREY, INK = "#466b5d", "#c0392b", "#96563f", "#566476", "#111821"
BADGE = {"pass": "green", "fail": "red", "n/a": "gray"}

st.set_page_config(page_title="Bench workshop", page_icon="🪑", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2.2rem; max-width: 1280px;}
    .request {font-size: 1.15rem; line-height: 1.5; border-left: 4px solid #466b5d; padding: .4rem 0 .4rem 1rem; margin: .2rem 0 1rem;}
    .pick {border: 1px solid rgba(128,128,128,.3); border-radius: 10px; padding: .7rem .9rem; margin-bottom: .6rem;}
    .st-key-case_list button {justify-content: flex-start; text-align: left; padding: .55rem .8rem;}
    .st-key-case_list button p {font-weight: 400; line-height: 1.4;}
    .st-key-case_list button[kind="primary"] {background: rgba(70,107,93,.10); border: 1.5px solid #466b5d; color: inherit;}
    .pick b {font-size: 1.02rem;} .pick small {opacity: .65;}
    .tool {border-left: 3px solid #b4c6d7; padding: .15rem 0 .2rem 1rem; margin-bottom: 1.4rem;}
    .tool-name {font-family: Consolas, Menlo, monospace; font-weight: 600; font-size: 1.02rem; margin-right: .6rem;}
    .param {font-family: Consolas, Menlo, monospace; font-size: .78rem; background: #e8eef4; color: #416784; border-radius: 5px;
            padding: .12rem .45rem; margin-right: .35rem; white-space: nowrap;}
    .param i {opacity: .7;}
    .tool-lead, .tool-more {font-size: .88rem; opacity: .68; line-height: 1.55;} .tool-lead {margin-top: .4rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Connecting to the model and Arize...")
def connect(provider: str) -> dict:
    env = setup_clients(provider)
    env["dataset_id"] = ensure_dataset(env["arize"], env["space_id"])
    env["org_id"] = env["arize"].spaces.get(space=env["space_id"]).organization_id
    return env


ROOT = Path(__file__).resolve().parent
REPO = "https://github.com/levelup-labs-ai/egen-workshop/blob/main/"


def src(target, marker: str | None = None) -> str:
    """A GitHub link to where something lives: a function or class, a file, or the line in a file that contains `marker`."""
    if isinstance(target, str):
        if not marker:
            return f"[`{target}`]({REPO}{target})"
        line = next(i for i, text in enumerate((ROOT / target).read_text().splitlines(), 1) if marker in text)
        return f"[`{target}` line {line}]({REPO}{target}#L{line})"
    lines, start = inspect.getsourcelines(target)
    path = Path(inspect.getsourcefile(target)).resolve().relative_to(ROOT)
    return f"[`{path}` lines {start}–{start + len(lines) - 1}]({REPO}{path}#L{start}-L{start + len(lines) - 1})"


def follow_along(items: dict) -> None:
    st.divider()
    st.caption("**Follow along in the code**  \n" + "  \n".join(f"{label}: {link}" for label, link in items.items()))


def name(cid: str) -> str:
    return tools.CONSULTANTS.get(cid, {}).get("name", cid)


def as_dict(value) -> dict:
    """Tool outputs arrive as dicts or as JSON text, depending on where the run came from."""
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


# ---------- what the agent did, as a picture ----------

def trace_graph(request: str, out: dict) -> str:
    """Graphviz DOT: request -> searches -> each consultant Bench looked at (and what it checked) -> picks."""
    picks = [p["consultant_id"] for p in out["recommended"]]
    looked: dict[str, dict] = {}   # consultant id -> {"availability": output or None, "profile": bool}
    searches = []                  # (node id, label, ids that came back)
    for i, call in enumerate(out["tool_calls"]):
        args, result = call["arguments"], as_dict(call["output"])
        if call["name"] == "search_consultants":
            where = f" in {args['location']}" if args.get("location") else ""
            label = f"search_consultants\\n{', '.join(args.get('skills', []))}{where}\\n{result.get('count', 0)} found"
            searches.append((f"s{i}", label, {r["id"] for r in result.get("results", [])}))
        elif call["name"] == "get_project":
            team = {m["consultant_id"] for p in result.get("projects", []) for m in p.get("team", [])}
            found = len(result.get("projects", []))
            searches.append((f"s{i}", f"get_project\\n{args.get('query', '')}\\n{found} project{'s' if found != 1 else ''}", team))
        else:
            entry = looked.setdefault(args.get("consultant_id", "?"), {"availability": None, "profile": False})
            if call["name"] == "get_availability":
                entry["availability"] = result
            else:
                entry["profile"] = True
    for cid in picks:
        looked.setdefault(cid, {"availability": None, "profile": False})

    lines = [
        'digraph { rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.7;',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, margin="0.14,0.08", color="#c7d1da", fillcolor="#f5f6f7", fontcolor="#111821"];',
        'edge [color="#566476", arrowsize=0.7];',
        f'req [label="Request", fillcolor="{INK}", fontcolor="white", color="{INK}"];',
    ]
    for node, label, _ in searches:
        lines.append(f'{node} [label="{label}", fillcolor="#e8eef4", color="#b4c6d7"];')
        lines.append(f"req -> {node};")
    for cid, seen in looked.items():
        avail = seen["availability"]
        if avail is None:
            status = "availability not checked"
        else:
            booked = sum(a["allocation_pct"] for a in avail.get("assignments", []))
            status = ("free" if not booked else f"{booked}% booked") + (", PTO" if avail.get("pto") else "")
        checks = status + ("\\nprofile read" if seen["profile"] else "")
        picked = cid in picks
        fill, border, font = (GREEN, GREEN, "white") if picked else ("#f5f6f7", "#c7d1da", INK)
        if picked and avail is None:
            fill, border = AMBER, AMBER
        lines.append(f'{cid} [label="{name(cid)} ({cid})\\n{checks}", fillcolor="{fill}", color="{border}", fontcolor="{font}"];')
        sources = [node for node, _, ids in searches if cid in ids] or ["req"]
        lines.append(f"{sources[0]} -> {cid};")
        if picked:
            lines.append(f'{cid} -> reply [color="{GREEN}", penwidth=1.6];')
    reply = f"Recommended {len(picks)}" if picks else "Recommended no one"
    lines.append(f'reply [label="{reply}", fillcolor="{INK}", fontcolor="white", color="{INK}"];')
    if not picks:
        lines.append(f"{searches[-1][0] if searches else 'req'} -> reply [style=dashed];")
    return "\n".join(lines) + "}"


def show_run(request: str, out: dict, case_id: str | None, env: dict, judge_model: str) -> None:
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.subheader("Bench's reply")
        st.write(out["message"])
        for p in out["recommended"]:
            st.markdown(f'<div class="pick"><b>{name(p["consultant_id"])}</b> <small>{p["consultant_id"]}</small><br>{p["reason"]}</div>',
                        unsafe_allow_html=True)
        if not out["recommended"]:
            st.caption("No one recommended.")
        if case_id:
            st.subheader("Checks")
            row = {"case_id": case_id}
            judge = make_grounded_judge(env["openai"], GROUNDED_JUDGE_PROMPT, judge_model)
            for check in (no_ineligible_pick, found_good_fit, judge):
                result = check(out, row)
                st.markdown(f":{BADGE[result.label]}-badge[{result.label}] **{check.__name__}**  \n{result.explanation}")
    with right:
        st.subheader("What Bench did")
        st.caption("Green: recommended. Amber: recommended without checking availability. Grey: looked at, not recommended.")
        st.graphviz_chart(trace_graph(request, out), width="stretch")
        with st.expander(f"All {len(out['tool_calls'])} tool calls, in order"):
            for i, call in enumerate(out["tool_calls"], 1):
                args = ", ".join(f"{k}={v}" for k, v in call["arguments"].items() if v is not None)
                st.markdown(f"**{i}. {call['name']}**({args})")
                st.json(as_dict(call["output"]), expanded=False)


@st.cache_data(ttl=60, show_spinner=False)
def past_runs(provider: str) -> dict:
    """Experiment name -> id for every run on the dataset, newest first."""
    env = connect(provider)
    return {e.name: e.id for e in list_experiments(env["arize"], env["dataset_id"])}


@st.cache_data(show_spinner="Loading the run from Arize...")
def run_outputs(provider: str, experiment_id: str) -> pd.DataFrame:
    env = connect(provider)
    return load_outputs(env["arize"], experiment_id, env["dataset_id"])


# ---------- sidebar ----------

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["azure", "openai"], help="azure = Egen's endpoint. openai = OpenAI directly.")
    default_model = "gpt-5.6-terra" if provider == "azure" else "gpt-6-luna"
    agent_model = st.text_input("Agent model", default_model)
    judge_model = st.text_input("Judge model", default_model)
    st.caption("Azure models: gpt-5.6-terra, gpt-5.4-mini")

try:
    env = connect(provider)
except RuntimeError as error:
    st.error(str(error))
    st.stop()

st.title("Bench, the staffing assistant")
st.caption(f"57 consultants, 13 projects, {len(CASES)} reference requests. In this world, today is Monday, Nov 16, 2026. "
           f"Every run is traced to the Arize project **{env['project_name']}**.")

problem_tab, data_tab, agent_tab, explore, evals_tab, dataset, arize_tab = st.tabs(
    ["1 · The problem", "2 · The data", "3 · The agent", "4 · Try it", "5 · The evaluators", "6 · Run the dataset", "7 · In Arize"])

# ---------- the problem ----------

with problem_tab:
    example = CASES["S01"]
    why_not = example["must_not_recommend"].values()
    tagged = tools.search_consultants(["BigQuery"])["count"]
    story_col, example_col = st.columns([6, 6], gap="large")
    with story_col:
        st.subheader("Who can we put on this?")
        st.markdown(
            "Delivery managers and account leads at Fernhill, a made-up consulting firm, ask this every week. A good answer means checking, for each person who might fit:\n"
            "- Do they have the skill, at the level the client asked for?\n"
            "- Are they free for those dates, or already booked on other work?\n"
            "- Is there PTO in the middle of the engagement?\n"
            "- Is the certification the contract requires still valid?\n"
            "- Are they in a timezone that works for the client?\n\n"
            "The facts are spread across profiles, project bookings and the PTO calendar. Get one wrong and a manager "
            "promises a client someone who turns out to be busy.")
        st.subheader("Why an agent?")
        st.markdown(
            "If every request came as a form (skill, level, dates), a database query would do. They don't. They come as messages, "
            "some of them vague, some asking things nobody planned for. Bench reads the message, decides what to look up, and writes the reply. "
            "That flexibility is also where it goes wrong, in ways that sound right.\n\n"
            "So the question for today: **how do we find out where it goes wrong, and measure it?**")
    with example_col:
        st.caption(f"One request, from {example['requested_by']}")
        st.markdown(f'<div class="request">{example["request"]}</div>', unsafe_allow_html=True)
        fits = [name(c) for c in example["good_fits"]]
        (caveat_id, caveat), = [(c, notes[0]) for c, notes in example["caveats"].items()]
        st.markdown(
            f"- **{tagged}** consultants have BigQuery on their profile\n"
            f"- **{sum('% on ' in w for w in why_not)}** of them are booked on other work in those months\n"
            f"- **{sum('mid-level' in w for w in why_not)}** are mid-level, not senior\n"
            f"- **{len(fits)}** fit: {' and '.join(fits)}\n"
            f"- {name(caveat_id).split()[0]} is out for a week in the middle ({caveat}), and the manager should hear that")
        st.caption("Every request in this workshop comes with an answer like this, worked out from the data. That's what Bench gets scored against.")
    follow_along({"The requests and their reference answers": src("workshop_helpers/data/reference_cases.json")})

# ---------- explore ----------

with explore:
    counts = Counter(c["case_type"] for c in CASES.values())
    kind = st.pills("Kind of request", list(counts), format_func=lambda k: f"{k.replace('_', ' ')} ({counts[k]})")
    cases = [c for c in CASES.values() if not kind or c["case_type"] == kind]
    selected = st.session_state.get("case_id")
    if selected not in {c["case_id"] for c in cases}:
        selected = None

    pick_col, detail_col = st.columns([5, 7], gap="large")
    with pick_col:
        st.caption(f"{len(cases)} request{'s' if len(cases) != 1 else ''}. Click one to open it.")
        with st.container(height=520, border=False, key="case_list"):
            if st.button("✏️  Write your own request", key="own", width="stretch", type="primary" if not selected else "secondary"):
                st.session_state["case_id"] = selected = None
                st.rerun()
            for c in cases:
                label = f"**{c['case_id']}** :gray-badge[{c['case_type'].replace('_', ' ')}]  \n{c['request']}"
                if st.button(label, key=f"case_{c['case_id']}", width="stretch", type="primary" if c["case_id"] == selected else "secondary"):
                    st.session_state["case_id"] = c["case_id"]
                    st.rerun()
        case = CASES[selected] if selected else None
    with detail_col:
        if case:
            st.markdown(f"**{case['case_id']}** · asked by {case['requested_by']}")
            st.markdown(f'<div class="request">{case["request"]}</div>', unsafe_allow_html=True)
            with st.expander("Reference answer (try guessing first)"):
                expected = {"shortlist": "**Bench should:** Give a shortlist", "no_fit": "**Bench should:** Say nobody fits",
                            "ask_clarifying": "No shortlist was written for this request."}
                st.markdown(expected[case["expected_behavior"]])
                if case["good_fits"]:
                    st.markdown("**Good fits:** " + ", ".join(f"{name(c)} ({c})" for c in case["good_fits"]))
                for cid, notes in case["caveats"].items():
                    st.markdown(f"**Must tell the manager about {name(cid)}:** {'; '.join(notes)}")
                if case["must_not_recommend"]:
                    items = "".join(f"<li>{name(c)} ({c}): {why}</li>" for c, why in case["must_not_recommend"].items())
                    st.markdown(f'<details><summary><b>Must not recommend</b> ({len(case["must_not_recommend"])} people)</summary><ul>{items}</ul></details>',
                                unsafe_allow_html=True)
                st.caption("The evaluators score Bench against this. For your own use case, reference answers like these are something you would have to build. Here they're given to you.")
            request = case["request"]
        else:
            request = st.text_area("Your own request", placeholder="Ask the way you'd ask on Slack, e.g. Need a Kafka person at 50% for Crestline from January",
                                   height=110)
        if st.button("Ask Bench", type="primary", disabled=not request):
            with st.spinner("Bench is looking..."):
                st.session_state["last_run"] = {"request": request, "case_id": case["case_id"] if case else None,
                                                "out": tools.run_agent(request, PROMPT_V1, agent_model)}

    last = st.session_state.get("last_run")
    if last and last["request"] == request:
        st.divider()
        show_run(last["request"], last["out"], last["case_id"], env, judge_model)
    follow_along({
        "The requests and their reference answers": src("workshop_helpers/data/reference_cases.json"),
        "Running the agent on one request": src(tools.run_agent),
        "The picture of what Bench did": src(trace_graph),
    })

# ---------- the data ----------

with data_tab:
    people = pd.DataFrame(tools.DATA["consultants"])
    projects = pd.DataFrame(tools.DATA["projects"])
    bookings = pd.DataFrame(tools.DATA["assignments"]).merge(
        projects[["id", "client", "name"]].rename(columns={"id": "project_id", "name": "project"}))
    bookings["consultant"] = bookings["consultant_id"].map(name)
    pto = pd.DataFrame(tools.DATA["pto"]).assign(consultant=lambda d: d["consultant_id"].map(name))

    st.markdown("What Fernhill knows about its people: who they are, what they're booked on, and when they're away. This is everything Bench can look up.")
    view = st.segmented_control("Show", ["Consultants", "Projects", "Who's booked when"], default="Consultants", label_visibility="collapsed")
    if view == "Consultants":
        find = st.text_input("Find", placeholder="Filter by name, skill, title, location or industry", label_visibility="collapsed")
        people["certifications"] = people["certifications"].map(lambda cs: [f"{c['name']} (expires {c['expires']})" for c in cs])
        if find:
            people = people[people.apply(lambda r: find.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
        st.caption(f"{len(people)} of {len(tools.CONSULTANTS)} consultants")
        st.dataframe(people[["id", "name", "title", "level", "location", "skills", "industries", "certifications", "notes"]],
                     hide_index=True, width="stretch", height=560,
                     column_config={"id": st.column_config.Column(width=50), "skills": st.column_config.ListColumn(width=320)})
    elif view == "Projects":
        st.dataframe(projects[["id", "client", "name", "industry", "start", "end", "skills", "summary"]], hide_index=True, width="stretch",
                     height=500, column_config={"id": st.column_config.Column(width=50), "skills": st.column_config.ListColumn(width=260)})
    else:
        st.caption("One row per consultant. Blue bars are project bookings: the darker the bar, the more of their time is taken. "
                   "Grey bars are PTO. An empty stretch means they're free. The dashed line is today (Nov 16, 2026). Hover a bar for details.")
        order = sorted(tools.CONSULTANTS.values(), key=lambda c: c["name"])
        y = alt.Y("consultant:N", sort=[c["name"] for c in order], title=None, axis=alt.Axis(labelLimit=170, labelOverlap=False, labelFontSize=11, ticks=False, domain=False, labelPadding=8),
                  scale=alt.Scale(domain=[c["name"] for c in order]))
        x = alt.X("start:T", title=None, scale=alt.Scale(domain=["2026-11-01", "2027-07-01"]), axis=alt.Axis(format="%b %Y", orient="top", grid=True, tickCount="month", domain=False, ticks=False))
        booked = alt.Chart(bookings).mark_bar(height=11, cornerRadius=3, clip=True).encode(
            x=x, x2="end:T", y=y,
            color=alt.Color("allocation_pct:Q", title="% of time booked", scale=alt.Scale(domain=[0, 100], range=["#e8eef4", "#416784"]),
                            legend=alt.Legend(orient="top", gradientLength=160)),
            tooltip=[alt.Tooltip("consultant", title="Consultant"), alt.Tooltip("client", title="Client"), alt.Tooltip("project", title="Project"),
                     alt.Tooltip("allocation_pct", title="% of time"), alt.Tooltip("start:T", title="From"), alt.Tooltip("end:T", title="Until")])
        off = alt.Chart(pto).mark_bar(height=5, cornerRadius=2, color="#6b7280", clip=True).encode(
            x=x, x2="end:T", y=y, tooltip=[alt.Tooltip("consultant", title="Consultant (PTO)"), alt.Tooltip("start:T", title="From"), alt.Tooltip("end:T", title="Until")])
        today = alt.Chart(pd.DataFrame({"start": [tools.TODAY]})).mark_rule(strokeDash=[4, 3], color="#111821").encode(x="start:T")
        st.altair_chart((booked + off + today).properties(height=alt.Step(21)).configure_view(stroke=None).configure_axis(gridColor="#dce2e7"),
                        width="stretch")
    follow_along({"The staffing data": src("workshop_helpers/data/staffing.json"),
                  "Where it's loaded": src("workshop_helpers/tools.py", "DATA = ")})

# ---------- the agent ----------

with agent_tab:
    st.markdown("Bench is one agent with a prompt and four tools. It decides which tools to call and in what order; nothing is scripted. "
                "The tools read a JSON file of consultants, projects, bookings and PTO.")
    box = 'shape=box, style="rounded,filled", fontname="Arial", fontsize=11, margin="0.16,0.1"'
    tool_nodes = "\n".join(f'{t.name} [label="{t.name}", fillcolor="#e8eef4", color="#b4c6d7"]; bench -> {t.name} [dir=both]; {t.name} -> data;'
                           for t in tools.TOOLS)
    st.graphviz_chart(f"""digraph {{ rankdir=LR; bgcolor="transparent"; nodesep=0.2; ranksep=0.8;
        node [{box}, color="#c7d1da", fillcolor="#f5f6f7", fontcolor="{INK}"]; edge [color="#566476", arrowsize=0.7];
        req [label="Manager's request"];
        bench [label="Bench\n{agent_model}\nprompt + 4 tools", fillcolor="{INK}", fontcolor="white", color="{INK}"];
        reply [label="Reply\nup to 3 picks, a reason for each\na message to the manager", fillcolor="{GREEN}", fontcolor="white", color="{GREEN}"];
        data [label="staffing.json\n{len(tools.CONSULTANTS)} consultants, {len(tools.DATA['projects'])} projects", shape=cylinder, style=filled];
        req -> bench; reply -> bench [dir=back]; {tool_nodes} }}""", width="stretch")

    st.subheader("The prompt")
    st.caption("A first draft. You'll change it in step 6.")
    st.code(PROMPT_V1, language=None, wrap_lines=True)
    tools_col, reply_col = st.columns([5, 7], gap="large")
    with tools_col:
        st.subheader("The tools")
        st.caption("Bench sees exactly this: the name, the inputs and the description.")
        for t in tools.TOOLS:
            inputs = "".join(f'<span class="param">{p}{" <i>optional</i>" if "anyOf" in spec else ""}</span>'
                             for p, spec in t.params_json_schema["properties"].items())
            lead, *rest = [html.escape(line) for line in t.description.splitlines() if line]
            st.markdown(f'<div class="tool"><div><span class="tool-name">{t.name}</span>{inputs}</div>'
                        f'<div class="tool-lead">{lead}</div><div class="tool-more">{"<br>".join(rest)}</div></div>', unsafe_allow_html=True)
    with reply_col:
        st.subheader("The reply format")
        st.caption("Bench must answer in this shape, which is what lets code check its picks: a list of picks, each with a consultant id "
                   "and a reason, plus a message to the manager. A good reply to the request in step 1 would look like this.")
        st.code(json.dumps({
            "recommended": [
                {"consultant_id": "C03", "reason": "Senior BigQuery engineer, free for the whole engagement."},
                {"consultant_id": "C13", "reason": "Senior BigQuery engineer, free full-time. On PTO Mar 1 to Mar 5."},
            ],
            "message": "Hannah Kim and Kevin Park both fit and can start Jan 11. Kevin is out Mar 1 to Mar 5, so plan around that week.",
        }, indent=2), language="json", wrap_lines=True)
    follow_along({
        "The agent": src(tools.build_agent),
        "The prompt": src("app.py", "PROMPT_V1 = "),
        "The tools": ", ".join(src(f) for f in tools.TOOL_FUNCTIONS),
        "The reply format": src(tools.StaffingReply),
        "Running the agent on one request": src(tools.run_agent),
    })

# ---------- the evaluators ----------

with evals_tab:
    st.markdown(f"Trying requests one at a time doesn't tell you how often Bench is wrong. So each of the {len(CASES)} requests in step 4 has a "
                "reference answer, and three evaluators score Bench's reply against it. These are the things we knew we wanted to measure "
                "before reading a single reply. Two are plain code. The third is an LLM judge, because it has to read what Bench wrote.")
    st.caption(f"The examples use the request from step 1: {CASES['S01']['request']}")
    fake = lambda picks: {"recommended": [{"consultant_id": c, "reason": ""} for c in picks], "message": "", "tool_calls": []}  # noqa: E731
    verdict = lambda label, text: f":{BADGE[label]}-badge[{label}] {text}"  # noqa: E731

    def code_examples(check, good, bad):
        lines = []
        for picks in (good, bad):
            result = check(fake(picks), {"case_id": "S01"})
            lines.append(f"Bench picks {' and '.join(name(c) for c in picks)}  \n{verdict(result.label, result.explanation)}")
        return lines

    evaluators = [
        ("no_ineligible_pick", "code", "Nobody Bench recommends is ruled out: booked too heavily for those dates, too junior, "
         "or missing a certification the request requires.", code_examples(no_ineligible_pick, ["C03"], ["C03", "C01"]), no_ineligible_pick, None),
        ("found_good_fit", "code", "At least one of Bench's picks is on the reference list of good fits. When nobody fits, Bench should recommend no one.",
         code_examples(found_good_fit, ["C13"], ["C04"]), found_good_fit, None),
        ("grounded", "LLM judge", "Everything the message says about a consultant is backed by what the tools returned. "
         "Opinions like \"strong fit\" don't count.",
         [f"Message says \"Kevin Park has no PTO in the window\", and the tools showed none  \n{verdict('pass', 'Nothing in the message conflicts with the tool results.')}",
          f"Message says \"Kevin Park is fully available\", but the tools showed PTO Mar 1 to Mar 5  \n{verdict('fail', 'Claims full availability despite PTO in the window.')}"],
         make_grounded_judge, GROUNDED_JUDGE_PROMPT),
    ]
    for title, kind, rule, examples, source, judge_prompt in evaluators:
        st.divider()
        title_col, rule_col, example_col = st.columns([3, 4, 5], gap="large")
        title_col.markdown(f"**{title}**  \n:{'violet' if judge_prompt else 'blue'}-badge[{kind}]")
        rule_col.markdown(rule)
        example_col.markdown("\n\n".join(examples))
        title_col.caption(src(source))
        if judge_prompt:
            with st.expander("The judge's prompt"):
                st.code(judge_prompt, language=None, wrap_lines=True)
    st.divider()
    st.markdown("These three won't catch everything. The failures nobody thought to measure are the ones you find by reading replies, in step 6 and in Arize.")
    follow_along({"The judge's prompt": src("app.py", "GROUNDED_JUDGE_PROMPT = "), "How the judge is called": src(evals._judge)})

# ---------- dataset ----------

with dataset:
    st.markdown("Run Bench over all the reference requests as an Arize experiment, scored by the three evaluators. "
                "To test a fix, change the prompt and run again. Each run gets its own experiment, named by the time it started.")
    prompt = st.text_area("Bench's prompt", PROMPT_V1, height=170)
    if st.button(f"Run all {len(CASES)} requests", type="primary"):
        with st.spinner(f"Running {len(CASES)} requests. This takes about 5 minutes."):
            task = lambda dataset_row: json.dumps(tools.run_agent(dataset_row["request"], prompt, agent_model))  # noqa: E731
            evaluators = [no_ineligible_pick, found_good_fit, make_grounded_judge(env["openai"], GROUNDED_JUDGE_PROMPT, judge_model)]
            env["dataset_id"] = ensure_dataset(env["arize"], env["space_id"])  # look it up again: it may have been deleted since the app started
            run_experiment(env["arize"], env["space_id"], env["dataset_id"], "bench", task, evaluators)
            past_runs.clear()

    runs = past_runs(provider)
    shown = url = None
    if runs:
        st.divider()
        shown = st.selectbox("Showing run", list(runs), help="Every run on this dataset, newest first. They are read back from Arize.")
        results = run_outputs(provider, runs[shown])
        url = (f"https://app.arize.com/organizations/{quote(env['org_id'])}/spaces/{quote(env['space_id'])}"
               f"/datasets/{quote(env['dataset_id'])}/experiments/compare?experimentId={quote(runs[shown])}")
        st.markdown(f"The three evaluators' scores for **{shown}** are in Arize. [Open it in Arize]({url})")
        view = pd.DataFrame({
            "id": results["case_id"],
            "request": results["case_id"].map(lambda c: CASES[c]["request"]),
            "recommended": results["output"].map(lambda o: ", ".join(name(p["consultant_id"]) for p in json.loads(o)["recommended"]) or "no one"),
        }).sort_values("id").reset_index(drop=True)
        st.caption("Click a row to see the reply and what Bench did.")
        opened = st.dataframe(view, hide_index=True, width="stretch", height=380, on_select="rerun", selection_mode="single-row",
                              column_config={"id": st.column_config.Column(width=50), "request": st.column_config.Column(width=560)})
        if opened.selection.rows:
            row = view.iloc[opened.selection.rows[0]]
            st.markdown(f'<div class="request">{row["request"]}</div>', unsafe_allow_html=True)
            out = json.loads(results.set_index("case_id").loc[row["id"], "output"])
            show_run(row["request"], out, None, env, judge_model)
        st.markdown("**Next:** read and annotate these replies in Arize. The steps are in tab 7.")
    follow_along({
        "Uploading the requests to Arize as a dataset": src(experiments.ensure_dataset),
        "Running an experiment": src(experiments.run_experiment),
        "Where this page starts the run": src("app.py", '"bench", task, evaluators)'),
        "The evaluators": src("workshop_helpers/evals.py"),
    })

# ---------- in Arize ----------


EXAMPLE_JUDGE = """You are checking one reply from Bench, a staffing assistant, for one specific problem.

Request from the manager:
{request}

Bench's reply:
{output}

The problem: the request leaves out something Bench needs before it can pick people (the role, the skills, the dates,
or how much of their time), and Bench filled it in itself and recommended people anyway instead of asking.

Answer "yes" if this reply has that problem.
Answer "no" if the request was specific enough, or if Bench asked the manager for the missing detail.
Give your reason in one sentence."""


def step(title: str, body: str, *images: str, example: str | None = None) -> None:
    st.subheader(title)
    text_col, image_col = st.columns([6, 5], gap="large")
    text_col.markdown(body)
    if example:
        with text_col.expander("An example judge, for one pattern. Open it after you've picked your own."):
            st.markdown("The pattern: **Bench guessed instead of asking.** The judge is named `guessed_instead_of_asking_judge`.")
            st.code(example, language=None, wrap_lines=True)
    for image in images:
        image_col.image(str(ROOT / "assets" / "arize" / image), width="stretch")
    st.divider()


with arize_tab:
    if not shown:
        st.info("Run the dataset in tab 6 first. These steps work on that run.")
    else:
        st.markdown(f"Everything on this page happens in Arize, on the run selected in tab 6: **{shown}**. [Open it in Arize]({url})")
        st.divider()
        step("1 · Make the rows readable", """
- Click **Columns**, top right of the table.
- Tick `request` and `output.message`.

Each row now shows what was asked and what Bench replied.""", "01_columns.png")
        step("2 · Set up the two annotation fields", """
You do this once.

- Click the **annotate icon** on any row. It's the small person-and-pencil icon at the top right of the output cell.
- Click **Add / Remove Configs** → **Add New Annotation**.
- Name it `verdict`, keep **Categorical Options**, and type two labels: `fail` with score 0 and `pass` with score 1. Click **Create**.
- Add a second one named `note`, with the type **Freeform Text**. Click **Create**.
- Back under **Add / Remove Configs**, tick `verdict` and `note` so they show in the panel.
- On the first row you annotate, fill in **both** fields, with a note even if the reply passes. A field only stays on the following rows once it has one saved value.""", "02_configs_menu.png", "03_new_config.png")
        step("3 · Annotate the rows", f"""
Take about five minutes and annotate as many rows as you can. You don't need all {len(CASES)}.

The three evaluators have checked the facts: who is booked, who is senior enough, whether the message matches the data. Don't re-check them.

The annotation panel shows only the first lines of Bench's raw output. So read `request` and `output.message` in the table row first, then click the annotate icon on that row. If you're unsure about a reply, its reference answer is in tab 4.

You sent this request and got this reply. **Could you act on it as it stands?** Fail it if:

- it answers a different question from the one you asked
- it filled in something you never said (a role, a skill, dates)
- it doesn't tell you who to pick, or why

On a fail, write one sentence in `note` on the first thing that's wrong. Be specific: *"Lists three people but doesn't say which one to pick"*, not *"bad reply"*.

**Next** moves to the next row. Annotations save on their own, and **Next** waits a second while they do.""", "04_annotate_panel.png")
        step("4 · Group your notes into patterns", f"""
- Click **Alyx**, top right. It's Arize's built-in assistant.
- Ask it to read your notes. You can paste this:

> I annotated the rows of the experiment {shown} with a verdict (pass or fail) and a free-text note. Read my notes on the rows I marked fail and group them into failure patterns. For each pattern, say how many rows it covers and whether one of the existing evaluators (no_ineligible_pick, found_good_fit, grounded) already catches it.

- Compare its groups with your own impression. Pick the biggest pattern that none of the three evaluators catches.""", "05_alyx.png")
        step("5 · Build a judge for that pattern", f"""
You found the pattern in the rows you read. To know how often it happens, on this run and on every run after it, something has to check each reply for it. Nobody will read every reply by hand, so build a judge.

The judge runs on a model set up in your Arize space. If Alyx can't find one, add it under **Settings** → **AI Providers** first.

- Give the pattern a short name, for example `no_clear_pick`.
- Ask Alyx to build an **LLM-as-a-judge evaluator** for it. Name it after the pattern with `_judge` on the end. Give it two labels, `yes` and `no`, and say in plain words when the answer is `yes`.
- Read the prompt it drafts. Then click **Accept**.
- Click **Add Evaluator**, top right of the experiment page, and pick your evaluator under **Evaluator Hub**.
- Under *Pick the examples to evaluate*, change **Dataset** to **Experiment** and select `{shown}`.
- Under *Connect the eval to your examples*, map `{{request}}` to `request` and the output variable to `output`.
- Click **Run Evaluator(s)**. The result shows on each row next to the other three.""", "06_judge_draft.png", "07_add_evaluator.png", "08_run_on_experiment.png", example=EXAMPLE_JUDGE)
        step("6 · Label the pattern yourself", f"""
The judge now gives every row a `yes` or a `no`. You don't know yet whether to believe it. To find out, answer the same question yourself.

Your `verdict` won't do for this. It says whether a reply is usable, not *why*.

- Click the **annotate icon** → **Add / Remove Configs** → **Add New Annotation**. Use the pattern's name (the judge's name without `_judge`) and keep **Categorical Options**.
- Under **Optimization Direction** choose **Minimize**, because a `yes` is the bad outcome. The two label rows swap when you do: type `yes` in the red row (score 1) and `no` in the green row (score 0). Click **Create**. The new field shows in the panel straight away.
- Go through the rows and mark `yes` if the reply has this pattern and `no` if it doesn't. Aim for all {len(CASES)}.

Label from your own notes, before you look at what the judge said on that row. If you copy the judge, the next step tells you nothing.

It's quick. Rows you passed are almost always `no`. A row you failed for a different reason is also `no`.""", "09_pattern_field.png", "10_label_row.png")
        step("7 · Check alignment", """
Alignment is how often the judge agrees with you. Row by row, compare the judge's `yes` or `no` with the label you gave in step 6.

Don't compare it with `verdict`. A reply can fail for a different reason, and the judge is right to say `no` on it.

Where the two differ, read the judge's reason. Sometimes the judge is wrong, and you fix its prompt. Sometimes you missed something.

The judge is usually off on the first try. If there's time, fix it: tell Alyx which rows differ and why, ask it to update the judge's prompt, and click **Accept**. Then run the judge again yourself: click **Evaluators** at the top of the experiment page, then **Run** next to your judge. Open **Advanced** and choose **Override evaluation labels**, or the run skips every row that already has a label. Then check again. One or two rounds is normal, and it doesn't have to reach 100%.

This is the step people skip. Before you use a judge to improve Bench, make sure the judge is right.""", "11_alignment_row.png", "12_rerun_override.png")
        step("8 · Fix the prompt and run again", f"""
- Go back to tab 6, change Bench's prompt to address the pattern, and run all {len(CASES)} requests again.
- In Arize, open the dataset `{DATASET_NAME}`, click **Evaluators** (top right), then **Run** next to your judge. In the dialog keep **Experiment** selected, check the new run is ticked, and click **Run Evaluator**.
- Compare the two runs on the dataset's **Experiment Analysis** tab: open **Metrics** under the chart and tick your judge. It should say `yes` on fewer rows. Arize counts the judge's `yes` as the higher score, so it shows that drop in red. Here the drop is the improvement.
- One prompt change rarely fixes every row, and the three evaluators' scores move by a row or so between runs. They should not drop by more than that.""", "13_two_runs.png")

