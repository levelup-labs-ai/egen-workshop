"""Staffing data, the four agent tools, and the agent runner."""

import asyncio
import concurrent.futures
import contextvars
import json
from pathlib import Path

from agents import Agent, OpenAIResponsesModel, Runner, function_tool
from openai import AsyncOpenAI
from pydantic import BaseModel

MODEL = "gpt-6-luna"
CONNECTION: dict = {}  # AsyncOpenAI kwargs; setup_clients() fills this for the Azure endpoint

DATA = json.loads((Path(__file__).parent / "data" / "staffing.json").read_text())
TODAY = DATA["today"]
CONSULTANTS = {c["id"]: c for c in DATA["consultants"]}
PROJECTS = {p["id"]: p for p in DATA["projects"]}

def _overlaps(start: str, end: str, window_start: str, window_end: str) -> bool:
    return start <= window_end and end >= window_start  # ISO dates compare correctly as strings


def search_consultants(skills: list[str], location: str | None = None) -> dict:
    """Find consultants whose skill tags match any of the given skills.

    Matching is on exact skill tags (case-insensitive), e.g. "BigQuery", "Kafka", "Spring Boot".
    It does not search job titles, certifications, or industries; those are in get_consultant.
    Optionally filter by location text, e.g. "Chicago" or "India".
    Returns profile basics only. Use get_availability to check whether someone is free.
    """
    terms = {skill.lower() for skill in skills}

    results = []
    for c in CONSULTANTS.values():
        matched = [s for s in c["skills"] if s.lower() in terms]
        if not matched:
            continue
        if location and location.lower() not in c["location"].lower():
            continue
        results.append({k: c[k] for k in ("id", "name", "title", "level", "location")} | {"matched_skills": matched})
    results.sort(key=lambda r: -len(r["matched_skills"]))
    return {"count": len(results), "results": results}


def get_consultant(consultant_id: str) -> dict:
    """Get a consultant's full profile: skills, certifications, industries, timezone, and notes."""
    c = CONSULTANTS.get(consultant_id)
    return c if c else {"error": f"No consultant with id {consultant_id}"}


def get_availability(consultant_id: str, start_date: str, end_date: str) -> dict:
    """List a consultant's project assignments and PTO that overlap a date window.

    Dates are YYYY-MM-DD. allocation_pct is the share of the consultant's time booked on that project.
    """
    if consultant_id not in CONSULTANTS:
        return {"error": f"No consultant with id {consultant_id}"}
    assignments = [
        {
            "project_id": a["project_id"],
            "project": PROJECTS[a["project_id"]]["name"],
            "client": PROJECTS[a["project_id"]]["client"],
            "allocation_pct": a["allocation_pct"],
            "start": a["start"],
            "end": a["end"],
        }
        for a in DATA["assignments"]
        if a["consultant_id"] == consultant_id and _overlaps(a["start"], a["end"], start_date, end_date)
    ]
    pto = [
        {"start": p["start"], "end": p["end"]}
        for p in DATA["pto"]
        if p["consultant_id"] == consultant_id and _overlaps(p["start"], p["end"], start_date, end_date)
    ]
    return {"consultant_id": consultant_id, "window": {"start": start_date, "end": end_date},
            "assignments": assignments, "pto": pto}


def get_project(query: str) -> dict:
    """Look up projects by id, project name, or client name, with the current team.

    Every word in the query must appear in the project's id, name, or client, e.g. "Northwind claims".
    """
    words = query.lower().split()
    matches = [p for p in PROJECTS.values()
               if all(w in f"{p['id']} {p['name']} {p['client']}".lower() for w in words)]
    if not matches:
        return {"error": f"No project matching '{query}'"}
    out = []
    for p in matches:
        team = [
            {"consultant_id": a["consultant_id"], "name": CONSULTANTS[a["consultant_id"]]["name"], "role": a["role"],
             "allocation_pct": a["allocation_pct"], "start": a["start"], "end": a["end"]}
            for a in DATA["assignments"] if a["project_id"] == p["id"]
        ]
        out.append(p | {"team": team})
    return {"projects": out}


TOOL_FUNCTIONS = [search_consultants, get_consultant, get_availability, get_project]
TOOLS = [function_tool(f) for f in TOOL_FUNCTIONS]


class Pick(BaseModel):
    consultant_id: str
    reason: str


class StaffingReply(BaseModel):
    recommended: list[Pick]
    message: str


def build_agent(instructions: str, model=MODEL) -> Agent:
    return Agent(
        name="Bench",
        instructions=instructions,
        tools=TOOLS,
        model=model,
        output_type=StaffingReply,
    )


def _tool_calls(result) -> list[dict]:
    calls, by_id = [], {}
    for item in result.new_items:
        if item.type == "tool_call_item":
            raw = item.raw_item
            entry = {"name": raw.name, "arguments": json.loads(raw.arguments or "{}"), "output": None}
            by_id[raw.call_id] = entry
            calls.append(entry)
        elif item.type == "tool_call_output_item":
            entry = by_id.get(item.raw_item["call_id"])
            if entry is not None:
                entry["output"] = item.output
    return calls


def _run(request: str, instructions: str, model: str) -> dict:
    async def go():
        # A fresh client per run: each run gets its own event loop, and a client can't be shared across loops.
        async with AsyncOpenAI(**CONNECTION) as client:
            agent = build_agent(instructions, OpenAIResponsesModel(model, client))
            return await Runner.run(agent, request, max_turns=25)

    result = asyncio.run(go())
    reply: StaffingReply = result.final_output
    return {
        "recommended": [p.model_dump() for p in reply.recommended],
        "message": reply.message,
        "tool_calls": _tool_calls(result),
    }


def run_agent(request: str, instructions: str, model: str = MODEL) -> dict:
    """Run the agent on one request. Uses a worker thread so it also works inside Colab's event loop."""
    from .setup import route_agent_traces

    route_agent_traces()
    ctx = contextvars.copy_context()  # carry the current trace into the worker thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, _run, request, instructions, model).result()
