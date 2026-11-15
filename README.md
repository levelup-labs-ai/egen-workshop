# Egen workshop: evaluating a staffing agent

Hands-on notebook for the Egen session (Nov 16, 2026). Attendees run a pre-built agent over a reference dataset scored by three evaluators, annotate the replies in Arize, find error patterns, and write an LLM judge in Arize for one the evaluators miss.

## The agent

**Bench** is an internal staffing assistant. A delivery manager asks for someone ("senior BigQuery engineer, full-time from Jan 11") and Bench replies with a shortlist and a message.

It's one OpenAI Agents SDK agent with four tools over a JSON file. The notebook's first settings cell picks the provider (Egen's Azure endpoint by default, or OpenAI directly) and the models for the agent and the judge (`gpt-5.6-terra` by default):

| Tool | What it does |
|---|---|
| `search_consultants(skills, location?)` | Exact skill-tag match: "GCP" does not find people tagged "BigQuery" |
| `get_consultant(id)` | Full profile: certifications, industries, timezone, notes |
| `get_availability(id, start, end)` | Assignments and PTO that overlap a window |
| `get_project(query)` | Project details and current team |

## Files

- `Egen_Staffing_Evals_Workshop.ipynb`: the notebook (runs in Colab)
- `workshop_helpers/data/staffing.json`: 57 consultants, 13 projects, assignments, PTO. "Today" is 2026-11-16.
- `workshop_helpers/data/reference_cases.json`: 25 requests, each with good fits, people who must not be recommended (and why), and the caveats a manager must hear
- `workshop_helpers/tools.py`: tools and agent runner
- `workshop_helpers/evals.py`: two code checks (`no_ineligible_pick`, `found_good_fit`) and one LLM judge (`grounded`)
- `scripts/check_data.py`: confirms the reference answers agree with the tools

## Run locally

```bash
uv venv && uv pip install openai==2.31.0 openai-agents==0.13.6 arize==8.64.0 arize-otel==0.12.0 \
  openinference-instrumentation-openai-agents==1.4.1 pandas==2.3.3 python-dotenv==1.2.2 ipywidgets==8.1.9
cp .env.example .env   # Arize keys, plus the Azure endpoint and key (or OPENAI_API_KEY)
python scripts/check_data.py
```

## The app

The same exercises without notebook cells: browse the reference requests, run Bench on one and see what it did, then run and score the whole dataset.

```bash
uv pip install streamlit==1.67.0
streamlit run app.py
```
