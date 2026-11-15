"""Load secrets (Colab or .env), connect Arize tracing, and return clients."""

import os
import threading
from datetime import datetime

from arize import ArizeClient
from arize.otel import register
from openai import OpenAI
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry import trace

from . import tools

PROJECT_NAME = f"Egen-Staffing-Workshop-{datetime.now():%Y-%m-%d}"
SECRETS = {
    "openai": ("OPENAI_API_KEY", "ARIZE_API_KEY", "ARIZE_SPACE_ID"),
    "azure": ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "ARIZE_API_KEY", "ARIZE_SPACE_ID"),
}


def _load_secrets(keys) -> dict:
    """Environment variables or a .env file first; Colab's Secrets panel if those don't have everything."""
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    secrets = {k: os.environ.get(k) for k in keys}
    if not all(secrets.values()):
        try:
            from google.colab import userdata
            secrets = {k: userdata.get(k) for k in keys}
        except ImportError:
            pass
    return secrets


def setup_clients(provider: str = "openai", project_name: str = PROJECT_NAME) -> dict:
    """provider "openai" calls OpenAI directly; "azure" calls the OpenAI-compatible endpoint in AZURE_OPENAI_ENDPOINT."""
    secrets = _load_secrets(SECRETS[provider])
    missing = [k for k, v in secrets.items() if not v]
    if missing:
        raise RuntimeError(f"Missing secrets: {missing}. Add them in Colab's Secrets panel (key icon) or a .env file.")
    if provider == "azure":
        conn = {"base_url": secrets["AZURE_OPENAI_ENDPOINT"].rstrip("/"), "api_key": secrets["AZURE_OPENAI_API_KEY"]}
        tools.CONNECTION = conn  # the agent
        openai_client = OpenAI(**conn)  # the judges
    else:
        os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]
        tools.CONNECTION = {}
        openai_client = OpenAI()

    _TRACING.update(secrets=secrets, default_project=project_name)
    route_agent_traces()

    return {
        "openai": openai_client,
        "arize": ArizeClient(api_key=secrets["ARIZE_API_KEY"]),
        "space_id": secrets["ARIZE_SPACE_ID"],
        "project_name": project_name,
    }


_TRACING: dict = {"secrets": None, "default_project": None, "project": None}
_LOCK = threading.Lock()


def route_agent_traces() -> None:
    """Send agent spans to the Arize project of the span we're running under.

    Inside experiments.run that's the experiment's trace project, so each experiment row shows the full
    agent trace (LLM calls and tool calls). Outside an experiment it's the workshop project.
    """
    if not _TRACING["secrets"]:
        return  # setup_clients() not called; run untraced
    parent = trace.get_current_span()
    project = getattr(getattr(parent, "resource", None), "attributes", {}).get("openinference.project.name")
    project = project or _TRACING["default_project"]
    with _LOCK:
        if project == _TRACING["project"]:
            return
        provider = register(
            space_id=_TRACING["secrets"]["ARIZE_SPACE_ID"],
            api_key=_TRACING["secrets"]["ARIZE_API_KEY"],
            project_name=project,
            set_global_tracer_provider=False,
            verbose=False,
        )
        instrumentor = OpenAIAgentsInstrumentor()
        instrumentor.uninstrument()
        instrumentor.instrument(tracer_provider=provider)
        _TRACING["project"] = project
