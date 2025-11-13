from agents.Teams import initialize_team
from agno.os import AgentOS
from fastapi import FastAPI


def run_team(session_state: dict) -> tuple[AgentOS, FastAPI]:
    team = initialize_team(session_state)
    agent_os = AgentOS(
        id="agentos-research-assistant",
        teams=[team],
    )
    app = agent_os.get_app()
    return agent_os, app
