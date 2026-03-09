import json
import uuid

import psycopg
from agno.os import AgentOS
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger as log

from agents.Teams import initialize_team
from Config import config
from dbs.IngestToDB import ensure_gap_reports_table
from gap_analysis.models import GapReport
from gap_analysis.report import generate_dashboard_html


async def _run_gap_pipeline(
    report_id: str,
    query: str,
    max_records: int,
    days_back: int,
    filter_mode: str = "picos",
    filter_description: str = "",
) -> None:
    """Run gap analysis pipeline in background and store the result."""
    from dbs.IngestToDB import store_gap_report
    from gap_analysis.fetch import FilterMode
    from gap_analysis.pipeline import gap_analysis_flow

    try:
        async with await psycopg.AsyncConnection.connect(
            config.GAP_REPORTS_DB_URL
        ) as conn:
            await conn.execute(
                "UPDATE gap_reports SET status = %s WHERE id = %s",
                ("running", report_id),
            )
            await conn.commit()

        report = await gap_analysis_flow(
            query=query,
            max_records=max_records,
            days_back=days_back,
            filter_mode=FilterMode(filter_mode),
            filter_description=filter_description,
        )
        # Overwrite the auto-generated id so it matches the one we returned
        report.id = report_id
        await store_gap_report(report)
    except Exception as exc:
        log.error(f"Gap analysis pipeline failed for {report_id}: {exc}")
        try:
            async with await psycopg.AsyncConnection.connect(
                config.GAP_REPORTS_DB_URL
            ) as conn:
                await conn.execute(
                    "UPDATE gap_reports SET status = %s WHERE id = %s",
                    ("failed", report_id),
                )
                await conn.commit()
        except Exception as db_exc:
            log.error(f"Failed to update status for {report_id}: {db_exc}")


def _register_gap_routes(app: FastAPI) -> None:
    """Attach gap-analysis dashboard routes to the FastAPI app."""

    @app.get("/gap-analysis", response_class=JSONResponse)
    async def list_gap_reports():
        """List past gap analysis reports."""
        async with await psycopg.AsyncConnection.connect(
            config.GAP_REPORTS_DB_URL
        ) as conn:
            await ensure_gap_reports_table(conn)
            cur = await conn.execute(
                """
                SELECT id, query, created_at, scope, executive_summary, status
                FROM gap_reports
                ORDER BY created_at DESC
                """,
            )
            rows = await cur.fetchall()

        reports = [
            {
                "id": row[0],
                "query": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "scope": row[3],
                "executive_summary": row[4],
                "status": row[5],
            }
            for row in rows
        ]
        return reports

    @app.get("/gap-analysis/{report_id}", response_class=HTMLResponse)
    async def get_gap_dashboard(report_id: str):
        """Serve the interactive HTML dashboard for a specific gap report."""
        async with await psycopg.AsyncConnection.connect(
            config.GAP_REPORTS_DB_URL
        ) as conn:
            cur = await conn.execute(
                "SELECT report_json FROM gap_reports WHERE id = %s",
                (report_id,),
            )
            row = await cur.fetchone()

        if row is None or row[0] is None:
            return HTMLResponse(content="<h1>Report not found</h1>", status_code=404)

        report_data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        report = GapReport.model_validate(report_data)
        html = generate_dashboard_html(report)
        return HTMLResponse(content=html)

    @app.post("/gap-analysis", response_class=JSONResponse, status_code=202)
    async def trigger_gap_analysis(
        body: dict,
        background_tasks: BackgroundTasks,
    ):
        """Trigger a new gap analysis pipeline in the background."""
        query = body.get("query", "")
        if not query:
            return JSONResponse(content={"error": "query is required"}, status_code=400)

        max_records = body.get("max_records", config.GAP_DEFAULT_SCOPE)
        days_back = body.get("days_back", config.GAP_DEFAULT_DAYS)
        filter_mode = body.get("filter_mode", "picos")
        filter_description = body.get("filter_description", "")
        report_id = str(uuid.uuid4())

        if filter_mode not in ("picos", "llm", "none"):
            return JSONResponse(
                content={"error": "filter_mode must be 'picos', 'llm', or 'none'"},
                status_code=400,
            )

        # Create a placeholder row so the list endpoint shows it immediately
        async with await psycopg.AsyncConnection.connect(
            config.GAP_REPORTS_DB_URL
        ) as conn:
            await ensure_gap_reports_table(conn)
            await conn.execute(
                """
                INSERT INTO gap_reports (id, query, scope, status)
                VALUES (%s, %s, %s, %s)
                """,
                (report_id, query, max_records, "pending"),
            )
            await conn.commit()

        background_tasks.add_task(
            _run_gap_pipeline,
            report_id,
            query,
            max_records,
            days_back,
            filter_mode,
            filter_description,
        )

        return {"report_id": report_id, "status": "pending"}


async def _ensure_db_tables() -> None:
    """Create required tables at startup so agents can query them."""
    try:
        async with await psycopg.AsyncConnection.connect(
            config.GAP_REPORTS_DB_URL
        ) as conn:
            await ensure_gap_reports_table(conn)
    except Exception as e:
        log.warning(f"Could not ensure gap_reports table at startup: {e}")


def run_team(session_state: dict) -> tuple[AgentOS, FastAPI]:
    team = initialize_team(session_state)
    agent_os = AgentOS(
        id="agentos-research-assistant",
        teams=[team],
    )
    app = agent_os.get_app()
    _register_gap_routes(app)

    @app.on_event("startup")
    async def _startup():
        await _ensure_db_tables()

    return agent_os, app
