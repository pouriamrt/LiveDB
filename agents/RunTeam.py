import asyncio
import json
import uuid

import psycopg
from agno.os import AgentOS
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger as log

from agents.Teams import initialize_team
from Config import config
from gap_analysis.models import GapReport
from gap_analysis.report import generate_dashboard_html


async def _run_gap_pipeline(
    report_id: str, query: str, max_records: int, start_day: int
) -> None:
    """Run gap analysis pipeline in background and store the result."""
    from dbs.IngestToDB import store_gap_report
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
            start_day=start_day,
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
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gap_reports (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    scope INTEGER,
                    report_json JSONB,
                    executive_summary TEXT,
                    status TEXT DEFAULT 'completed'
                )
                """,
            )
            await conn.commit()
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
        start_day = body.get("start_day", config.GAP_DEFAULT_DAYS)
        report_id = str(uuid.uuid4())

        # Create a placeholder row so the list endpoint shows it immediately
        async with await psycopg.AsyncConnection.connect(
            config.GAP_REPORTS_DB_URL
        ) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gap_reports (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    scope INTEGER,
                    report_json JSONB,
                    executive_summary TEXT,
                    status TEXT DEFAULT 'completed'
                )
                """,
            )
            await conn.execute(
                """
                INSERT INTO gap_reports (id, query, scope, status)
                VALUES (%s, %s, %s, %s)
                """,
                (report_id, query, max_records, "pending"),
            )
            await conn.commit()

        background_tasks.add_task(
            asyncio.to_thread,
            lambda: asyncio.run(
                _run_gap_pipeline(report_id, query, max_records, start_day)
            ),
        )

        return {"report_id": report_id, "status": "pending"}


def run_team(session_state: dict) -> tuple[AgentOS, FastAPI]:
    team = initialize_team(session_state)
    agent_os = AgentOS(
        id="agentos-research-assistant",
        teams=[team],
    )
    app = agent_os.get_app()
    _register_gap_routes(app)
    return agent_os, app
