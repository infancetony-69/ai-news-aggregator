import os
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

load_dotenv()

from app.database.repository import Repository

# In-memory pipeline status
_pipeline_status = {
    "running": False,
    "last_run": None,
    "last_result": None
}


def _run_pipeline_sync(hours: int = 24, top_n: int = 10):
    from app.daily_runner import run_daily_pipeline
    if _pipeline_status["running"]:
        return
    _pipeline_status["running"] = True
    try:
        result = run_daily_pipeline(hours=hours, top_n=top_n)
        _pipeline_status["last_result"] = result
        _pipeline_status["last_run"] = datetime.now().isoformat()
    finally:
        _pipeline_status["running"] = False


# Scheduler: runs pipeline every day at 8:00 AM
scheduler = BackgroundScheduler()
scheduler.add_job(
    _run_pipeline_sync,
    trigger="cron",
    hour=8,
    minute=0,
    id="daily_pipeline",
    replace_existing=True
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create tables on startup (safe to run multiple times)
    from app.database.create_tables import create_tables
    create_tables()
    scheduler.start()
    print("Scheduler started — pipeline will run daily at 8:00 AM")
    yield
    scheduler.shutdown()
    print("Scheduler stopped")


app = FastAPI(
    title="AI News Aggregator",
    description="Scrapes, summarizes, and emails AI news digests daily",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRequest(BaseModel):
    hours: int = 24
    top_n: int = 10


@app.get("/")
def root():
    return {"message": "AI News Aggregator API", "docs": "/docs"}


@app.post("/run-pipeline")
def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    if _pipeline_status["running"]:
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    background_tasks.add_task(_run_pipeline_sync, request.hours, request.top_n)
    return {"message": "Pipeline started", "hours": request.hours, "top_n": request.top_n}


@app.get("/status")
def get_status():
    next_run = None
    job = scheduler.get_job("daily_pipeline")
    if job and job.next_run_time:
        next_run = job.next_run_time.isoformat()
    return {
        "running": _pipeline_status["running"],
        "last_run": _pipeline_status["last_run"],
        "last_result": _pipeline_status["last_result"],
        "next_scheduled_run": next_run
    }


@app.get("/digests")
def get_digests(hours: int = 24):
    repo = Repository()
    digests = repo.get_recent_digests(hours=hours)
    return {"count": len(digests), "digests": digests}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
