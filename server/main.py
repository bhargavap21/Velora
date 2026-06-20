"""
FastAPI backend for StratRL.

episode_core.py lives one level up — sys.path is patched before the import.
Run from the project root: uvicorn server.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Dict
from uuid import uuid4

sys.path.insert(0, "..")

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(title="StratRL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SEED_MAP: dict[str, dict] = {
    "tsla_rsi": {
        "ticker": "TSLA",
        "indicators": [{"type": "RSI", "period": 14, "name": "rsi_14"}],
        "entry_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 70}],
        "exit_conditions": [{"indicator": "rsi_14", "operator": "<", "value": 50}],
        "stop_loss": 0.10,
        "take_profit": 0.10,
    },
    "nvda_sma_rsi": {
        "ticker": "NVDA",
        "indicators": [
            {"type": "SMA", "period": 20, "name": "sma_20"},
            {"type": "RSI", "period": 14, "name": "rsi_14"},
        ],
        "entry_conditions": [
            {"indicator": "sma_20", "operator": ">", "value": "close"},
            {"indicator": "rsi_14", "operator": "<", "value": 55},
        ],
        "exit_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 55}],
        "stop_loss": 0.08,
        "take_profit": 0.08,
    },
    "aapl_sma_rsi": {
        "ticker": "AAPL",
        "indicators": [
            {"type": "SMA", "period": 10, "name": "sma_10"},
            {"type": "RSI", "period": 14, "name": "rsi_14"},
        ],
        "entry_conditions": [
            {"indicator": "sma_10", "operator": ">", "value": "close"},
            {"indicator": "rsi_14", "operator": "<", "value": 55},
        ],
        "exit_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 65}],
        "stop_loss": 0.05,
        "take_profit": 0.12,
    },
    "spy_ema_rsi": {
        "ticker": "SPY",
        "indicators": [
            {"type": "EMA", "period": 20, "name": "ema_20"},
            {"type": "RSI", "period": 14, "name": "rsi_14"},
        ],
        "entry_conditions": [
            {"indicator": "ema_20", "operator": ">", "value": "close"},
            {"indicator": "rsi_14", "operator": "<", "value": 50},
        ],
        "exit_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 65}],
        "stop_loss": 0.05,
        "take_profit": 0.15,
    },
    "tsla_sma_rsi": {
        "ticker": "TSLA",
        "indicators": [
            {"type": "SMA", "period": 20, "name": "sma_20"},
            {"type": "RSI", "period": 14, "name": "rsi_14"},
        ],
        "entry_conditions": [
            {"indicator": "sma_20", "operator": ">", "value": "close"},
            {"indicator": "rsi_14", "operator": "<", "value": 45},
        ],
        "exit_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 65}],
        "stop_loss": 0.07,
        "take_profit": 0.20,
    },
}

_runs: Dict[str, dict] = {}

_PARSE_SYSTEM = """You are a trading strategy parser. Convert the user's natural language description into a structured trading strategy JSON.

Available tickers: TSLA, NVDA, AAPL, SPY. Pick the most relevant one if not specified.

Available indicators:
- RSI: {"type": "RSI", "period": 14, "name": "rsi_14"} — momentum oscillator 0-100, oversold < 30, overbought > 70
- SMA: {"type": "SMA", "period": 20, "name": "sma_20"} — simple moving average of close price
- EMA: {"type": "EMA", "period": 20, "name": "ema_20"} — exponential moving average
- MACD: {"type": "MACD", "period": 12, "name": "macd"} — also creates "macd_signal" column
- BB: {"type": "BB", "period": 20, "name": "bb_20"} — creates "bb_20_upper", "bb_20_mid", "bb_20_lower"

Condition format: {"indicator": "name", "operator": ">"|"<"|">="|"<=", "value": number_or_"close"}
- For "price above SMA": {"indicator": "sma_20", "operator": "<", "value": "close"} (SMA < close means price is above SMA)
- For "RSI below 30": {"indicator": "rsi_14", "operator": "<", "value": 30}

Stop loss and take profit are decimal fractions: 0.05 = 5%, 0.10 = 10%.

Return ONLY valid JSON, no prose:
{
  "strategy": {
    "ticker": "...",
    "indicators": [...],
    "entry_conditions": [...],
    "exit_conditions": [...],
    "stop_loss": 0.05,
    "take_profit": 0.15
  },
  "summary": "One plain-English sentence describing what this strategy does."
}"""


class ParseRequest(BaseModel):
    description: str


class RunRequest(BaseModel):
    seeds: list[str] = []
    strategy: dict | None = None


def _run_episodes(run_id: str, seeds: list[dict]) -> None:
    # Runs episodes locally (no Modal) so the live SSE stream gets true per-turn
    # events as they happen. Modal is used for the offline batch job (modal_runner.py),
    # not the interactive demo — and isn't installed in this venv (hud-python's
    # a2a-sdk requires protobuf>=5.29.5, which conflicts with modal's protobuf<5.0).
    from episode_core import run_episode_events

    def run_one(seed: dict) -> None:
        try:
            for event in run_episode_events(seed):
                _runs[run_id]["events"].append(event)
                if event["type"] == "episode_complete":
                    _runs[run_id]["results"].append(event)
        except Exception as e:
            _runs[run_id]["events"].append({"type": "error", "message": str(e)})

    threads = [threading.Thread(target=run_one, args=(seed,)) for seed in seeds]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _runs[run_id]["status"] = "complete"


@app.post("/parse")
async def parse_strategy(body: ParseRequest):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=_PARSE_SYSTEM,
        messages=[{"role": "user", "content": body.description}],
    )
    text = response.content[0].text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise HTTPException(status_code=500, detail="Failed to parse Claude response as JSON")


@app.post("/run")
async def start_run(body: RunRequest, background_tasks: BackgroundTasks):
    if body.strategy:
        seeds = [body.strategy]
    elif "all" in body.seeds:
        seeds = list(_SEED_MAP.values())
    else:
        seeds = [_SEED_MAP[name] for name in body.seeds if name in _SEED_MAP]

    if not seeds:
        raise HTTPException(
            status_code=400,
            detail="No valid seeds. Available: " + ", ".join(_SEED_MAP),
        )

    run_id = f"run_{uuid4().hex[:8]}"
    _runs[run_id] = {"status": "running", "events": [], "results": []}
    background_tasks.add_task(_run_episodes, run_id, seeds)

    return {"run_id": run_id}


@app.get("/stream/{run_id}")
async def stream_run(run_id: str, request: Request):
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        sent = 0
        while True:
            events = _runs[run_id]["events"]
            while sent < len(events):
                yield {"data": json.dumps(events[sent])}
                sent += 1
            if _runs[run_id]["status"] == "complete":
                break
            await asyncio.sleep(0.3)

    return EventSourceResponse(event_generator())


@app.get("/results/{run_id}")
async def get_results(run_id: str):
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    run = _runs[run_id]
    if run["status"] != "complete":
        raise HTTPException(status_code=202, detail="Run still in progress")
    return {"run_id": run_id, "results": run["results"]}
