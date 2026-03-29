"""
Evaluation pipeline configuration.
Reads secrets from the repo-root .env (loaded by main.py or explicitly here).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root so this module can be imported stand-alone
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# ── API keys ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
SERPER_API_KEY: str = os.environ["SERPER_API_KEY"]

# ── Model ────────────────────────────────────────────────────────────────────
MODEL: str = "gpt-4o-mini"
# LangChain-style ID used when calling L-MARS directly
LMARS_MODEL_ID: str = "openai:gpt-4o-mini"
TEMPERATURE: float = 0.0

# ── Token budgets ─────────────────────────────────────────────────────────────
MAX_TOKENS_ZERO_SHOT: int = 10
MAX_TOKENS_COT: int = 1024
MAX_TOKENS_RAG: int = 10
MAX_TOKENS_LMARS: int = 2048

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR: Path = _ROOT / "data"
BAREXAM_CSV: Path = DATA_DIR / "barexam_qa.csv"
LEGALSEARCHQA_JSON: Path = DATA_DIR / "legalsearchqa.json"

# ── Results paths ─────────────────────────────────────────────────────────────
EVAL_DIR: Path = _ROOT / "eval"
RESULTS_DIR: Path = EVAL_DIR / "results"
PILOT_DIR: Path = RESULTS_DIR / "pilot"
FULL_DIR: Path = RESULTS_DIR / "full"

for _d in (PILOT_DIR, FULL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Serper cache ──────────────────────────────────────────────────────────────
# Top-SERPER_FETCH_K results are always fetched and stored; top-k is sliced at
# query time.  If an existing cache entry has fewer results than requested k,
# it is transparently re-fetched with SERPER_FETCH_K.
SERPER_CACHE_DIR: Path = EVAL_DIR / "cache" / "serper"
SERPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Misc ──────────────────────────────────────────────────────────────────────
PILOT_SEED: int = 42
PILOT_N: int = 50
RATE_SLEEP_GPT: float = 0.1    # seconds between OpenAI calls
RATE_SLEEP_SERPER: float = 0.5  # seconds between live Serper calls
SERPER_TOP_K: int = 5           # default top-k returned to callers
SERPER_FETCH_K: int = 10        # how many results to fetch & cache (max ablation k)
SERPER_MAX_CHARS: int = 2000
