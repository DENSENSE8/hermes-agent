#!/usr/bin/env python3
"""
Hermes Bridge API — lightweight FastAPI server on port 8780.
Exposes Hermes agent functionality to Garisek-OS via HTTP.

Primary-agent endpoints (HERMES-PRIMARY-AGENT-PLAN phases 3/4/5):
- POST /api/turn       — full flow: Honcho pre → model route → Ollama/Mac/Claude → Honcho post + Qdrant save
- POST /api/turn/sse   — same flow, SSE streaming response (for Phase 4 bridge inversion)
"""

import json
import os
import sqlite3
import time
import uuid
import yaml
from datetime import datetime
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Load .env from project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(Path.home() / ".hermes" / ".env")  # Hermes-scoped keys (QDRANT_API_KEY, etc.)

app = FastAPI(title="Hermes Bridge", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ────────────────────────────────────────────────────────────────
HERMES_DIR = PROJECT_DIR
MEMORY_FILE = HERMES_DIR / "MEMORY.md"
USER_FILE = HERMES_DIR / "USER.md"
SOUL_FILE = HERMES_DIR / "SOUL.md"
SKILLS_DIR = HERMES_DIR / "skills"
REPORTS_DIR = Path.home() / "hermes-reports"

# ── Chat log storage (SQLite) ───────────────────────────────────────────
DB_PATH = Path.home() / ".hermes" / "bridge.db"


def _get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        source TEXT,
        query TEXT,
        response TEXT,
        model TEXT,
        node TEXT,
        metadata TEXT
    )"""
    )
    conn.commit()
    return conn


# ── Health ───────────────────────────────────────────────────────────────
_start_time = time.time()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "hermes-bridge",
        "uptime_s": int(time.time() - _start_time),
        "version": "0.1.0",
    }


# ── Chat log (sidecar endpoint) ─────────────────────────────────────────
@app.post("/api/chat-log")
async def receive_chat_log(request: Request):
    body = await request.json()
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO chat_logs (timestamp, source, query, response, model, node, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                body.get("timestamp", datetime.utcnow().isoformat()),
                body.get("source", "unknown"),
                body.get("query", ""),
                body.get("response", ""),
                body.get("model", ""),
                body.get("node", ""),
                json.dumps(body.get("metadata", {})),
            ),
        )
        conn.commit()
        return {"ok": True, "logged": True}
    finally:
        conn.close()


@app.get("/api/chat-log")
async def get_chat_logs(limit: int = 50, source: str = None):
    """Retrieve recent conversation logs in JSON format."""
    conn = _get_db()
    try:
        if source:
            cur = conn.execute(
                "SELECT id, timestamp, source, query, response, model, node, metadata FROM chat_logs WHERE source = ? ORDER BY id DESC LIMIT ?",
                (source, limit),
            )
        else:
            cur = conn.execute(
                "SELECT id, timestamp, source, query, response, model, node, metadata FROM chat_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        rows = cur.fetchall()
        logs = []
        for row in rows:
            meta = {}
            try:
                meta = json.loads(row[7]) if row[7] else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            logs.append({
                "id": row[0],
                "timestamp": row[1],
                "source": row[2],
                "query": row[3],
                "response": row[4],
                "model": row[5],
                "node": row[6],
                "metadata": meta,
            })
        return {"logs": logs, "count": len(logs)}
    finally:
        conn.close()


# ── Memory ───────────────────────────────────────────────────────────────
@app.get("/api/memory")
async def list_memory():
    entries = {}
    for name, path in [
        ("MEMORY", MEMORY_FILE),
        ("USER", USER_FILE),
        ("SOUL", SOUL_FILE),
    ]:
        if path.exists():
            entries[name] = path.read_text(errors="replace")[:5000]
    return {"entries": entries}


# ── Skills ───────────────────────────────────────────────────────────────
@app.get("/api/skills")
async def list_skills():
    if not SKILLS_DIR.exists():
        return {"skills": []}
    skills = []
    for d in sorted(SKILLS_DIR.iterdir()):
        if d.is_dir():
            skill_md = d / "SKILL.md"
            desc = ""
            if skill_md.exists():
                text = skill_md.read_text(errors="replace")
                for line in text.splitlines():
                    if line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip().strip('"')
                        break
            skills.append({"name": d.name, "description": desc})
    return {"skills": skills}


# ── Run (execute agent task) ─────────────────────────────────────────────
@app.post("/api/run")
async def run_task(request: Request):
    body = await request.json()
    task = body.get("task", "")
    if not task:
        raise HTTPException(status_code=400, detail="Missing task field")

    # Try to call Ollama for a simple completion
    ollama_base = os.getenv("OPENAI_BASE_URL", "http://192.168.50.74:11434/v1")
    ollama_base = ollama_base.replace("/v1", "")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(
                f"{ollama_base}/api/generate",
                json={
                    "model": "qwen3:8b",
                    "prompt": task,
                    "stream": False,
                },
            )
            if res.status_code == 200:
                data = res.json()
                return {
                    "ok": True,
                    "response": data.get("response", ""),
                    "model": "qwen3:8b",
                }
    except Exception as e:
        return {"ok": False, "error": str(e), "detail": "Ollama not reachable"}

    return {"ok": False, "error": "No inference backend available"}


# ── Reports ──────────────────────────────────────────────────────────────
@app.get("/api/reports")
async def list_reports():
    if not REPORTS_DIR.exists():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return {"reports": []}
    reports = []
    for f in sorted(REPORTS_DIR.glob("*.json"), reverse=True)[:20]:
        try:
            data = json.loads(f.read_text())
            reports.append(
                {
                    "filename": f.name,
                    "timestamp": data.get("timestamp"),
                    "type": data.get("type"),
                    "summary": data.get("summary", ""),
                }
            )
        except Exception:
            pass
    return {"reports": reports}


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3/4/5 — Primary agent turn pipeline
# Honcho pre → model route → inference → Honcho commit + Qdrant save
# ════════════════════════════════════════════════════════════════════════════

HERMES_HOME = Path.home() / ".hermes"
HERMES_CONFIG_PATH = HERMES_HOME / "config.yaml"
HONCHO_URL = os.getenv("HONCHO_URL", "http://localhost:8800")
HONCHO_WORKSPACE = os.getenv("HONCHO_WORKSPACE", "avion")
MEMORY_SIDECAR_URL = os.getenv("MEMORY_SIDECAR_URL", "http://localhost:8791")
NEMOCLAW_ABSTRACT_URL = os.getenv("NEMOCLAW_ABSTRACT_URL", "http://localhost:8765/api/abstract")
VECTOR_INGEST_URL = os.getenv("VECTOR_INGEST_URL", "http://127.0.0.1:9880/ingest")


def _load_hermes_config() -> dict:
    """Load ~/.hermes/config.yaml. Cached per-call — cheap enough."""
    try:
        return yaml.safe_load(HERMES_CONFIG_PATH.read_text()) or {}
    except Exception:
        return {}


def _emit_log(event: str, **fields):
    """Fire-and-forget structured log to Vector (→ Loki). Never raises."""
    try:
        payload = {"service": "hermes-bridge", "event": event,
                   "ts": datetime.utcnow().isoformat() + "Z", **fields}
        httpx.post(VECTOR_INGEST_URL, json=payload, timeout=2.0)
    except Exception:
        pass


async def _honcho_get_user_model(peer_name: str) -> str:
    """Fetch dialectic user-model snapshot. Returns plain text to inject into prompt."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            # Ensure workspace
            r = await c.post(f"{HONCHO_URL}/v3/workspaces/list",
                             json={"filter": {"name": HONCHO_WORKSPACE}})
            items = r.json().get("items", []) if r.status_code == 200 else []
            if not items:
                r = await c.post(f"{HONCHO_URL}/v3/workspaces",
                                 json={"name": HONCHO_WORKSPACE, "metadata": {}})
                ws_id = r.json()["id"]
            else:
                ws_id = items[0]["id"]
            # Ensure peer
            r = await c.post(f"{HONCHO_URL}/v3/workspaces/{ws_id}/peers/list",
                             json={"filter": {"name": peer_name}})
            items = r.json().get("items", []) if r.status_code == 200 else []
            if not items:
                r = await c.post(f"{HONCHO_URL}/v3/workspaces/{ws_id}/peers",
                                 json={"name": peer_name, "metadata": {}})
                peer_id = r.json()["id"]
            else:
                peer_id = items[0]["id"]
            # Dialectic query
            r = await c.post(
                f"{HONCHO_URL}/v3/workspaces/{ws_id}/peers/{peer_id}/chat",
                json={"queries": ["What should I know about this user?"]},
            )
            if r.status_code == 200:
                data = r.json()
                content = data.get("content") or data.get("response") or []
                if isinstance(content, list):
                    return "\n".join(str(x) for x in content if x)
                return str(content)
    except Exception as e:
        _emit_log("honcho.get.fail", error=str(e), peer=peer_name)
    return ""


async def _honcho_commit_turn(peer_name: str, session_name: str, query: str, response: str):
    """Commit a user→assistant turn. Fire-and-forget; failures logged not raised."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            # Re-resolve workspace/peer (could cache but keeping simple)
            r = await c.post(f"{HONCHO_URL}/v3/workspaces/list",
                             json={"filter": {"name": HONCHO_WORKSPACE}})
            ws_id = r.json()["items"][0]["id"]
            # User peer
            r = await c.post(f"{HONCHO_URL}/v3/workspaces/{ws_id}/peers/list",
                             json={"filter": {"name": peer_name}})
            user_peer_id = r.json()["items"][0]["id"]
            # Assistant peer
            r = await c.post(f"{HONCHO_URL}/v3/workspaces/{ws_id}/peers/list",
                             json={"filter": {"name": "hermes"}})
            items = r.json().get("items", [])
            if not items:
                r = await c.post(f"{HONCHO_URL}/v3/workspaces/{ws_id}/peers",
                                 json={"name": "hermes", "metadata": {}})
                asst_peer_id = r.json()["id"]
            else:
                asst_peer_id = items[0]["id"]
            # Session
            r = await c.post(f"{HONCHO_URL}/v3/workspaces/{ws_id}/sessions/list",
                             json={"filter": {"name": session_name}})
            items = r.json().get("items", [])
            if not items:
                r = await c.post(
                    f"{HONCHO_URL}/v3/workspaces/{ws_id}/sessions",
                    json={
                        "name": session_name, "metadata": {},
                        "peers": {
                            user_peer_id: {"observe_me": True, "observe_others": False},
                            asst_peer_id: {"observe_me": False, "observe_others": True},
                        },
                    },
                )
                session_id = r.json()["id"]
            else:
                session_id = items[0]["id"]
            # Messages
            await c.post(
                f"{HONCHO_URL}/v3/workspaces/{ws_id}/sessions/{session_id}/messages",
                json={"messages": [
                    {"content": query, "peer_id": user_peer_id, "metadata": {}},
                    {"content": response, "peer_id": asst_peer_id, "metadata": {}},
                ]},
            )
    except Exception as e:
        _emit_log("honcho.commit.fail", error=str(e), peer=peer_name)


def _pick_model(query: str, preferred: str | None = None) -> tuple[str, dict, str]:
    """Return (alias, model_config_dict, reason) per model-routing skill rules.

    If `preferred` is a known alias, pin to it. Otherwise apply routing heuristic.
    """
    cfg = _load_hermes_config()
    aliases = cfg.get("models", {})
    if preferred and preferred in aliases:
        return preferred, aliases[preferred], f"user-override: /model {preferred}"

    q = query.lower()
    # Critical: production code paths, security/auth/crypto
    critical_markers = ("rewrite", "refactor production", "auth token", "credential",
                        "sql injection", "security audit", "cve", "/src/", "nemoclaw-fork/scripts/")
    if any(m in q for m in critical_markers):
        if "critical" in aliases:
            return "critical", aliases["critical"], "matched critical markers"

    # Deep: multi-step reasoning, code review of diffs
    deep_markers = ("review this diff", "compare these", "plan the", "multi-step",
                    "explain why", "trace through")
    if any(m in q for m in deep_markers) or len(query) > 1500:
        if "deep" in aliases:
            return "deep", aliases["deep"], "matched deep markers / long query"

    # Default: fast
    if "fast" in aliases:
        return "fast", aliases["fast"], "default"
    # Fallback: legacy model: block
    default_model = cfg.get("model", {}).get("default", "qwen3:8b")
    return "legacy", {"provider": "openai-compatible",
                      "api_base": os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11437/v1"),
                      "model": default_model}, "fallback to legacy default"


async def _pii_abstract(text: str) -> tuple[str, bool]:
    """Send text through NemoClaw abstractor. Returns (abstracted_text, safe)."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(NEMOCLAW_ABSTRACT_URL, json={"text": text})
            if r.status_code == 200:
                d = r.json()
                return d.get("abstracted", text), bool(d.get("safe", False))
    except Exception as e:
        _emit_log("pii.abstract.fail", error=str(e))
    return text, False  # fail-closed: treat as unsafe


async def _call_model(alias: str, model_cfg: dict, prompt: str, context: str,
                      _fallback_depth: int = 0) -> str:
    """Call the selected backend with graceful degradation.

    Fallback chain: critical → deep → fast. Each hop emits a structured log so
    dashboards can show how often escalated routes actually reach their target.
    """
    provider = model_cfg.get("provider", "openai-compatible")

    async def _fallback_to(target: str, reason: str) -> str:
        if _fallback_depth >= 2:
            return f"[all routes exhausted ({reason})]"
        cfg = _load_hermes_config().get("models", {})
        if target in cfg:
            _emit_log("route.fallback", from_alias=alias, to=target, reason=reason)
            return await _call_model(target, cfg[target], prompt, context,
                                     _fallback_depth=_fallback_depth + 1)
        return f"[no {target} route configured ({reason})]"

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return await _fallback_to("deep", "ANTHROPIC_API_KEY missing")
        # Gate PII first
        abstracted, safe = await _pii_abstract(prompt)
        if not safe:
            _emit_log("pii.gate.blocked", alias=alias)
            return await _fallback_to("deep", "PII gate unavailable/failed")
        try:
            async with httpx.AsyncClient(timeout=120.0) as c:
                r = await c.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json={
                        "model": model_cfg.get("model", "claude-sonnet-4-6"),
                        "max_tokens": model_cfg.get("max_tokens", 8192),
                        "system": context,
                        "messages": [{"role": "user", "content": abstracted}],
                    },
                )
                if r.status_code == 200:
                    return r.json()["content"][0]["text"]
                return await _fallback_to("deep", f"anthropic {r.status_code}")
        except Exception as e:
            return await _fallback_to("deep", f"anthropic exc: {type(e).__name__}")

    # openai-compatible (fast, deep, or legacy) — shorter timeout so fallback is quick
    base = model_cfg.get("api_base", "http://172.24.96.1:11434/v1")
    timeout_s = 30.0 if alias == "fast" else 60.0
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as c:
            r = await c.post(
                f"{base}/chat/completions",
                json={
                    "model": model_cfg.get("model", "qwen3:8b-nothink"),
                    "max_tokens": model_cfg.get("max_tokens", 2048),
                    "messages": [
                        {"role": "system", "content": context},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            # Non-200 — fall back only if we're on deep (fast is the floor)
            if alias == "deep":
                return await _fallback_to("fast", f"{alias} {r.status_code}")
            return f"[{alias} error {r.status_code}: {r.text[:200]}]"
    except Exception as e:
        if alias == "deep":
            return await _fallback_to("fast", f"{alias} unreachable: {type(e).__name__}")
        return f"[{alias} exception: {type(e).__name__}: {e}]"


async def _qdrant_save(query: str, response: str, peer_name: str, session_name: str, model: str):
    """Save the turn to Memory Sidecar for vector indexing. Fire-and-forget."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            await c.post(
                f"{MEMORY_SIDECAR_URL}/api/memory/save",
                json={
                    "text": f"Q: {query}\nA: {response}",
                    "session_id": session_name,
                    "source": "hermes-primary",
                    "query": query,
                    "response": response,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "metadata": {"peer": peer_name, "model": model},
                },
            )
    except Exception as e:
        _emit_log("qdrant.save.fail", error=str(e))


@app.post("/api/turn")
async def run_turn(request: Request):
    """Primary-agent turn (Phase 3+5 activated; feed from Phase 4 bridges).

    Request body:
      {
        "query": "...",                    # required
        "user_id": "telegram-7838951371",  # stable peer id for Honcho
        "session_id": "telegram-chat-42",  # stable session
        "surface": "telegram|discord|garisek",
        "preferred_model": "fast|deep|critical"  # optional /model override
      }

    Response:
      { "ok": true, "response": "...", "model_alias": "deep", "turn_id": "...",
        "timing_ms": {...}, "honcho_context_chars": 340 }
    """
    body = await request.json()
    query = body.get("query") or body.get("task", "")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query field")

    peer_name = body.get("user_id", "anonymous")
    session_name = body.get("session_id", f"adhoc-{int(time.time())}")
    surface = body.get("surface", "unknown")
    preferred = body.get("preferred_model")
    turn_id = str(uuid.uuid4())
    t0 = time.time()

    _emit_log("turn.start", turn_id=turn_id, peer=peer_name, session=session_name,
              surface=surface, query_len=len(query))

    # 1. Honcho pre-hook
    t_honcho_start = time.time()
    honcho_context = await _honcho_get_user_model(peer_name)
    t_honcho_ms = int((time.time() - t_honcho_start) * 1000)

    # 2. Model routing
    alias, model_cfg, reason = _pick_model(query, preferred)
    _emit_log("model.route", turn_id=turn_id, chosen=alias, reason=reason,
              model=model_cfg.get("model"))

    # 3. Build system prompt (identity + user model + SOUL)
    soul = (HERMES_HOME / "SOUL.md").read_text(errors="replace")[:3000] if (HERMES_HOME / "SOUL.md").exists() else ""
    system_parts = ["You are Hermes, a self-improving agent."]
    if soul: system_parts.append(f"--- IDENTITY (SOUL.md) ---\n{soul}")
    if honcho_context: system_parts.append(f"--- WHAT YOU KNOW ABOUT THIS USER ---\n{honcho_context}")
    context = "\n\n".join(system_parts)

    # 4. Call backend
    t_model_start = time.time()
    response_text = await _call_model(alias, model_cfg, query, context)
    t_model_ms = int((time.time() - t_model_start) * 1000)

    # 5. Post-hooks: Honcho commit + Qdrant save (both fire-and-forget)
    import asyncio
    asyncio.create_task(_honcho_commit_turn(peer_name, session_name, query, response_text))
    asyncio.create_task(_qdrant_save(query, response_text, peer_name, session_name,
                                     model_cfg.get("model", alias)))

    total_ms = int((time.time() - t0) * 1000)
    _emit_log("turn.complete", turn_id=turn_id, peer=peer_name, surface=surface,
              model=model_cfg.get("model"), alias=alias,
              duration_ms=total_ms, honcho_ms=t_honcho_ms, model_ms=t_model_ms,
              query_len=len(query), response_len=len(response_text))

    return {
        "ok": True,
        "turn_id": turn_id,
        "response": response_text,
        "model_alias": alias,
        "model": model_cfg.get("model"),
        "routing_reason": reason,
        "timing_ms": {"total": total_ms, "honcho": t_honcho_ms, "model": t_model_ms},
        "honcho_context_chars": len(honcho_context),
    }


@app.post("/api/turn/sse")
async def run_turn_sse(request: Request):
    """SSE-streaming variant of /api/turn. Event types: token, meta, done, error.

    For Phase 4 bridge inversion — Discord/Telegram/Garisek can stream tokens live.
    """
    body = await request.json()
    query = body.get("query") or body.get("task", "")
    if not query:
        raise HTTPException(status_code=400, detail="Missing query field")

    peer_name = body.get("user_id", "anonymous")
    session_name = body.get("session_id", f"adhoc-{int(time.time())}")
    surface = body.get("surface", "unknown")
    preferred = body.get("preferred_model")
    turn_id = str(uuid.uuid4())

    async def event_stream():
        import asyncio
        t0 = time.time()
        try:
            honcho_context = await _honcho_get_user_model(peer_name)
            alias, model_cfg, reason = _pick_model(query, preferred)
            yield f"event: meta\ndata: {json.dumps({'turn_id': turn_id, 'alias': alias, 'reason': reason})}\n\n"

            # Streaming inference — openai-compatible `stream: true`
            base = model_cfg.get("api_base")
            if model_cfg.get("provider") == "openai-compatible" and base:
                soul = (HERMES_HOME / "SOUL.md").read_text(errors="replace")[:3000] if (HERMES_HOME / "SOUL.md").exists() else ""
                ctx_parts = ["You are Hermes, a self-improving agent."]
                if soul: ctx_parts.append(f"--- IDENTITY ---\n{soul}")
                if honcho_context: ctx_parts.append(f"--- USER MODEL ---\n{honcho_context}")
                accumulated = ""
                async with httpx.AsyncClient(timeout=120.0) as c:
                    async with c.stream("POST", f"{base}/chat/completions",
                                         json={"model": model_cfg.get("model"),
                                               "max_tokens": model_cfg.get("max_tokens", 2048),
                                               "messages": [{"role": "system", "content": "\n\n".join(ctx_parts)},
                                                            {"role": "user", "content": query}],
                                               "stream": True,
                                               "reasoning_effort": "none"}) as resp:
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                                if delta:
                                    accumulated += delta
                                    yield f"event: token\ndata: {json.dumps({'t': delta})}\n\n"
                            except Exception:
                                continue
                # Fire post-hooks
                asyncio.create_task(_honcho_commit_turn(peer_name, session_name, query, accumulated))
                asyncio.create_task(_qdrant_save(query, accumulated, peer_name, session_name,
                                                  model_cfg.get("model", alias)))
                total_ms = int((time.time() - t0) * 1000)
                _emit_log("turn.complete", turn_id=turn_id, peer=peer_name, surface=surface,
                          alias=alias, duration_ms=total_ms, response_len=len(accumulated),
                          streamed=True)
                yield f"event: done\ndata: {json.dumps({'turn_id': turn_id, 'total_ms': total_ms, 'response_len': len(accumulated)})}\n\n"
            else:
                # Non-streaming providers (anthropic) — emit full block
                full = await _call_model(alias, model_cfg, query, "\n".join([f"{honcho_context}"]))
                yield f"event: token\ndata: {json.dumps({'t': full})}\n\n"
                asyncio.create_task(_honcho_commit_turn(peer_name, session_name, query, full))
                asyncio.create_task(_qdrant_save(query, full, peer_name, session_name,
                                                  model_cfg.get("model", alias)))
                yield f"event: done\ndata: {json.dumps({'turn_id': turn_id})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e), 'turn_id': turn_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")




@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible shim so Garisek HERMES_WEBUI_URL can use the bridge."""
    body = await request.json()
    messages = body.get("messages") or []
    system_bits = []
    user_bits = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if role == "system":
            system_bits.append(str(content))
        elif role == "user":
            user_bits.append(str(content))
        elif role == "assistant":
            user_bits.append(f"(assistant previously said)\n{content}")
    query = "\n\n".join(user_bits).strip() or body.get("prompt") or ""
    if system_bits and query:
        query = f"[context]\n{chr(10).join(system_bits)}\n\n[user]\n{query}"
    if not query:
        raise HTTPException(status_code=400, detail="No user message")

    class _Req:
        async def json(self_inner):
            return {
                "query": query,
                "user_id": body.get("user") or "garisek",
                "session_id": body.get("session_id") or f"webui-{int(time.time())}",
                "surface": "garisek",
            }

    result = await run_turn(_Req())
    text = result.get("response") or ""
    return {
        "id": result.get("turn_id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.get("model") or body.get("model") or "hermes",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
async def openai_models():
    return {
        "object": "list",
        "data": [{"id": "hermes", "object": "model", "owned_by": "hermes"}],
    }


if __name__ == "__main__":
    print(f"[hermes-bridge] Starting on port 8780 ...")
    print(f"[hermes-bridge] Project dir: {PROJECT_DIR}")
    print(f"[hermes-bridge] Primary endpoints: /api/turn, /api/turn/sse")
    uvicorn.run(app, host="0.0.0.0", port=8780, log_level="info")
