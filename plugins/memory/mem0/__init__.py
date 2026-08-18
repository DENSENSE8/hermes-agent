"""Mem0 memory plugin — MemoryProvider interface (self-hosted + cloud).

Local-first memory with LLM fact extraction, semantic search with reranking,
and automatic deduplication. Supports both modes:
  - Self-hosted: Qdrant + Ollama (default, no API key needed)
  - Cloud: Mem0 Platform API (set MEM0_API_KEY)

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC,
then extended for self-hosted mode (garisek-os).

Config via environment variables:
  MEM0_MODE           — 'local' (default) or 'cloud'
  MEM0_API_KEY        — Mem0 Platform API key (cloud mode only)
  MEM0_USER_ID        — User identifier (default: hermes-user)
  MEM0_AGENT_ID       — Agent identifier (default: hermes)
  MEM0_QDRANT_URL     — Qdrant URL (default: http://localhost:6333)
  MEM0_QDRANT_API_KEY ��� Qdrant API key (optional)
  MEM0_OLLAMA_URL     — Ollama URL (default: http://localhost:11434)
  MEM0_LLM_MODEL      — LLM model for fact extraction (default: qwen3:8b)
  MEM0_EMBED_MODEL    — Embedding model (default: snowflake-arctic-embed:m)
  MEM0_EMBED_DIMS     — Embedding dimensions (default: 768)
  MEM0_COLLECTION     — Qdrant collection name (default: mem0-memories)

Or via $HERMES_HOME/mem0.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "mode": os.environ.get("MEM0_MODE", "local"),
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": True,
        "keyword_search": False,
        # Local mode settings
        "qdrant_url": os.environ.get("MEM0_QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.environ.get("MEM0_QDRANT_API_KEY", ""),
        "ollama_url": os.environ.get("MEM0_OLLAMA_URL", "http://localhost:11434"),
        "llm_model": os.environ.get("MEM0_LLM_MODEL", "qwen3:8b"),
        "embed_model": os.environ.get("MEM0_EMBED_MODEL", "snowflake-arctic-embed:m"),
        "embed_dims": int(os.environ.get("MEM0_EMBED_DIMS", "768")),
        "collection": os.environ.get("MEM0_COLLECTION", "mem0-memories"),
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


def _build_local_mem0_config(cfg: dict) -> dict:
    """Build mem0 Memory.from_config() dict for self-hosted Qdrant + Ollama."""
    mem0_cfg = {
        "version": "v1.1",
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": cfg["qdrant_url"],
                "collection_name": cfg["collection"],
                "embedding_model_dims": cfg["embed_dims"],
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": cfg["llm_model"],
                "ollama_base_url": cfg["ollama_url"],
                "temperature": 0.2,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": cfg["embed_model"],
                "ollama_base_url": cfg["ollama_url"],
                "embedding_dims": cfg["embed_dims"],
            },
        },
    }

    if cfg.get("qdrant_api_key"):
        mem0_cfg["vector_store"]["config"]["api_key"] = cfg["qdrant_api_key"]

    return mem0_cfg


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory — self-hosted (Qdrant + Ollama) or cloud."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._mode = "local"
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        mode = cfg.get("mode", "local")
        if mode == "cloud":
            return bool(cfg.get("api_key"))
        # Local mode — available if mem0ai is installed
        try:
            import mem0  # noqa: F401
            return True
        except ImportError:
            return False

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "mode", "description": "Mode: 'local' (self-hosted) or 'cloud'", "default": "local", "choices": ["local", "cloud"]},
            {"key": "api_key", "description": "Mem0 Platform API key (cloud mode only)", "secret": True, "required": False, "env_var": "MEM0_API_KEY"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "qdrant_url", "description": "Qdrant URL (local mode)", "default": "http://localhost:6333", "env_var": "MEM0_QDRANT_URL"},
            {"key": "ollama_url", "description": "Ollama URL (local mode)", "default": "http://localhost:11434", "env_var": "MEM0_OLLAMA_URL"},
            {"key": "llm_model", "description": "LLM for fact extraction", "default": "qwen3:8b", "env_var": "MEM0_LLM_MODEL"},
            {"key": "embed_model", "description": "Embedding model", "default": "snowflake-arctic-embed:m", "env_var": "MEM0_EMBED_MODEL"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def _get_client(self):
        """Thread-safe client — local Memory or cloud MemoryClient."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                if self._mode == "cloud":
                    from mem0 import MemoryClient
                    self._client = MemoryClient(api_key=self._api_key)
                else:
                    from mem0 import Memory
                    local_cfg = _build_local_mem0_config(self._config)
                    self._client = Memory.from_config(local_cfg)
                    logger.info("Mem0 local memory initialized (Qdrant + Ollama)")
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._mode = self._config.get("mode", "local")
        self._api_key = self._config.get("api_key", "")
        self._user_id = self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)

    def system_prompt_block(self) -> str:
        mode_label = "self-hosted" if self._mode == "local" else "cloud"
        return (
            f"# Mem0 Memory ({mode_label})\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                results = client.search(
                    query=query,
                    user_id=self._user_id,
                    rerank=self._rerank,
                    limit=5,
                )
                if results:
                    # Local Memory returns dict with 'results' key; cloud returns list
                    items = results if isinstance(results, list) else results.get("results", results)
                    lines = [r.get("memory", "") for r in items if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                client.add(messages, user_id=self._user_id, agent_id=self._agent_id)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 temporarily unavailable (circuit breaker). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return json.dumps({"error": str(e)})

        if tool_name == "mem0_profile":
            try:
                memories = client.get_all(user_id=self._user_id)
                self._record_success()
                # Local returns dict with 'results'; cloud returns list
                items = memories if isinstance(memories, list) else memories.get("results", memories)
                if not items:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in items if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return json.dumps({"error": f"Failed to fetch profile: {e}"})

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = client.search(
                    query=query, user_id=self._user_id,
                    rerank=rerank, limit=top_k,
                )
                self._record_success()
                items = results if isinstance(results, list) else results.get("results", results)
                if not items:
                    return json.dumps({"result": "No relevant memories found."})
                out = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in items]
                return json.dumps({"results": out, "count": len(out)})
            except Exception as e:
                self._record_failure()
                return json.dumps({"error": f"Search failed: {e}"})

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return json.dumps({"error": "Missing required parameter: conclusion"})
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return json.dumps({"error": f"Failed to store: {e}"})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
