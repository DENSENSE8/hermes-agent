---
name: paperclip-memory
description: "Unified memory agent skill — routes all memory operations (save, recall, reflect) across nemoclaw, openclaw, paperclip, and sidecar sources through the Memory Brain Sidecar"
user-invocable: true
metadata:
  openclaw:
    emoji: "\U0001F4CE"
    requires:
      bins: ["curl", "node"]
      env: ["MEMORY_SIDECAR_URL", "PAPERCLIP_MEMORY_MODE"]
---

# Paperclip Memory — Unified Agent Memory Router

You are the central memory agent for the Garisek-OS ecosystem. All memory operations from any source are routed through you via the Memory Brain Sidecar (`:8791`).

## Registered Sources

| Source | Description | Collections |
|--------|-------------|-------------|
| `nemoclaw` | NemoClaw RAG pipeline, auto-triage, code improvement | hermes-memory, hermes-skills |
| `openclaw` | OpenClaw skills, hooks, apply_patch, process mgmt | hermes-memory, hermes-skills |
| `paperclip` | Paperclip orchestration — managed agent tasks/heartbeats | hermes-memory, agent-sessions |
| `sidecar` | Memory Brain Sidecar internal operations | hermes-memory, hermes-skills, agent-sessions |

## Memory Operations

### Save (any source)
Route through `POST http://localhost:8791/api/memory/save` with:
```json
{
  "source": "<source-tag>",
  "query": "<user query or task description>",
  "response": "<agent response or result>",
  "model": "<model used>",
  "session_id": "<session identifier>",
  "metadata": {"project": "garisek-os"}
}
```

### Recall (hybrid: Qdrant vector + Honcho dialectic)
```json
POST http://localhost:8791/api/memory/recall
{"query": "<search query>", "limit": 5, "source": "<optional source filter>"}
```

### Reflect (extract patterns from recent interactions)
```json
POST http://localhost:8791/api/memory/reflect
{"since_hours": 6}
```

### List Sources
```
GET http://localhost:8791/api/memory/sources
```

## When Invoked

When the user invokes `/paperclip-memory`:

1. Check health of all backends: `GET http://localhost:8791/health`
2. List registered sources: `GET http://localhost:8791/api/memory/sources`
3. Report status of unified memory system
4. If the user provides a query, perform hybrid recall across all sources
5. If the user asks to reflect, trigger reflection and report insights

## Session Persistence

Paperclip sessions are automatically saved to the `agent-sessions` Qdrant collection on every heartbeat. Sessions can be resumed via the `--resume` flag with the stored session ID.

Session persistence directory: `/home/avion/hermes-agent/.paperclip-sessions/`
