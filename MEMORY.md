# Hermes Memory

## Cluster Infrastructure
- **Windows host**: Tailscale 100.64.222.195, Ollama :11434 (qwen3:8b, snowflake-arctic-embed:m)
- **WSL (Ubuntu 22.04)**: user `avion`, Tailscale 100.69.4.21
- **Mac M5 Pro**: Tailscale 100.64.38.223, MLX Server :8080 (Qwen2.5-32B, Qwen2.5-Coder-32B)
- **Jetson Nano Orin**: SSH `mikes@100.86.54.103`, gemma:2b — often offline

## Services
- 3100: Paperclip dashboard | 5433: PostgreSQL | 6333: Qdrant | 6379: Redis
- 7233: Temporal | 8000: Kong/Supabase | 8765: NemoClaw RAG | 8780: Hermes Bridge
- 8790: UAM Audio | 8791: Memory Sidecar | 8800: Honcho | 9090: Prometheus
- 18789: OpenClaw Gateway | 54329: Paperclip embedded PG

## OpenClaw
- Gateway on port 18789, Ed25519 device auth, 5 providers, 13+ custom skills
- Channels: Telegram (@AvionBirdbot) + Discord

## Key Patterns
- WSL2 path translation: Windows `E:\foo` → `/mnt/e/foo` (toWslPath in @paperclipai/shared)
- Paperclip .env: `~/.paperclip/instances/default/.env` (primary), `paperclip/server/.env` (fallback)
- Memory pipeline: save → Qdrant (vector) + Honcho (dialectic) + Hermes (chat-log)
- Honcho v3 API: workspaces → peers → sessions/conclusions
