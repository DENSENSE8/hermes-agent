#!/usr/bin/env python3
"""
YouTube RAG Query Tool

Embeds a question using Snowflake Arctic, searches the youtube_chunk
table via pgvector cosine similarity, and returns the top matching
transcript chunks with video metadata.

Usage:
    python youtube_rag_query.py "What did they say about X?"
    python youtube_rag_query.py "topic keywords" --channel UCxxx --top 10
    python youtube_rag_query.py "question" --list-channels

Examples:
    python youtube_rag_query.py "how to deploy to production"
    python youtube_rag_query.py "machine learning basics" --top 5
    python youtube_rag_query.py --list-channels
"""

import argparse
import json
import sys
from typing import Optional

import psycopg2
import psycopg2.extras
import requests

# ─── Config ───────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://100.64.222.195:11434"
EMBED_MODEL = "qwen3-embedding:0.6b"
DB_URL = "postgresql://jarvis:Jarvis_a9d61331dd828a80ad258fc8@100.64.222.195:5433/garisek"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama Snowflake Arctic."""
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL,
        "input": text,
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def list_channels(cur):
    """List all ingested channels."""
    cur.execute("""
        SELECT c.id, c.name, c.total_videos, c.last_scraped_at,
               count(DISTINCT v.id) FILTER (WHERE v.has_transcript) AS transcribed,
               count(DISTINCT ch.id) AS chunks
        FROM youtube_channel c
        LEFT JOIN youtube_video v ON v.channel_id = c.id
        LEFT JOIN youtube_chunk ch ON ch.channel_id = c.id
        GROUP BY c.id
        ORDER BY c.last_scraped_at DESC
    """)
    rows = cur.fetchall()
    if not rows:
        print("No channels ingested yet. Run youtube_rag_ingest.py first.")
        return

    print(f"{'Channel ID':<26} {'Name':<30} {'Videos':>7} {'Transcripts':>12} {'Chunks':>8} {'Last Scraped'}")
    print("-" * 110)
    for row in rows:
        ch_id, name, total, scraped_at, transcribed, chunks = row
        name_short = (name[:28] + "..") if len(name) > 30 else name
        scraped = scraped_at.strftime("%Y-%m-%d %H:%M") if scraped_at else "never"
        print(f"{ch_id:<26} {name_short:<30} {total or 0:>7} {transcribed:>12} {chunks:>8} {scraped}")


def search_chunks(cur, query: str, channel_id: Optional[str] = None,
                  top_k: int = 5) -> list[dict]:
    """Embed query and search for similar chunks."""
    emb = get_embedding(query)
    emb_str = "[" + ",".join(str(x) for x in emb) + "]"

    channel_filter = ""
    if channel_id:
        channel_filter = "AND ch.channel_id = %s"

    sql = f"""
        SELECT ch.content, ch.chunk_index, ch.video_id,
               v.title, v.url,
               1 - (ch.embedding <=> %s::vector) AS similarity
        FROM youtube_chunk ch
        JOIN youtube_video v ON v.id = ch.video_id
        WHERE ch.embedding IS NOT NULL {channel_filter}
        ORDER BY ch.embedding <=> %s::vector
        LIMIT %s
    """
    if channel_id:
        cur.execute(sql, (emb_str, channel_id, emb_str, top_k))
    else:
        cur.execute(sql, (emb_str, emb_str, top_k))

    results = []
    for row in cur.fetchall():
        content, chunk_idx, vid_id, title, url, similarity = row
        results.append({
            "similarity": round(float(similarity), 4),
            "video_title": title,
            "video_url": url,
            "video_id": vid_id,
            "chunk_index": chunk_idx,
            "content": content,
        })
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query YouTube RAG pipeline")
    parser.add_argument("query", nargs="?", help="Search query / question")
    parser.add_argument("--channel", "-c", default=None,
                        help="Filter by channel ID (UCxxx)")
    parser.add_argument("--top", "-k", type=int, default=5,
                        help="Number of results (default: 5)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted text")
    parser.add_argument("--list-channels", action="store_true",
                        help="List all ingested channels")
    args = parser.parse_args()

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    if args.list_channels:
        list_channels(cur)
        cur.close()
        conn.close()
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    print(f"Searching for: \"{args.query}\"")
    if args.channel:
        print(f"Filtering to channel: {args.channel}")
    print()

    results = search_chunks(cur, args.query, channel_id=args.channel, top_k=args.top)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        if not results:
            print("No results found. Ingest a channel first with youtube_rag_ingest.py")
        else:
            for i, r in enumerate(results, 1):
                sim_pct = r["similarity"] * 100
                print(f"━━━ Result {i} ({sim_pct:.1f}% match) ━━━")
                print(f"Video: {r['video_title']}")
                print(f"URL:   {r['video_url']}")
                print(f"Chunk: #{r['chunk_index']}")
                print(f"\n{r['content'][:600]}")
                if len(r["content"]) > 600:
                    print(f"... ({len(r['content'])} chars total)")
                print()

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
