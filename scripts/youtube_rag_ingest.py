#!/usr/bin/env python3
"""
YouTube Channel RAG Ingest Pipeline

Scrapes all videos from a YouTube channel, fetches transcripts,
chunks the text, embeds via Snowflake Arctic (Ollama), and stores
in Postgres with pgvector for later retrieval.

Usage:
    python youtube_rag_ingest.py <channel_id_or_url> [--limit N] [--chunk-size 400] [--overlap 50]

Examples:
    python youtube_rag_ingest.py UCxxxxxxxxxxxxxxxxxxxxxx
    python youtube_rag_ingest.py "https://www.youtube.com/@ChannelName" --limit 10
    python youtube_rag_ingest.py UCxxxxxxxxxxxxxxxxxxxxxx --chunk-size 300 --overlap 40
"""

import argparse
import json
import re
import sys
import time
from typing import Optional

import psycopg2
import psycopg2.extras
import requests
import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi

# ─── Config ───────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://100.64.222.195:11434"
EMBED_MODEL = "qwen3-embedding:0.6b"
EMBED_DIM = 1024

DB_URL = "postgresql://jarvis:Jarvis_a9d61331dd828a80ad258fc8@100.64.222.195:5433/garisek"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_channel_id(input_str: str) -> str:
    """Extract channel ID from URL or return as-is if already an ID."""
    input_str = input_str.strip()
    # Direct channel ID
    if re.match(r'^UC[\w-]{22}$', input_str):
        return input_str
    # URL with /channel/UCxxx
    m = re.search(r'/channel/(UC[\w-]{22})', input_str)
    if m:
        return m.group(1)
    # For @handle URLs, scrapetube can resolve them — return as-is
    return input_str


def get_channel_videos(channel_input: str, limit: Optional[int] = None):
    """Get all video metadata from a channel using scrapetube."""
    kwargs = {}
    if channel_input.startswith("UC") and len(channel_input) == 24:
        kwargs["channel_id"] = channel_input
    elif channel_input.startswith("http"):
        kwargs["channel_url"] = channel_input
    else:
        # Might be a handle like @ChannelName
        kwargs["channel_url"] = f"https://www.youtube.com/{channel_input}"

    videos = []
    for i, v in enumerate(scrapetube.get_channel(**kwargs)):
        if limit and i >= limit:
            break
        vid_id = v.get("videoId", "")
        title = ""
        if "title" in v and "runs" in v["title"]:
            title = v["title"]["runs"][0].get("text", "")
        elif "title" in v and "simpleText" in v["title"]:
            title = v["title"]["simpleText"]

        length_text = ""
        if "lengthText" in v and "simpleText" in v["lengthText"]:
            length_text = v["lengthText"]["simpleText"]

        videos.append({
            "id": vid_id,
            "title": title,
            "duration": length_text,
            "url": f"https://www.youtube.com/watch?v={vid_id}",
        })
    return videos


def fetch_transcript(video_id: str) -> Optional[str]:
    """Fetch transcript text for a video. Returns None if unavailable."""
    try:
        ytt = YouTubeTranscriptApi()
        segments = ytt.fetch(video_id)
        return " ".join(seg.text for seg in segments)
    except Exception:
        return None


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama Snowflake Arctic."""
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL,
        "input": text,
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts in one call."""
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL,
        "input": texts,
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["embeddings"]


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(channel_input: str, limit: Optional[int] = None,
                 chunk_size: int = 400, overlap: int = 50):
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    cur = conn.cursor()

    # 1. Resolve channel
    channel_id = extract_channel_id(channel_input)
    print(f"[1/5] Scraping video list from channel: {channel_input}")
    videos = get_channel_videos(channel_input, limit=limit)
    print(f"       Found {len(videos)} videos")

    if not videos:
        print("No videos found. Check the channel ID/URL.")
        return

    # Resolve actual channel name from YouTube page title
    channel_name = channel_input
    try:
        resolve_url = f"https://www.youtube.com/channel/{channel_id}" if (
            channel_id.startswith("UC") and len(channel_id) == 24
        ) else channel_input
        r = requests.get(resolve_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        m = re.search(r'<title>(.+?)\s*-\s*YouTube</title>', r.text)
        if m:
            channel_name = m.group(1)
    except Exception:
        pass

    # If we came in with a URL, resolve the channel_id from first video
    if not (channel_id.startswith("UC") and len(channel_id) == 24) and videos:
        try:
            for v in scrapetube.get_channel(
                channel_url=channel_input if channel_input.startswith("http")
                else f"https://www.youtube.com/{channel_input}",
            ):
                channel_id = v.get("channelId", channel_input[:24])
                break
        except Exception:
            channel_id = channel_input[:24]

    # 2. Upsert channel record
    cur.execute("""
        INSERT INTO youtube_channel (id, name, url, total_videos, last_scraped_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (id) DO UPDATE SET
            total_videos = EXCLUDED.total_videos,
            last_scraped_at = now()
    """, (channel_id, channel_name, channel_input, len(videos)))
    conn.commit()

    # 3. Fetch transcripts
    print(f"[2/5] Fetching transcripts...")
    transcript_count = 0
    skip_count = 0

    for i, video in enumerate(videos):
        vid_id = video["id"]

        # Check if already ingested
        cur.execute("SELECT has_transcript FROM youtube_video WHERE id = %s", (vid_id,))
        row = cur.fetchone()
        if row is not None:
            skip_count += 1
            continue

        transcript = fetch_transcript(vid_id)
        has_transcript = transcript is not None

        cur.execute("""
            INSERT INTO youtube_video (id, channel_id, title, url, duration,
                                       transcript_text, segment_count, has_transcript)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            vid_id, channel_id, video["title"], video["url"],
            video["duration"], transcript, 0, has_transcript,
        ))
        conn.commit()

        if has_transcript:
            transcript_count += 1
        progress = f"({i+1}/{len(videos)})"
        status = "OK" if has_transcript else "no transcript"
        print(f"       {progress} {video['title'][:60]} — {status}")

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    print(f"       Transcripts fetched: {transcript_count}, skipped (already ingested): {skip_count}")

    # 4. Chunk + Embed
    print(f"[3/5] Chunking transcripts (size={chunk_size}, overlap={overlap})...")
    cur.execute("""
        SELECT v.id, v.transcript_text
        FROM youtube_video v
        LEFT JOIN youtube_chunk c ON c.video_id = v.id
        WHERE v.channel_id = %s AND v.has_transcript = true AND c.id IS NULL
    """, (channel_id,))
    to_embed = cur.fetchall()
    print(f"       {len(to_embed)} videos to chunk+embed")

    total_chunks = 0
    for vid_id, transcript in to_embed:
        chunks = chunk_text(transcript, chunk_size=chunk_size, overlap=overlap)

        # Batch embed (up to 32 at a time to avoid timeout)
        batch_size = 32
        all_embeddings = []
        for b in range(0, len(chunks), batch_size):
            batch = chunks[b:b + batch_size]
            embeddings = get_embeddings_batch(batch)
            all_embeddings.extend(embeddings)

        # Insert chunks
        for idx, (chunk, emb) in enumerate(zip(chunks, all_embeddings)):
            emb_str = "[" + ",".join(str(x) for x in emb) + "]"
            cur.execute("""
                INSERT INTO youtube_chunk (video_id, channel_id, chunk_index,
                                           content, token_estimate, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
            """, (vid_id, channel_id, idx, chunk, len(chunk.split()), emb_str))

        conn.commit()
        total_chunks += len(chunks)
        print(f"       {vid_id}: {len(chunks)} chunks embedded")

    print(f"[4/5] Total chunks created: {total_chunks}")

    # 5. Summary
    cur.execute("SELECT count(*) FROM youtube_chunk WHERE channel_id = %s", (channel_id,))
    total_in_db = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM youtube_video WHERE channel_id = %s AND has_transcript = true", (channel_id,))
    vids_with_transcript = cur.fetchone()[0]

    print(f"[5/5] Done!")
    print(f"       Channel: {channel_name}")
    print(f"       Videos with transcripts: {vids_with_transcript}/{len(videos)}")
    print(f"       Total chunks in DB: {total_in_db}")
    print(f"       Ready for RAG queries via youtube_rag_query.py")

    cur.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest YouTube channel into RAG pipeline")
    parser.add_argument("channel", help="YouTube channel ID (UCxxx), URL, or @handle")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Max videos to process (default: all)")
    parser.add_argument("--chunk-size", type=int, default=400,
                        help="Words per chunk (default: 400)")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Overlap words between chunks (default: 50)")
    args = parser.parse_args()

    run_pipeline(args.channel, limit=args.limit,
                 chunk_size=args.chunk_size, overlap=args.overlap)
