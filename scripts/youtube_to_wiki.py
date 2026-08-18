#!/usr/bin/env python3
"""
Auto-file YouTube transcripts into LLM-Wiki raw/transcripts/.

Creates one markdown file per video with YAML frontmatter.
Updates wiki/index.md and wiki/log.md. Skips videos already filed.

Usage:
    python youtube_to_wiki.py [--channel CHANNEL_ID] [--wiki-path /home/avion/wiki]
"""

import argparse
import os
import re
from datetime import datetime

import psycopg2

DB_URL = "postgresql://jarvis:Jarvis_a9d61331dd828a80ad258fc8@100.64.222.195:5433/garisek"
DEFAULT_WIKI = os.path.expanduser("~/wiki")


def slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text[:80].rstrip('-')


def file_transcript(wiki_path: str, video: dict) -> str | None:
    """Create a markdown file for a video transcript. Returns filename or None if skipped."""
    raw_dir = os.path.join(wiki_path, "raw", "transcripts")
    os.makedirs(raw_dir, exist_ok=True)

    channel_slug = slugify(video["channel_name"])
    title_slug = slugify(video["title"] or video["id"])
    filename = f"{channel_slug}-{title_slug}.md"
    filepath = os.path.join(raw_dir, filename)

    # Skip if already filed
    if os.path.exists(filepath):
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    transcript = video["transcript_text"] or ""

    content = f"""---
title: "{(video['title'] or '').replace('"', "'")}"
created: {today}
updated: {today}
type: transcript
tags: [transcript, youtube, {channel_slug}]
sources: [{video['url']}]
channel: "{video['channel_name']}"
video_id: "{video['id']}"
duration: "{video['duration'] or 'unknown'}"
---

# {video['title']}

**Channel:** {video['channel_name']}
**URL:** {video['url']}
**Duration:** {video['duration'] or 'unknown'}

## Transcript

{transcript}
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filename


def update_index(wiki_path: str, new_files: list[str]):
    """Add new transcript files to the index under a Transcripts section."""
    index_path = os.path.join(wiki_path, "index.md")
    if not os.path.exists(index_path):
        return

    content = open(index_path, "r", encoding="utf-8").read()

    # Update total count
    # Count all .md files in wiki subdirs
    total = 0
    for d in ["entities", "concepts", "comparisons", "queries"]:
        dp = os.path.join(wiki_path, d)
        if os.path.isdir(dp):
            total += len([f for f in os.listdir(dp) if f.endswith(".md")])

    today = datetime.now().strftime("%Y-%m-%d")
    content = re.sub(
        r'Total pages: \d+',
        f'Total pages: {total}',
        content,
    )
    content = re.sub(
        r'Last updated: \S+',
        f'Last updated: {today}',
        content,
    )

    # Add Transcripts section if not present
    if "## Transcripts" not in content:
        content = content.rstrip() + f"\n\n## Transcripts (raw)\n"

    # Append new files
    for f in new_files:
        ref = f"- [[raw/transcripts/{f}]]"
        if ref not in content:
            content += f"\n{ref}"

    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def update_log(wiki_path: str, count: int, channel_name: str):
    """Append to the wiki log."""
    log_path = os.path.join(wiki_path, "log.md")
    today = datetime.now().strftime("%Y-%m-%d")
    entry = f"\n## {today} ingest | Auto-filed {count} transcripts from {channel_name}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


def main():
    parser = argparse.ArgumentParser(description="Auto-file YouTube transcripts into LLM-Wiki")
    parser.add_argument("--channel", "-c", default=None, help="Channel ID to file (default: all)")
    parser.add_argument("--wiki-path", "-w", default=DEFAULT_WIKI, help="Wiki directory path")
    args = parser.parse_args()

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    where = "WHERE v.has_transcript = true AND v.transcript_text IS NOT NULL"
    params: list = []
    if args.channel:
        where += " AND v.channel_id = %s"
        params.append(args.channel)

    cur.execute(f"""
        SELECT v.id, v.title, v.url, v.duration, v.transcript_text,
               v.channel_id, ch.name AS channel_name
        FROM youtube_video v
        JOIN youtube_channel ch ON ch.id = v.channel_id
        {where}
        ORDER BY v.scraped_at
    """, params)

    videos = cur.fetchall()
    columns = [d[0] for d in cur.description]
    video_dicts = [dict(zip(columns, row)) for row in videos]

    print(f"Found {len(video_dicts)} videos with transcripts")

    new_files: list[str] = []
    channels_seen: set[str] = set()

    for v in video_dicts:
        filename = file_transcript(args.wiki_path, v)
        if filename:
            new_files.append(filename)
            channels_seen.add(v["channel_name"])
            print(f"  Filed: {filename}")

    if new_files:
        update_index(args.wiki_path, new_files)
        for ch_name in channels_seen:
            ch_count = sum(1 for v in video_dicts if v["channel_name"] == ch_name
                          and f"{slugify(ch_name)}-{slugify(v['title'] or v['id'])}.md" in new_files)
            update_log(args.wiki_path, ch_count, ch_name)

    print(f"Done. Filed {len(new_files)} new transcripts, skipped {len(video_dicts) - len(new_files)}.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
