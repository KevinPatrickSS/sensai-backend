"""Async SQLite-backed simple memory store for small persistent records.

Usage:
  await init_db()
  await save_memory(key, payload, tags=[...])
  m = await get_memory(key)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, List, Optional

import aiosqlite

DB_PATH = Path(__file__).parent.parent / "memories.db"


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
        CREATE TABLE IF NOT EXISTS memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          key TEXT UNIQUE NOT NULL,
          value TEXT NOT NULL,
          tags TEXT,
          created_at REAL NOT NULL,
          updated_at REAL
        )
        """
        )
        await db.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)")
        await db.commit()


async def save_memory(key: str, value: Any, tags: Optional[List[str]] = None) -> None:
    payload = json.dumps(value, ensure_ascii=False)
    now = time.time()
    tags_s = json.dumps(tags) if tags else None
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO memories (key, value, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, tags=excluded.tags, updated_at=excluded.updated_at
            """,
            (key, payload, tags_s, now, now),
        )
        await db.commit()


async def get_memory(key: str) -> Optional[Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT value FROM memories WHERE key = ?", (key,))
        row = await cur.fetchone()
        await cur.close()
        return json.loads(row[0]) if row else None


async def query_memories(tag: str) -> List[Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT value FROM memories WHERE tags LIKE ?", (f'%{tag}%',))
        rows = await cur.fetchall()
        await cur.close()
        return [json.loads(r[0]) for r in rows]


async def delete_memory(key: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM memories WHERE key = ?", (key,))
        await db.commit()
