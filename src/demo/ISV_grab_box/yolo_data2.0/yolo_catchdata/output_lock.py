"""Advisory single-writer locks for dataset output roots."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import IO


def acquire_output_lock(output_root: Path, owner: str) -> IO[str]:
    """Hold an exclusive non-blocking lock until the returned handle is closed.

    The lock file is intentionally persistent; the kernel lock is released
    automatically on normal exit, exception, SIGTERM, or SIGKILL.  Therefore a
    stale file never blocks a later run, while its JSON payload remains useful
    for diagnosing which command last owned the directory.
    """

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".collection.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        holder = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            f"输出目录已有采集进程持锁：{root}\nLOCK_HOLDER={holder}"
        ) from exc

    payload = {
        "pid": os.getpid(),
        "owner": str(owner),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root),
    }
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()
    return handle


__all__ = ["acquire_output_lock"]
