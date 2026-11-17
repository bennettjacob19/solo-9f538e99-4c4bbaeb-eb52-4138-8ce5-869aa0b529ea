"""Crash-safe persistence helpers.

All state is written with the classic atomic-rename protocol:

1. write to a temporary file in the *same* directory,
2. ``flush()`` + ``fsync()`` the file,
3. ``os.replace`` onto the final name,
4. ``fsync`` the parent directory so the rename itself is durable.

A reader therefore only ever observes the previous complete file or the new
complete file, never a half-written one. This is what makes "save failure keeps
the previous complete round" hold even across a power loss / container kill.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

# Optional fault-injection hook used by the write-failure tests. It is inert
# unless STORAGE_FAULT_FILE points at a JSON marker file, e.g.
#   {"match": "status.json", "contains": "\"confirmed_round\":5", "count": 1}
# The next ``skip`` writes whose basename contains ``match`` succeed; the
# following ``count`` matching writes raise OSError *before* touching the
# filesystem, exactly like a failing disk. ``contains`` optionally restricts
# the fault to payloads with a given substring, which lets a test target the
# exact confirmation write rather than the numerous progress writes. Only the
# test harness writes this marker; the training code never creates it.
FAULT_FILE_ENV = "STORAGE_FAULT_FILE"


def _consume_fault(path: str, text: str) -> None:
    marker = os.environ.get(FAULT_FILE_ENV)
    if not marker or not os.path.exists(marker):
        return
    try:
        with open(marker, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except (OSError, ValueError):
        return
    match = str(spec.get("match", ""))
    if match and match not in os.path.basename(path):
        return
    contains = spec.get("contains")
    if contains and str(contains) not in text:
        return
    skip = int(spec.get("skip", 0))
    if skip > 0:
        spec["skip"] = skip - 1
        with open(marker, "w", encoding="utf-8") as f:
            json.dump(spec, f)
        return
    count = int(spec.get("count", 1))
    if count <= 1:
        try:
            os.unlink(marker)
        except FileNotFoundError:
            pass
    else:
        spec["count"] = count - 1
        with open(marker, "w", encoding="utf-8") as f:
            json.dump(spec, f)
    raise OSError(f"injected durable-write failure for {path}")


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: str, text: str) -> None:
    _consume_fault(path, text)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(directory)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, separators=(",", ":"),
                                       sort_keys=True))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def exists(path: str) -> bool:
    return os.path.exists(path)


def if_exists_read_json(path: str) -> Any | None:
    if not os.path.exists(path):
        return None
    return read_json(path)
