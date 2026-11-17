"""Coordinator: FastAPI web app that orchestrates the two training workers.

Correctness contract
--------------------
``status.json`` is the **single durable confirmation record**. It holds the
last confirmed round together with the matching ``w``/``v`` and the loss
history. A round becomes visible to the web UI, to the loss history and to the
weight download *only after* that document has been atomically replaced on
disk. Until then every observer keeps seeing the previous confirmed round.

Round protocol (synchronous)
----------------------------
1. ask both workers for the per-sample residuals of their shard at the last
   confirmed ``w``;
2. concatenate the residuals in canonical dataset row order and rebuild the
   full-data average gradient with one matrix product;
3. send the identical gradient to both workers, which apply one momentum step
   and *durably persist* it before acknowledging;
4. only when both acknowledgements match do we persist the new
   ``status.json`` (the commit point). A failure at any step leaves the
   previous confirmed round intact; the in-flight round is simply redone.

Recovery
--------
Every (re)start opens a new fencing generation and calls ``/prepare`` on both
workers with the coordinator's last confirmed round. ``/prepare`` rolls a
worker forward or **backward** to that exact state, so a worker that already
applied an unconfirmed update discards it and recomputes it from the confirmed
state. Because the recomputation uses identical float64 operations, the
recovered trajectory is bit-identical to uninterrupted training.

Stale messages are fenced by ``task_id`` + ``gen``, commits are idempotent for
an already persisted round, and a commit for any round other than
``current + 1`` is rejected - so no round can be skipped or applied twice.

The downloadable weight artifact is *derived* from ``status.json``. It is
written best-effort after the confirmation commits; if that export fails the
download endpoint rebuilds it from the confirmed status, so the download can
never be ahead of the confirmed round.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               Response)
from pydantic import BaseModel, Field

import lrmath
from storage import (atomic_write_json, exists,
                     if_exists_read_json, read_json)

DATA_DIR = os.environ.get("COORD_DATA", "/data")
JOB_PATH = os.path.join(DATA_DIR, "job.json")
STATUS_PATH = os.path.join(DATA_DIR, "status.json")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
LATEST_WEIGHTS = os.path.join(WEIGHTS_DIR, "weights_latest.json")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

WORKERS = {
    "worker1": os.environ.get("WORKER1_URL", "http://worker1:8000"),
    "worker2": os.environ.get("WORKER2_URL", "http://worker2:8000"),
}
HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_ROWS, MIN_ROWS = 200, 2
N_FEATURES = 2

# Fields owned by the durable confirmation record. Progress updates may never
# change them in memory - that is what keeps the UI/loss/download frozen at the
# previous confirmed round when a confirmation write fails.
CONFIRMED_FIELDS = ("confirmed_round", "w", "v", "loss_history")


class Ctx:
    """In-memory runtime state, always rebuildable from durable files."""

    def __init__(self) -> None:
        self.job: Optional[Dict[str, Any]] = None
        self.status: Optional[Dict[str, Any]] = None
        self.task: Optional[asyncio.Task] = None
        self.client: Optional[httpx.AsyncClient] = None
        self.worker_live: Dict[str, bool] = {k: False for k in WORKERS}


CTX = Ctx()
# Serializes task lifecycle API calls (create/start/pause/reset). The lock is
# held across worker preparation so a second Start/Resume can never run two
# training loops or open two generations for the same group.
OP_LOCK = asyncio.Lock()


# ============================================================ small utilities
def now_ms() -> int:
    return int(time.time() * 1000)


def httpc() -> httpx.AsyncClient:
    assert CTX.client is not None
    return CTX.client


def save_status(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Persist first, publish second.

    If the atomic write fails the exception propagates and ``CTX.status`` is
    left untouched, so no observer can see the attempted state. This is the
    single publish point for a newly confirmed round.
    """
    atomic_write_json(STATUS_PATH, doc)
    CTX.status = doc
    return doc


def progress(**fields: Any) -> Dict[str, Any]:
    """Update volatile UI fields in memory only.

    ``confirmed_round``/``w``/``v``/``loss_history`` are copied from the last
    durably saved document and never changed here. Event/progress text may be
    lost on restart; that is harmless and keeps the durable record clean.
    """
    if CTX.status is None:
        raise HTTPException(404, "no training task yet")
    doc = copy.deepcopy(CTX.status)
    for key, value in fields.items():
        if key in CONFIRMED_FIELDS:
            raise RuntimeError(f"progress update may not change {key}")
        doc[key] = value
    CTX.status = doc
    return doc


def require_job() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if CTX.job is None or CTX.status is None:
        raise HTTPException(404, "no training task yet; upload a CSV first")
    return CTX.job, CTX.status


def split_indices(n: int) -> Tuple[List[int], List[int]]:
    """Deterministic, mutually-exclusive contiguous shard assignment."""
    cut = (n + 1) // 2  # worker1 gets the extra row when n is odd
    return list(range(0, cut)), list(range(cut, n))


def sanitize_status(status: Dict[str, Any],
                    job: Optional[Dict[str, Any]]) -> bool:
    """Drop anything that is not covered by ``confirmed_round``.

    Guards against a status file written by an older/buggy build (or a manual
    edit): the loss history, weights and download must always line up with the
    last confirmed round. Returns True when the document changed.
    """
    before = json.dumps(status, sort_keys=True)
    confirmed = int(status.get("confirmed_round", 0))
    hist = [p for p in status.get("loss_history", [])
            if isinstance(p, dict) and int(p.get("round", -1)) <= confirmed]
    # Keep one entry per round (last written wins), sorted ascending.
    dedup: Dict[int, Dict[str, Any]] = {}
    for point in hist:
        dedup[int(point["round"])] = point
    status["loss_history"] = [dedup[k] for k in sorted(dedup)]
    if (job is not None and confirmed == 0
            and not status["loss_history"]):
        x = np.asarray(job["x"], dtype=np.float64)
        y = np.asarray(job["y"], dtype=np.float64)
        status["loss_history"] = [{
            "round": 0,
            "loss": lrmath.bce_loss(lrmath.augment(x), y,
                                    lrmath.zero_buffers()),
            "at": int(status.get("created_at", now_ms())),
        }]
    status["total_rounds"] = int(
        (job or {}).get("rounds", status.get("total_rounds", 0)))
    status["confirmed_round"] = confirmed
    status["running_round"] = None
    return json.dumps(status, sort_keys=True) != before


# ============================================================ CSV validation
def parse_csv(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, f"CSV must be UTF-8 encoded: {exc}")
    reader = list(csv.reader(io.StringIO(text)))
    reader = [r for r in reader if any(c.strip() for c in r)]  # drop blanks
    if len(reader) < MIN_ROWS:
        raise HTTPException(400, "CSV needs 2-200 data rows (plus an optional "
                                 "header row)")

    header, data_rows = reader[0], reader[1:]
    # Detect an optional header row: if the first row has a non-numeric cell,
    # treat it as the header; otherwise every row is data (no header).
    def _row_is_numeric(row: List[str]) -> bool:
        if len(row) != N_FEATURES + 1:
            return False
        try:
            float(row[0]), float(row[1])
        except ValueError:
            return False
        return row[2].strip() in ("0", "1")

    first_is_header = not _row_is_numeric(header)
    rows = data_rows if first_is_header else reader

    if not (MIN_ROWS <= len(rows) <= MAX_ROWS):
        raise HTTPException(400, f"row count must be in [{MIN_ROWS},{MAX_ROWS}],"
                                 f" got {len(rows)}")
    xs: List[List[float]] = []
    ys: List[float] = []
    for i, row in enumerate(rows):
        if len(row) != N_FEATURES + 1:
            raise HTTPException(400, f"row {i+1}: expected 3 columns "
                                     f"(x1,x2,label), got {len(row)}")
        try:
            x1, x2 = float(row[0]), float(row[1])
        except ValueError:
            raise HTTPException(400, f"row {i+1}: non-numeric feature value")
        label = row[2].strip()
        if label not in ("0", "1"):
            raise HTTPException(400, f"row {i+1}: label must be 0 or 1")
        if not (np.isfinite(x1) and np.isfinite(x2)):
            raise HTTPException(400, f"row {i+1}: feature must be finite")
        xs.append([x1, x2])
        ys.append(float(label))
    if len(set(ys)) < 2:
        raise HTTPException(400, "labels must contain both classes 0 and 1")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


# ============================================================ weight artifacts
def weights_payload(task_id: str, round_: int, w: List[float],
                    v: List[float], loss: Optional[float]) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "round": round_, "confirmed_round": round_, "loss": loss,
        "weights": {"bias": w[0], "w1": w[1], "w2": w[2]},
        "momentum": {"v_bias": v[0], "v1": v[1], "v2": v[2]},
        "saved_at": now_ms(),
    }


def _weights_path(round_: int) -> str:
    return os.path.join(WEIGHTS_DIR, f"weights_round_{round_:04d}.json")


def write_weights_artifact(task_id: str, round_: int, w: List[float],
                           v: List[float], loss: Optional[float]) -> None:
    """Export the confirmed round's weights. Derived, best-effort.

    The immutable per-round file is written first; the ``latest`` pointer is
    swapped last. Both writes are atomic. The real commit point is
    ``status.json``, so a failure here cannot make a download inconsistent -
    the download endpoint rebuilds this file from the confirmed status.
    """
    payload = weights_payload(task_id, round_, w, v, loss)
    atomic_write_json(_weights_path(round_), payload)
    atomic_write_json(LATEST_WEIGHTS, payload)


# ============================================================ worker messaging
async def worker_call(worker_id: str, path: str, payload: Dict[str, Any],
                      tolerate: bool = False) -> Dict[str, Any]:
    url = WORKERS[worker_id] + path
    try:
        r = await httpc().post(url, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            raise HTTPException(r.status_code,
                                f"{worker_id} {path} -> {r.status_code}: "
                                f"{r.text[:200]}")
        CTX.worker_live[worker_id] = True
        return r.json()
    except (httpx.HTTPError, HTTPException) as exc:
        CTX.worker_live[worker_id] = False
        if tolerate:
            return {"_error": str(exc)}
        raise


async def worker_health(worker_id: str) -> Dict[str, Any]:
    try:
        r = await httpc().get(WORKERS[worker_id] + "/health",
                              timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        CTX.worker_live[worker_id] = True
        return r.json()
    except Exception as exc:  # noqa: BLE001 - surfaced as status field
        CTX.worker_live[worker_id] = False
        return {"worker_id": worker_id, "alive": False, "error": str(exc)}


# ============================================================ training loop
def commit_round(rnd: int, w: List[float], v: List[float],
                 loss: float) -> Dict[str, Any]:
    """Publish round ``rnd`` iff the confirmation record lands on disk.

    The candidate document is built from the last confirmed document; only a
    successful atomic write updates ``CTX.status``. If the write raises, the
    caller still observes the previous round everywhere (web, loss history,
    download) even though both workers may have advanced.
    """
    job, cur = require_job()
    doc = copy.deepcopy(cur)
    doc["confirmed_round"] = rnd
    doc["w"] = w
    doc["v"] = v
    # Rebuild from the confirmed history rather than appending to a live list,
    # so a retried/duplicated call can never add a second entry for a round.
    hist = [p for p in cur["loss_history"] if int(p["round"]) < rnd]
    hist.append({"round": rnd, "loss": loss, "at": now_ms()})
    doc["loss_history"] = hist
    doc["running_round"] = None
    doc["last_error"] = None
    doc["last_event"] = f"round {rnd}: confirmed (both sides persisted)"
    save_status(doc)                      # <-- the commit point
    try:
        write_weights_artifact(job["task_id"], rnd, w, v, loss)
    except Exception as exc:              # derived export; download can rebuild
        progress(last_event=(f"round {rnd}: confirmed; weight export failed "
                             f"({exc}); download will rebuild it"))
    return CTX.status


async def run_training() -> None:
    """Round machine. Progress never advances past the last confirmed round."""
    job, _ = require_job()
    total_rounds = int(job["rounds"])
    try:
        while int(CTX.status["confirmed_round"]) < total_rounds:
            rnd = int(CTX.status["confirmed_round"]) + 1
            gen = int(CTX.status["gen"])
            progress(status="running", running_round=rnd,
                     last_event=f"round {rnd}: requesting shard gradients")

            payload = {"task_id": job["task_id"], "gen": gen, "round": rnd}
            g1, g2 = await asyncio.gather(
                worker_call("worker1", "/gradient", payload),
                worker_call("worker2", "/gradient", payload),
            )
            # Both gradients must be computed at the same confirmed round
            # within the same fenced generation.
            for g in (g1, g2):
                if (g.get("task_id") != job["task_id"]
                        or g.get("gen") != gen
                        or g.get("from_round") != rnd - 1):
                    raise RuntimeError(f"fence/round mismatch in gradient: {g}")

            n1, n2 = int(g1["n"]), int(g2["n"])
            if n1 + n2 != int(job["n"]):
                raise RuntimeError("shard sizes do not cover the full dataset")
            x = np.asarray(job["x"], dtype=np.float64)
            y = np.asarray(job["y"], dtype=np.float64)
            x_aug = lrmath.augment(x)
            # Worker1's shard precedes worker2's in canonical row order; one
            # matrix product over the concatenated residuals is identical to
            # the single-process full-data gradient, bit for bit.
            residuals = np.asarray(
                list(g1["residuals"]) + list(g2["residuals"]),
                dtype=np.float64)
            if residuals.shape[0] != len(y):
                raise RuntimeError("residual count mismatch")
            full_grad = lrmath.full_average_gradient(x_aug, residuals)
            progress(last_event=(f"round {rnd}: rebuilt full-data average "
                                 f"gradient over {n1 + n2} samples"))

            commit_payload = {"task_id": job["task_id"], "gen": gen,
                              "round": rnd,
                              "grad": lrmath.to_list(full_grad)}
            c1, c2 = await asyncio.gather(
                worker_call("worker1", "/commit", commit_payload),
                worker_call("worker2", "/commit", commit_payload),
            )
            for c in (c1, c2):
                if (c.get("gen") != gen or c.get("round") != rnd):
                    raise RuntimeError(f"fence/round mismatch in commit: {c}")
            w_exp, v_exp = lrmath.momentum_update(
                CTX.status["w"], CTX.status["v"], full_grad)
            if not (np.array_equal(np.asarray(c1["w"]), w_exp)
                    and np.array_equal(np.asarray(c2["w"]), w_exp)
                    and np.array_equal(np.asarray(c1["v"]), v_exp)
                    and np.array_equal(np.asarray(c2["v"]), v_exp)):
                raise RuntimeError("worker state divergence detected")

            loss = lrmath.bce_loss(x_aug, y, w_exp)
            commit_round(rnd, lrmath.to_list(w_exp), lrmath.to_list(v_exp),
                         loss)

        doc = copy.deepcopy(CTX.status)
        doc["status"] = "completed"
        doc["running_round"] = None
        doc["last_error"] = None
        doc["last_event"] = f"all {total_rounds} rounds confirmed"
        save_status(doc)
    except asyncio.CancelledError:
        # Pause/cancel is handled by the /api/pause handler, which re-reads
        # the (unchanged) confirmed state after the task has stopped.
        raise
    except Exception as exc:  # noqa: BLE001 - any worker failure pauses group
        doc = copy.deepcopy(CTX.status)
        if int(doc["confirmed_round"]) >= total_rounds:
            # Every round is durably confirmed; only the completion marker
            # failed to land. Do not report a training failure for this.
            doc["status"] = "completed"
            doc["last_event"] = f"all {total_rounds} rounds confirmed"
        else:
            doc["status"] = "paused"
            doc["last_event"] = (f"group paused: round "
                                 f"{doc['confirmed_round'] + 1} not confirmed; "
                                 "restart the dead worker and press Resume")
        doc["running_round"] = None
        doc["last_error"] = f"{type(exc).__name__}: {exc}"
        try:
            save_status(doc)
        except Exception as exc2:  # disk still unavailable
            # Expose the pause in memory; confirmed fields did not advance, so
            # the web/loss/download still describe the previous round. The
            # durable file says "running" and is normalized to "paused" on the
            # next process start.
            doc["last_error"] += f" (status not persisted: {exc2})"
            CTX.status = doc


# ============================================================ worker alignment
def _shard_rows(job: Dict[str, Any], worker_id: str) -> List[Dict[str, Any]]:
    idx = job["shards"][worker_id]
    return [{"x": job["x"][i], "y": job["y"][i]} for i in idx]


async def _ensure_worker_task(job: Dict[str, Any], worker_id: str) -> None:
    """Make sure the worker holds *this* task's shard.

    A previous failed ``/load`` (or a worker that was reset) must not wedge a
    later resume: if the worker reports a different task id or an unexpected
    shard size, the coordinator re-sends the shard before aligning it.
    """
    health = await worker_health(worker_id)
    rows = _shard_rows(job, worker_id)
    if (health.get("alive") and health.get("task_id") == job["task_id"]
            and int(health.get("n", -1)) == len(rows)):
        return
    res = await worker_call(worker_id, "/load",
                            {"task_id": job["task_id"],
                             "worker_id": worker_id, "rows": rows})
    if (res.get("task_id") != job["task_id"]
            or int(res.get("n", -1)) != len(rows)):
        raise RuntimeError(f"{worker_id}: shard reload failed: {res}")


async def _prepare_workers(gen: int) -> None:
    """Install a fresh fencing generation and align both workers to the
    coordinator's last confirmed round (durable rollback/catch-up)."""
    job, st = require_job()
    await asyncio.gather(*(_ensure_worker_task(job, w) for w in WORKERS))
    payload = {"task_id": job["task_id"], "gen": int(gen),
               "round": int(st["confirmed_round"]),
               "w": st["w"], "v": st["v"]}
    p1, p2 = await asyncio.gather(
        worker_call("worker1", "/prepare", payload),
        worker_call("worker2", "/prepare", payload),
    )
    for p in (p1, p2):
        if (p.get("gen") != gen
                or p.get("round") != st["confirmed_round"]):
            raise RuntimeError(f"prepare fence/round mismatch: {p}")


def ensure_not_busy() -> None:
    if CTX.status and CTX.status["status"] in ("running", "preparing"):
        raise HTTPException(409, "a round is currently in progress")


# ============================================================ API: schemas
class DelayIn(BaseModel):
    seconds: float = Field(..., ge=0, le=120)


# ============================================================ application
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    CTX.client = httpx.AsyncClient()
    CTX.job = if_exists_read_json(JOB_PATH)
    status = if_exists_read_json(STATUS_PATH)
    if status is not None:
        changed = sanitize_status(status, CTX.job)
        if (int(status.get("total_rounds", 0)) > 0
                and int(status["confirmed_round"])
                >= int(status["total_rounds"])
                and status.get("status") != "completed"):
            # Crash after the final confirmation but before the completion
            # write: every round is durably confirmed, so finish the group.
            status["status"] = "completed"
            status["running_round"] = None
            status["last_event"] = (
                f"all {status['total_rounds']} rounds were already confirmed")
            changed = True
        elif status.get("status") in ("running", "preparing"):
            # Coordinator died mid-round/mid-prepare: nothing after the last
            # confirmed round may count. The next resume re-aligns both
            # workers and redoes the unconfirmed round.
            status["status"] = "paused"
            status["running_round"] = None
            status["last_event"] = (
                "coordinator restarted while a round was in flight; "
                "unconfirmed round will be redone")
            changed = True
        CTX.status = status
        if changed:
            try:
                save_status(status)
            except OSError:
                pass
    yield
    if CTX.task and not CTX.task.done():
        CTX.task.cancel()
        with suppress(asyncio.CancelledError):
            await CTX.task
    await CTX.client.aclose()


app = FastAPI(title="Fault-tolerant logistic regression trainer",
              lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/status")
async def status() -> JSONResponse:
    health = await asyncio.gather(*(worker_health(w) for w in WORKERS))
    out: Dict[str, Any] = {"workers": {h["worker_id"]: h for h in health}}
    if CTX.status:
        out.update(CTX.status)
    out["worker_live"] = CTX.worker_live
    out["weights_available"] = CTX.status is not None
    return JSONResponse(out)


@app.post("/api/job")
async def create_job(file: UploadFile = File(...),
                     rounds: int = 10) -> JSONResponse:
    if not (1 <= rounds <= 100):
        raise HTTPException(400, "rounds must be in [1,100]")
    async with OP_LOCK:
        if CTX.status and CTX.status["status"] != "completed":
            raise HTTPException(409, "a task already exists; reset it first")
        raw = await file.read()
        x, y = parse_csv(raw)
        n = len(y)
        idx1, idx2 = split_indices(n)

        task_id = uuid.uuid4().hex
        rows_all = [{"x": [float(x[i, 0]), float(x[i, 1])], "y": float(y[i])}
                    for i in range(n)]
        shards = {
            "worker1": [{"x": r["x"], "y": r["y"]}
                        for r in (rows_all[i] for i in idx1)],
            "worker2": [{"x": r["x"], "y": r["y"]}
                        for r in (rows_all[i] for i in idx2)],
        }
        job = {
            "task_id": task_id, "rounds": int(rounds), "n": n,
            "x": [[float(a), float(b)] for a, b in x],
            "y": [float(v) for v in y],
            "shards": {"worker1": idx1, "worker2": idx2},
            "created_at": now_ms(),
            "hyper": {"lr": lrmath.LEARNING_RATE,
                      "momentum": lrmath.MOMENTUM},
        }
        # Durable first, before touching any worker.
        atomic_write_json(JOB_PATH, job)
        CTX.job = job  # keep in-memory context in sync with the durable file
        zero = lrmath.to_list(lrmath.zero_buffers())
        status_doc = {
            "task_id": task_id, "gen": 0,
            "status": "idle", "confirmed_round": 0,
            "total_rounds": int(rounds),
            "w": zero, "v": zero,
            "loss_history": [{"round": 0,
                              "loss": lrmath.bce_loss(lrmath.augment(x), y,
                                                      zero),
                              "at": now_ms()}],
            "running_round": None, "last_event": "task created",
            "last_error": None, "created_at": now_ms(),
        }
        save_status(status_doc)
        try:
            write_weights_artifact(task_id, 0, zero, zero,
                                   status_doc["loss_history"][0]["loss"])
        except Exception:
            progress(last_event="task created; round-0 weight export failed "
                                "(download will rebuild it)")

        # Push the mutually-exclusive shards to the two independent processes.
        results = await asyncio.gather(*(
            worker_call(wid, "/load",
                        {"task_id": task_id, "worker_id": wid,
                         "rows": shards[wid]}, tolerate=True)
            for wid in WORKERS))
        for wid, res in zip(WORKERS, results):
            if "_error" in res:
                doc = copy.deepcopy(CTX.status)
                doc["status"] = "paused"
                doc["last_error"] = f"shard load to {wid} failed: {res['_error']}"
                doc["last_event"] = ("task created but a worker is "
                                     "unreachable; Resume will retry the load")
                save_status(doc)
                return JSONResponse(doc, status_code=202)
        doc = copy.deepcopy(CTX.status)
        doc["last_event"] = "shards loaded on both workers; press Start"
        save_status(doc)
        return JSONResponse(doc)


@app.post("/api/start")
async def start() -> JSONResponse:
    async with OP_LOCK:
        job, st = require_job()
        ensure_not_busy()
        if st["status"] not in ("idle", "paused"):
            raise HTTPException(409, f"cannot start from status {st['status']}")
        if CTX.task is not None and not CTX.task.done():
            raise HTTPException(409, "a training loop is already active")
        if int(st["confirmed_round"]) >= int(job["rounds"]):
            raise HTTPException(409, "all rounds are already confirmed")

        new_gen = int(st["gen"]) + 1  # every (re)start opens a new group
        prev = copy.deepcopy(st)      # durable state to fall back to
        prep = copy.deepcopy(st)
        prep["gen"] = new_gen
        prep["status"] = "preparing"
        prep["last_error"] = None
        prep["running_round"] = None
        prep["last_event"] = f"opening training generation {new_gen}"
        # Visible while preparing, but not durable yet: the generation is only
        # published together with the running state after alignment succeeds.
        CTX.status = prep
        try:
            await _prepare_workers(new_gen)
        except Exception as exc:  # noqa: BLE001
            doc = copy.deepcopy(prev)
            doc["status"] = "paused"
            doc["running_round"] = None
            doc["last_error"] = f"worker alignment failed: {exc}"
            doc["last_event"] = ("could not align both workers to the last "
                                 "confirmed round; nothing was published")
            try:
                save_status(doc)
            except OSError as exc2:
                doc["last_error"] += f" (status not persisted: {exc2})"
                CTX.status = doc
            raise HTTPException(503, doc["last_error"])

        run = copy.deepcopy(CTX.status)
        run["status"] = "running"
        run["last_event"] = f"generation {new_gen} prepared; running"
        try:
            save_status(run)
        except OSError as exc:
            # Workers were aligned to the (new) generation, but the group may
            # not be published. Keep the previous durable generation/status;
            # a later Resume re-aligns the workers (idempotent) and retries.
            doc = copy.deepcopy(prev)
            doc["status"] = "paused"
            doc["running_round"] = None
            doc["last_error"] = (f"could not persist the start of generation "
                                 f"{new_gen}: {exc}")
            doc["last_event"] = ("workers aligned but start not committed; "
                                 "press Resume again")
            CTX.status = doc
            try:
                save_status(doc)
            except OSError:
                pass
            raise HTTPException(503, doc["last_error"])
        # No await between the durable publish and task creation, so a pause
        # request cannot interleave before the loop is registered.
        CTX.task = asyncio.create_task(run_training())
        return JSONResponse(CTX.status)


@app.post("/api/resume")
async def resume() -> JSONResponse:
    # Resume == guarded restart: same entry point, new generation, roll both
    # workers to the confirmed round, then continue. No round is skipped.
    return await start()


@app.post("/api/pause")
async def pause() -> JSONResponse:
    async with OP_LOCK:
        job, st = require_job()
        if st["status"] not in ("running", "preparing"):
            raise HTTPException(409, "not running")
        task = CTX.task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        CTX.task = None
        cur = CTX.status
        if cur["status"] != "paused":
            doc = copy.deepcopy(cur)
            doc["status"] = "paused"
            doc["running_round"] = None
            doc["last_event"] = ("paused between rounds; resume redoes nothing "
                                 "confirmed")
            try:
                save_status(doc)
            except OSError as exc:
                CTX.status = doc
                raise HTTPException(500, f"could not persist pause: {exc}")
        return JSONResponse(CTX.status)


@app.post("/api/reset")
async def reset() -> JSONResponse:
    async with OP_LOCK:
        ensure_not_busy()
        if CTX.task is not None and not CTX.task.done():
            raise HTTPException(409, "a training loop is still active")
        for p in (JOB_PATH, STATUS_PATH):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        for name in ("weights_latest.json",):
            try:
                os.unlink(os.path.join(WEIGHTS_DIR, name))
            except FileNotFoundError:
                pass
        CTX.job = None
        CTX.status = None
        return JSONResponse({"ok": True, "event": "task cleared"})


@app.post("/api/workers/{worker_id}/delay")
async def set_worker_delay(worker_id: str, body: DelayIn) -> JSONResponse:
    if worker_id not in WORKERS:
        raise HTTPException(404, "unknown worker")
    res = await worker_call(worker_id, "/delay",
                            {"seconds": body.seconds}, tolerate=True)
    return JSONResponse({"worker_id": worker_id, "response": res})


@app.post("/api/workers/delay")
async def set_both_delay(body: DelayIn) -> JSONResponse:
    results = await asyncio.gather(*(
        worker_call(w, "/delay", {"seconds": body.seconds}, tolerate=True)
        for w in WORKERS))
    return JSONResponse(dict(zip(WORKERS, results)))


@app.post("/api/workers/{worker_id}/kill")
async def kill_worker(worker_id: str) -> JSONResponse:
    if worker_id not in WORKERS:
        raise HTTPException(404, "unknown worker")
    # Fire-and-forget SIGKILL. The in-flight round (if any) fails and the group
    # pauses; the entrypoint supervisor respawns the process fresh.
    asyncio.create_task(
        httpc().post(WORKERS[worker_id] + "/kill", timeout=2.0))
    CTX.worker_live[worker_id] = False
    return JSONResponse({"worker_id": worker_id,
                         "event": "SIGKILL sent; group will pause"})


@app.get("/api/weights")
async def download_weights() -> Response:
    """Download the weights of the last **confirmed** round.

    The content is rebuilt from ``status.json`` whenever the derived artifact
    is missing or stale (e.g. its export failed). It is therefore impossible
    for the download to be ahead of the confirmation record.
    """
    job, st = require_job()
    rnd = int(st["confirmed_round"])
    w = [float(v) for v in st["w"]]
    v = [float(v) for v in st["v"]]
    hist = [p for p in st["loss_history"] if int(p["round"]) == rnd]
    loss = float(hist[-1]["loss"]) if hist else None
    payload = weights_payload(job["task_id"], rnd, w, v, loss)
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    filename = f"weights_round_{rnd:04d}.json"
    path = _weights_path(rnd)
    stale = True
    if exists(path):
        try:
            disk = read_json(path)
            stale = (disk.get("task_id") != job["task_id"]
                     or int(disk.get("confirmed_round", -1)) != rnd
                     or [disk["weights"][k] for k in ("bias", "w1", "w2")]
                     != w
                     or [disk["momentum"][k]
                         for k in ("v_bias", "v1", "v2")] != v)
        except (OSError, ValueError, KeyError):
            stale = True
    if stale:
        try:
            atomic_write_json(path, payload)
        except OSError:
            return Response(
                text, media_type="application/json",
                headers={"Content-Disposition":
                         f'attachment; filename="{filename}"'})
    try:
        atomic_write_json(LATEST_WEIGHTS, payload)
    except OSError:
        pass
    return FileResponse(path, media_type="application/json",
                        filename=filename)


@app.get("/api/reference")
async def reference() -> JSONResponse:
    """Single-process full-data reference trajectory (for verification)."""
    job, _ = require_job()
    x = np.asarray(job["x"], dtype=np.float64)
    y = np.asarray(job["y"], dtype=np.float64)
    ws, vs, losses = lrmath.reference_train(x, y, int(job["rounds"]))
    return JSONResponse({
        "rounds": int(job["rounds"]),
        "weights": [lrmath.to_list(w) for w in ws],
        "momentum": [lrmath.to_list(v) for v in vs],
        "loss": losses,
    })
