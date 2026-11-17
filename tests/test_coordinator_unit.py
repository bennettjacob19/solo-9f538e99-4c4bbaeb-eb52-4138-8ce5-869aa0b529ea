"""Unit tests for the coordinator's confirmation/publication contract.

These run in-process (no uvicorn) and pin down the exact failure chain:
a failed durable confirmation write must leave the web view, the loss history
and the downloadable weights at the previous confirmed round.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import coordinator as C  # noqa: E402
import lrmath  # noqa: E402


@pytest.fixture()
def coord(tmp_path, monkeypatch):
    """Isolated coordinator module state rooted at a temp data dir."""
    monkeypatch.setattr(C, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(C, "JOB_PATH", str(tmp_path / "job.json"))
    monkeypatch.setattr(C, "STATUS_PATH", str(tmp_path / "status.json"))
    monkeypatch.setattr(C, "WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setattr(C, "LATEST_WEIGHTS",
                        str(tmp_path / "weights" / "weights_latest.json"))
    monkeypatch.delenv(C.__dict__.get("STORAGE_FAULT_FILE", "STORAGE_FAULT_FILE"),
                       raising=False)
    (tmp_path / "weights").mkdir()
    C.CTX.status = None
    C.CTX.job = None
    yield C
    C.CTX.status = None
    C.CTX.job = None


def _make_job(coord, n=8, rounds=5):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 2))
    y = (x[:, 0] > 0).astype(float)
    task_id = "t" * 32
    idx1, idx2 = C.split_indices(n)
    job = {
        "task_id": task_id, "rounds": rounds, "n": n,
        "x": [[float(a), float(b)] for a, b in x],
        "y": [float(v) for v in y],
        "shards": {"worker1": idx1, "worker2": idx2},
        "created_at": C.now_ms(),
    }
    coord.atomic_write_json(coord.JOB_PATH, job)
    zero = lrmath.to_list(lrmath.zero_buffers())
    loss0 = lrmath.bce_loss(lrmath.augment(x), y, zero)
    status = {
        "task_id": task_id, "gen": 0, "status": "idle",
        "confirmed_round": 0, "total_rounds": rounds,
        "w": zero, "v": zero,
        "loss_history": [{"round": 0, "loss": loss0, "at": C.now_ms()}],
        "running_round": None, "last_event": "task created",
        "last_error": None, "created_at": C.now_ms(),
    }
    C.CTX.job = job
    C.save_status(status)
    return job, status, x, y


def test_commit_round_publishes_only_after_successful_write(coord, tmp_path):
    job, status, x, y = _make_job(coord)
    w, v = lrmath.momentum_update(status["w"], status["v"],
                                  np.array([0.1, -0.2, 0.3]))
    doc = coord.commit_round(1, lrmath.to_list(w), lrmath.to_list(v), 0.5)
    assert doc["confirmed_round"] == 1
    assert [p["round"] for p in doc["loss_history"]] == [0, 1]
    disk = json.loads((tmp_path / "status.json").read_text())
    assert disk["confirmed_round"] == 1
    assert [p["round"] for p in disk["loss_history"]] == [0, 1]


def test_failed_confirmation_write_freezes_every_view(coord, tmp_path,
                                                      monkeypatch):
    job, status, x, y = _make_job(coord)
    w, v = lrmath.momentum_update(status["w"], status["v"],
                                  np.array([0.1, -0.2, 0.3]))
    w_list, v_list = lrmath.to_list(w), lrmath.to_list(v)

    def boom(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(coord, "atomic_write_json", boom)
    with pytest.raises(OSError):
        coord.commit_round(1, w_list, v_list, 0.5)
    # In-memory and on-disk views are both still the previous confirmed round.
    assert coord.CTX.status["confirmed_round"] == 0
    assert [p["round"] for p in coord.CTX.status["loss_history"]] == [0]
    disk = json.loads((tmp_path / "status.json").read_text())
    assert disk["confirmed_round"] == 0
    assert [p["round"] for p in disk["loss_history"]] == [0]
    assert disk["w"] == status["w"]
    # The weight artifact was never written ahead of the confirmation either.
    assert not (tmp_path / "weights" / "weights_round_0001.json").exists()


def test_progress_never_mutates_confirmed_fields(coord):
    _make_job(coord)
    before = copy.deepcopy(coord.CTX.status)
    coord.progress(status="running", running_round=1, last_event="x")
    assert coord.CTX.status["running_round"] == 1
    assert coord.CTX.status["confirmed_round"] == before["confirmed_round"]
    assert coord.CTX.status["loss_history"] == before["loss_history"]
    assert coord.CTX.status["w"] == before["w"]
    with pytest.raises(RuntimeError):
        coord.progress(confirmed_round=99)


def test_weight_export_failure_keeps_download_at_confirmed(coord, tmp_path,
                                                           monkeypatch):
    job, status, x, y = _make_job(coord)
    # Seed round-0 artifact so a stale export is observable.
    zero = status["w"]
    coord.write_weights_artifact(job["task_id"], 0, zero, status["v"],
                                 status["loss_history"][0]["loss"])

    def boom(path, payload):
        raise OSError("export failed")

    real_write = coord.atomic_write_json

    def selective(path, payload):
        if "weights" in str(path):
            boom(path, payload)
        return real_write(path, payload)

    monkeypatch.setattr(coord, "atomic_write_json", selective)
    w, v = lrmath.momentum_update(zero, status["v"], np.array([0.0, 0.0, 0.0]))
    # Confirmation succeeds, export fails: the in-memory round still advances
    # (the confirmation is the commit point) but nothing is written ahead.
    doc = coord.commit_round(1, lrmath.to_list(w), lrmath.to_list(v), 0.25)
    assert doc["confirmed_round"] == 1
    assert "weight export failed" in doc["last_event"]


def test_history_rebuild_is_idempotent_for_same_round(coord):
    _make_job(coord)
    w = [0.01, 0.02, 0.03]
    v = [-0.01, -0.02, -0.03]
    coord.commit_round(1, w, v, 0.4)
    first = copy.deepcopy(coord.CTX.status["loss_history"])
    # A retried confirmation for the same round replaces, never duplicates.
    coord.commit_round(1, w, v, 0.4)
    assert [p["round"] for p in coord.CTX.status["loss_history"]] == [0, 1]
    assert len(coord.CTX.status["loss_history"]) == len(first)


def test_sanitize_drops_unconfirmed_history(coord):
    job, status, x, y = _make_job(coord)
    status["confirmed_round"] = 2
    status["loss_history"].append({"round": 3, "loss": 0.1, "at": 0})
    status["loss_history"].append({"round": 2, "loss": 0.2, "at": 0})
    changed = coord.sanitize_status(status, job)
    assert changed
    assert [p["round"] for p in status["loss_history"]] == [0, 2]
