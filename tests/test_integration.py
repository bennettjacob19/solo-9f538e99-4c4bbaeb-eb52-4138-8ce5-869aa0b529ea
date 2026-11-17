"""End-to-end fault-injection tests against real uvicorn processes.

Spins up one coordinator + two independent worker processes (each with its own
temporary data directory, exactly like the Docker topology) and verifies:

1. clean distributed training equals the single-process reference trajectory
   bit-for-bit;
2. killing a worker mid-training pauses the group, and resume reproduces the
   exact uninterrupted trajectory (no skipped / double-applied round);
3. redoing an unconfirmed round after coordinator restart stays exact;
4. stale-generation messages are fenced (409) and cannot advance a new group;
5. downloaded weights only ever reflect confirmed rounds.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
ROUNDS = 30


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Service:
    def __init__(self, name, module, port, env, health_path="/health"):
        self.name, self.module, self.port, self.env = name, module, port, env
        self.health_path = health_path
        self.proc: subprocess.Popen | None = None

    def start(self):
        import site
        env = dict(os.environ)
        env.update(self.env)
        # Use -P (cwd excluded from sys.path, guarding against planted modules)
        # rather than -I, which would also drop PYTHONPATH and user site
        # packages. The app dir and locally installed deps are supplied here.
        extra = [str(APP), site.getusersitepackages()]
        env["PYTHONPATH"] = os.pathsep.join(extra) + os.pathsep + env.get(
            "PYTHONPATH", "")
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "uvicorn",
             f"{self.module}:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=APP, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self._wait_up()

    def _wait_up(self, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"{self.name} exited early:\n{out}")
            try:
                r = httpx.get(
                    f"http://127.0.0.1:{self.port}{self.health_path}",
                    timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.15)
        raise TimeoutError(f"{self.name} did not come up")

    def hard_kill(self):
        assert self.proc
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture()
def coord_env(tmp_path):
    """Extra environment for the coordinator process (tests override this)."""
    return {}


@pytest.fixture()
def cluster(tmp_path, coord_env):
    p1, p2, pc = _free_port(), _free_port(), _free_port()
    d1, d2, dc = (tmp_path / "w1"), (tmp_path / "w2"), (tmp_path / "coord")
    for d in (d1, d2, dc):
        d.mkdir()
    w1 = Service("worker1", "worker", p1,
                 {"WORKER_ID": "worker1", "WORKER_DATA": str(d1)})
    w2 = Service("worker2", "worker", p2,
                 {"WORKER_ID": "worker2", "WORKER_DATA": str(d2)})
    coord = Service("coordinator", "coordinator", pc, {
        "COORD_DATA": str(dc),
        "WORKER1_URL": f"http://127.0.0.1:{p1}",
        "WORKER2_URL": f"http://127.0.0.1:{p2}",
        **coord_env,
    }, health_path="/api/status")
    w1.start(); w2.start(); coord.start()
    yield {
        "coord": f"http://127.0.0.1:{pc}",
        "coord_svc": coord, "w1": w1, "w2": w2,
        "paths": {"w1": d1, "w2": d2, "coord": dc},
    }
    for s in (w1, w2, coord):
        s.stop()


def _dataset_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 2))
    z = 1.2 * x[:, 0] - 0.6 * x[:, 1]
    y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    p = tmp_path / "data.csv"
    lines = ["x1,x2,label"]
    for (a, b), c in zip(x, y):
        lines.append(f"{a:.10f},{b:.10f},{c}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _wait_status(base, predicate, timeout=40):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = httpx.get(base + "/api/status", timeout=5).json()
        if predicate(last):
            return last
        time.sleep(0.25)
    raise AssertionError(f"condition not met; last={json.dumps(last)[:600]}")


def _create_and_start(base, csv_path, rounds=ROUNDS):
    with open(csv_path, "rb") as f:
        r = httpx.post(base + f"/api/job?rounds={rounds}",
                       files={"file": ("data.csv", f, "text/csv")},
                       timeout=15)
    r.raise_for_status()
    httpx.post(base + "/api/start", timeout=10).raise_for_status()


# --------------------------------------------------------------------- tests
def test_clean_run_matches_reference(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    st = _wait_status(base, lambda d: d["status"] == "completed")
    assert st["confirmed_round"] == ROUNDS

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    # Exact, bit-for-bit equality with uninterrupted single-process training.
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert np.allclose(st["w"], ref["weights"][-1], atol=0.0, rtol=0.0)
    assert np.allclose(st["v"], ref["momentum"][-1], atol=0.0, rtol=0.0)

    # Weights download corresponds to the confirmed round only.
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == ROUNDS
    assert wdoc["weights"]["w1"] == st["w"][1]


def test_worker_kill_and_resume_is_numerically_exact(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    # Let some rounds confirm, then SIGKILL worker1 mid-flight.
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 8)
    confirmed_at_kill = st["confirmed_round"]
    cluster["w1"].hard_kill()

    st = _wait_status(base, lambda d: d["status"] == "paused")
    assert st["confirmed_round"] == confirmed_at_kill  # nothing half-counted

    # The killed worker's durable state never exceeds the confirmed round by
    # more than one (a commit may have landed just before the kill).
    state1 = json.loads((cluster["paths"]["w1"] / "state.json").read_text())
    assert state1["round"] <= confirmed_at_kill + 1

    # Restart the worker process (fresh generation-neutral process).
    cluster["w1"].start()
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed")

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k], f"round {k} differs"
    np.testing.assert_array_equal(np.asarray(st["w"]),
                                  np.asarray(ref["weights"][-1]))
    # Exactly one loss entry per round: no duplicate application.
    rounds_seen = [p["round"] for p in st["loss_history"]]
    assert rounds_seen == list(range(ROUNDS + 1))


def test_coordinator_restart_mid_round_redoes_without_skip(cluster, tmp_path):
    base = cluster["coord"]
    # Add delay so a round is almost certainly in flight at kill time.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=12)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    confirmed = st["confirmed_round"]
    cluster["coord_svc"].hard_kill()
    time.sleep(0.8)  # let at least one delayed in-flight round drift
    cluster["coord_svc"].start()

    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["status"] == "paused"
    assert st["confirmed_round"] == confirmed
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert [p["round"] for p in st["loss_history"]] == list(range(13))


def test_stale_generation_messages_are_fenced(cluster, tmp_path):
    base = cluster["coord"]
    # Slow each round down so the pause reliably lands while training runs.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.5}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=10)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    old_gen = st["gen"]
    task_id = st["task_id"]

    # Pause and resume -> a strictly newer generation is installed.
    deadline = time.time() + 15
    while True:
        r = httpx.post(base + "/api/pause", timeout=5)
        if r.status_code == 200:
            break
        if httpx.get(base + "/api/status").json()["status"] == "completed":
            raise AssertionError("training completed before pause took effect")
        assert time.time() < deadline
        time.sleep(0.3)
    _wait_status(base, lambda d: d["status"] == "paused")
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st2 = _wait_status(base, lambda d: d["gen"] > old_gen)
    new_gen = st2["gen"]

    # A late message from the old training group must be rejected (409) and
    # must not mutate any state. Reach the worker directly on its own port.
    w1_port = cluster["w1"].port
    r = httpx.post(f"http://127.0.0.1:{w1_port}/gradient",
                   json={"task_id": task_id, "gen": old_gen,
                         "round": st2["confirmed_round"] + 1},
                   timeout=5)
    assert r.status_code == 409

    # Unknown task id is also rejected.
    r2 = httpx.post(f"http://127.0.0.1:{w1_port}/commit",
                    json={"task_id": "deadbeef", "gen": new_gen,
                          "round": 1, "grad": [0, 0, 0]}, timeout=5)
    assert r2.status_code in (409, 410)

    final = _wait_status(base, lambda d: d["status"] == "completed")
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(final["loss_history"]):
        assert point["loss"] == ref["loss"][k]


def test_full_cluster_reboot_keeps_durable_state(cluster, tmp_path):
    base = cluster["coord"]
    # Slow rounds so the kill reliably lands mid-training, not after completion.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=15)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    confirmed = st["confirmed_round"]
    # Kill every process, then bring the whole cluster back up.
    cluster["coord_svc"].hard_kill()
    cluster["w1"].hard_kill()
    cluster["w2"].hard_kill()
    for s in (cluster["w1"], cluster["w2"], cluster["coord_svc"]):
        s.start()
    st = httpx.get(base + "/api/status", timeout=10).json()
    # More rounds may have confirmed between observation and the kill; the
    # durable invariant that matters is that progress is never lost and the
    # reboot resumes from a confirmed round >= the one we observed.
    assert st["confirmed_round"] >= confirmed
    assert st["status"] in ("paused", "idle", "completed")
    if st["status"] != "completed":
        httpx.post(base + "/api/resume", timeout=15).raise_for_status()
        st = _wait_status(base, lambda d: d["status"] == "completed",
                          timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert [p["round"] for p in st["loss_history"]] == list(range(16))


def test_csv_validation_rules(cluster, tmp_path):
    base = cluster["coord"]
    good = _dataset_csv(tmp_path, n=20)
    # Below the 2-row minimum: header + a single data row -> 400.
    p = tmp_path / "small.csv"
    p.write_text("x1,x2,label\n1,2,0\n")
    with open(p, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("small.csv", f, "text/csv")})
    assert r.status_code == 400
    # The minimum allowed size (2 rows, both classes) is accepted.
    pmin = tmp_path / "min.csv"
    pmin.write_text("1,2,0\n3,4,1\n")  # no header, exactly 2 rows
    with open(pmin, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=3",
                       files={"file": ("min.csv", f, "text/csv")})
    assert r.status_code in (200, 202)
    httpx.post(base + "/api/reset", timeout=5)
    # Bad label.
    p2 = tmp_path / "badlabel.csv"
    lines = good.read_text().splitlines()
    head, body = lines[0], lines[1:]
    body[0] = body[0][:-1] + "9"
    p2.write_text("\n".join([head] + body))
    with open(p2, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("b.csv", f, "text/csv")})
    assert r.status_code == 400
    # Rounds out of range.
    with open(good, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=101",
                       files={"file": ("g.csv", f, "text/csv")})
    assert r.status_code == 400
    # Over the 200-row maximum.
    big = tmp_path / "big.csv"
    rows = ["x1,x2,label"] + [f"{i},1,{i % 2}" for i in range(201)]
    big.write_text("\n".join(rows))
    with open(big, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("big.csv", f, "text/csv")})
    assert r.status_code == 400


def test_download_weights_only_confirmed_and_save_keeps_last(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=12)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 5)
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == st["confirmed_round"]
    # On-disk immutable artifact plus latest pointer exist.
    cdir = cluster["paths"]["coord"] / "weights"
    assert (cdir / f"weights_round_{st['confirmed_round']:04d}.json").exists()
    assert (cdir / "weights_latest.json").exists()
    final = _wait_status(base, lambda d: d["status"] == "completed")
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == final["confirmed_round"] == 12


def test_worker_commit_dedup_and_prepare_rollback(cluster, tmp_path):
    """Direct worker protocol: a retried commit applies once, and /prepare
    rolls an unconfirmed advance back to the last confirmed round."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=8)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    task_id, gen = st["task_id"], st["gen"]
    w1_port = cluster["w1"].port
    url = f"http://127.0.0.1:{w1_port}"

    h = httpx.get(url + "/health", timeout=5).json()
    at = h["round"]
    # Gradient for the next round.
    httpx.post(url + "/gradient",
               json={"task_id": task_id, "gen": gen, "round": at + 1},
               timeout=5).raise_for_status()
    grad = [0.01, -0.02, 0.03]
    c1 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": gen,
                          "round": at + 1, "grad": grad}, timeout=5).json()
    assert c1["round"] == at + 1 and c1["dedup"] is False
    w_after, v_after = c1["w"], c1["v"]
    # Retried commit for the same round must NOT apply the step a second time.
    c2 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": gen,
                          "round": at + 1, "grad": grad}, timeout=5).json()
    assert c2["dedup"] is True and c2["w"] == w_after and c2["v"] == v_after

    # A newer generation's /prepare rolls the worker back to the confirmed
    # round (discarding the unconfirmed local advance above).
    new_gen = gen + 1
    p = httpx.post(url + "/prepare",
                   json={"task_id": task_id, "gen": new_gen,
                         "round": at, "w": st["w"], "v": st["v"]},
                   timeout=5).json()
    assert p["gen"] == new_gen and p["round"] == at
    h2 = httpx.get(url + "/health", timeout=5).json()
    assert h2["round"] == at and h2["gen"] == new_gen

    # Old-generation traffic is now fenced.
    r = httpx.post(url + "/gradient",
                   json={"task_id": task_id, "gen": gen, "round": at + 1},
                   timeout=5)
    assert r.status_code == 409
    # No skipping rounds: committing round at+2 before at+1 is rejected.
    r2 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": new_gen,
                          "round": at + 2, "grad": grad}, timeout=5)
    assert r2.status_code == 409


def test_delay_endpoint_survives_and_rounds_still_exact(cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.2}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=6)
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]


# ------------------------------------------------- durability fault injection
def _arm_fault(path: Path, match: str, count: int = 1, skip: int = 0,
               contains: str | None = None) -> None:
    spec = {"match": match, "count": count, "skip": skip}
    if contains is not None:
        spec["contains"] = contains
    path.write_text(json.dumps(spec), encoding="utf-8")


def _assert_confirmed_view(base, cdir: Path, confirmed: int, total: int):
    """Web, loss history, durable status and download all describe ``confirmed``."""
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["confirmed_round"] == confirmed
    assert [p["round"] for p in st["loss_history"]] == list(range(confirmed + 1))
    disk = json.loads((cdir / "status.json").read_text())
    assert disk["confirmed_round"] == confirmed
    assert [p["round"] for p in disk["loss_history"]] == list(
        range(confirmed + 1))
    wdoc = httpx.get(base + "/api/weights", timeout=10).json()
    assert wdoc["confirmed_round"] == confirmed
    assert wdoc["weights"]["bias"] == st["w"][0]
    assert wdoc["weights"]["w1"] == st["w"][1]
    assert wdoc["weights"]["w2"] == st["w"][2]
    return st


@pytest.fixture()
def fault_cluster(tmp_path, coord_env):
    """Cluster whose coordinator has an armed storage-fault marker file."""
    marker = tmp_path / "fault.json"
    coord_env["STORAGE_FAULT_FILE"] = str(marker)
    # Reuse the standard fixture machinery with the extra env.
    p1, p2, pc = _free_port(), _free_port(), _free_port()
    d1, d2, dc = (tmp_path / "w1"), (tmp_path / "w2"), (tmp_path / "coord")
    for d in (d1, d2, dc):
        d.mkdir()
    w1 = Service("worker1", "worker", p1,
                 {"WORKER_ID": "worker1", "WORKER_DATA": str(d1)})
    w2 = Service("worker2", "worker", p2,
                 {"WORKER_ID": "worker2", "WORKER_DATA": str(d2)})
    coord = Service("coordinator", "coordinator", pc, {
        "COORD_DATA": str(dc),
        "WORKER1_URL": f"http://127.0.0.1:{p1}",
        "WORKER2_URL": f"http://127.0.0.1:{p2}",
        **coord_env,
    }, health_path="/api/status")
    w1.start(); w2.start(); coord.start()
    yield {
        "coord": f"http://127.0.0.1:{pc}",
        "coord_svc": coord, "w1": w1, "w2": w2,
        "paths": {"w1": d1, "w2": d2, "coord": dc},
        "marker": marker,
    }
    for s in (w1, w2, coord):
        s.stop()


def _pause_now(base: str, timeout: float = 20) -> dict:
    """Pause deterministically; returns the paused status document."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.post(base + "/api/pause", timeout=5)
        if r.status_code == 200:
            break
        st = httpx.get(base + "/api/status", timeout=5).json()
        if st["status"] not in ("running", "preparing"):
            break
        time.sleep(0.1)
    return _wait_status(base, lambda d: d["status"] in ("paused", "completed"),
                        timeout=timeout)


def test_status_write_failure_freezes_web_loss_and_download(
        fault_cluster, tmp_path):
    """The reported chain: workers applied round k, the confirmation write
    fails. Until a write succeeds, web/loss/download must all stay at k-1;
    resume must realign both workers and redo the unconfirmed round exactly."""
    base = fault_cluster["coord"]
    cdir = fault_cluster["paths"]["coord"]
    rounds = 14
    # Slow enough that a paused round is a clean between-rounds boundary.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.05}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=rounds)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    st = _pause_now(base)
    before = st["confirmed_round"]
    assert 3 <= before < rounds
    # Pause commits the confirmed state; make sure it is on disk cleanly.
    _assert_confirmed_view(base, cdir, before, rounds)

    # Fail exactly the confirmation write for the next round. The payload is
    # compact JSON with sorted keys, so this substring only matches a status
    # document whose confirmed_round is ``before + 1``.
    _arm_fault(fault_cluster["marker"], "status.json",
               contains=f'"confirmed_round":{before + 1},')

    httpx.post(base + "/api/resume", timeout=20).raise_for_status()

    st = _wait_status(base, lambda d: d["status"] == "paused", timeout=40)
    confirmed = st["confirmed_round"]
    # The web, loss history, durable status and download are frozen together
    # at the last round whose confirmation actually landed.
    assert confirmed in (before, before + 1)
    st = _assert_confirmed_view(base, cdir, confirmed, rounds)
    assert st["last_error"]
    # At least one side may already have applied the unconfirmed round; that
    # advance must be invisible and must be rolled back, not confirmed.
    for w in ("w1", "w2"):
        h = httpx.get(
            f"http://127.0.0.1:{fault_cluster[w].port}/health",
            timeout=5).json()
        assert h["round"] <= confirmed + 1

    # A retried save (Resume) must roll both workers back to ``confirmed``,
    # redo the round and match uninterrupted training bit for bit.
    httpx.post(base + "/api/resume", timeout=20).raise_for_status()
    final = _wait_status(base, lambda d: d["status"] == "completed",
                         timeout=90)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    assert final["confirmed_round"] == rounds
    assert [p["round"] for p in final["loss_history"]] == list(
        range(rounds + 1))                              # no duplicate entries
    for k, point in enumerate(final["loss_history"]):
        assert point["loss"] == ref["loss"][k], f"round {k} differs"
    np.testing.assert_array_equal(np.asarray(final["w"]),
                                  np.asarray(ref["weights"][-1]))
    np.testing.assert_array_equal(np.asarray(final["v"]),
                                  np.asarray(ref["momentum"][-1]))
    _assert_confirmed_view(base, cdir, rounds, rounds)


def test_status_write_failure_then_coordinator_restart_does_not_skip(
        fault_cluster, tmp_path):
    """Repeat the failure and reboot the coordinator before resuming: the
    durable record must still be the last confirmed round, and the recovered
    run must equal uninterrupted training."""
    base = fault_cluster["coord"]
    cdir = fault_cluster["paths"]["coord"]
    rounds = 12
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.05}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=rounds)
    _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    st = _pause_now(base)
    before = st["confirmed_round"]
    _arm_fault(fault_cluster["marker"], "status.json",
               contains=f'"confirmed_round":{before + 1},')
    httpx.post(base + "/api/resume", timeout=20).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "paused", timeout=40)
    confirmed = st["confirmed_round"]
    assert confirmed in (before, before + 1)
    _assert_confirmed_view(base, cdir, confirmed, rounds)

    # Restart the coordinator (and one worker) while the workers may be ahead.
    fault_cluster["coord_svc"].hard_kill()
    fault_cluster["w2"].hard_kill()
    time.sleep(0.3)
    fault_cluster["coord_svc"].start()
    fault_cluster["w2"].start()
    st = httpx.get(base + "/api/status", timeout=10).json()
    assert st["status"] == "paused"
    assert st["confirmed_round"] == confirmed
    assert [p["round"] for p in st["loss_history"]] == list(
        range(confirmed + 1))

    # Resume twice: a duplicate request must never create a second loop.
    httpx.post(base + "/api/resume", timeout=20).raise_for_status()
    r = httpx.post(base + "/api/resume", timeout=20)
    assert r.status_code in (200, 409)
    final = _wait_status(base, lambda d: d["status"] == "completed",
                         timeout=90)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    assert [p["round"] for p in final["loss_history"]] == list(
        range(rounds + 1))
    for k, point in enumerate(final["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    np.testing.assert_array_equal(np.asarray(final["w"]),
                                  np.asarray(ref["weights"][-1]))
    # A completed group cannot be resumed into extra rounds.
    assert httpx.post(base + "/api/resume", timeout=10).status_code == 409


def test_completed_group_recovers_after_crash_before_completion_mark(
        cluster, tmp_path):
    """Crash after the last round was confirmed but before the 'completed'
    status landed must not leave an unresumable paused group."""
    base = cluster["coord"]
    cdir = cluster["paths"]["coord"]
    rounds = 5
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.05}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=rounds)
    _wait_status(base, lambda d: d["status"] == "completed", timeout=60)
    # Simulate the crash window: all rounds confirmed, status still 'running'.
    doc = json.loads((cdir / "status.json").read_text())
    doc["status"] = "running"
    (cdir / "status.json").write_text(json.dumps(doc))
    cluster["coord_svc"].hard_kill()
    cluster["coord_svc"].start()
    st = httpx.get(base + "/api/status", timeout=10).json()
    assert st["status"] == "completed"
    assert st["confirmed_round"] == rounds
    _assert_confirmed_view(base, cdir, rounds, rounds)


def test_startup_repairs_status_ahead_of_confirmed_invariants(
        cluster, tmp_path):
    """A status file carrying loss entries beyond confirmed_round (as the old
    buggy build could persist) is repaired on startup so the web/loss/download
    cannot show an unconfirmed round."""
    base = cluster["coord"]
    cdir = cluster["paths"]["coord"]
    rounds = 4
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.05}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=rounds)
    _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    st = _pause_now(base)
    confirmed = st["confirmed_round"]
    doc = json.loads((cdir / "status.json").read_text())
    # Inject an unconfirmed trailing loss point, mimicking the old defect.
    doc["loss_history"].append(
        {"round": confirmed + 1, "loss": 0.123, "at": 0})
    (cdir / "status.json").write_text(json.dumps(doc))
    cluster["coord_svc"].hard_kill()
    cluster["coord_svc"].start()
    st = _assert_confirmed_view(base, cdir, confirmed, rounds)
    assert st["status"] == "paused"


def test_weight_export_failure_never_advances_download(
        fault_cluster, tmp_path):
    """If publishing the weight file fails after confirmation, the download
    must still correspond to the confirmed round (rebuilt from status.json)."""
    base = fault_cluster["coord"]
    cdir = fault_cluster["paths"]["coord"]
    rounds = 10
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.05}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=rounds)
    _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    st = _pause_now(base)
    before = st["confirmed_round"]
    # Fail the next immutable per-round weight export (status commit succeeds).
    _arm_fault(fault_cluster["marker"], "weights_round_")

    httpx.post(base + "/api/resume", timeout=20).raise_for_status()
    st = _wait_status(base, lambda d: d["confirmed_round"] >= before + 1,
                      timeout=40)
    assert not fault_cluster["marker"].exists(), "export fault never fired"
    st = _pause_now(base)
    confirmed = st["confirmed_round"]
    # Download is never ahead of, and always equals, the confirmed round.
    wdoc = httpx.get(base + "/api/weights", timeout=10).json()
    assert wdoc["confirmed_round"] == confirmed
    assert wdoc["weights"]["w1"] == st["w"][1]
    # The rebuilt file now exists for the confirmed round.
    assert (cdir / "weights" /
            f"weights_round_{confirmed:04d}.json").exists()
    # The regenerated latest pointer matches the confirmed round too.
    latest = json.loads((cdir / "weights" / "weights_latest.json").read_text())
    assert latest["confirmed_round"] == confirmed

    httpx.post(base + "/api/resume", timeout=20).raise_for_status()
    final = _wait_status(base, lambda d: d["status"] == "completed",
                         timeout=90)
    _assert_confirmed_view(base, cdir, rounds, rounds)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(final["loss_history"]):
        assert point["loss"] == ref["loss"][k]
