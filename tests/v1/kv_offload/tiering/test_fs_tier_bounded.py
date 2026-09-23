# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the bounded fs KV tier on qwen38/v0.27.1 (vLLM 0.27.1 + local patches).

Covers the three changes the NVMe secondary tier needs before a live test:
  1. promotion returns HIT_PENDING (upstream #51840): the Mamba/GDN
     one-chunk-window lookup stops promoting every earlier GDN snapshot;
  2. a failed promotion marks its keys as a miss (upstream #49328, Python
     only): an unreadable block file no longer livelocks the request;
  3. ``max_bytes``: bounded fs tier with LRU eviction (cf. open upstream
     #54327), synchronous index lookups, startup rescan, metrics.

Run from a tree whose vllm/ is this branch (e.g. a copy of the venv wheel):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=<pytest-target>:<stage> python -m pytest \
    -q tests/v1/kv_offload/tiering/test_fs_tier_bounded.py
Against the unpatched venv every test here fails (red); patched, all pass.

Sections 4-5 cover patches-local/fs-tier-0271-storecap-breaker.patch (applied
on top): the in-flight store cap (``max_inflight_store_bytes``) and the circuit
breaker with probe recovery. Red on a venv with only fs-tier-0271.patch.
"""

import errno
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm.v1.kv_offload.tiering.fs.manager as fsm
from tests.v1.kv_offload.tiering.test_fs_tier import (
    _BLOCK_ELEMENTS,
    _make_offloading_spec,
    _page_aligned_zero_tensor,
    drain,
    key,
    lookup_and_wait,
    make_job,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.example.manager import ExampleSecondaryTierManager
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_SPEC = _make_offloading_spec()
_CTX = ReqContext(req_id="r0")
_END = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
_BS = _BLOCK_ELEMENTS * 4  # bytes per block row (float32 rows of the helper)


def _tier(tmp_path, n_blocks=8, **kw) -> tuple[FileSystemTierManager, torch.Tensor]:
    tensor = _page_aligned_zero_tensor(n_blocks, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=2,
        n_write_threads=2,
        **kw,
    )
    return tier, tensor


def _store(tier, job_id, ids, slots=None):
    tier.submit_store(make_job(job_id, [key(i) for i in ids], slots))
    return drain(tier)


def _exists(tier, i):
    return os.path.exists(tier.file_mapper.get_file_name(key(i)))


# ---------------------------------------------------------------------------
# 1. HIT_PENDING on promotion: GDN window lookup promotes one chunk, not all
# ---------------------------------------------------------------------------


class _RecordingTier(ExampleSecondaryTierManager):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.loaded = []

    def submit_load(self, job_metadata):
        self.loaded.extend(job_metadata.keys)
        super().submit_load(job_metadata)


def _hybrid_lookup_until_resolved(n_chunks: int, primary_blocks: int):
    """Drive the connector's real _lookup() over scheduler steps for a request
    whose 4 groups (1 attention prefix group + 3 GDN one-chunk windows, the
    Qwen3.8 layout) are all on a secondary tier only. Returns the resolved hit
    and the promoted keys per group."""
    tpc = 832
    region = MagicMock()
    region.create_kv_memoryview.return_value = memoryview(
        np.zeros((primary_blocks, 16), dtype=np.int8)
    )
    primary = CPUPrimaryTierOffloadingManager(
        num_blocks=primary_blocks, mmap_region=region
    )
    sec = _RecordingTier(
        offloading_spec=MagicMock(), primary_kv_view=memoryview(b"x"), tier_type="ex"
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[sec])
    groups = [
        GroupOffloadConfig(g, tpc, tpc, 1, MagicMock(), None if g == 0 else 1)
        for g in range(4)
    ]
    keys = [
        [make_offload_key(f"c{c}".encode(), g) for c in range(n_chunks)]
        for g in range(4)
    ]
    for group_keys in keys:
        for k in group_keys:
            sec.blocks[k] = True
    ctx = ReqContext(req_id="hyb")
    manager.on_new_request(ctx)
    stub = SimpleNamespace(
        manager=manager,
        config=SimpleNamespace(kv_group_configs=groups, sync_load=True),
        _lookup_groups=(0, 1, 2, 3),
        _sliding_window_groups=(1, 2, 3),
        _mamba_align_size=tpc,
        _chunks_being_loaded=set(),
        _events_tracker=MagicMock(),
    )
    for name in ("_maximal_prefix_lookup", "_sliding_window_lookup"):
        setattr(
            stub,
            name,
            getattr(OffloadingConnectorScheduler, name).__get__(stub),
        )
    req_status = SimpleNamespace(
        num_locally_computed_tokens=0,
        req=SimpleNamespace(num_tokens=n_chunks * tpc + 100, request_id="hyb"),
        req_context=ctx,
        group_states=[SimpleNamespace(offload_keys=k) for k in keys],
    )
    hit = None
    for _ in range(10):
        hit = OffloadingConnectorScheduler._lookup(stub, req_status)
        manager.on_schedule_end(_END)
        if hit is not None:
            break
    per_group = [sum(1 for k in sec.loaded if k in set(g)) for g in keys]
    return hit, per_group


def test_gdn_window_lookup_promotes_only_boundary_chunk():
    hit, per_group = _hybrid_lookup_until_resolved(n_chunks=48, primary_blocks=400)
    assert hit == 48 * 832
    # attention: every chunk; each GDN group: only the chunk at the boundary.
    # (0.27.1 promoted all 48 GDN snapshots per group: [48, 48, 48, 48].)
    assert per_group == [48, 1, 1, 1]


def test_gdn_window_lookup_fits_small_primary():
    # 48-chunk hit on a 60-block CPU tier: needs 51 slots with the fix, but
    # 192 without it, where promotions fail and the hit collapses.
    hit, per_group = _hybrid_lookup_until_resolved(n_chunks=48, primary_blocks=60)
    assert hit == 48 * 832
    assert per_group == [48, 1, 1, 1]


# ---------------------------------------------------------------------------
# 2. failed promotion -> miss (no livelock)
# ---------------------------------------------------------------------------


def test_failed_load_marks_miss_unbounded(tmp_path):
    tier, _ = _tier(tmp_path)
    try:
        assert all(r.success for r in _store(tier, 1, [1, 2], [0, 1]))
        assert lookup_and_wait(tier, [key(1), key(2)]) == [LookupResult.HIT] * 2
        os.remove(tier.file_mapper.get_file_name(key(2)))  # unreadable now
        tier.submit_load(make_job(2, [key(1), key(2)], [3, 4], is_promotion=True))
        results = drain(tier)
        assert [r.success for r in results] == [False]
        # 0.27.1 kept the cached True verdict -> HIT -> re-promote forever.
        assert tier.lookup(key(2), _CTX) is LookupResult.MISS
        stats = tier.get_stats().reduce()
        assert stats["vllm:kv_offload_fs_load_failures"] == 1
    finally:
        tier.shutdown()


def test_mark_miss_survives_late_probe_result(tmp_path):
    tier, _ = _tier(tmp_path)
    try:
        _store(tier, 1, [1], [0])
        lm = tier._lookup_manager
        assert tier.lookup(key(1), _CTX) is LookupResult.RETRY  # probe queued
        lm.mark_miss([key(1)])  # e.g. evicted or failed while probing
        tier.on_schedule_end(_END)  # probe runs, finds the file (True)
        deadline = time.monotonic() + 2
        while lm._pending_results.empty() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert tier.lookup(key(1), _CTX) is LookupResult.MISS
    finally:
        tier.shutdown()


# ---------------------------------------------------------------------------
# 3. bounded mode
# ---------------------------------------------------------------------------


def test_bounded_lru_eviction_and_accounting(tmp_path):
    tier, _ = _tier(tmp_path, max_bytes=3 * _BS)
    try:
        for i in range(5):
            assert all(r.success for r in _store(tier, i, [i], [i]))
        assert [_exists(tier, i) for i in range(5)] == [False, False, True, True, True]
        assert tier._cache_bytes == 3 * _BS and tier._reserved_bytes == 0
        # index lookups are synchronous: no RETRY round trip
        assert tier.lookup(key(0), _CTX) is LookupResult.MISS
        assert tier.lookup(key(4), _CTX) is LookupResult.HIT
        stats = tier.get_stats().reduce()
        assert stats["vllm:kv_offload_fs_cache_bytes"] == 3 * _BS
        assert stats["vllm:kv_offload_fs_cache_blocks"] == 3
        assert stats["vllm:kv_offload_fs_evicted_bytes"] == 2 * _BS
        assert stats["vllm:kv_offload_fs_store_bytes"] == 5 * _BS
        # counters reset, gauges stay
        again = tier.get_stats().reduce()
        assert "vllm:kv_offload_fs_store_bytes" not in again
        assert again["vllm:kv_offload_fs_cache_blocks"] == 3
    finally:
        tier.shutdown()


def test_bounded_touch_and_restore_refresh_recency(tmp_path):
    tier, _ = _tier(tmp_path, max_bytes=3 * _BS)
    try:
        for i in range(3):
            _store(tier, i, [i], [i])
        tier.touch([key(0)], _CTX)  # 0 becomes most recent
        _store(tier, 10, [1], [1])  # re-store of a present block: no write
        _store(tier, 11, [3], [3])  # needs one slot -> evicts LRU = 2
        assert [_exists(tier, i) for i in range(4)] == [True, True, False, True]
    finally:
        tier.shutdown()


def test_bounded_pinned_blocks_not_evicted(tmp_path):
    tier, _ = _tier(tmp_path, max_bytes=2 * _BS)
    try:
        _store(tier, 1, [1, 2], [0, 1])
        # in-flight promotion pins 1 and 2 until get_finished_jobs sees it
        tier.submit_load(make_job(2, [key(1), key(2)], [4, 5], is_promotion=True))
        tier.submit_store(make_job(3, [key(3)], [2]))
        tier.drain_jobs()
        # store job 3 was admitted while 1,2 were pinned: skipped, not failed
        results = {r.job_id: r.success for r in tier.get_finished_jobs()}
        assert results == {2: True, 3: True}
        assert not _exists(tier, 3) and _exists(tier, 1) and _exists(tier, 2)
        assert tier.get_stats().reduce()["vllm:kv_offload_fs_stores_skipped"] == 1
        # unpinned now: the next store evicts the LRU block
        _store(tier, 4, [3], [2])
        assert _exists(tier, 3) and tier._cache_bytes == 2 * _BS
        assert sum(_exists(tier, i) for i in (1, 2)) == 1
    finally:
        tier.shutdown()


def test_bounded_failed_store_keeps_only_landed_blocks(tmp_path, monkeypatch):
    real = fsm.batch_store_block

    def first_then_fail(paths, view, offsets, block_size, use_o_direct=True):
        real(paths[:1], view, offsets[:1], block_size, use_o_direct)
        raise OSError("disk full")

    tier, _ = _tier(tmp_path, max_bytes=4 * _BS)
    try:
        monkeypatch.setattr(fsm, "batch_store_block", first_then_fail)
        results = _store(tier, 1, [1, 2], [0, 1])
        assert [r.success for r in results] == [False]
        assert tier._reserved_bytes == 0 and not tier._writing
        assert tier.lookup(key(1), _CTX) is LookupResult.HIT
        assert tier.lookup(key(2), _CTX) is LookupResult.MISS
        assert tier._cache_bytes == _BS
    finally:
        tier.shutdown()


def test_bounded_failed_load_drops_missing_file(tmp_path):
    tier, _ = _tier(tmp_path, max_bytes=4 * _BS)
    try:
        _store(tier, 1, [1, 2], [0, 1])
        os.remove(tier.file_mapper.get_file_name(key(1)))
        tier.submit_load(make_job(2, [key(1), key(2)], [3, 4], is_promotion=True))
        assert [r.success for r in drain(tier)] == [False]
        assert tier.lookup(key(1), _CTX) is LookupResult.MISS
        assert tier.lookup(key(2), _CTX) is LookupResult.HIT
        assert tier._cache_bytes == _BS and not tier._pinned
    finally:
        tier.shutdown()


def test_bounded_startup_rescan(tmp_path):
    tier, _ = _tier(tmp_path, max_bytes=8 * _BS)
    for i in range(3):
        _store(tier, i, [i], [i])
    paths = [tier.file_mapper.get_file_name(key(i)) for i in range(3)]
    tier.shutdown()
    now = time.time()
    for age, p in zip((300, 100, 200), paths, strict=True):  # oldest: block 0
        os.utime(p, (now - age, now - age))
    stray = paths[0] + "_123.tmp"
    open(stray, "wb").close()

    tier2, _ = _tier(tmp_path, max_bytes=2 * _BS)  # must evict one at startup
    try:
        assert not os.path.exists(stray)
        assert [os.path.exists(p) for p in paths] == [False, True, True]
        assert tier2._cache_bytes == 2 * _BS
        assert tier2.lookup(key(1), _CTX) is LookupResult.HIT
        assert tier2.lookup(key(0), _CTX) is LookupResult.MISS
    finally:
        tier2.shutdown()


def test_bounded_roundtrip_data_integrity(tmp_path):
    tier, tensor = _tier(tmp_path, max_bytes=4 * _BS)
    try:
        tensor[0].copy_(torch.rand(_BLOCK_ELEMENTS))
        ref = tensor[0].clone()
        _store(tier, 1, [7], [0])
        tier.submit_load(make_job(2, [key(7)], [5], is_promotion=True))
        assert all(r.success for r in drain(tier))
        assert torch.equal(tensor[5], ref)
        stats = tier.get_stats().reduce()
        assert stats["vllm:kv_offload_fs_load_bytes"] == _BS
    finally:
        tier.shutdown()


def test_factory_and_metric_definitions(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {"type": "fs", "root_dir": str(tmp_path), "max_bytes": "1e9"},
        memoryview(tensor.numpy()),
        _SPEC,
    )
    try:
        assert tier._max_bytes == 10**9
    finally:
        tier.shutdown()
    defs = FileSystemTierManager.build_metric_definitions({})
    assert "vllm:kv_offload_fs_load_bytes" in defs
    with pytest.raises(ValueError):
        _tier(tmp_path, max_bytes=-1)


# ---------------------------------------------------------------------------
# 4. store backlog cap (fs-tier-0271-storecap-breaker.patch, patch A)
# ---------------------------------------------------------------------------

_CTX_B = ReqContext(req_id="r1")


def _blocking_store(monkeypatch):
    """Make store jobs wait for the returned event before writing."""
    real = fsm.batch_store_block
    gate = threading.Event()

    def wait_then_store(*args, **kwargs):
        assert gate.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(fsm, "batch_store_block", wait_then_store)
    return gate


def _failing(monkeypatch, name, err=errno.EIO):
    def fail(paths, *args, **kwargs):
        raise OSError(err, os.strerror(err), paths[0])

    monkeypatch.setattr(fsm, name, fail)


def test_cap_accounting_released_once_on_success_and_failure(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path, max_inflight_store_bytes=2 * _BS)
    try:
        gate = _blocking_store(monkeypatch)
        tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
        assert tier._inflight_store_bytes == 2 * _BS
        gate.set()
        assert [r.success for r in drain(tier)] == [True]
        assert tier._inflight_store_bytes == 0
        _failing(monkeypatch, "batch_store_block")
        tier.submit_store(make_job(2, [key(3)], [2]))
        assert tier._inflight_store_bytes == _BS
        assert [r.success for r in drain(tier)] == [False]
        assert tier._inflight_store_bytes == 0
        assert list(tier.get_finished_jobs()) == []  # nothing released twice
        assert tier._inflight_store_bytes == 0 and not tier._store_job_bytes
    finally:
        gate.set()
        tier.shutdown()


def test_cap_declines_sticky_per_request_and_cleans_up(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path, max_inflight_store_bytes=2 * _BS)
    try:
        gate = _blocking_store(monkeypatch)
        # an empty backlog always admits one job, however large
        assert tier.accepts_store([key(i) for i in range(5)], _CTX)
        tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
        assert not tier.accepts_store([key(3)], _CTX)  # 3 blocks > cap
        gate.set()
        drain(tier)
        assert tier._inflight_store_bytes == 0
        # backlog drained, but r0 now has a hole: its later chunks stay off
        assert not tier.accepts_store([key(4), key(5)], _CTX)
        assert tier.accepts_store([key(4)], _CTX_B)  # other requests unaffected
        stats = tier.get_stats().reduce()
        assert stats["vllm:kv_offload_fs_stores_dropped"] == 3
        tier.on_request_finished(_CTX)
        assert not tier._dropping_reqs
        tier.on_request_finished(ReqContext(req_id="never-seen"))  # no raise
        assert tier.accepts_store([key(6)], _CTX)
    finally:
        gate.set()
        tier.shutdown()


def test_no_cap_is_unchanged(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path)
    try:
        gate = _blocking_store(monkeypatch)
        tier.submit_store(make_job(1, [key(i) for i in range(6)], list(range(6))))
        assert tier.accepts_store([key(7)], _CTX)
        gate.set()
        drain(tier)
        stats = tier.get_stats().reduce()
        assert "vllm:kv_offload_fs_stores_dropped" not in stats
    finally:
        gate.set()
        tier.shutdown()


def test_cap_bounded_counts_only_new_blocks(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path, max_bytes=8 * _BS, max_inflight_store_bytes=_BS)
    try:
        _store(tier, 1, [1, 2], [0, 1])  # on disk
        gate = _blocking_store(monkeypatch)
        tier.submit_store(make_job(2, [key(3)], [2]))
        assert tier.accepts_store([key(1), key(2)], _CTX)  # nothing to write
        assert not tier.accepts_store([key(4)], _CTX)
        gate.set()
        drain(tier)
    finally:
        gate.set()
        tier.shutdown()


def _manager_with_fs(tmp_path, n_blocks=4, **kw):
    tensor = _page_aligned_zero_tensor(n_blocks, _BLOCK_ELEMENTS)
    region = MagicMock()
    region.create_kv_memoryview.return_value = memoryview(tensor.numpy())
    primary = CPUPrimaryTierOffloadingManager(num_blocks=n_blocks, mmap_region=region)
    tier = FileSystemTierManager(
        offloading_spec=_SPEC,
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=1,
        n_write_threads=1,
        **kw,
    )
    return TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier]), tier


def _gpu_store(manager, keys, ctx):
    assert manager.prepare_store(keys, ctx) is not None
    manager.complete_store(keys, ctx)


def test_manager_declined_batch_is_not_pinned(tmp_path, monkeypatch):
    manager, tier = _manager_with_fs(tmp_path, max_inflight_store_bytes=_BS)
    policy = manager.primary_tier._policy
    try:
        gate = _blocking_store(monkeypatch)
        manager.on_new_request(_CTX)
        _gpu_store(manager, [key(1)], _CTX)
        assert policy.get(key(1)).ref_cnt == 1  # pinned by the fs job
        n_jobs = len(manager._transfer_jobs)
        _gpu_store(manager, [key(2)], _CTX)  # over the cap: declined
        assert policy.get(key(2)).ref_cnt == 0
        assert manager.primary_tier._num_evictable_cache_blocks == 1  # key(2)
        assert len(manager._transfer_jobs) == n_jobs
        manager.on_schedule_end(_END)  # end of step
        gate.set()
        tier.drain_jobs()
        manager.on_schedule_end(_END)  # next step polls the finished job
        assert policy.get(key(1)).ref_cnt == 0
        _gpu_store(manager, [key(3)], _CTX)  # sticky for r0
        assert not manager._transfer_jobs
        manager.on_new_request(_CTX_B)
        _gpu_store(manager, [key(4)], _CTX_B)  # new request: stored
        assert policy.get(key(4)).ref_cnt == 1
        manager.on_request_finished(_CTX)
        assert "r0" not in tier._dropping_reqs and "r0" not in manager._req_state
        tier.drain_jobs()
        manager.on_schedule_end(_END)
        stats = manager.get_stats().reduce()
        assert stats["vllm:kv_offload_fs_stores_dropped"] == 2
    finally:
        gate.set()
        tier.shutdown()


# ---------------------------------------------------------------------------
# 5. circuit breaker (fs-tier-0271-storecap-breaker.patch, patch B)
# ---------------------------------------------------------------------------


def _fail_loads(tier, n, first_job_id=100):
    for i in range(n):
        tier.submit_load(make_job(first_job_id + i, [key(1)], [5], is_promotion=True))
        assert [r.success for r in drain(tier)] == [False]


def _warnings(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(
        fsm.logger, "warning", lambda msg, *a: lines.append(msg % a if a else msg)
    )
    return lines


def test_breaker_trips_at_threshold_not_before(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path, max_bytes=8 * _BS, breaker_consecutive_failures=3)
    try:
        _store(tier, 1, [1], [0])
        lines = _warnings(monkeypatch)
        _failing(monkeypatch, "batch_load_block")
        _fail_loads(tier, 2)
        assert not tier._tripped
        assert tier.get_stats().reduce()["vllm:kv_offload_fs_disabled"] == 0
        _fail_loads(tier, 1, first_job_id=200)
        assert tier._tripped
        assert len(lines) == 1 and "fs KV tier disabled" in lines[0]
        assert "Input/output error" in lines[0]
        stats = tier.get_stats().reduce()
        assert stats["vllm:kv_offload_fs_disabled"] == 1
        assert stats["vllm:kv_offload_fs_breaker_trips"] == 1
    finally:
        tier.shutdown()


def test_breaker_success_resets_and_enoent_does_not_count(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path, max_bytes=8 * _BS, breaker_consecutive_failures=3)
    try:
        _store(tier, 1, [1, 2], [0, 1])
        real_load = fsm.batch_load_block
        _failing(monkeypatch, "batch_load_block")
        _fail_loads(tier, 2)
        monkeypatch.setattr(fsm, "batch_load_block", real_load)
        tier.submit_load(make_job(50, [key(2)], [5], is_promotion=True))
        assert [r.success for r in drain(tier)] == [True]  # resets the count
        _failing(monkeypatch, "batch_load_block")
        _fail_loads(tier, 2, first_job_id=200)
        _failing(monkeypatch, "batch_load_block", errno.ENOENT)
        _fail_loads(tier, 3, first_job_id=300)  # stale entries, disk is fine
        assert not tier._tripped and tier._consecutive_failures == 2
        tier2, _ = _tier(tmp_path / "off", breaker_consecutive_failures=0)
        try:
            _failing(monkeypatch, "batch_load_block")
            _fail_loads(tier2, 5)
            assert not tier2._tripped
            assert "vllm:kv_offload_fs_disabled" not in tier2.get_stats().reduce()
        finally:
            tier2.shutdown()
    finally:
        tier.shutdown()


def test_tripped_tier_misses_and_submits_nothing(tmp_path, monkeypatch):
    tier, _ = _tier(
        tmp_path,
        max_bytes=8 * _BS,
        breaker_consecutive_failures=1,
        breaker_probe_interval_s=3600,
    )
    try:
        _store(tier, 1, [1, 2], [0, 1])
        _failing(monkeypatch, "batch_load_block")
        _fail_loads(tier, 1)
        assert tier._tripped
        entries = dict(tier._entries)
        pool = tier._pool
        tier._pool = MagicMock(wraps=pool)
        assert tier.lookup(key(2), _CTX) is LookupResult.MISS
        assert not tier.accepts_store([key(3)], _CTX)
        tier.submit_store(make_job(10, [key(3)], [2]))
        tier.submit_load(make_job(11, [key(2)], [3], is_promotion=True))
        results = {r.job_id: r.success for r in tier.get_finished_jobs()}
        assert results == {10: False, 11: False}
        tier._pool.enqueue_store.assert_not_called()
        tier._pool.enqueue_load.assert_not_called()
        tier.on_schedule_end(_END)  # probe not due yet
        assert tier._probe_thread is None
        tier._pool = pool
        assert dict(tier._entries) == entries and _exists(tier, 2)  # index kept
        assert tier._inflight_store_bytes == 0 and not tier._pinned
    finally:
        tier.shutdown()


def test_inflight_store_completes_normally_after_trip(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path, max_bytes=8 * _BS, breaker_consecutive_failures=1)
    try:
        _store(tier, 1, [1], [0])
        gate = _blocking_store(monkeypatch)
        tier.submit_store(make_job(2, [key(2)], [1]))  # in flight
        _failing(monkeypatch, "batch_load_block")
        tier.submit_load(make_job(3, [key(1)], [5], is_promotion=True))
        deadline = time.monotonic() + 5
        results = []
        while not results and time.monotonic() < deadline:
            results = list(tier.get_finished_jobs())
            time.sleep(0.01)
        assert [(r.job_id, r.success) for r in results] == [(3, False)]
        assert tier._tripped
        assert list(tier.get_finished_jobs()) == []  # job 2 still in flight
        gate.set()
        assert [(r.job_id, r.success) for r in drain(tier)] == [(2, True)]
        assert tier._inflight_store_bytes == 0 and _exists(tier, 2)
    finally:
        gate.set()
        tier.shutdown()


def _trip(tier, monkeypatch):
    real = fsm.batch_load_block
    _failing(monkeypatch, "batch_load_block")
    _fail_loads(tier, tier._breaker_threshold)
    monkeypatch.setattr(fsm, "batch_load_block", real)
    assert tier._tripped


def _wait_probe(tier):
    thread = tier._probe_thread
    assert thread is not None
    thread.join(10)
    assert not thread.is_alive()


def test_probe_success_reenables(tmp_path, monkeypatch):
    tier, _ = _tier(
        tmp_path,
        max_bytes=8 * _BS,
        breaker_consecutive_failures=2,
        breaker_probe_interval_s=0,
    )
    try:
        _store(tier, 1, [1], [0])
        lines = _warnings(monkeypatch)
        _trip(tier, monkeypatch)
        tier.on_schedule_end(_END)  # starts the probe
        _wait_probe(tier)
        assert tier._tripped  # only the next step collects the result
        tier.on_schedule_end(_END)
        assert not tier._tripped and tier._probe_thread is None
        assert any("fs KV tier re-enabled" in line for line in lines)
        assert not os.path.exists(tier._probe_path)
        assert tier.lookup(key(1), _CTX) is LookupResult.HIT
        assert tier.accepts_store([key(2)], _CTX_B)
        assert tier.get_stats().reduce()["vllm:kv_offload_fs_disabled"] == 0
    finally:
        tier.shutdown()


def test_probe_hang_keeps_tripped_without_blocking(tmp_path, monkeypatch):
    tier, _ = _tier(
        tmp_path, breaker_consecutive_failures=1, breaker_probe_interval_s=0
    )
    try:
        _trip(tier, monkeypatch)
        stuck = threading.Event()
        real_fsync = os.fsync
        monkeypatch.setattr(fsm.os, "fsync", lambda fd: stuck.wait(30))
        started = []
        real_start = threading.Thread.start
        monkeypatch.setattr(
            threading.Thread, "start", lambda t: (started.append(t), real_start(t))
        )
        t0 = time.monotonic()
        for _ in range(20):
            tier.on_schedule_end(_END)
            time.sleep(0.005)
        assert time.monotonic() - t0 < 2
        assert tier._tripped and len(started) == 1  # one probe, never stacked
        assert tier.lookup(key(1), _CTX) is LookupResult.MISS
        monkeypatch.setattr(fsm.os, "fsync", real_fsync)
        stuck.set()
        _wait_probe(tier)
        tier.on_schedule_end(_END)
        assert not tier._tripped  # finished within the timeout
    finally:
        tier.shutdown()


def test_slow_or_failed_probe_stays_tripped(tmp_path, monkeypatch):
    tier, _ = _tier(
        tmp_path,
        breaker_consecutive_failures=1,
        breaker_probe_interval_s=0,
        breaker_probe_timeout_s=0,
    )
    try:
        _trip(tier, monkeypatch)
        tier.on_schedule_end(_END)
        _wait_probe(tier)
        tier.on_schedule_end(_END)  # collects "took ... > 0 s"
        assert tier._tripped and tier._probe_thread is None
        tier._probe_timeout = 30.0

        def broken_fsync(fd):
            raise OSError(errno.EIO, "fsync failed")

        monkeypatch.setattr(fsm.os, "fsync", broken_fsync)
        tier.on_schedule_end(_END)
        _wait_probe(tier)
        tier.on_schedule_end(_END)
        assert tier._tripped
        assert not os.path.exists(tier._probe_path)  # cleaned up on failure
    finally:
        tier.shutdown()


def test_enospc_shrinks_max_bytes_instead_of_tripping(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path, max_bytes=20 * _BS, breaker_consecutive_failures=1)
    try:
        for i in range(12):
            _store(tier, i, [i], [i % 8])
        lines = _warnings(monkeypatch)
        _failing(monkeypatch, "batch_store_block", errno.ENOSPC)
        tier.submit_store(make_job(20, [key(20)], [5]))
        tier.submit_store(make_job(21, [key(21)], [6]))
        assert [r.success for r in drain(tier)] == [False, False]
        assert not tier._tripped and tier._consecutive_failures == 0
        assert tier._max_bytes == int(12 * _BS * 0.95)
        assert len(lines) == 1 and "ENOSPC" in lines[0]  # 2nd: same episode
        monkeypatch.undo()
        assert [r.success for r in _store(tier, 22, [22], [7])] == [True]
        assert tier._cache_bytes <= tier._max_bytes  # evicted to fit
        assert tier.get_stats().reduce()["vllm:kv_offload_fs_evicted_bytes"] > 0
    finally:
        tier.shutdown()


def test_storecap_breaker_config_and_metric_definitions(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "max_inflight_store_bytes": "5e9",
            "breaker_consecutive_failures": 8,
            "breaker_probe_interval_s": 300,
        },
        memoryview(tensor.numpy()),
        _SPEC,
    )
    try:
        assert tier._max_inflight_store_bytes == 5 * 10**9
        assert tier._breaker_threshold == 8 and tier._probe_interval == 300.0
    finally:
        tier.shutdown()
    defs = FileSystemTierManager.build_metric_definitions({})
    for name in ("stores_dropped", "disabled", "breaker_trips"):
        assert f"vllm:kv_offload_fs_{name}" in defs
    with pytest.raises(ValueError, match="max_inflight_store_bytes"):
        _tier(tmp_path, max_inflight_store_bytes=-1)


# ---------------------------------------------------------------------------
# 6. review fixes: done-at-once empty jobs, stall trip, bounded eviction per
#    call + shrink floor, device guard
# ---------------------------------------------------------------------------


def test_empty_store_job_not_queued_and_partial_counts_pins(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path, max_bytes=8 * _BS, max_inflight_store_bytes=2 * _BS)
    try:
        _store(tier, 1, [1, 2], [0, 1])  # on disk
        gate = _blocking_store(monkeypatch)
        pool = tier._pool
        tier._pool = MagicMock(wraps=pool)
        tier.submit_store(make_job(2, [key(1), key(2)], [0, 1]))  # all on disk
        tier._pool.enqueue_store.assert_not_called()
        assert tier._inflight_store_bytes == 0 and not tier._store_job_writes
        assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(2, True)]
        # partial job: one block to write, but two primary blocks pinned
        tier.submit_store(make_job(3, [key(1), key(3)], [0, 2]))
        assert tier._inflight_store_bytes == 2 * _BS
        assert not tier.accepts_store([key(4)], _CTX)  # 2 + 1 blocks > cap
        assert tier.accepts_store([key(1), key(3)], _CTX_B)  # nothing to write
        tier._pool = pool
        gate.set()
        assert [r.success for r in drain(tier)] == [True]
        assert tier._inflight_store_bytes == 0 and not tier._store_job_pinned
        stats = tier.get_stats().reduce()
        assert stats["vllm:kv_offload_fs_store_bytes"] == 3 * _BS  # written only
    finally:
        gate.set()
        tier.shutdown()


def test_manager_releases_pins_of_ondisk_batch_behind_backlog(tmp_path, monkeypatch):
    manager, tier = _manager_with_fs(
        tmp_path, max_bytes=8 * _BS, max_inflight_store_bytes=_BS
    )
    policy = manager.primary_tier._policy
    try:
        manager.on_new_request(_CTX)
        _gpu_store(manager, [key(1)], _CTX)
        tier.drain_jobs()
        manager.on_schedule_end(_END)
        manager.on_schedule_end(_END)
        assert policy.get(key(1)).ref_cnt == 0 and tier._entries  # on disk
        gate = _blocking_store(monkeypatch)
        _gpu_store(manager, [key(2)], _CTX)  # backlog: blocked write
        # cascade key(1) again (already on disk) while the gate is shut
        manager.on_new_request(_CTX_B)
        assert manager._submit_store_to_tier(tier, [key(1)], _CTX_B)
        assert policy.get(key(1)).ref_cnt == 1  # pinned for the fs job
        manager.on_schedule_end(_END)
        manager.on_schedule_end(_END)  # next poll: released, write still stuck
        assert policy.get(key(1)).ref_cnt == 0
        assert policy.get(key(2)).ref_cnt == 1
    finally:
        gate.set()
        tier.shutdown()


def test_stall_trips_without_fabricating_completions(tmp_path, monkeypatch):
    tier, _ = _tier(
        tmp_path,
        max_bytes=8 * _BS,
        breaker_stall_s=0.6,
        breaker_probe_interval_s=0,
    )
    try:
        real = fsm.batch_store_block
        gate = threading.Event()
        stuck_path = tier.file_mapper.get_file_name(key(1))

        def hang_on_key1(paths, *args, **kwargs):
            if stuck_path in paths:
                assert gate.wait(10)
            return real(paths, *args, **kwargs)

        monkeypatch.setattr(fsm, "batch_store_block", hang_on_key1)
        lines = _warnings(monkeypatch)
        tier._last_progress = 0.0  # idle for a long time before the hang
        tier.submit_store(make_job(1, [key(1)], [0]))
        time.sleep(0.4)
        tier.submit_store(make_job(2, [key(2)], [1]))  # a healthy slow disk
        deadline = time.monotonic() + 5
        done = []
        while not done and time.monotonic() < deadline:
            done = list(tier.get_finished_jobs())
        assert [r.job_id for r in done] == [2]
        time.sleep(0.3)  # job 1 ran > 0.6 s, but job 2 finished < 0.6 s ago
        assert list(tier.get_finished_jobs()) == [] and not tier._tripped
        time.sleep(0.5)
        assert list(tier.get_finished_jobs()) == []  # nothing fabricated
        assert tier._tripped
        assert len(lines) == 1 and "fs KV tier disabled: I/O stalled" in lines[0]
        assert tier.lookup(key(2), _CTX) is LookupResult.MISS
        assert not tier.accepts_store([key(3)], _CTX)
        gate.set()
        assert [(r.job_id, r.success) for r in drain(tier)] == [(1, True)]
        tier.on_schedule_end(_END)  # probe
        _wait_probe(tier)
        tier.on_schedule_end(_END)
        assert not tier._tripped
        assert list(tier.get_finished_jobs()) == [] and not tier._tripped  # grace
        # a queued job that never started is not a stall
        tier2, _ = _tier(tmp_path / "t2", breaker_stall_s=0.01)
        try:
            tier2._last_progress = 0.0
            tier2._job_started[99] = [0.0]
            tier2.get_finished_jobs()
            assert not tier2._tripped
            del tier2._job_started[99]
        finally:
            tier2.shutdown()
    finally:
        gate.set()
        tier.shutdown()


def test_admission_evicts_boundedly_after_shrink(tmp_path, monkeypatch):
    monkeypatch.setattr(fsm, "_EVICT_EXTRA_BLOCKS", 1)
    tier, _ = _tier(tmp_path, max_bytes=20 * _BS)
    try:
        for i in range(10):
            _store(tier, i, [i], [i % 8])
        tier._max_bytes = 4 * _BS  # as if shrunk: 6 blocks over the cap
        _store(tier, 20, [20], [0])
        # evicted this job's block + 1 extra, and the job still wrote
        assert tier._cache_bytes == 9 * _BS and _exists(tier, 20)
        _store(tier, 21, [21], [1])
        assert tier._cache_bytes == 8 * _BS and _exists(tier, 21)
        stats = tier.get_stats().reduce()
        assert "vllm:kv_offload_fs_stores_skipped" not in stats
        # unchanged when not over the cap: evict exactly what the job needs
        tier._max_bytes = 9 * _BS
        _store(tier, 22, [22, 23], [2, 3])
        assert tier._cache_bytes == 9 * _BS
    finally:
        tier.shutdown()


def test_enospc_at_floor_counts_toward_breaker(tmp_path, monkeypatch):
    tier, _ = _tier(tmp_path, max_bytes=4 * _BS, breaker_consecutive_failures=2)
    try:
        assert tier._min_max_bytes == 2 * _BS
        _store(tier, 1, [1, 2], [0, 1])
        _failing(monkeypatch, "batch_store_block", errno.ENOSPC)
        _store(tier, 2, [3], [2])
        assert tier._max_bytes == 2 * _BS  # floored, not 0.95 * 2 blocks
        assert tier._consecutive_failures == 0
        _store(tier, 3, [4], [3])  # still full at the floor: a real failure
        assert tier._consecutive_failures == 1 and tier._max_bytes == 2 * _BS
        _store(tier, 4, [5], [4])
        assert tier._tripped
    finally:
        tier.shutdown()


def test_device_guard_blocks_writes_and_probe(tmp_path, monkeypatch):
    import shutil

    tier, _ = _tier(
        tmp_path,
        max_bytes=8 * _BS,
        breaker_consecutive_failures=2,
        breaker_probe_interval_s=0,
    )
    try:
        _store(tier, 1, [1], [0])
        tier._storage_dev += 1  # as if the drive was swapped under the dir
        tier.submit_load(make_job(2, [key(1)], [5], is_promotion=True))
        tier.submit_load(make_job(3, [key(1)], [6], is_promotion=True))
        assert [r.success for r in drain(tier)] == [False, False]
        assert tier._tripped  # ENODEV counts, unlike ENOENT
        assert _exists(tier, 1)  # never touched
        tier.on_schedule_end(_END)
        _wait_probe(tier)
        tier.on_schedule_end(_END)
        assert tier._tripped  # probe refuses the foreign device
        tier._storage_dev -= 1
        # unmounted: the directory is gone; nothing may recreate it
        shutil.rmtree(tier._storage_dir)
        tier.on_schedule_end(_END)
        _wait_probe(tier)
        tier.on_schedule_end(_END)
        assert tier._tripped and not os.path.exists(tier._storage_dir)
        tier._tripped = False
        assert [r.success for r in _store(tier, 4, [2], [1])] == [False]
        assert not os.path.exists(tier._storage_dir)
    finally:
        tier.shutdown()
