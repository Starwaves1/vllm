# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the bounded fs KV tier (``max_bytes``).

Covers the bounded fs tier with LRU eviction (cf. open upstream #54327):
synchronous index lookups, pins for in-flight loads, skip-not-fail when full,
startup rescan and the fs-tier metrics.

Run: CUDA_VISIBLE_DEVICES= python -m pytest -q \
    tests/v1/kv_offload/tiering/test_fs_tier_bounded.py
"""

import os
import time
from unittest.mock import MagicMock

import pytest
import torch

import vllm.v1.kv_offload.tiering.fs.manager as fsm
from tests.v1.kv_offload.tiering.test_fs_tier import (
    _BLOCK_ELEMENTS,
    _make_offloading_spec,
    _page_aligned_zero_tensor,
    drain,
    key,
    make_job,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_SPEC = _make_offloading_spec()
_CTX = ReqContext(req_id="r0")
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
# bounded mode
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


def test_request_finish_refreshes_fs_recency(tmp_path):
    """The connector no longer calls touch() (#51787): the tiering manager
    passes a finished request's keys to the fs tier, so chunks a request used
    from the CPU tier stay fresh on disk."""
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    region = MagicMock()
    region.create_kv_memoryview.return_value = memoryview(tensor.numpy())
    primary = CPUPrimaryTierOffloadingManager(num_chunks=4, mmap_region=region)
    tier = FileSystemTierManager(
        offloading_spec=_SPEC,
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fs",
        root_dir=str(tmp_path),
        max_bytes=3 * _BS,
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    try:
        for i in range(3):
            _store(tier, i, [i], [i])
        ctx = ReqContext(req_id="uses-0")
        manager.on_new_request(ctx)
        ctx.set_offload_key_position(key(0), 16)  # the request's only chunk
        manager.on_request_finished(ctx)
        _store(tier, 10, [3], [3])  # evicts the LRU file: 1, not 0
        assert [_exists(tier, i) for i in range(4)] == [True, False, True, True]
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
        assert tier.get_stats().reduce()["vllm:kv_offload_fs_load_failures"] == 1
    finally:
        tier.shutdown()


def test_bounded_partial_load_keeps_loaded_blocks(tmp_path):
    """Main's partial keep (#50321) survives bounded mode: the blocks read
    before the bad one are reported loaded, only the rest are dropped."""
    tier, _ = _tier(tmp_path, max_bytes=4 * _BS)
    try:
        _store(tier, 1, [1, 2], [0, 1])
        os.remove(tier.file_mapper.get_file_name(key(2)))
        tier.submit_load(make_job(2, [key(1), key(2)], [3, 4], is_promotion=True))
        [result] = drain(tier)
        assert not result.success
        assert tuple(result.successful_keys) == (key(1),)
        assert tier.lookup(key(1), _CTX) is LookupResult.HIT
        assert tier.lookup(key(2), _CTX) is LookupResult.MISS
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
