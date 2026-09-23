# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the write-back store policy of secondary KV tiers.

``store_policy: "write_back"`` on a secondary tier: blocks stored in the CPU
tier are not cascaded; once per scheduler step, while the CPU tier is filled to
the high watermark, the coldest blocks the tier lacks are written (with the
earlier chunks of their prefix and their KV-cache-group siblings) until at most
the low watermark of the CPU tier is unwritten.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.v1.kv_offload.tiering.fs.manager as fsm
import vllm.v1.kv_offload.tiering.manager as tm
from tests.v1.kv_offload.tiering.test_fs_tier import (
    _BLOCK_ELEMENTS,
    _make_offloading_spec,
    _page_aligned_zero_tensor,
    drain,
    make_job,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.tiering.base import (
    WRITEBACK_REQ_ID,
    StorePolicy,
    TieringOffloadingMetrics,
)
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

_SPEC = _make_offloading_spec()
_END = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
_BS = _BLOCK_ELEMENTS * 4  # bytes per block
_M = TieringOffloadingMetrics


def k(chunk: int, group: int = 0) -> OffloadKey:
    return make_offload_key(b"chunk%04d" % chunk, group)


class _Env:
    """A TieringOffloadingManager with a CPU tier of ``n_blocks`` and one
    bounded fs tier; GPU->CPU stores are simulated by prepare/complete_store
    with a per-key byte pattern written into the CPU slot."""

    def __init__(
        self,
        tmp_path,
        n_blocks=20,
        policy="write_back",
        high=0.5,
        low=0.25,
        cache_policy="lru",
        **tier_kw,
    ):
        self.tensor = _page_aligned_zero_tensor(n_blocks, _BLOCK_ELEMENTS)
        region = MagicMock()
        region.create_kv_memoryview.return_value = memoryview(self.tensor.numpy())
        self.primary = CPUPrimaryTierOffloadingManager(
            num_blocks=n_blocks, mmap_region=region, cache_policy=cache_policy
        )
        tier_kw.setdefault("max_bytes", 1000 * _BS)
        self.tier = FileSystemTierManager(
            offloading_spec=_SPEC,
            primary_kv_view=self.primary.get_kv_memoryview(),
            tier_type="fs",
            root_dir=str(tmp_path),
            n_read_threads=1,
            n_write_threads=1,
            **tier_kw,
        )
        self.tier.configure_store_policy(policy, high, low)
        self.manager = TieringOffloadingManager(
            primary_tier=self.primary, secondary_tiers=[self.tier]
        )
        self.wb = self.manager._writeback.get(self.tier)
        self.ctx = ReqContext(req_id="r0")
        self.manager.on_new_request(self.ctx)
        self.patterns: dict[OffloadKey, float] = {}

    @property
    def policy(self):
        return self.primary._policy

    def store(self, keys, complete=True):
        out = self.manager.prepare_store(keys, self.ctx)
        assert out is not None
        for key, bid in zip(out.keys_to_store, out.store_spec.block_ids):
            value = float(len(self.patterns) + 1)
            self.patterns[key] = value
            self.tensor[int(bid)] = value
        if complete:
            self.manager.complete_store(keys, self.ctx)
        return out

    def step(self):
        self.manager.on_schedule_end(_END)

    def settle(self):
        """Finish all fs I/O and let the manager consume the results."""
        self.tier.drain_jobs()
        self.step()

    def jobs(self):
        return list(self.manager._transfer_jobs.values())

    def pinned(self, keys):
        return [key for key in keys if self.policy.get(key).ref_cnt > 0]

    def stats(self):
        stats = self.manager.get_stats()
        return {} if stats is None else stats.reduce()

    def close(self):
        self.tier.shutdown()


@pytest.fixture
def env(tmp_path):
    envs = []

    def make(**kw):
        e = _Env(tmp_path / f"e{len(envs)}", **kw)
        envs.append(e)
        return e

    yield make
    for e in envs:
        e.close()


def _blocking_store(monkeypatch):
    real = fsm.batch_store_block
    gate = threading.Event()

    def wait_then_store(*args, **kwargs):
        assert gate.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(fsm, "batch_store_block", wait_then_store)
    return gate


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_default_is_write_through_and_unchanged(env):
    e = env(policy="write_through")
    assert e.tier.store_policy == StorePolicy.WRITE_THROUGH
    assert not e.manager._writeback and e.primary.eviction_listener is None
    e.store([k(0)])
    # write-through: complete_store cascades at once and pins the block
    assert [list(j.keys) for j in e.jobs()] == [[k(0)]]
    assert e.pinned([k(0)]) == [k(0)]
    e.step()  # prepare_store already polled the tier in this step
    e.settle()
    assert e.tier.is_stored(k(0)) and not e.pinned([k(0)])
    assert _M.WRITEBACK_DIRTY not in e.stats()


def test_factory_parses_store_policy(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    cfg = {"type": "fs", "root_dir": str(tmp_path), "max_bytes": 10 * _BS}
    tier = SecondaryTierFactory.create_secondary_tier(dict(cfg), view, _SPEC)
    assert tier.store_policy == "write_through"
    tier.shutdown()
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            **cfg,
            "store_policy": "write_back",
            "writeback_high_watermark": 0.9,
            "writeback_low_watermark": 0.6,
        },
        view,
        _SPEC,
    )
    assert tier.store_policy == "write_back"
    assert (tier.writeback_high_watermark, tier.writeback_low_watermark) == (0.9, 0.6)
    tier.shutdown()
    for bad in (
        {"store_policy": "write_around"},
        {"store_policy": "write_back", "writeback_low_watermark": 0.9},
        {"store_policy": "write_back", "writeback_high_watermark": 1.5},
    ):
        with pytest.raises(ValueError, match="store_policy|watermark"):
            SecondaryTierFactory.create_secondary_tier({**cfg, **bad}, view, _SPEC)
    defs = TieringOffloadingSpec.build_metric_definitions({})
    for name in (_M.WRITEBACK_FLUSHED, _M.WRITEBACK_LOST, _M.WRITEBACK_DIRTY):
        assert name in defs


def test_write_back_needs_iterable_cache_policy(env, monkeypatch):
    monkeypatch.setattr(LRUCachePolicy, "iter_evictable", CachePolicy.iter_evictable)
    with pytest.raises(ValueError, match="iter_evictable"):
        env()


# ---------------------------------------------------------------------------
# write-back core
# ---------------------------------------------------------------------------


def test_no_store_on_complete_store(env):
    e = env()  # 20 blocks, high 10, low 5
    e.store([k(i) for i in range(9)])
    assert not e.jobs() and not e.pinned([k(i) for i in range(9)])
    assert e.wb.dirty == {k(i) for i in range(9)}
    e.step()
    assert not e.jobs()  # 9 < high watermark (10 blocks)
    assert not any(e.tier.is_stored(k(i)) for i in range(9))
    assert e.stats()[_M.WRITEBACK_DIRTY] == 9


def test_flush_starts_at_high_and_stops_at_low(env):
    e = env()  # 20 blocks, high 10, low 5
    e.store([k(i) for i in range(10)])
    assert not e.jobs()  # nothing before the step's flusher runs
    e.step()
    cold = [k(i) for i in range(5)]
    assert [list(j.keys) for j in e.jobs()] == [cold]
    assert e.jobs()[0].req_context.req_id == WRITEBACK_REQ_ID
    assert e.pinned([k(i) for i in range(10)]) == cold  # ref+1 for the write
    e.step()
    assert len(e.jobs()) == 1  # 5 unwritten left, none in flight: at low
    e.settle()
    assert not e.jobs() and not e.pinned(cold)
    assert all(e.tier.is_stored(key) for key in cold)
    assert e.wb.dirty == {k(i) for i in range(5, 10)}
    e.step()
    assert not e.jobs()  # clean blocks are not written again
    stats = e.stats()
    assert stats[_M.WRITEBACK_FLUSHED] == 5
    assert stats[_M.WRITEBACK_DIRTY] == 5
    assert _M.WRITEBACK_LOST not in stats


def test_flushed_blocks_evicted_first_and_not_reflushed(env):
    e = env()  # 20 blocks, high 10, low 5
    e.store([k(i) for i in range(10)])
    e.step()
    e.settle()
    # written blocks went back to the LRU end: they are the next victims
    assert list(e.policy.evictable_blocks)[:5] == [k(i) for i in range(5)]
    e.store([k(i) for i in range(10, 20)])  # CPU tier now full
    e.step()
    # 15 unwritten -> down to 5: the 10 coldest unwritten, skipping clean ones
    assert sorted(key for j in e.jobs() for key in j.keys) == [
        k(i) for i in range(5, 15)
    ]
    e.settle()
    evicted = e.store([k(i) for i in range(20, 25)]).evicted_keys
    assert len(evicted) == 5 and all(e.tier.is_stored(key) for key in evicted)
    stats = e.stats()
    assert _M.WRITEBACK_LOST not in stats
    assert stats[_M.WRITEBACK_FLUSHED] == 15


def test_evicting_unwritten_block_counts_lost(env):
    e = env(n_blocks=4, high=1.0, low=0.5)
    e.store([k(i) for i in range(4)])  # full, but no step ran: nothing written
    evicted = e.store([k(4)]).evicted_keys
    assert evicted == [k(0)]
    assert k(0) not in e.wb.dirty
    stats = e.stats()
    assert stats[_M.WRITEBACK_LOST] == 1
    assert stats[_M.WRITEBACK_DIRTY] == 4
    assert e.stats()[_M.WRITEBACK_DIRTY] == 4  # gauge re-sent, counter reset
    assert _M.WRITEBACK_LOST not in e.stats()


def test_backlog_cap_stops_flushing_without_pinning(env, monkeypatch):
    monkeypatch.setattr(tm, "_WRITEBACK_JOB_BLOCKS", 2)
    e = env(max_inflight_store_bytes=2 * _BS)  # 20 blocks, high 10, low 5
    gate = _blocking_store(monkeypatch)
    try:
        e.store([k(i) for i in range(10)])
        e.step()
        # first job (2 blocks) fills the backlog; the next one is declined
        assert [list(j.keys) for j in e.jobs()] == [[k(0), k(1)]]
        assert e.pinned([k(i) for i in range(10)]) == [k(0), k(1)]
        assert not e.tier._dropping_reqs  # not sticky
        e.step()
        assert len(e.jobs()) == 1
        gate.set()
        e.settle()  # job done; the same step's flusher submits the next one
        assert [list(j.keys) for j in e.jobs()] == [[k(2), k(3)]]
        e.settle()
        assert [list(j.keys) for j in e.jobs()] == [[k(4)]]
        e.settle()
        assert not e.jobs() and len(e.wb.dirty) == 5
        stats = e.stats()
        assert stats[_M.WRITEBACK_FLUSHED] == 5
        assert "vllm:kv_offload_fs_stores_dropped" not in stats
    finally:
        gate.set()


def test_tripped_breaker_stops_flushing(env):
    e = env()
    e.store([k(i) for i in range(10)])
    e.tier._tripped = True
    e.step()
    assert not e.jobs() and len(e.wb.dirty) == 10
    assert not e.pinned([k(i) for i in range(10)])
    e.tier._tripped = False  # recovered (probe ok)
    e.step()
    assert len(e.jobs()) == 1


def test_failed_write_keeps_blocks_dirty(env, monkeypatch):
    e = env()

    def fail(paths, *args, **kwargs):
        raise OSError(5, "EIO", paths[0])

    monkeypatch.setattr(fsm, "batch_store_block", fail)
    e.store([k(i) for i in range(10)])
    e.step()
    assert len(e.jobs()) == 1
    e.tier.drain_jobs()
    monkeypatch.undo()
    e.step()  # failure consumed: unpinned, still dirty, offered again
    assert len(e.wb.dirty) == 10 and e.wb.n_flushed == 0
    assert [list(j.keys) for j in e.jobs()] == [[k(i) for i in range(5)]]
    assert not any(e.tier.is_stored(k(i)) for i in range(10))
    e.settle()
    assert len(e.wb.dirty) == 5


# ---------------------------------------------------------------------------
# hybrid groups, prefix order, pins
# ---------------------------------------------------------------------------


def test_whole_chunk_all_groups_flushed_together(env):
    e = env(high=0.5, low=0.45)  # 20 blocks, high 10, low 9
    # 3 chunks x 4 groups (1 attention + 3 GDN), stored group by group, so
    # the LRU order interleaves chunks: c0g0 c1g0 c2g0 c0g1 ...
    e.store([k(c, g) for g in range(4) for c in range(3)])
    e.step()
    # 12 unwritten, target 9: the coldest block is c0g0 and its whole chunk
    # (all 4 groups) goes in one job, although that overshoots by one
    assert [list(j.keys) for j in e.jobs()] == [[k(0, g) for g in range(4)]]


def test_unready_blocks_never_flushed(env):
    e = env(n_blocks=8, high=0.5, low=0.0)  # flush everything at >= 4 used
    e.store([k(0, g) for g in range(3)])
    e.store([k(0, 3)], complete=False)  # GPU->CPU copy in flight: ref_cnt -1
    assert e.policy.get(k(0, 3)).ref_cnt == -1
    e.wb.dirty.add(k(0, 3))  # even if it were marked dirty ...
    e.manager._writeback_groups[k(0, 3)[-4:]] = None  # ... and a known group
    e.step()
    assert [list(j.keys) for j in e.jobs()] == [[k(0, g) for g in range(3)]]
    assert e.policy.get(k(0, 3)).ref_cnt == -1
    e.wb.dirty.discard(k(0, 3))
    e.settle()
    e.manager.complete_store([k(0, 3)], e.ctx)
    e.step()
    assert [list(j.keys) for j in e.jobs()] == [[k(0, 3)]]


def test_prefix_flushed_with_cold_tail_oldest_first(env):
    e = env(high=0.3, low=0.15)  # 20 blocks, high 6, low 3
    chain = [k(c) for c in range(6)]
    e.store(chain)
    # the connector touches a request's keys in chunk order; LRU then holds
    # the tail as coldest: c5 c4 c3 c2 c1 c0
    e.manager.touch(chain, e.ctx)
    assert list(e.policy.evictable_blocks) == chain[::-1]
    e.step()
    # 3 blocks needed; the cold pick is c5, but c5 on disk is useless without
    # c0..c4 (prefix lookup stops at the first miss): the prefix goes oldest
    # first and the step stops at its budget, the rest follows later
    assert [list(j.keys) for j in e.jobs()] == [chain[:3]]
    e.settle()
    # every written block no request used meanwhile goes back to the LRU end,
    # prefix chunks included
    assert list(e.policy.evictable_blocks) == chain[:3] + chain[3:][::-1]


def test_one_step_pins_no_more_than_needed(env):
    """A long dirty prefix behind one cold block must not be pinned in one
    step: the flusher stops at its budget (plus at most one chunk's group
    siblings), so the CPU tier keeps enough evictable blocks to store."""
    e = env(n_blocks=200, high=0.75, low=0.6)  # high 150, low 120
    groups = [[k(c, g) for c in range(40)] for g in range(4)]
    e.store([key for group in groups for key in group])  # 160 blocks
    for group in groups:
        e.manager.touch(group, e.ctx)
    e.step()  # need = 160 - 120 = 40
    pinned = [key for j in e.jobs() for key in j.keys]
    assert 40 <= len(pinned) <= 40 + 3
    assert sorted(pinned) == sorted(k(c, g) for c in range(10) for g in range(4))
    # 40 free + 120 evictable: a 80-block store still fits
    assert e.manager.prepare_store([k(1000 + i) for i in range(80)], e.ctx)


def test_prefix_walk_stops_at_block_already_on_disk(env):
    e = env(high=0.3, low=0.1)  # 20 blocks, high 6, low 2
    e.tier.submit_store(make_job(900, [k(2)], [19]))  # c2 on disk already
    drain(e.tier)
    chain = [k(c) for c in range(6)]
    e.store(chain)
    assert k(2) not in e.wb.dirty  # is_stored: nothing to write
    e.manager.touch(chain, e.ctx)
    e.step()
    assert [list(j.keys) for j in e.jobs()] == [[k(3), k(4), k(5)]]


def test_block_used_while_flushing_is_not_demoted(env, monkeypatch):
    e = env()
    gate = _blocking_store(monkeypatch)
    try:
        e.store([k(i) for i in range(10)])
        e.step()
        e.manager.touch([k(0)], e.ctx)  # a request hits k0 mid-write
        gate.set()
        e.settle()
        order = list(e.policy.evictable_blocks)
        assert order[:4] == [k(i) for i in range(1, 5)]
        assert order[-1] == k(0)
    finally:
        gate.set()


def test_flushed_block_promoted_back_and_hits(env):
    e = env(n_blocks=8, high=0.5, low=0.0)
    e.store([k(i) for i in range(4)])
    e.step()
    e.settle()
    assert e.wb.dirty == set()
    e.store([k(i) for i in range(4, 8)])  # CPU tier full
    evicted = e.store([k(8)]).evicted_keys
    assert evicted == [k(0)]  # the written, demoted block went first
    assert e.manager.lookup(k(0), e.ctx) is LookupResult.HIT_PENDING  # promote
    e.step()  # submits the load
    e.settle()
    assert e.manager.lookup(k(0), e.ctx) is LookupResult.HIT
    spec = e.manager.prepare_load([k(0)], e.ctx)
    row = e.tensor[int(spec.block_ids[0])]
    assert torch.all(row == e.patterns[k(0)])
    e.manager.complete_load([k(0)], e.ctx)
    assert _M.WRITEBACK_LOST not in e.stats()


def test_reset_cache_drops_writeback_state(env):
    e = env()
    e.store([k(i) for i in range(10)])
    e.step()
    e.manager.touch([k(i) for i in range(10)], e.ctx)
    e.manager.reset_cache()
    wb = e.wb
    assert not (wb.dirty or wb.flushing or wb.touched)
    assert not e.manager._writeback_jobs and not e.manager._writeback_parent
    assert e.stats()[_M.WRITEBACK_LOST] == 5  # the 5 never written


def test_arc_policy_supported(env):
    e = env(cache_policy="arc")
    e.store([k(i) for i in range(10)])
    e.step()
    assert [sorted(j.keys) for j in e.jobs()] == [[k(i) for i in range(5)]]
    e.settle()
    assert len(e.wb.dirty) == 5


def test_request_touch_learns_links_only_for_cpu_blocks(env):
    e = env()
    e.store([k(0), k(1)])
    e.manager.touch([k(0), k(1), k(2)], e.ctx)  # k2 only on the GPU
    assert e.manager._writeback_parent == {k(1): k(0)}
    evicted = e.store([k(i) for i in range(2, 22)]).evicted_keys
    assert sorted(evicted) == [k(0), k(1)]
    assert e.manager._writeback_parent == {}  # pruned on eviction


def test_hybrid_request_hits_from_disk_after_eviction(env):
    """End to end through the connector's real _lookup(): a 4-group request
    (1 attention prefix group + 3 GDN one-chunk windows, the Qwen3.8 layout)
    written back, pushed out of the CPU tier, then fully served from disk."""
    tpc, n = 832, 6
    e = env(n_blocks=40, high=0.5, low=0.0)
    keys = [[k(c, g) for c in range(n)] for g in range(4)]
    e.store([key for group in keys for key in group])
    for group in keys:  # as OffloadingConnectorScheduler._touch does
        e.manager.touch(group, e.ctx)
    e.step()  # 24 blocks >= 20: write everything unwritten (low 0)
    for _ in range(5):
        e.settle()
    assert not e.wb.dirty
    e.store([k(100 + i) for i in range(40)])  # evicts the whole request
    assert all(e.policy.get(key) is None for group in keys for key in group)

    groups = [
        GroupOffloadConfig(g, tpc, tpc, 1, MagicMock(), None if g == 0 else 1)
        for g in range(4)
    ]
    stub = SimpleNamespace(
        manager=e.manager,
        config=SimpleNamespace(kv_group_configs=groups, sync_load=True),
        _lookup_groups=(0, 1, 2, 3),
        _sliding_window_groups=(1, 2, 3),
        _mamba_align_size=tpc,
        _chunks_being_loaded=set(),
        _events_tracker=MagicMock(),
    )
    for name in ("_maximal_prefix_lookup", "_sliding_window_lookup"):
        setattr(stub, name, getattr(OffloadingConnectorScheduler, name).__get__(stub))
    req_status = SimpleNamespace(
        num_locally_computed_tokens=0,
        req=SimpleNamespace(num_tokens=n * tpc + 100, request_id="r0"),
        req_context=e.ctx,
        group_states=[SimpleNamespace(offload_keys=group) for group in keys],
    )
    hit = None
    for _ in range(10):
        hit = OffloadingConnectorScheduler._lookup(stub, req_status)
        if hit is not None:
            break
        e.step()
        e.tier.drain_jobs()
    assert hit == n * tpc
    # every attention chunk, and the GDN state at the hit boundary, came back
    for key in keys[0] + [keys[g][-1] for g in (1, 2, 3)]:
        assert e.manager.lookup(key, e.ctx) is LookupResult.HIT
        spec = e.manager.prepare_load([key], e.ctx)
        assert torch.all(e.tensor[int(spec.block_ids[0])] == e.patterns[key])
        e.manager.complete_load([key], e.ctx)


def test_flush_jobs_stay_small(env):
    """Each write-back job pins its blocks until its last write lands and one
    long job counts toward the fs tier's stall detector (breaker_stall_s), so a
    large flush is split into jobs of at most _WRITEBACK_JOB_BLOCKS plus the
    group siblings of the last chunk."""
    e = env(n_blocks=200, high=0.5, low=0.0)
    e.store([k(c, g) for c in range(30) for g in range(4)])  # 120 blocks
    e.step()
    sizes = [len(j.keys) for j in e.jobs()]
    assert sum(sizes) == 120
    assert max(sizes) <= tm._WRITEBACK_JOB_BLOCKS + 3
    assert all(size % 4 == 0 for size in sizes)  # chunks never split
    for _ in range(len(sizes)):
        e.settle()
    assert not e.wb.dirty and not e.jobs()
