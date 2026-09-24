# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the write-back store policy of secondary KV tiers.

``store_policy: "write_back"`` on a secondary tier: chunks stored in the CPU
tier are not cascaded; once per scheduler step, while the CPU tier is filled to
the high watermark, the coldest chunks the tier lacks are written (with the
earlier chunks of their prefix and their KV-cache-group siblings) until at most
the low watermark of the CPU tier is unwritten. Reads first: one chunk per job,
at most _WRITEBACK_INFLIGHT_BYTES in flight, none while a promotion is in
flight. Most tests set that budget to one chunk (one job in flight).
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
    get_offload_group_idx,
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
        low=0.75,
        cache_policy="lru",
        inflight_chunks=1,
        **tier_kw,
    ):
        self.tensor = _page_aligned_zero_tensor(n_blocks, _BLOCK_ELEMENTS)
        region = MagicMock()
        region.create_kv_memoryview.return_value = memoryview(self.tensor.numpy())
        self.primary = CPUPrimaryTierOffloadingManager(
            num_chunks=n_blocks, mmap_region=region, cache_policy=cache_policy
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
        self.wb = self.manager._writeback.get(0)
        if self.wb is not None and inflight_chunks is not None:
            self.manager._writeback_max_inflight_chunks = inflight_chunks
        self.ctx = ReqContext(req_id="r0")
        self.manager.on_new_request(self.ctx)
        self.patterns: dict[OffloadKey, float] = {}

    @property
    def policy(self):
        return self.primary._policy

    def store(self, keys, complete=True):
        out = self.manager.prepare_store(keys, self.ctx)
        assert out is not None
        for key, bid in zip(out.keys_to_store, out.store_spec.chunk_ids):
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
        return [m.transfer_job for m in self.manager._transfer_jobs.values()]

    def lru_order(self):
        return list(self.primary.iter_evictable())

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


def _blocking_store(monkeypatch, name="batch_store_block"):
    real = getattr(fsm, name)
    gate = threading.Event()

    def wait_then_run(*args, **kwargs):
        assert gate.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(fsm, name, wait_then_run)
    return gate


def _flush(e):
    """Step until the flusher idles; return each write-back job's keys. At
    most one job is ever in flight (the default one-chunk budget)."""
    written = []
    e.step()
    while e.jobs():
        (job,) = e.jobs()
        written.append(list(job.keys))
        e.settle()  # job done; the same step's flusher submits the next one
    return written


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
        {"store_policy": "write_back", "writeback_low_watermark": 1.0},
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
    e = env()  # 20 blocks, high 10, cold window 5
    e.store([k(i) for i in range(9)])
    assert not e.jobs() and not e.pinned([k(i) for i in range(9)])
    assert e.wb.dirty == {k(i) for i in range(9)}
    e.step()
    assert not e.jobs()  # 9 < high watermark (10 blocks)
    assert not any(e.tier.is_stored(k(i)) for i in range(9))
    assert e.stats()[_M.WRITEBACK_DIRTY] == 9


def test_flush_starts_at_high_and_stops_at_low(env):
    e = env()  # 20 blocks, high 10, cold window 5
    e.store([k(i) for i in range(10)])
    assert not e.jobs()  # nothing before the step's flusher runs
    e.step()
    cold = [k(i) for i in range(5)]
    assert [list(j.keys) for j in e.jobs()] == [[k(0)]]  # one chunk per job
    assert e.jobs()[0].req_context.req_id == WRITEBACK_REQ_ID
    assert e.pinned([k(i) for i in range(10)]) == [k(0)]  # ref+1 for the write
    e.step()
    assert len(e.jobs()) == 1  # one job in flight at a time
    assert _flush(e) == [[key] for key in cold]  # 5 unwritten left: at low
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
    e = env()  # 20 blocks, high 10, cold window 5
    e.store([k(i) for i in range(10)])
    _flush(e)
    # written blocks went back to the LRU end: they are the next victims
    assert sorted(e.lru_order()[:5]) == [k(i) for i in range(5)]
    e.store([k(i) for i in range(10, 20)])  # CPU tier now full
    e.step()
    assert not e.jobs()  # the cold window is clean: nothing to write
    evicted = e.store([k(i) for i in range(20, 25)]).evicted_keys
    assert sorted(evicted) == [k(i) for i in range(5)]  # all already on disk
    # the next 5 victims are written
    assert _flush(e) == [[k(i)] for i in range(5, 10)]
    stats = e.stats()
    assert _M.WRITEBACK_LOST not in stats
    assert stats[_M.WRITEBACK_FLUSHED] == 10


def test_promoted_clean_blocks_do_not_starve_the_cold_end(env):
    """Chunks promoted from the tier arrive clean and become MRU once the
    request that loads them finishes. However many of them the CPU tier
    holds, the dirty chunks at the cold end (the next eviction victims) must
    still be written. Counting dirty chunks against the low watermark instead
    left the cold end dirty: evicted unwritten."""
    e = env(high=0.85, low=0.75)  # 20 chunks, high 17, cold window 5
    promoted = [k(100 + i) for i in range(5)]
    e.tier.submit_store(make_job(900, promoted, list(range(15, 20))))
    drain(e.tier)
    for key in promoted:
        assert e.manager.lookup(key, e.ctx) is LookupResult.HIT_PENDING
    e.step()  # submits the promotion (5 chunks used: below high)
    e.settle()
    assert not e.jobs()
    e.store([k(i) for i in range(15)])  # 15 dirty
    # another request loads the promoted chunks to the GPU; its finish makes
    # them MRU (#51787: request recency is applied at request finish)
    reader = ReqContext(req_id="reader")
    e.manager.on_new_request(reader)
    spec = e.manager.prepare_load(promoted, reader)
    assert len(spec.chunk_ids) == 5
    e.manager.complete_load(promoted, reader)
    e.manager.on_request_finished(reader)
    assert e.lru_order()[-5:] == promoted[::-1]
    assert len(e.wb.dirty) == 15 and e.primary.num_used_chunks() == 20
    assert _flush(e) == [[k(i)] for i in range(5)]
    evicted = e.store([k(i) for i in range(20, 25)]).evicted_keys
    assert sorted(evicted) == [k(i) for i in range(5)]
    assert _M.WRITEBACK_LOST not in e.stats()


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
    e = env(max_inflight_store_bytes=_BS)  # 20 blocks, high 10, cold window 5
    gate = _blocking_store(monkeypatch)
    try:
        e.store([k(i) for i in range(10)])
        # another store fills the tier's write backlog
        e.tier.submit_store(make_job(900, [k(100)], [19]))
        e.step()
        assert not e.jobs()  # declined
        assert not e.pinned([k(i) for i in range(10)])
        assert not e.tier._dropping_reqs  # not sticky
        gate.set()
        drain(e.tier)
        assert _flush(e) == [[k(i)] for i in range(5)]
        assert len(e.wb.dirty) == 5
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
    assert [list(j.keys) for j in e.jobs()] == [[k(0)]]
    e.tier.drain_jobs()
    monkeypatch.undo()
    e.step()  # failure consumed: unpinned, still dirty, offered again
    assert len(e.wb.dirty) == 10 and e.wb.n_flushed == 0
    assert [list(j.keys) for j in e.jobs()] == [[k(0)]]
    assert not any(e.tier.is_stored(k(i)) for i in range(10))
    _flush(e)
    assert len(e.wb.dirty) == 5


# ---------------------------------------------------------------------------
# hybrid groups, prefix order, pins
# ---------------------------------------------------------------------------


def test_whole_chunk_all_groups_flushed_together(env):
    e = env(high=0.5, low=0.96)  # 20 blocks, high 10, cold window 1
    # 3 chunks x 4 groups (1 attention + 3 GDN), stored group by group, so
    # the LRU order interleaves chunks: c0g0 c1g0 c2g0 c0g1 ...
    e.store([k(c, g) for g in range(4) for c in range(3)])
    e.step()
    # cold window of 1: the coldest block is c0g0, and its whole chunk (all 4
    # groups) goes in one job, although that overshoots the budget
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
    e = env(high=0.3, low=0.86)  # 20 blocks, high 6, cold window 3
    chain = [k(c) for c in range(6)]
    e.store(chain)
    # the connector touches a request's keys in chunk order; LRU then holds
    # the tail as coldest: c5 c4 c3 c2 c1 c0
    e.manager.touch(chain, e.ctx)
    assert e.lru_order() == chain[::-1]
    # the cold pick is c5, but c5 on disk is useless without c0..c4 (prefix
    # lookup stops at the first miss): the prefix goes oldest first, one
    # chunk per job, until the cold window of 3 is written
    assert _flush(e) == [[key] for key in chain[:3]]
    # every written block no request used meanwhile goes back to the LRU end,
    # prefix chunks included
    assert e.lru_order() == chain[:3][::-1] + chain[3:][::-1]


def test_one_step_pins_one_chunk(env):
    """A long dirty prefix behind one cold block is written one chunk (all
    its groups) per job, oldest first, one job in flight."""
    e = env(n_blocks=200, high=0.75, low=0.8)  # high 150, cold window 40
    groups = [[k(c, g) for c in range(40)] for g in range(4)]
    e.store([key for group in groups for key in group])  # 160 blocks
    for group in groups:
        e.manager.touch(group, e.ctx)
    e.step()
    e.step()
    assert [sorted(j.keys) for j in e.jobs()] == [[k(0, g) for g in range(4)]]
    written = _flush(e)
    assert [sorted(keys) for keys in written[:3]] == [
        [k(c, g) for g in range(4)] for c in range(3)
    ]


def test_prefix_walk_stops_at_block_already_on_disk(env):
    e = env(high=0.3, low=0.86)  # 20 blocks, high 6, cold window 3
    e.tier.submit_store(make_job(900, [k(2)], [19]))  # c2 on disk already
    drain(e.tier)
    chain = [k(c) for c in range(6)]
    e.store(chain)
    assert k(2) not in e.wb.dirty  # is_stored: nothing to write
    e.manager.touch(chain, e.ctx)
    assert _flush(e) == [[k(3)], [k(4)], [k(5)]]


def test_block_used_while_flushing_is_not_demoted(env, monkeypatch):
    e = env()
    gate = _blocking_store(monkeypatch)
    try:
        e.store([k(i) for i in range(10)])
        e.step()
        e.manager.touch([k(0)], e.ctx)  # a request hits k0 mid-write
        gate.set()
        assert _flush(e)[0] == [k(0)]
        order = e.lru_order()
        assert sorted(order[:5]) == [k(i) for i in range(1, 6)]
        assert order[-1] == k(0)
    finally:
        gate.set()


def test_flushed_block_promoted_back_and_hits(env):
    e = env(n_blocks=8, high=0.5, low=0.0)
    e.store([k(i) for i in range(4)])
    _flush(e)
    assert e.wb.dirty == set()
    e.store([k(i) for i in range(4, 8)])  # CPU tier full
    (victim,) = e.store([k(8)]).evicted_keys
    assert victim in [k(i) for i in range(4)]  # a written, demoted block
    assert e.manager.lookup(victim, e.ctx) is LookupResult.HIT_PENDING  # promote
    e.step()  # submits the load
    e.settle()
    assert e.manager.lookup(victim, e.ctx) is LookupResult.HIT
    spec = e.manager.prepare_load([victim], e.ctx)
    row = e.tensor[int(spec.chunk_ids[0])]
    assert torch.all(row == e.patterns[victim])
    e.manager.complete_load([victim], e.ctx)
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
    assert e.stats()[_M.WRITEBACK_LOST] == 9  # all but the one in flight


def test_arc_policy_supported(env):
    e = env(cache_policy="arc")
    e.store([k(i) for i in range(10)])
    assert sorted(key for keys in _flush(e) for key in keys) == [k(i) for i in range(5)]
    assert len(e.wb.dirty) == 5


def test_request_finish_learns_prefix_links(env):
    """Main's connector does not call touch(): links come from the finished
    request's keys, ordered by the positions the connector records."""
    e = env(high=0.3, low=0.86)  # 20 chunks, high 6, cold window 3
    chain = [k(c) for c in range(6)]
    e.store(chain)
    for c, key in enumerate(chain[::-1]):  # recorded out of order on purpose
        e.ctx.set_offload_key_position(key, (6 - c) * 16)
    e.manager.on_request_finished(e.ctx)
    assert e.manager._writeback_parent == {
        chain[c][:-4]: chain[c - 1][:-4] for c in range(1, 6)
    }
    # request finish made the head most recent: the tail is the cold pick,
    # and its prefix goes first
    assert e.lru_order() == chain[::-1]
    assert _flush(e) == [[key] for key in chain[:3]]


def test_request_touch_learns_links_only_for_cpu_blocks(env):
    e = env()
    e.store([k(0), k(1)])
    e.manager.touch([k(0), k(1), k(2)], e.ctx)  # k2 only on the GPU
    assert e.manager._writeback_parent == {k(1)[:-4]: k(0)[:-4]}
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
    for group in keys:  # prefix links, as request finish passes them
        e.manager.touch(group, e.ctx)
    assert len(_flush(e)) == n  # 24 blocks >= 20: write all (low 0), by chunk
    assert not e.wb.dirty
    e.store([k(100 + i) for i in range(40)])  # evicts the whole request
    assert all(e.policy.get(key) is None for group in keys for key in group)

    groups = [
        GroupOffloadConfig(
            group_idx=g,
            tokens_per_block=tpc,
            tokens_per_chunk=tpc,
            hashes_per_chunk=1,
            kv_event_group_spec=MagicMock(),
            sliding_window_size_in_chunks=None if g == 0 else 1,
            kv_cache_spec=MagicMock(),
            manager_cls=MagicMock(),
        )
        for g in range(4)
    ]
    stub = SimpleNamespace(
        manager=e.manager,
        config=SimpleNamespace(kv_group_configs=groups, supports_partial_tail=False),
        _lookup_groups=(0, 1, 2, 3),
        _sliding_window_groups=(1, 2, 3),
        _mamba_align_size=tpc,
        _chunks_being_loaded=set(),
        _events_tracker=MagicMock(),
    )
    for name in (
        "_lookup_complete_chunks",
        "_maximal_prefix_lookup",
        "_sliding_window_lookup",
    ):
        setattr(stub, name, getattr(OffloadingConnectorScheduler, name).__get__(stub))
    num_tokens = n * tpc + 100
    req_status = SimpleNamespace(
        num_locally_computed_tokens=0,
        max_load_tokens=None,
        partial_tail_boundary=None,
        req=SimpleNamespace(
            num_tokens=num_tokens, num_prompt_tokens=num_tokens, request_id="r0"
        ),
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
        assert torch.all(e.tensor[int(spec.chunk_ids[0])] == e.patterns[key])
        e.manager.complete_load([key], e.ctx)


def test_flush_jobs_are_one_chunk_one_at_a_time(env):
    """Reads first: a promotion queues behind at most one write-back job, and
    that job is a single chunk (its group siblings)."""
    e = env(n_blocks=200, high=0.5, low=0.0)
    e.store([k(c, g) for c in range(30) for g in range(4)])  # 120 blocks
    assert _flush(e) == [[k(c, g) for g in range(4)] for c in range(30)]
    assert not e.wb.dirty


def test_no_flush_while_promotion_in_flight(env, monkeypatch):
    """The flusher submits nothing while a load from the tier is in flight,
    and resumes once it is done."""
    e = env(n_blocks=8, high=0.5, low=0.0, inflight_chunks=4)
    e.tier.submit_store(make_job(900, [k(100)], [7]))
    drain(e.tier)  # k100: on disk only
    gate = _blocking_store(monkeypatch, "batch_load_block")
    try:
        assert e.manager.lookup(k(100), e.ctx) is LookupResult.HIT_PENDING
        e.store([k(i) for i in range(4)])  # 5 used >= high 4, all dirty
        e.step()  # submits the promotion; the flusher waits
        assert [j.is_promotion for j in e.jobs()] == [True]
        e.step()
        assert [j.is_promotion for j in e.jobs()] == [True]
        assert not e.wb.flushing
        gate.set()
        e.settle()  # load done; the same step's flusher resumes
        assert [list(j.keys) for j in e.jobs()] == [[k(i)] for i in range(4)]
        assert e.manager.lookup(k(100), e.ctx) is LookupResult.HIT
    finally:
        gate.set()


def test_prefix_walk_by_chunk_with_pruned_gdn_blocks(env):
    """GDN blocks stored only at some chunks (here every 8th and the last; on
    main only boundary states are offered), attention at every chunk. GDN keys
    are touched before attention, so they are the coldest; the walk from one
    must still go back along the chunk chain and write chunk 0 first, then
    the prefix in order."""
    n = 40
    keepers = set(range(7, n, 8)) | {n - 1}
    e = env(n_blocks=100, high=0.5, low=0.0)
    attention = [k(c, 0) for c in range(n)]
    gdn = [[k(c, g) for c in range(n)] for g in (1, 2, 3)]
    e.store(attention + [group[c] for group in gdn for c in sorted(keepers)])
    for group in gdn:  # the connector touches every group's full key list
        e.manager.touch(group, e.ctx)
    e.manager.touch(attention, e.ctx)
    assert get_offload_group_idx(e.lru_order()[0]) != 0
    order = []
    for _ in range(n + 5):
        e.step()
        for job in e.jobs():
            chunks = {bytes(key[:-4]) for key in job.keys}
            assert len(chunks) == 1  # one whole chunk per job
            order.append(chunks.pop())
        e.settle()
    assert order == [k(c)[:-4] for c in range(n)]
    assert not e.wb.dirty


def test_flush_budget_keeps_several_chunk_jobs_in_flight(env):
    """With no promotion in flight, one step submits one-chunk jobs in prefix
    order until the in-flight budget is used, so the disk isn't left idle
    between scheduler steps."""
    e = env(n_blocks=200, high=0.5, low=0.0, inflight_chunks=None)
    assert (
        e.manager._writeback_max_inflight_chunks == tm._WRITEBACK_INFLIGHT_BYTES // _BS
    )
    e.manager._writeback_max_inflight_chunks = 12  # 3 chunks of 4 groups
    groups = [[k(c, g) for c in range(30)] for g in range(4)]
    e.store([key for group in groups for key in group])  # 120 blocks
    for group in groups:  # LRU: the tail chunks are coldest
        e.manager.touch(group, e.ctx)
    e.step()
    chunk = [[k(c, g) for g in range(4)] for c in range(30)]
    assert [sorted(j.keys) for j in e.jobs()] == chunk[:3]
    e.step()
    assert len(e.jobs()) == 3  # budget used: nothing more until one finishes
    e.settle()
    assert [sorted(j.keys) for j in e.jobs()] == chunk[3:6]


def test_prod_config_accepted(tmp_path):
    """The deployed kv_connector_extra_config (qwen-vllm.service), with the
    fs tier moved to tmp_path and a small CPU tier: every key is accepted,
    including ones main no longer reads (mamba_keep_every_n_chunks)."""
    from tests.v1.kv_connector.unit.utils import create_vllm_config
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )

    extra_config = {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": 1 << 20,  # prod: 25769803776
        "sync_load": True,
        "mamba_keep_every_n_chunks": 8,
        "secondary_tiers": [
            {
                "type": "fs",
                "root_dir": str(tmp_path),  # prod: /mnt/kvcache/tier
                "max_bytes": 300000000000,
                "n_read_threads": 8,
                "n_write_threads": 4,
                "store_policy": "write_back",
                "writeback_high_watermark": 0.85,
                "writeback_low_watermark": 0.5,
                "max_inflight_store_bytes": 5000000000,
                "breaker_consecutive_failures": 8,
                "breaker_probe_interval_s": 300,
                "breaker_probe_timeout_s": 30,
                "breaker_stall_s": 600,
            }
        ],
    }
    vllm_config = create_vllm_config(block_size=4, max_num_batched_tokens=16)
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config=extra_config,
    )
    attn = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(
                size=attn.page_size_bytes * 16,
                layers=["layer"],
                layer_stride=attn.page_size_bytes * 16,
                block_stride=attn.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], attn)],
    )
    spec = TieringOffloadingSpec(build_offloading_config(vllm_config, kv_cache_config))
    manager = spec.get_manager()
    try:
        (tier,) = manager.secondary_tiers
        assert isinstance(tier, FileSystemTierManager)
        assert tier.store_policy == StorePolicy.WRITE_BACK
        assert (tier.writeback_high_watermark, tier.writeback_low_watermark) == (
            0.85,
            0.5,
        )
        assert tier._max_bytes == 300 * 10**9
        assert tier._max_inflight_store_bytes == 5 * 10**9
        assert (tier._breaker_threshold, tier._stall_s) == (8, 600.0)
        assert (tier._probe_interval, tier._probe_timeout) == (300.0, 30.0)
        assert tier.bp_detector is None  # no "backpressure" key: off
        assert 0 in manager._writeback
    finally:
        manager.shutdown()
