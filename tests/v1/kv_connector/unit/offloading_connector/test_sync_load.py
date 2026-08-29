# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OffloadingConnector sync_load mode.

With ``sync_load`` enabled, an offloaded-prefix hit is consumed through the
same admission path as a plain (cache-miss) request: the request is scheduled
immediately (never parked in WAITING_FOR_REMOTE_KVS, no up-front full-hit
reservation) and the worker blocks on the transfer in start_load_kv, before
the forward pass consumes the loaded KV.

The harness would also fail loudly if a sync load leaked ``finished_recving``
for a running request: the base scheduler asserts such requests are finished.
"""

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.request import RequestStatus

BLOCK_SIZE = 4


def _never_parked_checker(runner):
    """post_step_fn asserting no request ever enters the async-load states."""

    def check():
        for req in runner.scheduler.requests.values():
            assert req.status != RequestStatus.WAITING_FOR_REMOTE_KVS
        assert not runner.scheduler.finished_recving_kv_req_ids
        assert not runner.scheduler.failed_recving_kv_req_ids

    return check


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sync_load_serves_hit_without_parking(request_runner, async_scheduling):
    """A full offloaded-prefix hit is loaded synchronously and scheduled in
    the same step; the hit is capped at num_tokens - 1 so the request always
    has one token to compute. The loaded blocks also appear as waited-on
    (flushed) - proof the worker blocked before the forward pass."""
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        extra_config_overrides={"sync_load": True},
    )

    # Store a 4-chunk prompt.
    runner.new_request(token_ids=[0] * BLOCK_SIZE * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3))

    # Reload it from the offloaded medium with an empty GPU prefix cache.
    # All 4 chunks hit; the sync cap trims the hit to num_tokens - 1 = 15
    # tokens, which still loads all 4 chunks (the last token of block 3 is
    # recomputed as this step's compute chunk).
    runner.scheduler.reset_prefix_cache()
    runner.manager.lookup.return_value = LookupResult.HIT
    runner.new_request(token_ids=[0] * BLOCK_SIZE * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(0, 1, 2, 3),
        expected_flushed=(0, 1, 2, 3),
        post_step_fn=_never_parked_checker(runner),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sync_load_local_prefix_plus_offloaded_suffix(request_runner, async_scheduling):
    """The load boundary is derived positionally from the locally computed
    token count (block hashes are already assigned for sync loads): a GPU
    prefix-cache hit is skipped and only the offloaded continuation loads."""
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        extra_config_overrides={"sync_load": True},
    )

    # Store chunks 2..7 of an 8-chunk prompt.
    runner.new_request(token_ids=[7] * BLOCK_SIZE * 8)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(list(keys)[2:])
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(2, 3, 4, 5, 6, 7))
    runner.scheduler.reset_prefix_cache()

    # Warm the GPU prefix cache with the first 2 chunks only.
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.new_request(token_ids=[7] * BLOCK_SIZE * 2)
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    # Same 8-chunk prompt: blocks 0-1 hit the GPU prefix cache, chunks 2..7
    # hit the offloaded medium (the sync cap trims the last token; block 7 is
    # still fully loaded and its last token recomputed).
    stored_keys = {key for key, _ in runner.offloaded}
    runner.manager.lookup.side_effect = lambda key, req_context: (
        LookupResult.HIT if key in stored_keys else LookupResult.MISS
    )
    runner.new_request(token_ids=[7] * BLOCK_SIZE * 8)
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(2, 3, 4, 5, 6, 7),
        expected_flushed=(2, 3, 4, 5, 6, 7),
        post_step_fn=_never_parked_checker(runner),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sync_load_recomputes_partial_chunk_tail(request_runner, async_scheduling):
    """Hits stay chunk-granular under sync_load: a trailing partial chunk is
    recomputed, exactly as with async loads."""
    blocks_per_chunk = 3
    tokens_per_chunk = BLOCK_SIZE * blocks_per_chunk
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        blocks_per_chunk=blocks_per_chunk,
        extra_config_overrides={"sync_load": True},
    )

    # Store a 3-chunk prompt.
    runner.new_request(token_ids=[0] * tokens_per_chunk * 3)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=tuple(range(9)))

    # A prompt covering 2 chunks plus a partial third chunk: only the two
    # complete chunks load; the 8-token tail is recomputed.
    runner.scheduler.reset_prefix_cache()
    runner.manager.lookup.return_value = LookupResult.HIT
    runner.new_request(token_ids=[0] * (tokens_per_chunk * 2 + 2 * BLOCK_SIZE))
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=tuple(range(6)),
        expected_flushed=tuple(range(6)),
        post_step_fn=_never_parked_checker(runner),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sync_load_with_sliding_window_group(request_runner, async_scheduling):
    """Sync loads work with hybrid groups sharing one block size: the
    sliding-window group loads only its window, and the positional boundary
    derivation skips that group's leading null placeholder blocks."""
    sliding_window = 8  # -> 2 offloaded chunks (blocks_per_chunk=1)
    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            SlidingWindowSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=sliding_window,
            ),
        ),
    ]
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        kv_cache_groups=kv_cache_groups,
        extra_config_overrides={"sync_load": True},
    )

    # Store 3 full blocks (the 13th token leaves block 3 partial).
    runner.new_request(token_ids=[0] * (BLOCK_SIZE * 3 + 1))
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2))

    # Reload: the full-attention group loads blocks 0-2; the sliding-window
    # group (window = 2 chunks) loads only blocks 1-2, its block 0 being a
    # null placeholder the boundary derivation must skip.
    runner.scheduler.reset_prefix_cache()
    runner.manager.lookup.return_value = LookupResult.HIT
    runner.new_request(token_ids=[0] * (BLOCK_SIZE * 3 + 1))
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
        expected_flushed=((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
        post_step_fn=_never_parked_checker(runner),
    )


def test_sync_load_rejects_mixed_block_sizes(request_runner):
    """sync_load derives load boundaries positionally, which requires a
    uniform block size across KV cache groups; mixed sizes must fail fast."""
    tokens_per_hash = 4
    kv_cache_groups = [
        KVCacheGroupSpec(
            ["layer0"],
            FullAttentionSpec(
                block_size=tokens_per_hash * 3,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["layer1"],
            FullAttentionSpec(
                block_size=tokens_per_hash * 4,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
    ]
    with pytest.raises(ValueError, match="sync_load requires"):
        request_runner(
            block_size=tokens_per_hash,
            num_gpu_blocks=100,
            async_scheduling=False,
            kv_cache_groups=kv_cache_groups,
            extra_config_overrides={"sync_load": True},
        )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sync_load_then_store_of_new_chunks_same_step(request_runner, async_scheduling):
    """Regression test (production engine crash): a partial offloaded-prefix
    hit schedules through the plain path, so the SAME scheduling step that
    issues the sync load job also computes the miss tail past the hit and
    proposes storing it. _build_store_jobs must tolerate the request's own
    (already host-waited) sync load job in transfer_jobs instead of asserting
    that any pending job is a store."""
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        extra_config_overrides={"sync_load": True},
    )

    # Store a 4-chunk prompt.
    runner.new_request(token_ids=[3] * BLOCK_SIZE * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3))
    runner.scheduler.reset_prefix_cache()

    # A longer, 6-chunk prompt: chunks 0-3 hit the offloaded medium, chunks
    # 4-5 are computed in the admission step and become storable in that very
    # step (stores are proposed at schedule time and deferred to the next
    # step's submission). The proposal includes the loaded chunks 0-3; the
    # mock manager stores everything it is offered.
    runner.manager.lookup.side_effect = lambda key, req_context: (
        LookupResult.HIT
        if key in {k for k, _ in runner.offloaded}
        else LookupResult.MISS
    )
    runner.new_request(token_ids=[3] * BLOCK_SIZE * 6)
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(0, 1, 2, 3),
        expected_flushed=(0, 1, 2, 3),
        expected_stored=(4, 5),
        post_step_fn=_never_parked_checker(runner),
    )


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sync_load_back_to_back_requests(request_runner, async_scheduling):
    """Two consecutive sync-loaded requests, each storing its computed tail
    in its own admission step: the first request's load-job accounting must
    not leak into the second's store issuance."""
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        extra_config_overrides={"sync_load": True},
    )

    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.manager.lookup.side_effect = lambda key, req_context: (
        LookupResult.HIT
        if key in {k for k, _ in runner.offloaded}
        else LookupResult.MISS
    )

    # Seed the medium with a 4-chunk prompt (all misses, stored on the way).
    runner.new_request(token_ids=[6] * BLOCK_SIZE * 4)
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3))
    runner.scheduler.reset_prefix_cache()

    # First sync load: 6-chunk prompt, hits chunks 0-3, stores its tail.
    runner.new_request(token_ids=[6] * BLOCK_SIZE * 6)
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(0, 1, 2, 3),
        expected_flushed=(0, 1, 2, 3),
        expected_stored=(4, 5),
        post_step_fn=_never_parked_checker(runner),
    )
    runner.scheduler.reset_prefix_cache()

    # Second sync load, immediately after: 8-chunk prompt, hits chunks 0-5
    # (chunks 4-5 were stored by the previous request), stores its tail.
    runner.new_request(token_ids=[6] * BLOCK_SIZE * 8)
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_loaded=(0, 1, 2, 3, 4, 5),
        expected_flushed=(0, 1, 2, 3, 4, 5),
        expected_stored=(6, 7),
        post_step_fn=_never_parked_checker(runner),
    )


def test_sync_load_preempted_before_completion_ack(request_runner):
    """A sync-loaded request that is preempted before the worker's completion
    ack still has its load job in transfer_jobs. The preemption flush must
    not treat that job as a store (engine-killing AssertionError) and must
    not flush it: the load finished before its own step's forward pass, so
    only pending stores need flushing."""
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=False,
        extra_config_overrides={"sync_load": True},
    )

    # Store a 4-chunk prompt.
    runner.new_request(token_ids=[5] * BLOCK_SIZE * 4)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.run(decoded_tokens=[EOS_TOKEN_ID], expected_stored=(0, 1, 2, 3))
    runner.scheduler.reset_prefix_cache()

    # Admission step: issues the sync load job, which stays in transfer_jobs
    # until the worker's completion ack (not processed yet). The store
    # proposal is declined so this test isolates the preemption-flush path.
    runner.manager.lookup.return_value = LookupResult.HIT
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.new_request(token_ids=[5] * BLOCK_SIZE * 4)
    scheduler_output = runner.scheduler.schedule()
    (req_id,) = scheduler_output.num_scheduled_tokens
    connector_scheduler = runner.connector_scheduler
    req_status = connector_scheduler._req_status[req_id]
    (load_jid,) = req_status.transfer_jobs
    assert not connector_scheduler._jobs[load_jid].is_store

    # Preempt the request while its load job is still unacknowledged.
    preemption_step = SchedulerOutput.make_empty()
    preemption_step.preempted_req_ids = {req_id}
    meta = connector_scheduler.build_connector_meta(preemption_step)
    assert load_jid not in meta.jobs_to_flush
