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

from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from tests.v1.kv_connector.unit.utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_vllm_config,
)
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


@pytest.fixture(autouse=True)
def _v1_model_runner(monkeypatch):
    # sync_load is V1-only (the offloading scheduler rejects it under V2).
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

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


# ---------------------------------------------------------------------------
# Hybrid layout: one full-attention group (holding an MTP drafter layer, not
# annotated as a drafter group), three align-mode Mamba (GDN) groups, a
# uniform block size and MTP k=3. The worker is simulated: every store copies
# the source block's post-forward content and every load lands before any
# compute reads it. After a step the running Mamba slot (cdiv(computed, block)
# - 1) holds state@computed, so each stored Mamba chunk c must hold
# state@(c + 1) * block and a hit at boundary H must be served state@H.
# ---------------------------------------------------------------------------

HYBRID_BS = 832
HYBRID_K = 3
HYBRID_PROMPT = 4260  # 5 full chunks + 100 tokens
HYBRID_EXTRA = 1740  # turn 2 = turn 1 prompt + more
GDN_GROUPS = (1, 2, 3)


def _mtp_speculative_config():
    spec = MagicMock(name="mtp_speculative_config")
    spec.method = "mtp"
    spec.use_eagle.return_value = True
    spec.use_eagle_block_drop.return_value = True
    spec.max_num_new_slots_for_drafting = 0
    spec.uses_draft_model.return_value = False
    spec.use_dflash.return_value = False
    spec.use_dspark.return_value = False
    spec.num_speculative_tokens_per_batch_size = None
    spec.num_speculative_tokens = HYBRID_K
    return spec


def _hybrid_groups():
    groups = [
        KVCacheGroupSpec(
            ["attn_and_mtp"],
            FullAttentionSpec(
                block_size=HYBRID_BS, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        )
    ]
    for i in range(3):
        groups.append(
            KVCacheGroupSpec(
                [f"gdn{i}"],
                MambaSpec(
                    block_size=HYBRID_BS,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=HYBRID_K,
                ),
            )
        )
    return groups


def _build_hybrid_scheduler(sync_load, threshold, retention_interval):
    vllm_config = create_vllm_config(
        block_size=HYBRID_BS,
        max_num_batched_tokens=8192,
        max_model_len=65536,
        max_num_seqs=8,
        hf_overrides={"max_position_embeddings": 131072},
    )
    vllm_config.cache_config.mamba_cache_mode = "align"
    vllm_config.cache_config.prefix_cache_retention_interval = retention_interval
    vllm_config.scheduler_config.long_prefill_token_threshold = threshold
    vllm_config.scheduler_config.async_scheduling = False
    vllm_config.speculative_config = _mtp_speculative_config()
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config={
            "spec_name": "MockOffloadingSpec",
            "spec_module_path": (
                "tests.v1.kv_connector.unit.offloading_connector.utils"
            ),
            "sync_load": sync_load,
        },
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=400,
        kv_cache_tensors=[],
        kv_cache_groups=_hybrid_groups(),
        # what the GPU-side coordinator reads (get_kv_cache_configs copies it
        # from cache_config in the engine)
        prefix_cache_retention_interval=retention_interval,
    )
    vllm_config.cache_config.num_gpu_blocks = 400
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )


class _HybridSim:
    """Scheduler-side engine loop with a simulated worker (see above)."""

    def __init__(self, scheduler: Scheduler):
        self.s = scheduler
        self.cs = scheduler.connector.connector_scheduler
        self.sync_load = self.cs.config.sync_load
        # offload key -> (group, chunk, content); content is state@tokens for
        # Mamba groups and "kv" for attention.
        self.stored: dict = {}
        self.state_at: dict = {g: {} for g in GDN_GROUPS}  # block -> state@
        self.external: dict[str, int] = {}
        self.loads: list[tuple[str, list[tuple[int, int]]]] = []
        self.parked: set[str] = set()
        manager = self.cs.manager
        manager.lookup.side_effect = lambda key, ctx: (
            LookupResult.HIT if key in self.stored else LookupResult.MISS
        )
        manager.prepare_store.side_effect = lambda keys, ctx: generate_store_output(
            list(keys)
        )
        connector = scheduler.connector
        orig = connector.update_state_after_alloc

        def record(request, blocks, num_external_tokens):
            if num_external_tokens:
                self.external[request.request_id] = num_external_tokens
            return orig(request, blocks, num_external_tokens)

        connector.update_state_after_alloc = record
        init_none_hash(sha256)
        self.hasher = get_request_block_hasher(HYBRID_BS, sha256)
        self.next_id = 0

    def add(self, token_ids: list[int], max_tokens: int = 3) -> Request:
        self.next_id += 1
        params = SamplingParams(max_tokens=max_tokens)
        params.update_from_generation_config({}, EOS_TOKEN_ID)
        request = Request(
            request_id=f"r{self.next_id}",
            prompt_token_ids=token_ids,
            sampling_params=params,
            pooling_params=None,
            block_hasher=self.hasher,
        )
        self.s.add_request(request)
        return request

    def _key_index(self) -> dict:
        index = {}
        for req_status in self.cs._req_status.values():
            for g, group_state in enumerate(req_status.group_states):
                for c, key in enumerate(group_state.offload_keys):
                    index[key] = (g, c)
        return index

    def step(self):
        s = self.s
        so = s.schedule()
        meta = so.kv_connector_metadata
        index = self._key_index()
        for req in s.requests.values():
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.parked.add(req.request_id)
        for job in meta.load_jobs.values():
            keys = job.src_spec.offload_keys
            block_ids = [int(b) for b in job.dst_spec.block_ids]
            assert len(keys) == len(block_ids)
            loaded = []
            for key, block_id in zip(keys, block_ids):
                g, c = index[key]
                loaded.append((g, c))
                if g in GDN_GROUPS:
                    self.state_at[g][block_id] = self.stored[key][2]
            self.loads.append((job.req_id, loaded))
        # Forward: the running Mamba slot advances to state@computed.
        managers = s.kv_cache_manager.coordinator.single_type_managers
        for req_id in so.num_scheduled_tokens:
            computed = s.requests[req_id].num_computed_tokens
            for g in GDN_GROUPS:
                blocks = managers[g].req_to_blocks[req_id]
                block = blocks[cdiv(computed, HYBRID_BS) - 1]
                self.state_at[g][block.block_id] = computed
        # Stores copy the source block's post-forward content.
        for job in meta.store_jobs.values():
            keys = job.dst_spec.offload_keys
            block_ids = [int(b) for b in job.src_spec.block_ids]
            assert len(keys) == len(block_ids)
            for key, block_id in zip(keys, block_ids):
                g, c = index[key]
                if g in GDN_GROUPS:
                    content = self.state_at[g].get(block_id)
                    assert content == (c + 1) * HYBRID_BS, (
                        f"Mamba group {g} chunk {c} stored state@{content}"
                    )
                else:
                    content = "kv"
                self.stored[key] = (g, c, content)
        jobs = {j: 1 for j in list(meta.load_jobs) + list(meta.store_jobs)}
        out = create_model_runner_output(
            reqs=s.running,
            token_id=7,
            kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs=jobs),
        )
        if not self.sync_load and meta.load_jobs:
            out.kv_connector_output.finished_recving = {
                job.req_id for job in meta.load_jobs.values()
            }
        for i, req_id in enumerate(out.req_ids):
            request = s.requests[req_id]
            if request.num_computed_tokens < request.num_tokens:
                out.sampled_token_ids[i] = []  # mid-prefill: nothing sampled
        s.update_from_output(so, out)
        return so

    def run_to_completion(self, cap: int = 200):
        for _ in range(cap):
            if not self.s.requests and not self.cs._jobs and not self.s.finished_req_ids:
                return
            self.step()
        raise AssertionError("simulation did not drain")

    def admit(self, request: Request, cap: int = 4) -> int:
        """Step until `request` computes its first tokens; return the hit
        boundary (local + external tokens) it started from."""
        for _ in range(cap):
            before = request.num_computed_tokens
            so = self.step()
            if request.request_id in so.num_scheduled_tokens:
                n = so.num_scheduled_tokens[request.request_id]
                start = request.num_computed_tokens - n
                assert start >= before
                return start
        raise AssertionError("request was never scheduled")

    def check_loads(self, request: Request, boundary: int):
        """Every Mamba chunk loaded for `request` ends at the hit boundary and
        holds exactly state@boundary."""
        for req_id, loaded in self.loads:
            if req_id != request.request_id:
                continue
            for g, c in loaded:
                if g in GDN_GROUPS:
                    assert (c + 1) * HYBRID_BS == boundary, (g, c, boundary)
            for g, c, content in self.stored.values():
                if g in GDN_GROUPS and (g, c) in loaded:
                    assert content == boundary, (g, c, content, boundary)


def _turn_two(sync_load, threshold, retention_interval, reset_gpu):
    sim = _HybridSim(_build_hybrid_scheduler(sync_load, threshold, retention_interval))
    prompt_a = [1000 + i for i in range(HYBRID_PROMPT)]
    sim.add(prompt_a)
    # The scheduler drops long_prefill_token_threshold for a lone request; a
    # short companion (under one block, so it stores nothing) keeps turn 1
    # chunked at the threshold, as under concurrent load in production.
    sim.add([9000 + i for i in range(100)], max_tokens=40)
    sim.run_to_completion()
    gdn_stored = sorted((g, c) for (g, c, _) in sim.stored.values() if g in GDN_GROUPS)
    if reset_gpu:
        assert sim.s.reset_prefix_cache()
    request_b = sim.add(
        prompt_a + [5000 + i for i in range(HYBRID_EXTRA)], max_tokens=1
    )
    boundary = sim.admit(request_b)
    external = sim.external.get(request_b.request_id, 0)
    sim.check_loads(request_b, boundary)
    if sync_load:
        assert not sim.parked
    return sim, gdn_stored, boundary, external


@pytest.mark.parametrize("sync_load", [True, False], ids=["sync", "async"])
@pytest.mark.parametrize("retention_interval", [None, 0], ids=["dense", "ri0"])
def test_hybrid_mtp_offloaded_prefix_served(sync_load, retention_interval):
    """Turn 1 prefills in threshold-sized chunks, so every block boundary
    materializes a Mamba state. With the GPU cache reset, turn 2 is served all
    five offloaded chunks (loaded synchronously without parking under
    sync_load) together with the recurrent state at the hit boundary."""
    _, gdn_stored, boundary, external = _turn_two(
        sync_load, 512, retention_interval, reset_gpu=True
    )
    if retention_interval is None:
        assert gdn_stored == [(g, c) for g in GDN_GROUPS for c in range(5)]
        assert (boundary, external) == (5 * HYBRID_BS, 5 * HYBRID_BS)
    else:
        # Retention 0 keeps only the replay-boundary state: the prompt's last
        # full block minus the MTP (EAGLE) block drop, i.e. the end of chunk 3.
        assert gdn_stored == [(g, 3) for g in GDN_GROUPS]
        assert (boundary, external) == (4 * HYBRID_BS, 4 * HYBRID_BS)


@pytest.mark.parametrize("sync_load", [True, False], ids=["sync", "async"])
@pytest.mark.parametrize("retention_interval", [None, 0], ids=["dense", "ri0"])
def test_hybrid_mtp_offload_extends_gpu_hit(sync_load, retention_interval):
    """Turn 1 still resident on the GPU: the GPU hit stops one block short
    (EAGLE block drop) and the offload tier supplies the chunk past it, a
    local prefix plus a loaded suffix."""
    _, _, boundary, external = _turn_two(
        sync_load, 512, retention_interval, reset_gpu=False
    )
    if retention_interval is None:
        assert (boundary, external) == (5 * HYBRID_BS, HYBRID_BS)
    else:
        # No state past the replay boundary exists anywhere, so the GPU hit
        # at the end of chunk 3 is the whole hit and nothing is loaded.
        assert (boundary, external) == (4 * HYBRID_BS, 0)


def test_sync_load_rejected_on_v2_runner(monkeypatch):
    """The V2 runner reads a sync load's Mamba destination before the load
    lands, so the offloading scheduler refuses sync_load there."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with pytest.raises(ValueError, match="V1 model runner"):
        _build_hybrid_scheduler(True, 512, None)
