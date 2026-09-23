# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Align-mode Mamba block estimate must count the external (connector) range.

``MambaManager.get_num_blocks_to_allocate`` in ``align`` mode returns
``1 + num_speculative_blocks (+1 partial hit)`` for a first prefill. For the
same request the coordinator runs ``add_local_computed_blocks`` and then the
inherited base ``allocate_external_computed_blocks``, which pulls a real block
for the external range, before ``allocate_new_blocks``. The estimate is one
block short, so at a near-full pool ``allocate_new_blocks`` raises
``ValueError: Cannot get N free blocks from the pool`` and kills the engine.

The external draw is exactly one state block whatever the prefix length:
``add_local_computed_blocks`` pads ``(total - 1) // block_size`` null
placeholders first, since Mamba keeps only the last state.
"""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
NUM_SPECULATIVE_BLOCKS = 3
MAX_MODEL_LEN = 8192


def _build_manager(num_gpu_blocks: int) -> KVCacheManager:
    spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=NUM_SPECULATIVE_BLOCKS,
    )
    config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba0"], spec)],
    )
    return KVCacheManager(
        config,
        max_model_len=MAX_MODEL_LEN,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=True,
    )


def _make_request(prompt_len: int, rid: str = "r0") -> Request:
    return Request(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )


def _estimate_and_consume(mgr, rid, *, total_computed, num_new_tokens):
    """Drive the coordinator as allocate_slots does for a first prefill with
    `total_computed` external tokens and no local hit. Returns (estimate,
    blocks actually taken from the pool)."""
    coord = mgr.coordinator
    pool = mgr.block_pool
    num_tokens_main = total_computed + num_new_tokens

    free_before = pool.get_num_free_blocks()
    estimate = coord.get_num_blocks_to_allocate(
        request_id=rid,
        num_tokens=num_tokens_main,
        new_computed_blocks=([],),
        num_encoder_tokens=0,
        total_computed_tokens=total_computed,
        num_local_computed_tokens=0,
        num_tokens_main_model=num_tokens_main,
        apply_admission_cap=False,
    )
    coord.allocate_new_computed_blocks(
        request_id=rid,
        new_computed_blocks=([],),
        num_local_computed_tokens=0,
        num_external_computed_tokens=total_computed,
    )
    coord.allocate_new_blocks(rid, num_tokens_main, num_tokens_main, 0)
    consumed = free_before - pool.get_num_free_blocks()
    return estimate, consumed


def test_allocate_slots_never_raises_with_external_prefix():
    """Free blocks == the old estimate (4): the external draw used to exhaust
    the pool and the final allocate_new_blocks raised. The corrected estimate
    (5) exceeds the free count, so allocate_slots declines instead."""
    mgr = _build_manager(num_gpu_blocks=5)  # 4 free (block 0 is the null block)
    assert mgr.block_pool.get_num_free_blocks() == 4
    result = mgr.allocate_slots(
        _make_request(prompt_len=64),
        num_new_tokens=48,
        num_external_computed_tokens=16,
        has_scheduled_reqs=False,
    )
    assert result is None


def test_estimate_equals_actual_consumption():
    mgr = _build_manager(num_gpu_blocks=100)
    estimate, consumed = _estimate_and_consume(
        mgr, "inv0", total_computed=16, num_new_tokens=48
    )
    assert consumed == 5  # external (1) + state (1) + speculative (3)
    assert estimate == consumed


def test_large_external_prefix_estimate_adds_one_block():
    """A long external prefix adds one state block, not the whole prefix."""
    mgr = _build_manager(num_gpu_blocks=200)
    estimate, consumed = _estimate_and_consume(
        mgr, "big0", total_computed=BLOCK_SIZE * 50, num_new_tokens=48
    )
    assert consumed == 5
    assert estimate == consumed == 5


def test_fixed_estimate_admits_when_pool_has_room():
    mgr = _build_manager(num_gpu_blocks=6)  # 5 free == corrected estimate
    assert mgr.block_pool.get_num_free_blocks() == 5
    result = mgr.allocate_slots(
        _make_request(prompt_len=64),
        num_new_tokens=48,
        num_external_computed_tokens=16,
        delay_cache_blocks=True,
        has_scheduled_reqs=False,
    )
    assert result is not None
    assert mgr.block_pool.get_num_free_blocks() == 0
