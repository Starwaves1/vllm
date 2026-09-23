# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A step-skipped hybrid request keeps num_accepted_tokens (V1 runner).

A running hybrid request that the scheduler skips for one step (token budget
used up by other requests' prefill chunks) leaves the persistent batch. When
it comes back, InputBatch.add_request() sets num_accepted_tokens_cpu to 1, but
its GDN/Mamba state still sits in block column num_accepted - 1 of the last
verify step. _update_states keeps the count across the gap (not across
preemption).

CPU only: pinned memory is disabled process-wide and "hybrid + spec decode"
is faked on the runner, so the module patches global state and must run on
its own:

    VLLM_CPU_RUNNER_TEST=1 CUDA_VISIBLE_DEVICES= VLLM_USE_FLASHINFER_SAMPLER=0 \
        pytest tests/v1/worker/test_mamba_zero_draft_state_runner.py

Without VLLM_CPU_RUNNER_TEST=1 the module is skipped.
"""

import os

import pytest

if os.environ.get("VLLM_CPU_RUNNER_TEST") != "1":
    pytest.skip(
        "patches process-wide state: run on its own with VLLM_CPU_RUNNER_TEST=1",
        allow_module_level=True,
    )
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm.platforms import current_platform  # noqa: E402

# No GPU in this process: pinned host memory needs CUDA.
type(current_platform).is_pin_memory_available = classmethod(lambda cls: False)
import vllm.utils.torch_utils as _tu  # noqa: E402

_tu.PIN_MEMORY = False
import vllm.v1.utils as _v1_utils  # noqa: E402

# CpuGpuBuffer binds its pin_memory default at import time.
_cpu_gpu_buffer_init = _v1_utils.CpuGpuBuffer.__init__


def _unpinned_cpu_gpu_buffer_init(self, *size, pin_memory=False, **kwargs):
    _cpu_gpu_buffer_init(self, *size, pin_memory=False, **kwargs)


_v1_utils.CpuGpuBuffer.__init__ = _unpinned_cpu_gpu_buffer_init

import tempfile  # noqa: E402
import types  # noqa: E402


from vllm.config import (  # noqa: E402
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (  # noqa: E402
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.attention import Attention  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.sched.output import (  # noqa: E402
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu_input_batch import InputBatch  # noqa: E402
import vllm.v1.worker.gpu_model_runner as _gmr  # noqa: E402
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402

_gmr.PIN_MEMORY = False

pytestmark = pytest.mark.cpu_test


# CPU-only process: skip the CUDA device-property query.
GPUModelRunner._init_device_properties = lambda self: setattr(self, "num_sms", 1)
BLOCK_SIZE = 16
NUM_BLOCKS = 10


@pytest.fixture
def dist():
    fd, f = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{f}",
            local_rank=0, backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()
    if os.path.exists(f):
        os.unlink(f)


def _runner(hybrid: bool):
    model_config = ModelConfig(model="facebook/opt-125m", dtype="float16", seed=42)
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=BLOCK_SIZE, gpu_memory_utilization=0.9),
        scheduler_config=SchedulerConfig(
            max_num_seqs=10, max_num_batched_tokens=512, max_model_len=512,
            is_encoder_decoder=False, async_scheduling=False,
        ),
        parallel_config=ParallelConfig(),
    )
    with set_current_vllm_config(vllm_config):
        mc = vllm_config.model_config
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(
            mc.get_num_kv_heads(vllm_config.parallel_config), mc.get_head_size(), 0.1
        )
        runner = GPUModelRunner(vllm_config, "cpu")
        spec = FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=mc.get_num_kv_heads(runner.parallel_config),
            head_size=mc.get_head_size(),
            dtype=runner.kv_cache_dtype,
        )
        kv = KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=spec.page_size_bytes * NUM_BLOCKS,
                    layers=["layer.0"],
                    layer_stride=spec.page_size_bytes * NUM_BLOCKS,
                    block_stride=spec.page_size_bytes,
                )
            ],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)],
        )
        runner.kv_cache_config = kv
        runner.input_batch = InputBatch(
            max_num_reqs=runner.max_num_reqs,
            max_model_len=runner.max_model_len,
            max_num_batched_tokens=runner.max_num_tokens,
            device=runner.device,
            vocab_size=mc.get_vocab_size(),
            block_sizes=[BLOCK_SIZE],
            kernel_block_sizes=[BLOCK_SIZE],
            max_num_blocks_per_req=[NUM_BLOCKS],
        )
    if hybrid:
        # Pretend: hybrid model + speculative decoding (no CUDA events on CPU).
        runner.speculative_config = types.SimpleNamespace(use_ngram_gpu=lambda: False)
        runner.model_config = type(
            "HybridMC", (), {"__getattr__": lambda s, k: getattr(mc, k), "is_hybrid": True}
        )()
    return runner


def _out(new=(), cached=(), sched=None, preempted=None, resumed=()):
    new_reqs = [
        NewRequestData(
            req_id=r, prompt_token_ids=[1, 2, 3], mm_features=[],
            sampling_params=SamplingParams(), pooling_params=None,
            block_ids=([0],), num_computed_tokens=0, lora_request=None,
        )
        for r in new
    ]
    cached = list(cached)
    crd = (
        CachedRequestData(
            req_ids=cached, resumed_req_ids=set(resumed),
            new_token_ids=[[] for _ in cached], all_token_ids={},
            new_block_ids=[None if r not in resumed else ([0],) for r in cached],
            num_computed_tokens=[3 for _ in cached],
            num_output_tokens=[1 for _ in cached],
        )
        if cached
        else CachedRequestData.make_empty()
    )
    sched = sched or {r: 3 for r in new} | {r: 1 for r in cached}
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs, scheduled_cached_reqs=crd,
        num_scheduled_tokens=sched, total_num_scheduled_tokens=sum(sched.values()),
        scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
        num_common_prefix_blocks=[], finished_req_ids=set(),
        free_encoder_mm_hashes=[], preempted_req_ids=preempted,
    )


def _acc(runner, req_id):
    return int(runner.input_batch.num_accepted_tokens_cpu[runner.input_batch.req_id_to_index[req_id]])


def _set_acc(runner, req_id, n):
    runner.input_batch.num_accepted_tokens_cpu[runner.input_batch.req_id_to_index[req_id]] = n


def test_skipped_step_keeps_num_accepted(dist):
    r = _runner(hybrid=True)
    r._update_states(_out(new=["a", "b"]))
    _set_acc(r, "a", 2)
    _set_acc(r, "b", 3)  # b's last verify step accepted 2 drafts + bonus
    r._update_states(_out(cached=["a"]))  # b skipped: token budget
    assert "b" not in r.input_batch.req_id_to_index
    r._update_states(_out(cached=["a", "b"]))  # b back
    assert _acc(r, "b") == 3
    assert _acc(r, "a") == 2


def test_preempted_request_starts_fresh(dist):
    r = _runner(hybrid=True)
    r._update_states(_out(new=["a", "b"]))
    _set_acc(r, "b", 3)
    r._update_states(_out(cached=["a"], preempted={"b"}))
    r._update_states(_out(cached=["a", "b"], resumed={"b"}))
    assert _acc(r, "b") == 1


def test_non_hybrid_model_unchanged(dist):
    r = _runner(hybrid=False)
    r._update_states(_out(new=["a", "b"]))
    _set_acc(r, "b", 3)
    r._update_states(_out(cached=["a"]))
    r._update_states(_out(cached=["a", "b"]))
    assert _acc(r, "b") == 1
