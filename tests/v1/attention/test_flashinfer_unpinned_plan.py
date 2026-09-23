# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer plan staging buffers are pageable.

plan() fills the wrapper's pinned host staging buffer and issues an unfenced
cudaMemcpyAsync from it. The MTP drafter re-plans the same wrappers once per
draft position in one step; without a host sync in between, a later plan can
overwrite bytes an earlier queued copy has not read yet (Xid 31 illegal
memory access). A pageable staging buffer is consumed before cudaMemcpyAsync
returns. No GPU is used.
"""

import inspect
import json
import os
import types

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")
import flashinfer.decode as fi_decode  # noqa: E402
import flashinfer.prefill as fi_prefill  # noqa: E402

from vllm.v1.attention.backends import flashinfer as fib  # noqa: E402

pytestmark = pytest.mark.cpu_test

ENV = "VLLM_FLASHINFER_UNPINNED_PLAN_BUFFERS"
ATTR = "_pin_memory_int_workspace_buffer"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def _fake_wrapper(n=64):
    w = types.SimpleNamespace()
    setattr(w, ATTR, torch.empty(n, dtype=torch.uint8))
    return w


# --------------------------------------------------------------------------
# env gate
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        ("1", True),
        ("true", True),
        ("0", False),
        ("false", False),
        ("OFF", False),
        ("no", False),
    ],
)
def test_env_gate(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert fib.unpinned_plan_buffers_enabled() is expected


# --------------------------------------------------------------------------
# unpin helper
# --------------------------------------------------------------------------
def test_unpin_replaces_staging_buffer():
    w = _fake_wrapper(128)
    before = getattr(w, ATTR)
    assert fib.unpin_flashinfer_plan_buffers(w) == 1
    after = getattr(w, ATTR)
    assert after is not before
    assert after.shape == before.shape and after.dtype == before.dtype
    assert after.device.type == "cpu" and not after.is_pinned()


def test_unpin_walks_nested_wrappers():
    # BatchDCPPrefillWrapper layout
    dcp = types.SimpleNamespace(_context=_fake_wrapper(), _new_tokens=_fake_wrapper())
    assert fib.unpin_flashinfer_plan_buffers(dcp) == 2
    # MultiLevelCascadeAttentionWrapper layout (own buffer + list of children)
    cascade = _fake_wrapper()
    cascade._batch_prefill_wrappers = [_fake_wrapper(), _fake_wrapper()]
    assert fib.unpin_flashinfer_plan_buffers(cascade) == 3


def test_unpin_disabled_is_noop(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    w = _fake_wrapper()
    before = getattr(w, ATTR)
    assert fib.unpin_flashinfer_plan_buffers(w) == 0
    assert getattr(w, ATTR) is before


def test_unpin_tolerates_missing_attr_and_none():
    assert fib.unpin_flashinfer_plan_buffers(None) == 0
    assert fib.unpin_flashinfer_plan_buffers(types.SimpleNamespace()) == 0
    # a device-less / non-tensor value is left alone
    w = types.SimpleNamespace(**{ATTR: "not-a-tensor"})
    assert fib.unpin_flashinfer_plan_buffers(w) == 0


# --------------------------------------------------------------------------
# The installed flashinfer still has the hazard the patch targets. If a
# flashinfer upgrade renames the buffer, the helper would silently no-op; this
# fails first.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls",
    [
        fi_prefill.BatchPrefillWithPagedKVCacheWrapper,
        fi_prefill.BatchPrefillWithRaggedKVCacheWrapper,
        fi_decode.BatchDecodeWithPagedKVCacheWrapper,
    ],
)
def test_flashinfer_wrappers_stage_plan_in_pinned_attr(cls):
    init_src = inspect.getsource(cls.__init__)
    # plan() may delegate (decode: plan -> _plan_impl); check the whole class
    cls_src = inspect.getsource(cls)
    assert f"self.{ATTR} = torch.empty(" in init_src
    assert "pin_memory=True" in init_src
    assert f"self.{ATTR}," in cls_src


def test_flashinfer_plan_copy_is_unfenced():
    root = os.path.dirname(flashinfer.__file__)
    path = os.path.join(root, "data/include/flashinfer/attention/scheduler.cuh")
    src = open(path).read()
    for fn in ("inline cudaError_t DecodePlanImpl(", "inline cudaError_t PrefillPlanImpl("):
        start = src.index(fn)
        body = src[start : src.index("return cudaSuccess;", start)]
        assert "cudaMemcpyAsync(int_buffer, page_locked_int_buffer" in body
        assert "cudaStreamSynchronize" not in body
        assert "cudaEventSynchronize" not in body


# --------------------------------------------------------------------------
# Builder integration: real FlashInferMetadataBuilder.__init__ on CPU.
# --------------------------------------------------------------------------
def _tiny_vllm_config(tmp_path):
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_hidden_layers": 1,
        "vocab_size": 1024,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        seed=0,
        max_model_len=1024,
    )
    cache_config = CacheConfig(block_size=16, cache_dtype="auto")
    cache_config.kv_cache_layout = "NHD"
    cache_config.num_gpu_blocks = 100
    cache_config.num_cpu_blocks = 0
    scheduler_config = SchedulerConfig(
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        max_model_len=1024,
        is_encoder_decoder=False,
    )
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        scheduler_config=scheduler_config,
        device_config=DeviceConfig(),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
    )


def _make_builder(tmp_path, monkeypatch):
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.utils import PerLayerParameters
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    # The live engine runs the V1 GPUModelRunner (hybrid model); a plain llama
    # config would default to V2, which never pins these buffers anyway.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    vllm_config = _tiny_vllm_config(tmp_path)
    assert not vllm_config.use_v2_model_runner
    monkeypatch.setattr(fib, "can_use_trtllm_attention", lambda *a, **k: False)
    monkeypatch.setattr(
        fib,
        "get_per_layer_parameters",
        lambda *a, **k: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.125, has_sinks=False
            )
        },
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
    )
    with set_current_vllm_config(vllm_config):
        return fib.FlashInferMetadataBuilder(
            spec, ["layer.0"], vllm_config, torch.device("cpu")
        )


def test_builder_host_buffers_are_pageable(tmp_path, monkeypatch):
    """The builder's own reused host buffers are pageable (#54299), so they
    can be built on this CUDA-less box even with PIN_MEMORY forced on."""
    monkeypatch.setattr(fib, "PIN_MEMORY", True)
    builder = _make_builder(tmp_path, monkeypatch)
    assert not builder.paged_kv_indptr.cpu.is_pinned()
    assert not builder.paged_kv_last_page_len.cpu.is_pinned()


class _FakeFIWrapper:
    created: list = []

    def __init__(self, *args, **kwargs):
        setattr(self, ATTR, torch.empty(32, dtype=torch.uint8))
        self.original = getattr(self, ATTR)
        type(self).created.append(self)


@pytest.mark.parametrize("gate", ["1", "0"])
def test_builder_wrappers_get_unpinned(tmp_path, monkeypatch, gate):
    monkeypatch.setenv(ENV, "1")  # builder itself must construct on CPU
    builder = _make_builder(tmp_path, monkeypatch)
    monkeypatch.setenv(ENV, gate)
    _FakeFIWrapper.created = []
    monkeypatch.setattr(fib, "BatchPrefillWithPagedKVCacheWrapper", _FakeFIWrapper)
    monkeypatch.setattr(fib, "BatchDecodeWithPagedKVCacheWrapper", _FakeFIWrapper)
    monkeypatch.setattr(fib, "MultiLevelCascadeAttentionWrapper", _FakeFIWrapper)
    monkeypatch.setattr(fib, "get_flashinfer_layout_string", lambda layout: "NHD")
    builder._workspace_buffer = torch.zeros(16, dtype=torch.uint8)

    wrappers = [
        builder._get_prefill_wrapper(causal=True),
        builder._get_prefill_wrapper(causal=False),
        builder._get_decode_wrapper(4, use_cudagraph=False),
        builder._get_cascade_wrapper(),
    ]
    assert len(_FakeFIWrapper.created) == 4
    for w in wrappers:
        replaced = getattr(w, ATTR) is not w.original
        assert replaced is (gate == "1")
    # cached wrappers are returned as-is on later calls (no re-allocation)
    again = builder._get_decode_wrapper(4, use_cudagraph=False)
    assert again is wrappers[2]
    assert len(_FakeFIWrapper.created) == 4
