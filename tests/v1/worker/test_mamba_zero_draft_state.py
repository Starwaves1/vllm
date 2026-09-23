# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stale Mamba state after a zero-draft step under MTP (V1 runner, align mode).

After a verify step that accepted drafts, a hybrid (GDN/Mamba) request keeps
its running state in block column ``num_accepted - 1`` (temporal/SSM state)
and at token offset ``num_accepted - 1`` of its conv window. When the next
step schedules no drafts for it (the grammar truncated every draft, or the
token budget left no room), the runner classifies the row as non-spec and the
Mamba kernels read column 0 / offset 0 instead: a state missing the last
``num_accepted - 1`` tokens. ``preprocess_mamba`` shifts the state in-block
(src == dst, bias = num_accepted - 1) for such rows and resets num_accepted to
1; the V1 fused pre-copy kernel runs a src == dst copy when the bias is
non-zero.

The Triton kernels run on the CPU through the Triton interpreter, which must be
enabled before the kernels are imported, so run this file on its own:

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= pytest tests/v1/worker/test_mamba_zero_draft_state.py

Without TRITON_INTERPRET=1 the module is skipped.
"""

import os

import pytest

if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    pytest.skip(
        "needs TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= (run this file on its own)",
        allow_module_level=True,
    )

import types  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vllm.v1.worker.mamba_utils as _live_mu  # noqa: E402
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.attention.backends.registry import (  # noqa: E402
    MambaAttentionBackendEnum,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


mu = _live_mu

pytestmark = pytest.mark.cpu_test

NUM_BLOCKS = 12
NUM_SPEC = 3
CONV_LEN = 3 + NUM_SPEC  # (width - 1) + num_spec: the spec-decode conv window
CONV_DIM = 8
TEMPORAL_DIM = 16
ROW_COLS = 8
BLOCK = 16
DEV = torch.device("cpu")


class _Buf:
    """Minimal CpuGpuBuffer stand-in (both halves on CPU)."""

    def __init__(self, n: int, dtype: torch.dtype):
        self.cpu = torch.zeros(n, dtype=dtype)
        self.gpu = torch.zeros(n, dtype=dtype)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu)
        return self.gpu[:n].copy_(self.cpu[:n])


class _Harness:
    """One align-mode mamba layer (conv + temporal state), one request row."""

    def __init__(self, ds_layout: bool, fused: bool = True):
        torch.manual_seed(0)
        self.ds_layout = ds_layout
        self.fused = fused
        self.bt = torch.zeros(1, ROW_COLS, dtype=torch.int32)
        self.bt[0] = torch.arange(1, ROW_COLS + 1, dtype=torch.int32)
        conv_shape = (CONV_DIM, CONV_LEN) if ds_layout else (CONV_LEN, CONV_DIM)
        spec = MambaSpec(
            block_size=BLOCK,
            shapes=(conv_shape, (TEMPORAL_DIM,)),
            dtypes=(torch.float32, torch.float32),
            mamba_cache_mode="align",
            num_speculative_blocks=NUM_SPEC,
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        )
        self.kv_cfg = KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
        )
        self.conv = torch.randn(NUM_BLOCKS, *conv_shape)
        self.temporal = torch.randn(NUM_BLOCKS, TEMPORAL_DIM)
        self.copy_funcs = {
            MambaAttentionBackendEnum.GDN_ATTN: (
                get_conv_copy_spec,
                get_temporal_copy_spec,
            )
        }
        self.fwd = {"l0": types.SimpleNamespace(kv_cache=[self.conv, self.temporal])}
        self.copy_bufs = mu.MambaCopyBuffers.create(
            4, self.kv_cfg, self.copy_funcs, make_buffer=lambda n, dtype: _Buf(n, dtype)
        )
        self.ctx = mu.MambaSpecDecodeGPUContext.create(
            max_num_reqs=4,
            kv_cache_config=self.kv_cfg,
            copy_funcs=self.copy_funcs,
            device=DEV,
            make_buffer=lambda n, dtype: _Buf(n, dtype),
        )
        self.ctx.initialize_from_forward_context(
            self.kv_cfg, self.fwd, self.copy_funcs, [self.bt]
        )

    def snapshot(self):
        return self.conv.clone(), self.temporal.clone()

    def conv_window(self, conv, col: int, offset: int) -> torch.Tensor:
        """The (width - 1)-token window a kernel reads at ``offset``."""
        blk = int(self.bt[0, col])
        if self.ds_layout:
            return conv[blk, :, offset : offset + 3]
        return conv[blk, offset : offset + 3]

    def preprocess(
        self,
        *,
        num_computed: int,
        num_scheduled: int,
        drafts: list[int] | None,
        num_accepted: int,
        prev_state_idx: int | None,
    ):
        sched = types.SimpleNamespace(
            num_scheduled_tokens={"r0": num_scheduled},
            scheduled_spec_decode_tokens={"r0": drafts} if drafts else {},
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_cached_reqs=types.SimpleNamespace(resumed_req_ids=set()),
        )
        batch = types.SimpleNamespace(
            req_ids=["r0"],
            num_accepted_tokens_cpu=np.array([num_accepted], dtype=np.int32),
            block_table={0: types.SimpleNamespace(get_device_tensor=lambda n: self.bt)},
        )
        reqs = {
            "r0": types.SimpleNamespace(
                num_computed_tokens=num_computed,
                block_ids=[list(self.bt[0].tolist())],
            )
        }
        state_idx = {} if prev_state_idx is None else {"r0": prev_state_idx}
        # The scalar path's batch copy needs a GPU; record its decisions.
        calls = []
        orig = mu.do_mamba_copy_block
        if not self.fused:
            orig_collect = mu.collect_mamba_copy_meta

            def rec(copy_bufs, kv, funcs, gids, src, dst, bias, req, fwd):
                calls.append((src, dst, bias))
                return orig_collect(copy_bufs, kv, funcs, gids, src, dst, bias, req, fwd)

            mu.collect_mamba_copy_meta = rec
            mu.do_mamba_copy_block = lambda bufs: None
        try:
            mu.preprocess_mamba(
                sched,
                self.kv_cfg,
                types.SimpleNamespace(enable_prefix_caching=True),
                state_idx,
                batch,
                reqs,
                self.fwd,
                self.copy_funcs,
                self.copy_bufs,
                align_ctx=self.ctx if self.fused else None,
            )
        finally:
            if not self.fused:
                mu.collect_mamba_copy_meta = orig_collect
                mu.do_mamba_copy_block = orig
        return state_idx["r0"], int(batch.num_accepted_tokens_cpu[0]), calls


@pytest.fixture(autouse=True)
def _interpretable_cdiv(monkeypatch):
    """The Triton interpreter cannot run tl.cdiv (itself @triton.jit) from a
    nested @triton.jit helper such as _memcpy_u64_tiled; use plain integer
    arithmetic, which the interpreter evaluates the same way."""
    import triton.language as tl

    monkeypatch.setattr(tl, "cdiv", lambda x, div: (x + div - 1) // div)


@pytest.fixture(params=[False, True], ids=["SD", "DS"])
def ds_layout(request, monkeypatch):
    monkeypatch.setattr(mu, "is_conv_state_dim_first", lambda: request.param)
    return request.param


# The step before: 16 tokens computed, 3 drafts scheduled (4 tokens), all three
# accepted plus the bonus -> num_accepted = 4, num_computed = 20, running block
# column cdiv(20, 16) - 1 = 1. No block boundary was reached, so postprocess
# left the state in-block: SSM in column 1 + 3, conv window at offset 3.
PREV_COL = 1
NUM_ACCEPTED = 4


def test_zero_draft_step_reads_the_accepted_state(ds_layout):
    h = _Harness(ds_layout)
    conv0, temp0 = h.snapshot()
    want_ssm = temp0[int(h.bt[0, PREV_COL + NUM_ACCEPTED - 1])].clone()
    want_conv = h.conv_window(conv0, PREV_COL, NUM_ACCEPTED - 1).clone()

    col, acc, _ = h.preprocess(
        num_computed=20, num_scheduled=1, drafts=None,
        num_accepted=NUM_ACCEPTED, prev_state_idx=PREV_COL,
    )
    assert col == PREV_COL
    # Non-spec kernels read column 0 of the running block and conv offset 0.
    torch.testing.assert_close(h.temporal[int(h.bt[0, col])], want_ssm, rtol=0, atol=0)
    torch.testing.assert_close(h.conv_window(h.conv, col, 0), want_conv, rtol=0, atol=0)
    assert acc == 1


@pytest.mark.parametrize("num_accepted", [2, 3])
def test_zero_draft_partial_accept(ds_layout, num_accepted):
    h = _Harness(ds_layout)
    conv0, temp0 = h.snapshot()
    want_ssm = temp0[int(h.bt[0, PREV_COL + num_accepted - 1])].clone()
    want_conv = h.conv_window(conv0, PREV_COL, num_accepted - 1).clone()
    col, acc, _ = h.preprocess(
        num_computed=18, num_scheduled=1, drafts=None,
        num_accepted=num_accepted, prev_state_idx=PREV_COL,
    )
    torch.testing.assert_close(h.temporal[int(h.bt[0, col])], want_ssm, rtol=0, atol=0)
    torch.testing.assert_close(h.conv_window(h.conv, col, 0), want_conv, rtol=0, atol=0)
    assert acc == 1


def test_spec_step_is_untouched(ds_layout):
    """With drafts scheduled the spec kernels read via num_accepted: no copy."""
    h = _Harness(ds_layout)
    before = h.snapshot()
    col, acc, _ = h.preprocess(
        num_computed=20, num_scheduled=4, drafts=[11, 12, 13],
        num_accepted=NUM_ACCEPTED, prev_state_idx=PREV_COL,
    )
    assert col == PREV_COL
    torch.testing.assert_close(h.conv, before[0], rtol=0, atol=0)
    torch.testing.assert_close(h.temporal, before[1], rtol=0, atol=0)
    assert acc == NUM_ACCEPTED


def test_spec_step_with_truncated_drafts_is_untouched(ds_layout):
    """One grammar-valid draft left: still a spec row, still no copy."""
    h = _Harness(ds_layout)
    before = h.snapshot()
    col, acc, _ = h.preprocess(
        num_computed=20, num_scheduled=2, drafts=[11],
        num_accepted=NUM_ACCEPTED, prev_state_idx=PREV_COL,
    )
    torch.testing.assert_close(h.conv, before[0], rtol=0, atol=0)
    torch.testing.assert_close(h.temporal, before[1], rtol=0, atol=0)
    assert acc == NUM_ACCEPTED


def test_zero_draft_after_single_accept_is_noop(ds_layout):
    h = _Harness(ds_layout)
    before = h.snapshot()
    _, acc, _ = h.preprocess(
        num_computed=17, num_scheduled=1, drafts=None,
        num_accepted=1, prev_state_idx=PREV_COL,
    )
    torch.testing.assert_close(h.conv, before[0], rtol=0, atol=0)
    torch.testing.assert_close(h.temporal, before[1], rtol=0, atol=0)
    assert acc == 1


def test_new_request_is_noop(ds_layout):
    h = _Harness(ds_layout)
    before = h.snapshot()
    _, acc, _ = h.preprocess(
        num_computed=0, num_scheduled=10, drafts=None,
        num_accepted=1, prev_state_idx=None,
    )
    torch.testing.assert_close(h.conv, before[0], rtol=0, atol=0)
    torch.testing.assert_close(h.temporal, before[1], rtol=0, atol=0)
    assert acc == 1


def test_block_crossing_unchanged(ds_layout):
    """Zero-draft step that also crosses into the next block: the pre-existing
    cross-block copy (src = previous column) runs, as before the patch."""
    h = _Harness(ds_layout)
    conv0, temp0 = h.snapshot()
    # num_computed 32 + 1 scheduled -> cdiv(33, 16) - 1 = 2 != PREV_COL.
    want_ssm = temp0[int(h.bt[0, PREV_COL + 2])].clone()
    want_conv = h.conv_window(conv0, PREV_COL, 2).clone()
    col, acc, _ = h.preprocess(
        num_computed=32, num_scheduled=1, drafts=None,
        num_accepted=3, prev_state_idx=PREV_COL,
    )
    assert col == 2
    torch.testing.assert_close(h.temporal[int(h.bt[0, 2])], want_ssm, rtol=0, atol=0)
    torch.testing.assert_close(h.conv_window(h.conv, 2, 0), want_conv, rtol=0, atol=0)
    assert acc == 1


def test_scalar_path_same_decision():
    """align_ctx=None (scalar batch-memcpy path): same copy decision."""
    h = _Harness(ds_layout=False, fused=False)
    col, acc, calls = h.preprocess(
        num_computed=20, num_scheduled=1, drafts=None,
        num_accepted=NUM_ACCEPTED, prev_state_idx=PREV_COL,
    )
    assert calls == [(PREV_COL, PREV_COL, NUM_ACCEPTED - 1)]
    assert acc == 1
    h2 = _Harness(ds_layout=False, fused=False)
    _, acc2, calls2 = h2.preprocess(
        num_computed=20, num_scheduled=4, drafts=[1, 2, 3],
        num_accepted=NUM_ACCEPTED, prev_state_idx=PREV_COL,
    )
    assert calls2 == [] and acc2 == NUM_ACCEPTED


def test_v2_precopy_same_column_still_skipped(ds_layout):
    """V2 (idx_mapping) stages src_col = state_idx with the accepted bias for
    every row; an in-block row must stay a no-op there."""
    h = _Harness(ds_layout)
    before = h.snapshot()

    def t(v):
        return torch.tensor(v, dtype=torch.int32)

    h.ctx.run_fused_precopy(
        num_reqs=1,
        state_idx_gpu=t([PREV_COL]),
        src_col_gpu=t([PREV_COL]),
        token_bias_gpu=t([2]),
        idx_mapping=t([0]),
    )
    torch.testing.assert_close(h.conv, before[0], rtol=0, atol=0)
    torch.testing.assert_close(h.temporal, before[1], rtol=0, atol=0)


def test_v1_precopy_same_column_zero_bias_is_noop(ds_layout):
    h = _Harness(ds_layout)
    before = h.snapshot()

    def t(v):
        return torch.tensor(v, dtype=torch.int32)

    h.ctx.run_fused_precopy(
        num_reqs=1,
        state_idx_gpu=t([PREV_COL]),
        src_col_gpu=t([PREV_COL]),
        token_bias_gpu=t([0]),
        idx_mapping=None,
    )
    torch.testing.assert_close(h.conv, before[0], rtol=0, atol=0)
    torch.testing.assert_close(h.temporal, before[1], rtol=0, atol=0)


def test_runner_resyncs_num_accepted_after_preprocess():
    """gpu_model_runner copies num_accepted_tokens_cpu to the GPU after
    preprocess_mamba, so the reset reaches the kernels."""
    import vllm.v1.worker.gpu_model_runner as gmr

    with open(gmr.__file__) as f:
        src = f.read()
    tail = src.split("mamba_utils.preprocess_mamba(", 1)[1][:1500]
    assert "self.num_accepted_tokens.np[:num_reqs] = (" in tail
    assert "self.num_accepted_tokens.copy_to_gpu(num_reqs)" in tail
