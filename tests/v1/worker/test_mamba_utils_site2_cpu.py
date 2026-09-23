# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the mamba_utils.py ("site 2") hunks of #50021.

``_copy_mamba_state_block`` (shared by ``postprocess_mamba_fused_kernel`` and
``precopy_mamba_align_fused_kernel``) masks its four block-table loads to the
request row and returns before the state address math when the column is
outside ``[0, block_table_stride_req)`` or the loaded id is ``<= 0`` (the -1
sentinel or NULL_BLOCK_ID).

The Triton kernels run on the CPU through the Triton interpreter. Out-of-row
cases keep every stray read inside the allocated block-table storage, so on an
unpatched build they fail as assertion errors instead of crashing. The
interpreter has to be selected before Triton is imported, so run this file on
its own:

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= \
        pytest tests/v1/worker/test_mamba_utils_site2_cpu.py

Without TRITON_INTERPRET=1 the module is skipped.
"""

import os
import types

import pytest

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip(
        "needs TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= (run this file on its own)",
        allow_module_level=True,
    )

import torch  # noqa: E402

import vllm.v1.worker.mamba_utils as mu  # noqa: E402
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

NUM_BLOCKS = 8
CONV_WIDTH = 4
CONV_DIM = 32
TEMPORAL_DIM = 64
ROW_COLS = 4  # block_table_stride_req seen by the kernel
DTYPE = torch.float32
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
    """One mamba layer (conv + temporal, or temporal-only) bound to a block
    table whose kernel view is a row-slice of a larger storage tensor, so
    out-of-row reads land on real (distinguishable) block ids."""

    def __init__(
        self,
        *,
        ds_layout: bool,
        view_rows: int = 1,
        block_size: int = 16,
        temporal_only: bool = False,
    ):
        torch.manual_seed(0)
        self.ds_layout = ds_layout
        self.temporal_only = temporal_only
        # Storage has one extra row past the kernel's view.
        self.bt_storage = torch.zeros(view_rows + 1, ROW_COLS, dtype=torch.int32)
        self.bt_view = self.bt_storage[:view_rows]
        conv_shape = (CONV_DIM, CONV_WIDTH) if ds_layout else (CONV_WIDTH, CONV_DIM)
        if temporal_only:
            shapes = ((TEMPORAL_DIM,),)
            dtypes = (DTYPE,)
            self.copy_funcs = (get_temporal_copy_spec,)
        else:
            shapes = (conv_shape, (TEMPORAL_DIM,))
            dtypes = (DTYPE, DTYPE)
            self.copy_funcs = (get_conv_copy_spec, get_temporal_copy_spec)
        spec = MambaSpec(
            block_size=block_size,
            shapes=shapes,
            dtypes=dtypes,
            mamba_cache_mode="align",
        )
        self.kv_cfg = KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
        )
        self.conv = None if temporal_only else torch.randn(NUM_BLOCKS, *conv_shape)
        self.temporal = torch.randn(NUM_BLOCKS, TEMPORAL_DIM, dtype=DTYPE)
        states = [self.temporal] if temporal_only else [self.conv, self.temporal]
        fwd = {"l0": types.SimpleNamespace(kv_cache=states)}
        copy_funcs_by_type = {spec.mamba_type: self.copy_funcs}
        self.ctx = mu.MambaSpecDecodeGPUContext.create(
            max_num_reqs=4,
            kv_cache_config=self.kv_cfg,
            copy_funcs=copy_funcs_by_type,
            device=DEV,
            make_buffer=lambda n, dtype: _Buf(n, dtype),
        )
        self.ctx.initialize_from_forward_context(
            self.kv_cfg, fwd, copy_funcs_by_type, [self.bt_view]
        )
        assert self.ctx.block_table_stride_req == ROW_COLS

    def snapshot(self):
        conv = None if self.conv is None else self.conv.clone()
        return conv, self.temporal.clone()

    def precopy(self, src_cols, dst_cols, biases):
        n = len(src_cols)

        def t(v):
            return torch.tensor(v, dtype=torch.int32)

        self.ctx.run_fused_precopy(
            num_reqs=n,
            state_idx_gpu=t(dst_cols),
            src_col_gpu=t(src_cols),
            token_bias_gpu=t(biases),
            idx_mapping=None,
        )

    def postprocess(self, *, accepted, state_idx, scheduled, computed, draft):
        def t(v):
            return torch.tensor(v, dtype=torch.int32)

        self.ctx.run_fused_postprocess(
            num_reqs=len(accepted),
            num_accepted_tokens_gpu=t(accepted),
            mamba_state_idx_gpu=t(state_idx),
            num_scheduled_tokens_gpu=t(scheduled),
            num_computed_tokens_gpu=t(computed),
            num_draft_tokens_gpu=t(draft),
        )
        return self.ctx.num_accepted_tokens_out[: len(accepted)].clone()

    def reference_copy(self, conv_before, temporal_before, row, src, dst, bias):
        """Expected states after one in-range copy (pre-patch semantics)."""
        bt = self.bt_storage[row].tolist()
        conv = None if conv_before is None else conv_before.clone()
        temporal = temporal_before.clone()
        if conv is not None:
            s, d = bt[src], bt[dst]
            keep = CONV_WIDTH - bias
            if self.ds_layout:
                conv[d, :, :keep] = conv_before[s, :, bias:]
            else:
                conv[d, :keep] = conv_before[s, bias:]
        temporal[bt[dst]] = temporal_before[bt[src + bias]]
        return conv, temporal


@pytest.fixture(params=[False, True], ids=["SD", "DS"])
def ds_layout(request, monkeypatch):
    monkeypatch.setattr(mu, "is_conv_state_dim_first", lambda: request.param)
    return request.param


def _assert_states(h: _Harness, conv_exp, temporal_exp):
    if conv_exp is not None:
        torch.testing.assert_close(h.conv, conv_exp, rtol=0, atol=0)
    torch.testing.assert_close(h.temporal, temporal_exp, rtol=0, atol=0)


# --------------------------------------------------------------------------
# Environment / patch presence
# --------------------------------------------------------------------------


def test_cpu_only_interpreter():
    assert not torch.cuda.is_available()
    assert type(mu.postprocess_mamba_fused_kernel).__name__ == "InterpretedFunction"


def test_patch_is_applied():
    with open(mu.__file__) as f:
        src = f.read()
    body = src.split("def _copy_mamba_state_block(", 1)[1].split("@triton.jit", 1)[0]
    assert "mask=dst_col_ok, other=-1" in body
    assert body.count("mask=src_col_ok, other=-1") == 2
    assert "mask=tmp_col_ok, other=-1" in body
    assert body.count("<= 0:\n") == 4


# --------------------------------------------------------------------------
# Normal case: identical to pre-patch behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("src", "dst", "bias"), [(0, 2, 1), (1, 3, 0), (0, 1, 3)])
def test_precopy_in_range_matches_reference(ds_layout, src, dst, bias):
    h = _Harness(ds_layout=ds_layout)
    h.bt_storage[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    h.bt_storage[1] = torch.tensor([5, 6, 7, 5], dtype=torch.int32)
    before = h.snapshot()
    h.precopy([src], [dst], [bias])
    _assert_states(h, *h.reference_copy(*before, 0, src, dst, bias))


def test_postprocess_spec_decode_in_range_matches_reference(ds_layout):
    # 3 drafts, 3 accepted: running=31, new=33 -> aligned=32, bias=1,
    # dest=32//16-1=1, src=mamba_state_idx=2 -> state[bt[1]] <- bt[2]/bt[3].
    h = _Harness(ds_layout=ds_layout)
    h.bt_storage[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    before = h.snapshot()
    out = h.postprocess(
        accepted=[3], state_idx=[2], scheduled=[4], computed=[30], draft=[3]
    )
    _assert_states(h, *h.reference_copy(*before, 0, 2, 1, 1))
    assert out.tolist() == [3]  # src != dest: count preserved


def test_postprocess_two_rows_row_indexing_unchanged(ds_layout):
    # Row 1 copies with its own row base; row 0 is a no-op (src==dst, bias 0).
    h = _Harness(ds_layout=ds_layout, view_rows=2)
    h.bt_storage[0] = torch.tensor([1, 2, 0, 0], dtype=torch.int32)
    h.bt_storage[1] = torch.tensor([3, 4, 5, 6], dtype=torch.int32)
    before = h.snapshot()
    out = h.postprocess(
        accepted=[1, 3],
        state_idx=[1, 2],
        scheduled=[1, 4],
        computed=[31, 30],
        draft=[0, 3],
    )
    _assert_states(h, *h.reference_copy(*before, 1, 2, 1, 1))
    assert out.tolist() == [1, 3]


@pytest.mark.parametrize("accepted", [0, 1, 4])
def test_zero_draft_prefill_chunk_ending_on_832_boundary_is_noop(ds_layout, accepted):
    """The crash-step shape: no drafts, one prefill chunk that ends exactly on
    an 832-token mamba block boundary after a 2-block prefix hit. The
    postprocess copy is a no-op (src == dest, bias 0) before and after the
    patch. When the count is >= 1 the kernel resets it to 1."""
    h = _Harness(ds_layout=ds_layout, block_size=832)
    h.bt_storage[0] = torch.tensor([0, 2, 3, 4], dtype=torch.int32)  # align: col0 null
    before = h.snapshot()
    out = h.postprocess(
        accepted=[accepted],
        state_idx=[2],  # cdiv(1664 + 832, 832) - 1
        scheduled=[832],
        computed=[1664],
        draft=[0],
    )
    _assert_states(h, *before)
    assert out.tolist() == [accepted if accepted == 0 else 1]


# --------------------------------------------------------------------------
# Site-2 bounds: out-of-row columns and NULL_BLOCK_ID never reach address math
# --------------------------------------------------------------------------


def test_precopy_dst_col_past_row_is_skipped(ds_layout):
    h = _Harness(ds_layout=ds_layout)
    h.bt_storage[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    h.bt_storage[1, 0] = 5  # what an unbounded bt[0, 4] would read
    before = h.snapshot()
    h.precopy([0], [ROW_COLS], [0])
    _assert_states(h, *before)


def test_precopy_dst_null_block_is_skipped(ds_layout):
    h = _Harness(ds_layout=ds_layout)
    h.bt_storage[0] = torch.tensor([1, 2, 3, 0], dtype=torch.int32)
    before = h.snapshot()
    h.precopy([0], [3], [1])  # bt[3] == NULL_BLOCK_ID
    _assert_states(h, *before)


def test_precopy_src_col_past_row_is_skipped(ds_layout):
    h = _Harness(ds_layout=ds_layout)
    h.bt_storage[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    h.bt_storage[1, 0] = 5
    before = h.snapshot()
    h.precopy([ROW_COLS], [1], [0])
    _assert_states(h, *before)


def test_precopy_src_null_block_is_skipped(ds_layout):
    h = _Harness(ds_layout=ds_layout)
    h.bt_storage[0] = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    before = h.snapshot()
    h.precopy([0], [2], [0])  # bt[0] == NULL_BLOCK_ID (conv + temporal source)
    _assert_states(h, *before)


def test_precopy_temporal_src_plus_bias_past_row_skips_only_temporal(ds_layout):
    # src=2 is in the row, so the conv copy (which reads bt[src]) proceeds
    # exactly as before; the temporal read bt[src + bias] = bt[4] is out of
    # the row and must be skipped.
    h = _Harness(ds_layout=ds_layout)
    h.bt_storage[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    h.bt_storage[1, 0] = 5
    conv_before, temporal_before = h.snapshot()
    h.precopy([2], [3], [2])
    conv_exp = conv_before.clone()
    keep = CONV_WIDTH - 2
    if ds_layout:
        conv_exp[4, :, :keep] = conv_before[3, :, 2:]
    else:
        conv_exp[4, :keep] = conv_before[3, 2:]
    _assert_states(h, conv_exp, temporal_before)


def test_precopy_negative_temporal_col_is_skipped():
    # Temporal-only layer (conv would slide by a negative bias, which the
    # V1/V2 callers never produce). Row 1, src 0, bias -1 -> col -1, which
    # unbounded would read row 0's last column.
    h = _Harness(ds_layout=False, view_rows=2, temporal_only=True)
    h.bt_storage[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    h.bt_storage[1] = torch.tensor([5, 6, 7, 5], dtype=torch.int32)
    before = h.snapshot()
    # Row 0 is fresh (src -1 -> early return); row 1 is the probe.
    h.precopy([-1, 0], [0, 1], [0, -1])
    _assert_states(h, *before)


def test_postprocess_stale_accepted_count_dest_past_row_is_skipped(ds_layout):
    # Stale/oversized accepted count with no drafts: running=79, new=81,
    # aligned=80, dest=80//16-1=4 == row width -> out of row.
    h = _Harness(ds_layout=ds_layout)
    h.bt_storage[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    h.bt_storage[1, 0] = 5
    before = h.snapshot()
    h.postprocess(accepted=[3], state_idx=[1], scheduled=[3], computed=[76], draft=[0])
    _assert_states(h, *before)
