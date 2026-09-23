# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OffloadingConnector sync_load: loads land before the runner prepares the step.

Bug: with sync_load the request is admitted and scheduled in the same step, and
GPUModelRunner.execute_model runs
    handle_preemptions -> _update_states (queues zeroing of new attention
    blocks) -> mamba_utils.preprocess_mamba (align mode: copies GDN state from
    block (N-1)//block_size, the sync load's destination) ->
    maybe_get_kv_connector_output -> start_load_kv (submits + host-waits the
    sync load).
So the mamba state is copied from a stale recycled page before the load lands
(garbage output, MTP acceptance 0%). Moving the load earlier then needs the
scheduler to skip zeroing the sync-hit blocks (only the async admission branch
did that), or the zeroing kernel wipes the freshly loaded attention/MTP KV.

These tests pin:
  * sync load jobs are submitted AND host-waited before handle_preemptions
    returns, after the deferred stores + preemption flush;
  * start_kv_transfers never re-submits (or re-waits) them;
  * sync jobs still report no finished_recving and complete via
    completed_jobs; the async (sync_load=False) path is unchanged;
  * the scheduler keeps sync-hit blocks out of new_block_ids_to_zero;
  * an end-to-end simulation of the runner's step order (fake GPU memory)
    sees loaded data both at the mamba pre-copy and at the forward;
  * source guards: execute_model still calls handle_preemptions before
    _update_states / preprocess_mamba, and CPU->GPU loads wait on compute.
"""

import ast
import importlib.util

import pytest

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    MockLoadStoreSpec,
    generate_store_output,
    to_keys,
)
from tests.v1.kv_connector.unit.utils import EOS_TOKEN_ID
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LookupResult
from vllm.v1.request import RequestStatus


@pytest.fixture(autouse=True)
def _v1_model_runner(monkeypatch):
    # sync_load is V1-only (the offloading scheduler rejects it under V2).
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")


BLOCK_SIZE = 4


def _instrument(handler) -> list[tuple]:
    """Record submit_store / submit_load / wait calls on the mock worker."""
    log: list[tuple] = []
    orig_store, orig_load, orig_wait = (
        handler.submit_store,
        handler.submit_load,
        handler.wait,
    )

    def submit_store(job_id, src_spec, dst_spec):
        log.append(("submit_store", job_id))
        return orig_store(job_id, src_spec, dst_spec)

    def submit_load(job_id, src_spec, dst_spec):
        log.append(("submit_load", job_id))
        return orig_load(job_id, src_spec, dst_spec)

    def wait(job_ids):
        log.append(("wait", frozenset(job_ids)))
        return orig_wait(job_ids)

    handler.submit_store = submit_store
    handler.submit_load = submit_load
    handler.wait = wait
    return log


def _force_kv_cache_zeroing(runner) -> None:
    """The harness uses a uniform-precision attention-only cache, so zeroing
    is off; turn it on exactly as a hybrid (mamba) model would have it."""
    runner.scheduler.needs_kv_cache_zeroing = True
    for mgr in runner.scheduler.kv_cache_manager.coordinator.single_type_managers:
        mgr._record_new_block_ids = True


def _seed_then_admit_hit(
    runner, token: int, stored_chunks: int, prompt_chunks: int, zeroing=False
):
    """Offload a `stored_chunks` prompt, drop the GPU prefix cache, then admit
    a `prompt_chunks` prompt that hits the offloaded prefix. Returns the
    admission step's SchedulerOutput (worker side not yet run)."""
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(keys)
    )
    runner.new_request(token_ids=[token] * BLOCK_SIZE * stored_chunks)
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID], expected_stored=tuple(range(stored_chunks))
    )
    runner.scheduler.reset_prefix_cache()

    stored_keys = {key for key, _ in runner.offloaded}
    runner.manager.lookup.side_effect = lambda key, req_context: (
        LookupResult.HIT if key in stored_keys else LookupResult.MISS
    )
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    if zeroing:
        _force_kv_cache_zeroing(runner)
    runner.new_request(token_ids=[token] * BLOCK_SIZE * prompt_chunks)
    scheduler_output = runner.scheduler.schedule()
    runner._update_gpu_blocks()
    return scheduler_output


def _load_dst_block_ids(meta: OffloadingConnectorMetadata) -> set[int]:
    ids: set[int] = set()
    for job in meta.load_jobs.values():
        assert isinstance(job.dst_spec, GPULoadStoreSpec)
        ids.update(int(b) for b in job.dst_spec.block_ids)
    return ids


# --------------------------------------------------------------------------
# (a) worker: sync loads land inside handle_preemptions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_sync_load_lands_before_handle_preemptions_returns(
    request_runner, async_scheduling
):
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        extra_config_overrides={"sync_load": True},
    )
    so = _seed_then_admit_hit(runner, token=1, stored_chunks=4, prompt_chunks=6)
    meta = so.kv_connector_metadata
    assert isinstance(meta, OffloadingConnectorMetadata)
    load_jids = set(meta.load_jobs)
    assert len(load_jids) == 1
    (req_id,) = so.num_scheduled_tokens  # admitted + scheduled this step
    handler = runner.offloading_spec.handler
    log = _instrument(handler)

    runner.worker_connector.handle_preemptions(meta)
    # Before _update_states / preprocess_mamba could run: submitted + waited.
    assert [e for e in log if e[0] == "submit_load"] == [
        ("submit_load", j) for j in meta.load_jobs
    ]
    assert ("wait", frozenset(load_jids)) in log
    assert load_jids <= handler.flushed_jobs
    assert not load_jids & handler.waiting_jobs

    # start_load_kv must not submit (or wait on) them a second time.
    n = len(log)
    runner.worker_connector.bind_connector_metadata(meta)
    runner.worker_connector.start_load_kv(runner._dummy_ctx)
    assert log[n:] == []

    # Bookkeeping identical to the old in-start_kv_transfers sync path.
    finished_sending, finished_recving = runner.worker_connector.get_finished(
        so.finished_req_ids
    )
    assert finished_recving == set() and finished_sending == set()
    worker_meta = runner.worker_connector.build_connector_worker_meta()
    assert worker_meta is not None
    assert load_jids <= set(worker_meta.completed_jobs)
    assert not runner.worker_connector.connector_worker._load_jobs
    runner.worker_connector.clear_connector_metadata()
    assert runner.scheduler.requests[req_id].status == RequestStatus.RUNNING


def test_handle_preemptions_orders_stores_flush_then_sync_loads(request_runner):
    """Worker-level ordering with deferred stores, a preemption flush and two
    sync hits in the same step: stores (deferred + flushed) are submitted
    first, the flush is waited, then every sync load is submitted and waited;
    start_kv_transfers is a no-op for them afterwards."""
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=False,
        extra_config_overrides={"sync_load": True},
    )
    connector_worker = runner.worker_connector.connector_worker
    handler = runner.offloading_spec.handler
    log = _instrument(handler)

    def gpu(ids):
        return GPULoadStoreSpec(ids, group_sizes=[len(ids)], block_indices=[0])

    # A store deferred from the previous step's prepare_store_kv.
    connector_worker._unsubmitted_store_jobs.append(
        (200, gpu([1]), MockLoadStoreSpec(to_keys([90])))
    )
    meta = OffloadingConnectorMetadata(
        load_jobs={
            101: TransferJob("a", MockLoadStoreSpec(to_keys([1, 2])), gpu([7, 8])),
            102: TransferJob("b", MockLoadStoreSpec(to_keys([3])), gpu([9])),
        },
        store_jobs={
            201: TransferJob("c", gpu([3]), MockLoadStoreSpec(to_keys([91]))),
        },
        jobs_to_flush={201},
    )

    runner.worker_connector.handle_preemptions(meta)
    assert log == [
        ("submit_store", 200),
        ("submit_store", 201),
        ("wait", frozenset({201})),
        ("submit_load", 101),
        ("submit_load", 102),
        ("wait", frozenset({101, 102})),
    ]

    runner.worker_connector.bind_connector_metadata(meta)
    runner.worker_connector.start_load_kv(runner._dummy_ctx)
    assert len(log) == 6

    _, finished_recving = runner.worker_connector.get_finished(set())
    assert finished_recving == set()
    worker_meta = runner.worker_connector.build_connector_worker_meta()
    # The deferred store 200 was submitted but not flushed: still in flight.
    assert set(worker_meta.completed_jobs) == {201, 101, 102}
    assert handler.waiting_jobs == {200}
    assert not connector_worker._load_jobs
    runner.worker_connector.clear_connector_metadata()


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_async_load_path_unchanged(request_runner, async_scheduling):
    """sync_load=False: handle_preemptions issues no loads, start_kv_transfers
    submits them without waiting, and completion reports finished_recving."""
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
    )
    so = _seed_then_admit_hit(runner, token=2, stored_chunks=4, prompt_chunks=6)
    meta = so.kv_connector_metadata
    load_jids = set(meta.load_jobs)
    assert len(load_jids) == 1
    parked = [
        r
        for r in runner.scheduler.requests.values()
        if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    ]
    assert len(parked) == 1
    handler = runner.offloading_spec.handler
    log = _instrument(handler)

    runner.worker_connector.handle_preemptions(meta)
    assert log == []
    runner.worker_connector.bind_connector_metadata(meta)
    runner.worker_connector.start_load_kv(runner._dummy_ctx)
    assert log == [("submit_load", j) for j in meta.load_jobs]
    assert not load_jids & handler.flushed_jobs
    _, finished_recving = runner.worker_connector.get_finished(set())
    assert finished_recving == set()
    runner.worker_connector.clear_connector_metadata()

    runner.offloading_spec.complete_transfers()
    runner.worker_connector.bind_connector_metadata(
        OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
    )
    _, finished_recving = runner.worker_connector.get_finished(set())
    assert finished_recving == {parked[0].request_id}
    runner.worker_connector.clear_connector_metadata()


# --------------------------------------------------------------------------
# (b) scheduler: sync-hit blocks are not zeroed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sync_load", [True, False])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_load_destination_blocks_skip_zeroing(
    request_runner, async_scheduling, sync_load
):
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        extra_config_overrides={"sync_load": sync_load},
    )
    so = _seed_then_admit_hit(
        runner, token=3, stored_chunks=4, prompt_chunks=6, zeroing=True
    )
    meta = so.kv_connector_metadata
    dst = _load_dst_block_ids(meta)
    assert len(dst) == 4
    zeroed = set(so.new_block_ids_to_zero or ())
    assert not dst & zeroed, f"loaded blocks {sorted(dst & zeroed)} get zeroed"
    if sync_load:
        # Genuinely new blocks (the computed tail past the hit) still zero.
        (req_id,) = so.num_scheduled_tokens
        (req_blocks,) = runner.scheduler.kv_cache_manager.get_block_ids(req_id)
        assert set(req_blocks[4:]) and set(req_blocks[4:]) <= zeroed
    # The skip set is per-step.
    assert not runner.scheduler._skip_zero_block_ids


# --------------------------------------------------------------------------
# (a)+(b) end-to-end: simulate the runner's step order on fake GPU memory
# --------------------------------------------------------------------------


@pytest.mark.parametrize("async_scheduling", [True, False])
def test_runner_order_sim_state_copy_and_forward_see_loaded_kv(
    request_runner, async_scheduling
):
    """Fake GPU memory (block id -> tag). The simulated step follows
    GPUModelRunner.execute_model: handle_preemptions -> _update_states zeroing
    -> preprocess_mamba pre-copy from block (N-1)//block_size (a sync-load
    destination) -> bind + start_load_kv -> forward reads the loaded blocks.
    A host-waited load writes 'loaded' into its destination blocks."""
    runner = request_runner(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=100,
        async_scheduling=async_scheduling,
        extra_config_overrides={"sync_load": True},
    )
    so = _seed_then_admit_hit(
        runner, token=4, stored_chunks=4, prompt_chunks=6, zeroing=True
    )
    meta = so.kv_connector_metadata
    handler = runner.offloading_spec.handler
    fake_gpu: dict[int, str] = {}
    pending_loads: dict[int, list[int]] = {}
    orig_load, orig_wait = handler.submit_load, handler.wait


    def submit_load(job_id, src_spec, dst_spec):
        pending_loads[job_id] = [int(b) for b in dst_spec.block_ids]
        return orig_load(job_id, src_spec, dst_spec)

    def wait(job_ids):
        for jid in job_ids:
            for b in pending_loads.pop(jid, ()):
                fake_gpu[b] = "loaded"
        return orig_wait(job_ids)

    handler.submit_load = submit_load
    handler.wait = wait

    (new_req,) = so.scheduled_new_reqs
    n_computed = new_req.num_computed_tokens
    assert n_computed == 4 * BLOCK_SIZE
    (req_blocks,) = new_req.block_ids
    dst = _load_dst_block_ids(meta)

    # execute_model order
    runner.worker_connector.handle_preemptions(meta)
    for b in so.new_block_ids_to_zero or ():  # _update_states
        fake_gpu[b] = "zeroed"
    state_src = req_blocks[(n_computed - 1) // BLOCK_SIZE]  # preprocess_mamba
    copied_state = fake_gpu.get(state_src, "stale")
    runner.worker_connector.bind_connector_metadata(meta)
    runner.worker_connector.start_load_kv(runner._dummy_ctx)
    forward_view = {b: fake_gpu.get(b, "stale") for b in sorted(dst)}

    assert state_src in dst
    assert copied_state == "loaded", f"mamba pre-copy read a {copied_state} page"
    assert set(forward_view.values()) == {"loaded"}, forward_view
    runner.worker_connector.get_finished(set())
    runner.worker_connector.clear_connector_metadata()


# --------------------------------------------------------------------------
# source guards (no GPU needed)
# --------------------------------------------------------------------------


def _module_ast(modname: str) -> ast.Module:
    spec = importlib.util.find_spec(modname)
    assert spec is not None and spec.origin
    with open(spec.origin) as f:
        return ast.parse(f.read())


def _function(tree: ast.Module, cls: str, fn: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == fn:
                    return item
    raise AssertionError(f"{cls}.{fn} not found")


def _first_call_line(fn: ast.FunctionDef, attr: str) -> int:
    lines = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]
    assert lines, f"no call to {attr}"
    return min(lines)


def test_v1_runner_calls_handle_preemptions_before_state_prep():
    """The fix relies on this order in the V1 runner (the one hybrid mamba
    models run on): if it changes, the sync load must move with it."""
    fn = _function(
        _module_ast("vllm.v1.worker.gpu_model_runner"),
        "GPUModelRunner",
        "execute_model",
    )
    order = [
        _first_call_line(fn, name)
        for name in (
            "handle_preemptions",
            "_update_states",
            "preprocess_mamba",
            "maybe_get_kv_connector_output",
        )
    ]
    assert order == sorted(order), order


def test_cpu_to_gpu_loads_wait_on_compute_stream():
    """Port of upstream #50696: the transfer stream waits on the compute
    stream for loads too, not only for stores (GPU->CPU)."""
    fn = _function(
        _module_ast("vllm.v1.kv_offload.cpu.gpu_worker"),
        "SingleDirectionOffloadingHandler",
        "transfer_async",
    )
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and "gpu_to_cpu" in ast.unparse(node.test):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and ast.unparse(sub.func).endswith(
                    "wait_stream"
                ):
                    raise AssertionError("wait_stream(compute) gated on gpu_to_cpu")
    assert any(
        isinstance(node, ast.Call) and ast.unparse(node.func).endswith("wait_stream")
        for node in ast.walk(fn)
    )
