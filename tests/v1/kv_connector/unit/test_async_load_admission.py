# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded async-KV-load admission retries.

A request whose connector hit is loaded asynchronously is admitted only if its
full input sequence fits the pool. When it never can, the stock scheduler
retries it identically every step forever: it never parks in
WAITING_FOR_REMOTE_KVS and holds the head of the waiting queue.

While the engine is busy (a request is running, or an async load is in
flight) blocks will free, so the candidate keeps its hit and waits; failures
do not count. Only an idle engine, where no block will ever free, degrades
after ASYNC_LOAD_ADMIT_MAX_FAILS consecutive failures: the scheduler drops the
load and admits the request for local recompute via ordinary chunked prefill.
"""

from unittest.mock import patch

import pytest

from vllm.v1.core.sched.scheduler import ASYNC_LOAD_ADMIT_MAX_FAILS as N
from vllm.v1.request import RequestStatus

from .utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _step(scheduler, finished_recving=None, eos=()):
    """One engine step. Requests still prefilling sample nothing; requests in
    `eos` sample EOS, everyone else samples token 1."""
    so = scheduler.schedule()
    out = create_model_runner_output(
        reqs=list(scheduler.running),
        finished_recving=finished_recving,
        token_id=1,
    )
    for i, req_id in enumerate(out.req_ids):
        request = scheduler.requests[req_id]
        if req_id not in so.num_scheduled_tokens:
            out.sampled_token_ids[i] = []
        elif request.num_computed_tokens < request.num_tokens:
            out.sampled_token_ids[i] = []
        elif req_id in eos:
            out.sampled_token_ids[i] = [EOS_TOKEN_ID]
    scheduler.update_from_output(so, out)
    return so


def _make_scheduler(async_scheduling, num_blocks, **kwargs):
    vllm_config = create_vllm_config(**kwargs)
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    return create_scheduler(vllm_config, num_blocks=num_blocks)


def _setup_busy(async_scheduling, block_size, num_blocks, running_blocks):
    # A 2-block token budget: a degraded (dropped-hit) candidate's first chunk
    # fits next to the running request, so a dropped hit is observable.
    scheduler = _make_scheduler(
        async_scheduling, num_blocks, max_num_batched_tokens=2 * block_size
    )
    hog = create_request(
        request_id=1,
        block_size=block_size,
        num_tokens=block_size * running_blocks,
        max_tokens=1000,
    )
    scheduler.add_request(hog)
    while hog.num_computed_tokens < hog.num_prompt_tokens:
        _step(scheduler)
    assert hog.status == RequestStatus.RUNNING
    return scheduler, hog


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_busy_engine_never_drops_async_hit(async_scheduling):
    """A running request holds most of the pool; an async-load candidate's full
    sequence cannot fit yet. It waits with its hit and parks for the load once
    the pool frees, however many steps that takes."""
    bs = 16
    scheduler, hog = _setup_busy(async_scheduling, bs, num_blocks=12, running_blocks=6)
    cand = create_request(
        request_id=2, block_size=bs, num_tokens=bs * 6, do_remote_prefill=True
    )
    scheduler.add_request(cand)
    with patch.object(
        scheduler.connector,
        "get_num_new_matched_tokens",
        return_value=(bs * 5, True),
    ):
        for _ in range(4 * N):
            _step(scheduler)
            assert cand.status == RequestStatus.WAITING, "hit dropped while busy"
            assert cand.num_computed_tokens == 0
        assert hog.status == RequestStatus.RUNNING
        assert scheduler._async_load_admit_fails.get(cand.request_id, 0) == 0
        # The running request finishes; the candidate now fits and is parked
        # for its async load with the external hit intact.
        _step(scheduler, eos={hog.request_id})
        _step(scheduler)
    assert cand.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert cand.num_computed_tokens == bs * 5
    assert cand.request_id not in scheduler._async_load_admit_fails


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_idle_engine_degrades_after_n_failures(async_scheduling):
    """Nothing running, nothing in flight, and the full sequence can never fit
    the pool. The candidate degrades to chunked local recompute on the
    (N+1)-th pass instead of spinning forever."""
    bs = 16
    scheduler = _make_scheduler(async_scheduling, 12, max_num_batched_tokens=64)
    cand = create_request(
        request_id=3, block_size=bs, num_tokens=bs * 20, do_remote_prefill=True
    )
    scheduler.add_request(cand)
    with patch.object(
        scheduler.connector,
        "get_num_new_matched_tokens",
        return_value=(bs * 10, True),
    ):
        for i in range(N):
            _step(scheduler)
            assert cand.status == RequestStatus.WAITING
            assert scheduler._async_load_admit_fails[cand.request_id] == i + 1
        _step(scheduler)
    assert cand.status == RequestStatus.RUNNING
    assert cand.num_computed_tokens > 0  # first chunk computed locally
    assert cand.request_id not in scheduler._async_load_admit_fails


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_idle_countdown_starts_only_when_engine_goes_idle(async_scheduling):
    """Failures while busy never count: the candidate (which can never fit)
    survives any number of busy steps, and degrades only after N further
    consecutive idle failures once the running request has finished."""
    bs = 16
    scheduler, hog = _setup_busy(async_scheduling, bs, num_blocks=12, running_blocks=4)
    cand = create_request(
        request_id=4, block_size=bs, num_tokens=bs * 20, do_remote_prefill=True
    )
    scheduler.add_request(cand)
    with patch.object(
        scheduler.connector,
        "get_num_new_matched_tokens",
        return_value=(bs * 10, True),
    ):
        for _ in range(3 * N):
            _step(scheduler)
            assert cand.status == RequestStatus.WAITING, "hit dropped while busy"
        assert scheduler._async_load_admit_fails.get(cand.request_id, 0) == 0
        _step(scheduler, eos={hog.request_id})
        assert not scheduler.running
        for i in range(N):
            _step(scheduler)
            assert cand.status == RequestStatus.WAITING
            assert scheduler._async_load_admit_fails[cand.request_id] == i + 1
        _step(scheduler)
    assert cand.status == RequestStatus.RUNNING
    assert cand.request_id not in scheduler._async_load_admit_fails


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_inflight_async_load_counts_as_busy(async_scheduling):
    """Nothing is running, but an admitted async load is still in flight
    (WAITING_FOR_REMOTE_KVS). Its blocks will be released once it lands and
    runs, so a second candidate held back by the in-flight reservation keeps
    its hit."""
    scheduler = _make_scheduler(async_scheduling, 8)  # usable 7
    bs = scheduler.block_size
    req_a = create_request(
        request_id=5,
        block_size=bs,
        num_tokens=bs * 4,
        do_remote_prefill=True,
        num_remote_blocks=1,
        max_tokens=1,
    )
    req_b = create_request(
        request_id=6,
        block_size=bs,
        num_tokens=bs * 5,
        do_remote_prefill=True,
        num_remote_blocks=1,
        max_tokens=1,
    )
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    hits = {req_a.request_id: (bs, True), req_b.request_id: (bs * 4, True)}
    with patch.object(
        scheduler.connector,
        "get_num_new_matched_tokens",
        side_effect=lambda request, _: hits[request.request_id],
    ):
        for _ in range(3 * N):
            _step(scheduler)
            assert req_a.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            assert req_b.status == RequestStatus.WAITING, "hit dropped while busy"
            assert req_b.num_computed_tokens == 0
        assert not scheduler.running
        assert scheduler._async_load_admit_fails.get(req_b.request_id, 0) == 0
        # req_a's load lands and it runs to completion, freeing the pool.
        _step(scheduler, finished_recving={req_a.request_id})
        for _ in range(8):
            if req_b.status != RequestStatus.WAITING:
                break
            _step(scheduler, eos={req_a.request_id})
    assert req_a.is_finished()
    assert req_b.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req_b.num_computed_tokens == bs * 4


def test_abort_clears_admission_failure_count():
    """A candidate aborted while it keeps failing admission does not leak its
    failure count."""
    bs = 16
    scheduler = _make_scheduler(False, 12, max_num_batched_tokens=64)
    cand = create_request(
        request_id=7, block_size=bs, num_tokens=bs * 20, do_remote_prefill=True
    )
    scheduler.add_request(cand)
    with patch.object(
        scheduler.connector,
        "get_num_new_matched_tokens",
        return_value=(bs * 10, True),
    ):
        _step(scheduler)
        _step(scheduler)
    assert scheduler._async_load_admit_fails[cand.request_id] == 2
    scheduler.finish_requests(cand.request_id, RequestStatus.FINISHED_ABORTED)
    assert cand.request_id not in scheduler._async_load_admit_fails
