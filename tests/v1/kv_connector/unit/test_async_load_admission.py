# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded async-KV-load admission retries.

A request whose connector hit is loaded asynchronously is admitted only if its
full input sequence fits the pool. When it never can, the stock scheduler
retries it identically every step forever: it never parks in
WAITING_FOR_REMOTE_KVS and holds the head of the waiting queue. After
ASYNC_LOAD_ADMIT_MAX_FAILS consecutive failures the scheduler drops the load
and admits the request for local recompute via ordinary chunked prefill.
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
