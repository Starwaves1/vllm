# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingManager: Multi-tier KV cache offloading orchestrator.

This manager coordinates between a CPU primary tier (with direct GPU access)
and zero or more secondary tiers (Storage, Network, etc.) to provide
hierarchical KV cache offloading.

Key Design Principles:
1. Always offload to all tiers — When a block is stored to the primary tier,
   it is cascaded to ALL secondary tiers
2. Primary tier is the gateway — Secondary tiers cannot access GPU memory
   directly; all data flows through the CPU primary tier
3. Staged promotion — Blocks in secondary tiers must be promoted to the
   primary tier before GPU can access them
4. Transparent retry mechanism — Return None from lookup() to signal
   "data is being promoted, try later"
5. ref_cnt as eviction protection — primary.prepare_read() increments ref_cnt,
   protecting blocks from eviction until complete_read() is called

Write-back tiers (store_policy "write_back") are the exception to 1: blocks are
not cascaded on store. Once per step, while the primary tier is filled to its
high watermark, the coldest blocks such a tier lacks are written to it (see
_writeback_step), so a block evicted later is still on the secondary tier.
"""

import time
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.base import (
    WRITEBACK_REQ_ID,
    JobId,
    JobMetadata,
    ParentManager,
    SecondaryTierManager,
    StorePolicy,
    TieringOffloadingMetrics,
)

logger = init_logger(__name__)


@dataclass
class PendingPromotion:
    """Accumulator for blocks awaiting submit_load() for one (tier, request)."""

    req_context: ReqContext
    keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


# Max blocks per write-back store job. A job's blocks stay pinned until its
# last block is written, so small jobs release CPU slots sooner.
_WRITEBACK_JOB_BLOCKS = 16


@dataclass
class _WritebackState:
    """Write-back bookkeeping for one secondary tier (scheduler thread)."""

    tier: SecondaryTierManager
    # Primary-tier occupancy (blocks) at which flushing starts, and the number
    # of unwritten blocks it flushes down to.
    high_blocks: int
    low_blocks: int
    # Blocks in the primary tier that this tier does not hold yet. Includes
    # the blocks being written (``flushing``).
    dirty: set[OffloadKey] = field(default_factory=set)
    # Blocks pinned by an in-flight write-back job.
    flushing: set[OffloadKey] = field(default_factory=set)
    # Blocks the flusher picked as cold (not pulled in as a prefix or group
    # sibling): moved back to the LRU end when their write finishes.
    demote: set[OffloadKey] = field(default_factory=set)
    # Flushing blocks read or touched by a request meanwhile: not demoted.
    touched: set[OffloadKey] = field(default_factory=set)
    n_flushed: int = 0
    n_lost: int = 0


@dataclass(slots=True)
class RequestState:
    req_context: ReqContext
    pending_primary_stores: int = 0
    is_finished: bool = False
    request_level_tiers: set[SecondaryTierManager] | None = None
    sync_lookup_delay: float = 0.0
    # time.monotonic() of this request's first deferred secondary-tier lookup;
    # None once consumed (observed) or while no secondary lookup is pending.
    secondary_lookup_start_time: float | None = None


class CPUPrimaryTierOffloadingManager(CPUOffloadingManager):
    """CPUOffloadingManager with a primary/secondary transfer interface.

    The inherited prepare_store/complete_store/prepare_load/complete_load are the
    GPU-facing OffloadingManager interface. These aliases expose the same operations
    from the secondary tier perspective, where read/write refers to secondary
    accessing primary. This avoids confusion when reading TieringOffloadingManager
    code (e.g. calling prepare_load inside a cascade/store path would be misleading).
    """

    def __init__(
        self,
        num_blocks: int,
        mmap_region: SharedOffloadRegion,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
    ):
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,
            cache_policy_module_path=cache_policy_module_path,
            enable_events=enable_events,
        )
        self._mmap_region = mmap_region
        # read/write is for CPU<->secondary transfers,
        # load/store is for CPU<->GPU transfers.
        # These aliases avoid calling prepare_load inside a store path.
        self.prepare_read = self.prepare_load
        self.complete_read = self.complete_load
        self.prepare_write = self.prepare_store
        self.complete_write = self.complete_store

        self._kv_memoryview = mmap_region.create_kv_memoryview()
        # Called with the keys of blocks evicted by prepare_store/prepare_write.
        self.eviction_listener: Callable[[list[OffloadKey]], None] | None = None

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        result = super().prepare_store(keys, req_context)
        if (
            result is not None
            and result.evicted_keys
            and self.eviction_listener is not None
        ):
            self.eviction_listener(result.evicted_keys)
        return result

    # --- write-back support (read-only views of the cache policy) ---

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def num_used_blocks(self) -> int:
        """Slots holding a block (ready, being written, or pinned)."""
        return self._num_allocated_blocks - len(self._free_list)

    def get_block(self, key: OffloadKey) -> BlockStatus | None:
        return self._policy.get(key)

    def supports_writeback(self) -> bool:
        return type(self._policy).iter_evictable is not CachePolicy.iter_evictable

    def iter_evictable(self) -> Iterator[OffloadKey]:
        """Evictable blocks, next eviction victim first. Do not change the
        primary tier while iterating."""
        return self._policy.iter_evictable()

    def demote(self, keys: Iterable[OffloadKey]) -> None:
        """Make evictable ``keys`` the next eviction victims."""
        self._policy.demote(keys)

    def get_kv_memoryview(self) -> memoryview:
        """Return the memoryview over the primary tier's KV cache buffer.

        The view has shape (num_blocks, row_stride_bytes) and is backed by the
        SharedOffloadRegion mmap.  Secondary tiers address block *b* as
        ``view[b]``.
        """
        return self._kv_memoryview

    @override
    def shutdown(self) -> None:
        super().shutdown()
        self._kv_memoryview.release()
        self._mmap_region.cleanup()


class _SecondaryTierFacingParent(ParentManager):
    """Wrapper that implements ParentManager by delegating to the
    TieringOffloadingManager with exclude_tier set to the origin tier."""

    __slots__ = ("_m", "_origin")

    def __init__(
        self,
        manager: "TieringOffloadingManager",
        tier: SecondaryTierManager,
    ):
        self._m = manager
        self._origin = tier

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return self._m.on_new_request(req_context, exclude_tier=self._origin)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return self._m.lookup(key, req_context, exclude_tier=self._origin)

    def create_store_job(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> JobMetadata:
        return self._m.create_store_job(keys, req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        return self._m.on_request_finished(req_context, exclude_tier=self._origin)


class TieringOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a CPU primary tier (with direct GPU access)
    and zero or more secondary tiers (Storage, Network, etc.) to provide
    hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished_jobs()
      - Secondary tiers return JobResult objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs
    """

    def __init__(
        self,
        primary_tier: CPUPrimaryTierOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
    ):
        """
        Initialize the TieringOffloadingManager.

        Args:
            primary_tier: The primary tier manager (CPU-based).
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
        """
        self.primary_tier: CPUPrimaryTierOffloadingManager = primary_tier
        self.secondary_tiers = secondary_tiers or []

        self._job_id_counter: int = 0
        # Job tracking: maps job_id to metadata for all in-flight transfers.
        # JobMetadata.is_promotion distinguishes direction:
        #   True:  secondary → primary (promotion)
        #   False: primary → secondary (cascade)
        self._transfer_jobs: dict[JobId, JobMetadata] = {}

        # Pending promotion requests accumulated during lookup() calls; flushed
        # as one batched submit_load() per (tier, request) in on_schedule_end().
        # Outer key: tier. Inner key: req_context.req_id — the same ReqContext
        # object is reused for all block lookups of a given request per engine step.
        self._pending_load_submissions: dict[
            SecondaryTierManager, dict[str, PendingPromotion]
        ] = {}

        # Gate for once-per-step execution of _maybe_process_finished_jobs().
        # Reset at the end of each step in on_schedule_end().
        self._processed_jobs_this_step: bool = False

        # Per-request state for prepared GPU->primary stores and finalization.
        # Secondary tiers are finalized only after pending primary stores reach
        # complete_store(), since complete_store() can still submit cascades.
        self._req_state: dict[str, RequestState] = {}

        # Cached ParentManager wrappers for each secondary tier.
        self._tier_parents: dict[SecondaryTierManager, _SecondaryTierFacingParent] = {
            tier: _SecondaryTierFacingParent(self, tier)
            for tier in self.secondary_tiers
        }

        # Buffers manager-level observations (e.g. lookup delay) between
        # get_stats() calls; merged in and reset each time get_stats() runs.
        self._stats = OffloadingConnectorStats()

        # Write-back tiers (store_policy "write_back"), see _writeback_step().
        self._writeback: dict[SecondaryTierManager, _WritebackState] = {}
        for tier in self.secondary_tiers:
            if tier.store_policy != StorePolicy.WRITE_BACK:
                continue
            if not primary_tier.supports_writeback():
                raise ValueError(
                    f"store_policy 'write_back' on secondary tier "
                    f"'{tier.tier_type}' needs a primary cache policy that "
                    "implements CachePolicy.iter_evictable() (lru, arc)"
                )
            n = primary_tier.num_blocks
            self._writeback[tier] = _WritebackState(
                tier=tier,
                high_blocks=max(1, int(n * tier.writeback_high_watermark)),
                low_blocks=int(n * tier.writeback_low_watermark),
            )
        self._writeback_jobs: dict[JobId, _WritebackState] = {}
        # Prefix links learned from touch(): key -> key of the preceding chunk
        # of the same KV cache group, for keys in the primary tier. A block
        # hash chains its parent's, so a link never changes. Pruned on primary
        # eviction; reset if it ever outgrows the primary (failed stores).
        self._writeback_parent: dict[OffloadKey, OffloadKey] = {}
        # Group-index suffixes of the keys seen (in order), to find a chunk's
        # siblings.
        self._writeback_groups: dict[bytes, None] = {}
        self._writeback_ctx = ReqContext(req_id=WRITEBACK_REQ_ID)
        if self._writeback:
            primary_tier.eviction_listener = self._on_primary_evicted

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _maybe_process_finished_jobs(self):
        """
        Poll secondary tiers for completed jobs (at most once per step).

        Guarded by _processed_jobs_this_step: the first call in an engine step
        does the actual polling; subsequent calls are no-ops. The flag is reset
        in on_schedule_end() at the end of each step.
        """
        if self._processed_jobs_this_step:
            return
        self._processed_jobs_this_step = True
        self._process_finished_jobs()

    def _process_finished_jobs(self):
        """
        Unconditionally poll all secondary tiers for completed jobs.

        This method:
        1. Calls get_finished_jobs() on each secondary tier
        2. For completed stores (primary→secondary): calls primary.complete_read()
           to decrement ref_cnt
        3. For completed loads (secondary→primary): calls primary.complete_write()
           to make blocks available
        """
        for i, tier in enumerate(self.secondary_tiers):
            for completed_job in tier.get_finished_jobs():
                job_id = completed_job.job_id
                job_metadata = self._transfer_jobs.pop(job_id, None)
                assert job_metadata is not None, (
                    f"Finished job_id {job_id} from tier #{i}"
                    f" ({tier.tier_type}) not in _transfer_jobs"
                )

                if job_metadata.is_promotion:
                    # secondary→primary transfer (promotion) completed.
                    # Make blocks available in primary tier.
                    self.primary_tier.complete_write(
                        job_metadata.keys,
                        job_metadata.req_context,
                        completed_job.success,
                    )
                else:
                    # primary→secondary transfer completed.
                    # Decrement ref_cnt on primary blocks.
                    self.primary_tier.complete_read(
                        job_metadata.keys, job_metadata.req_context
                    )
                    wb = self._writeback_jobs.pop(job_id, None)
                    if wb is not None:
                        self._finish_writeback_job(
                            wb, job_id, job_metadata.keys, completed_job.success
                        )

    @override
    def lookup(
        self,
        key: OffloadKey,
        req_context: ReqContext,
        *,
        exclude_tier: SecondaryTierManager | None = None,
    ) -> LookupResult:
        """
        Check whether a single block is offloaded and ready.

        Algorithm:
            1. Process any completed async jobs first.
            2. Query primary tier — short-circuit on hit or in-flight.
            3. On primary miss, query secondary tiers — stop on first
               hit and initiate promotion.

        Args:
            key: Block hash to look up.
            req_context: Per-request context.

        Returns:
            HIT       — block is ready in the primary tier.
            HIT_PENDING — block found but not yet readable (write
                        in-flight on the primary tier).
            RETRY     — promotion started or a secondary tier is busy.
            MISS      — block not found in any tier, or primary is full
                        and cannot accept a promotion.
        """
        # Poll first so a promotion that finished since the last call is
        # already reflected as HIT (not stale HIT_PENDING/MISS) below, and
        # so blocks freed by cascade or promotion completions are evictable
        # in time for a promotion this lookup may initiate.
        self._maybe_process_finished_jobs()

        req_state = self._req_state.get(req_context.req_id)

        primary_hit = self.primary_tier.lookup(key, req_context)
        if primary_hit is LookupResult.HIT:
            return LookupResult.HIT
        if primary_hit is LookupResult.HIT_PENDING:
            return LookupResult.HIT_PENDING

        lookup_start = time.monotonic()
        any_retry = False
        for tier in self.secondary_tiers:
            if tier is exclude_tier:
                continue
            if not req_context.load_tier_filter.allows(tier.medium, tier.locality):
                continue
            result = tier.lookup(key, req_context)
            if result is LookupResult.HIT:
                promoted = self._initiate_promotion(tier, key, req_context)
                self._accumulate_lookup_sync_delay(req_state, lookup_start)
                if (
                    req_state is not None
                    and promoted
                    and req_state.secondary_lookup_start_time is None
                ):
                    req_state.secondary_lookup_start_time = lookup_start
                # HIT_PENDING, not RETRY (backport of upstream #51840): the block
                # is known to exist and is on its way to the primary tier. RETRY
                # made the connector's sliding-window (Mamba/GDN, SWA) lookup
                # keep scanning backward and promote every earlier chunk of the
                # group from the secondary tier, although only the chunk at the
                # hit boundary is ever loaded (4x disk reads for a 3-GDN-group
                # hybrid model, and CPU-tier slots wasted on unread state).
                return LookupResult.MISS if not promoted else LookupResult.HIT_PENDING
            if result is LookupResult.RETRY:
                any_retry = True

        self._accumulate_lookup_sync_delay(req_state, lookup_start)
        if any_retry:
            if req_state is not None and req_state.secondary_lookup_start_time is None:
                req_state.secondary_lookup_start_time = lookup_start
            return LookupResult.RETRY
        return LookupResult.MISS

    def _accumulate_lookup_sync_delay(
        self, req_state: RequestState | None, start_time: float
    ) -> None:
        """Accumulate secondary-tier lookup time until allocation or finish."""
        if req_state is not None:
            req_state.sync_lookup_delay += time.monotonic() - start_time

    def _maybe_observe_lookup_sync_delay(self, req_state: RequestState) -> None:
        delay = req_state.sync_lookup_delay
        if delay == 0:
            return
        req_state.sync_lookup_delay = 0.0
        self._stats.observe_histogram(
            TieringOffloadingMetrics.LOOKUP_SYNC_DELAY,
            delay,
        )

    def _maybe_observe_lookup_async_delay(self, req_state: RequestState) -> None:
        """Flush a pending deferred secondary-tier lookup timer, if any."""
        start_time = req_state.secondary_lookup_start_time
        if start_time is None:
            return
        req_state.secondary_lookup_start_time = None
        self._stats.observe_histogram(
            TieringOffloadingMetrics.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def _initiate_promotion(
        self,
        tier: SecondaryTierManager,
        key: OffloadKey,
        req_context: ReqContext,
    ) -> bool:
        """
        Queue a block for promotion from a secondary tier to the primary tier.

        Allocates space in the primary tier immediately (sets ref_cnt=-1 so
        subsequent lookups within the same step see the slot as in-flight),
        then defers the actual submit_load() call to _flush_pending_promotions()
        so all blocks queued during one engine step are submitted as a single
        batched job.

        Args:
            tier: The secondary tier to promote from
            key: Block to promote
            req_context: Per-request context forwarded to primary.prepare_write().

        Returns:
            True if promotion was initiated, False if primary tier is full.
        """
        # Allocate space in primary tier for promoted block.
        # Must happen immediately so primary.lookup() returns None (in-flight)
        # for this key on any subsequent lookup() call within the same step,
        # preventing duplicate promotion attempts.
        primary_write_result = self.primary_tier.prepare_write([key], req_context)

        if primary_write_result is None:
            # Primary tier is full; caller should treat the block as unavailable
            # rather than retrying indefinitely.
            return False

        store_spec = primary_write_result.store_spec
        assert isinstance(store_spec, CPULoadStoreSpec)
        # Defer submit_load to on_schedule_end(). Group by (tier, request) so
        # each request's blocks are submitted as one batched job per tier.
        tier_pending = self._pending_load_submissions.setdefault(tier, {})
        ctx_id = req_context.req_id
        if ctx_id not in tier_pending:
            tier_pending[ctx_id] = PendingPromotion(
                keys=[], block_ids=[], req_context=req_context
            )
        entry = tier_pending[ctx_id]
        entry.keys.extend(primary_write_result.keys_to_store)
        entry.block_ids.extend(store_spec.block_ids)
        return True

    def _flush_pending_promotions(self) -> None:
        """Submit one batched submit_load() per (tier, request).

        Called from on_schedule_end() at the end of each scheduler step,
        flushing all promotion requests deferred during lookup().
        """
        if not self._pending_load_submissions:
            return

        for tier, pending_by_ctx in self._pending_load_submissions.items():
            for entry in pending_by_ctx.values():
                job_id = self._next_job_id()
                job_metadata = JobMetadata(
                    job_id=job_id,
                    keys=entry.keys,
                    block_ids=np.array(entry.block_ids, dtype=np.int64),
                    is_promotion=True,
                    req_context=entry.req_context,
                )
                self._transfer_jobs[job_id] = job_metadata
                tier.submit_load(job_metadata)

        self._pending_load_submissions.clear()

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        Prepare blocks to be loaded from primary tier to GPU.

        Callers only pass keys already confirmed HIT by lookup() earlier this
        step.

        This increments ref_cnt on the blocks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            keys: Blocks to prepare for loading.
            req_context: Per-request context.

        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        if self._writeback:
            self._note_writeback_use(keys)
        return self.primary_tier.prepare_load(keys, req_context)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used in all tiers.

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context.
        """
        self.primary_tier.touch(keys, req_context)
        for tier in self.secondary_tiers:
            tier.touch(keys, req_context)
        if self._writeback:
            self._note_writeback_use(keys, learn_prefix=True)

    @override
    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as done loading from primary tier to GPU.

        This decrements ref_cnt on the blocks in the primary tier, allowing
        them to be evicted again.

        Args:
            keys: Blocks that finished loading.
            req_context: Per-request context.
        """
        self.primary_tier.complete_load(keys, req_context)

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare blocks to be stored from GPU to primary tier.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed async transfers have their ref_cnt decremented
        before the primary tier makes eviction decisions.

        For request-level tiers, blocks already present in the primary tier
        are immediately cascaded via submit_store().

        Args:
            keys: Blocks to prepare for storing.
            req_context: Per-request context.

        Returns:
            PrepareStoreOutput describing where to store blocks and what was
            evicted, or None if store cannot proceed.
        """
        # Step 1: Poll for completed async jobs FIRST
        # _process_finished_jobs() handles two kinds of completions here:
        #  - Cascade completions (store to a secondary tier, either a local
        #    cascade or a store job created for a remote requester via
        #    create_store_job()): decrements ref_cnt on the primary blocks
        #    that were read, making them evictable again once ref_cnt hits 0.
        #  - Promotion completions (secondary->primary loads): sets a
        #    not-yet-ready block's ref_cnt from -1 to 0 via complete_write(),
        #    making it evictable for the first time.
        # Both must be accounted for before the eviction decision below.
        self._maybe_process_finished_jobs()

        # Step 2: Store to primary tier (new blocks only).
        # Cascading of these newly-stored blocks to ALL secondary tiers
        # happens later in complete_store(), after the GPU→Primary transfer
        # completes.
        primary_result = self.primary_tier.prepare_store(keys, req_context)

        if primary_result is None:
            return None

        if primary_result.keys_to_store:
            state = self._req_state[req_context.req_id]
            state.pending_primary_stores += 1

        # Step 3: For request-level tiers, cascade blocks already in primary
        request_level_tiers = self._req_state[req_context.req_id].request_level_tiers
        if request_level_tiers:
            keys_to_store_set = set(primary_result.keys_to_store)
            keys_already_in_primary = tuple(
                k for k in keys if k not in keys_to_store_set
            )
            if keys_already_in_primary:
                self._cascade_existing_blocks_to_request_level_tiers(
                    keys_already_in_primary, req_context, request_level_tiers
                )

        return primary_result

    def _cascade_existing_blocks_to_request_level_tiers(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        request_level_tiers: set[SecondaryTierManager],
    ) -> None:
        """
        For tiers that requested request-level policy, submit_store() for
        blocks that are already present in the primary tier.
        """
        # Filter out keys that are not ready in primary (e.g. in-flight)
        ready_keys = tuple(
            k
            for k in keys
            if self.primary_tier.lookup(k, req_context) is LookupResult.HIT
        )
        if not ready_keys:
            return

        for tier in request_level_tiers:
            self._submit_store_to_tier(tier, ready_keys, req_context)

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        """
        Mark blocks as done storing from GPU to primary tier.

        This is where secondary tier cascading happens — after blocks are
        confirmed to be in the primary tier, they are cascaded to ALL
        secondary tiers.

        For each secondary tier:
        1. Call primary.prepare_read() to get LoadStoreSpec AND increment
           ref_cnt (protecting blocks during async transfer)
        2. Call tier.submit_store() to start async transfer: primary→secondary
        3. Track the job in _store_jobs dictionary

        Args:
            keys: Blocks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
            req_context: Per-request context forwarded to primary.prepare_read().
        """
        # Step 1: Complete store in primary tier (makes blocks loadable)
        self.primary_tier.complete_store(keys, req_context, success)

        if success:
            # Step 2: Cascade to ALL secondary tiers
            # For each secondary tier, call primary.prepare_read() to get the
            # LoadStoreSpec AND to increment ref_cnt (protecting blocks from
            # eviction during the async transfer). One prepare_read() call per
            # secondary tier.
            for tier in self.secondary_tiers:
                wb = self._writeback.get(tier)
                if wb is None:
                    self._submit_store_to_tier(tier, keys, req_context)
                else:
                    self._mark_dirty(wb, keys)

        # Note: The async transfers are now in flight. Their completion is
        # tracked via get_finished_jobs() / _maybe_process_finished_jobs().
        req_id = req_context.req_id
        state = self._req_state[req_id]
        assert state.pending_primary_stores > 0
        state.pending_primary_stores -= 1
        self._maybe_finalize_request(req_id)

    def _submit_store_to_tier(
        self,
        tier: SecondaryTierManager,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> JobMetadata | None:
        """Pin ``keys`` in the primary tier and submit a store job to ``tier``,
        unless the tier declines the batch (accepts_store(), e.g. its write
        backlog is full or it is disabled). Returns the submitted job, or None
        if the batch was declined (and not pinned)."""
        if not tier.accepts_store(keys, req_context):
            return None
        job_metadata = self.create_store_job(keys, req_context)
        tier.submit_store(job_metadata)
        return job_metadata

    def create_store_job(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> JobMetadata:
        """Pin blocks in the primary tier and create a tracked store job.

        Calls prepare_read() to increment ref_cnt (protecting blocks
        from eviction during the async transfer), allocates a job ID,
        and registers the job in _transfer_jobs.

        The caller is responsible for the actual data transfer and
        reporting completion via get_finished_jobs().
        """
        primary_blocks_spec = self.primary_tier.prepare_read(keys, req_context)
        assert isinstance(primary_blocks_spec, CPULoadStoreSpec)
        job_id = self._next_job_id()
        job_metadata = JobMetadata(
            job_id=job_id,
            keys=keys,
            block_ids=primary_blocks_spec.block_ids,
            is_promotion=False,
            req_context=req_context,
        )
        self._transfer_jobs[job_id] = job_metadata
        return job_metadata

    # ------------------------------------------------------------------
    # Write-back (store_policy "write_back")
    # ------------------------------------------------------------------

    def _mark_dirty(self, wb: _WritebackState, keys: Collection[OffloadKey]) -> None:
        """Blocks just stored in the primary tier: remember the ones ``wb``'s
        tier lacks instead of cascading them."""
        get_block = self.primary_tier.get_block
        is_stored = wb.tier.is_stored
        for key in keys:
            block = get_block(key)
            if block is None or not block.is_ready or is_stored(key):
                continue
            wb.dirty.add(key)
            self._writeback_groups.setdefault(key[-4:])

    def _on_primary_evicted(self, keys: list[OffloadKey]) -> None:
        for wb in self._writeback.values():
            dirty = wb.dirty
            for key in keys:
                if key in dirty:
                    dirty.discard(key)
                    wb.n_lost += 1
        parent = self._writeback_parent
        for key in keys:
            parent.pop(key, None)

    def _note_writeback_use(
        self, keys: Collection[OffloadKey], learn_prefix: bool = False
    ) -> None:
        """Learn prefix links from a request's ordered keys (touch() gets one
        KV cache group's keys in chunk order), and note flushing blocks that
        are in use, so they are not demoted when their write finishes."""
        flushing = [wb for wb in self._writeback.values() if wb.flushing]
        if learn_prefix:
            parent = self._writeback_parent
            get_block = self.primary_tier.get_block
            prev = None
            for key in keys:
                if (
                    prev is not None
                    and key not in parent
                    and get_block(key) is not None
                ):
                    parent[key] = prev
                prev = key
            if len(parent) > 2 * self.primary_tier.num_blocks:
                parent.clear()  # links are re-learned on the next touches
        for wb in flushing:
            for key in keys:
                if key in wb.flushing:
                    wb.touched.add(key)

    def _flushable(
        self, wb: _WritebackState, key: OffloadKey, taken: set[OffloadKey]
    ) -> bool:
        if key not in wb.dirty or key in wb.flushing or key in taken:
            return False
        block = self.primary_tier.get_block(key)
        # ref_cnt -1: still being written from the GPU (or promoted); never
        # read it. Dirty blocks are always ready; this is a safety net.
        return block is not None and block.is_ready

    def _plan_writeback_unit(
        self, wb: _WritebackState, key: OffloadKey, taken: set[OffloadKey]
    ) -> list[list[OffloadKey]]:
        """``key`` plus the unwritten earlier chunks of its prefix (oldest
        first), each with its unwritten siblings of the other KV cache groups.

        A secondary-tier hit needs every chunk before it (prefix lookup stops
        at the first miss) and, for a hybrid model, the blocks of all groups
        at the hit boundary. The prefix walk stops at the first chunk that is
        already written, being written, or not in the primary tier."""
        parent = self._writeback_parent
        chain: list[OffloadKey] = []
        k: OffloadKey | None = key
        while k is not None and self._flushable(wb, k, taken):
            chain.append(k)
            taken.add(k)
            k = parent.get(k)
        chain.reverse()
        unit: list[list[OffloadKey]] = []
        groups = self._writeback_groups
        for k in chain:
            chunk = [k]
            block_hash = k[:-4]
            for group in groups:
                sibling = OffloadKey(block_hash + group)
                if sibling != k and self._flushable(wb, sibling, taken):
                    chunk.append(sibling)
                    taken.add(sibling)
            unit.append(chunk)
        return unit

    def _writeback_step(self, wb: _WritebackState) -> None:
        """Once per scheduler step. While the primary tier holds at least
        ``high_blocks`` blocks, write the coldest unwritten blocks (in the
        primary's eviction order) to ``wb.tier`` until at most ``low_blocks``
        are left unwritten, or the tier declines (store backlog cap, breaker).
        Eviction itself never waits: a block evicted before its write
        started is lost to the tier (counted in writeback_lost_blocks)."""
        primary = self.primary_tier
        if primary.num_used_blocks() < wb.high_blocks:
            return
        need = len(wb.dirty) - len(wb.flushing) - wb.low_blocks
        if need <= 0:
            return
        # Plan first: pinning below changes the evictable set being iterated.
        # Each planned chunk: (keys of the chunk's groups, picked as cold?).
        chunks: list[tuple[list[OffloadKey], bool]] = []
        taken: set[OffloadKey] = set()
        for key in primary.iter_evictable():
            if key not in wb.dirty or key in taken or key in wb.flushing:
                continue
            unit = self._plan_writeback_unit(wb, key, taken)
            for chunk in unit:
                chunks.append((chunk, chunk[0] == key))
                need -= len(chunk)
            if need <= 0:
                break
        # Submit oldest prefix chunks first, in jobs of about
        # _WRITEBACK_JOB_BLOCKS; a chunk's group siblings share a job.
        batch: list[OffloadKey] = []
        demote: list[OffloadKey] = []
        for chunk, is_cold in chunks:
            batch.extend(chunk)
            if is_cold:
                demote.extend(chunk)
            if len(batch) >= _WRITEBACK_JOB_BLOCKS:
                if not self._submit_writeback(wb, batch, demote):
                    return
                batch, demote = [], []
        if batch:
            self._submit_writeback(wb, batch, demote)

    def _submit_writeback(
        self, wb: _WritebackState, keys: list[OffloadKey], demote: list[OffloadKey]
    ) -> bool:
        job = self._submit_store_to_tier(wb.tier, keys, self._writeback_ctx)
        if job is None:
            return False
        self._writeback_jobs[job.job_id] = wb
        wb.flushing.update(keys)
        wb.demote.update(demote)
        return True

    def _finish_writeback_job(
        self,
        wb: _WritebackState,
        job_id: JobId,
        keys: Collection[OffloadKey],
        success: bool,
    ) -> None:
        """A write-back job finished (its blocks are already unpinned)."""
        primary = self.primary_tier
        to_demote: list[OffloadKey] = []
        for key in keys:
            wb.flushing.discard(key)
            stored = wb.tier.is_stored(key)
            if stored is None:
                stored = success
            if stored and key in wb.dirty:
                wb.dirty.discard(key)
                wb.n_flushed += 1
            used = key in wb.touched
            wb.touched.discard(key)
            if key in wb.demote:
                wb.demote.discard(key)
                block = primary.get_block(key)
                if not used and block is not None and block.ref_cnt == 0:
                    to_demote.append(key)
        # Unpinning made them most recently used; put the cold blocks back at
        # the LRU end so the next evictions take written blocks first.
        if to_demote:
            primary.demote(to_demote)

    @override
    def on_new_request(
        self,
        req_context: ReqContext,
        *,
        exclude_tier: SecondaryTierManager | None = None,
    ) -> RequestOffloadingContext:
        """
        Query each secondary tier for its offload policy preference.

        Returns REQUEST_LEVEL if ANY secondary tier wants request-level.
        Only stores REQUEST_LEVEL tier decisions for use in prepare_store.
        """
        state = RequestState(req_context=req_context)
        for tier in self.secondary_tiers:
            if tier is exclude_tier:
                continue
            tier_ctx = tier.on_new_request(req_context)
            if tier_ctx.policy == OffloadPolicy.REQUEST_LEVEL:
                if state.request_level_tiers is None:
                    state.request_level_tiers = set()
                state.request_level_tiers.add(tier)
        self._req_state[req_context.req_id] = state

        policy = (
            OffloadPolicy.REQUEST_LEVEL
            if state.request_level_tiers
            else OffloadPolicy.BLOCK_LEVEL
        )
        return RequestOffloadingContext(policy=policy)

    @override
    def on_request_finished(
        self,
        req_context: ReqContext,
        *,
        exclude_tier: SecondaryTierManager | None = None,
    ) -> None:
        self.primary_tier.on_request_finished(req_context)
        state = self._req_state[req_context.req_id]
        state.is_finished = True
        self._maybe_finalize_request(req_context.req_id, exclude_tier)

    def _maybe_finalize_request(
        self,
        req_id: str,
        exclude_tier: SecondaryTierManager | None = None,
    ) -> None:
        """Finalize secondary tiers once no more store cascades can be submitted.

        Finalization means forwarding on_request_finished() to secondary tiers.
        It is delayed until pending GPU->primary stores finish, since their
        complete_store() callbacks may still submit primary->secondary stores.
        """
        state = self._req_state[req_id]
        if not state.is_finished:
            return
        if state.pending_primary_stores != 0:
            return

        for tier in self.secondary_tiers:
            if tier is exclude_tier:
                continue
            tier.on_request_finished(state.req_context)
        self._maybe_observe_lookup_sync_delay(state)
        self._maybe_observe_lookup_async_delay(state)
        del self._req_state[req_id]

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        """End-of-schedule hook: process finished jobs, flush deferred
        promotions, and reset the per-step gate.

        Called once per scheduler step from
        OffloadingConnectorScheduler.build_connector_meta().
        """
        # Catch-all poll: guarantees jobs are processed even on steps where
        # lookup()/prepare_store() were never called (e.g. no requests
        # scheduled but a tier still has_pending_work()).
        self._maybe_process_finished_jobs()

        for tier in self.secondary_tiers:
            tier.serve_external_requests(self._tier_parents[tier])

        # Reset the per-step gate AFTER serve_external_requests so that
        # lookup() calls within it skip redundant _process_finished_jobs().
        self._processed_jobs_this_step = False

        self._flush_pending_promotions()
        for wb in self._writeback.values():
            self._writeback_step(wb)
        for tier in self.secondary_tiers:
            tier.on_schedule_end(context)

        for req_id in context.new_req_ids:
            state = self._req_state.get(req_id)
            if state is None:
                continue
            self._maybe_observe_lookup_sync_delay(state)
            self._maybe_observe_lookup_async_delay(state)

    @override
    def has_pending_work(self) -> bool:
        # In-flight primary<->secondary transfers (pending promotions are
        # translated to transfer jobs in on_schedule_end), plus any work the
        # secondary tiers themselves still have outstanding.
        return bool(self._transfer_jobs) or any(
            tier.has_pending_work() for tier in self.secondary_tiers
        )

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        """Yield events owned by the primary and secondary tiers.

        Yields:
            New OffloadingEvents collected by each tier since the last call.
        """
        yield from self.primary_tier.take_events()
        for tier in self.secondary_tiers:
            yield from tier.take_events()

    @override
    def reset_cache(self) -> None:
        """Reset transfer bookkeeping and primary-tier cache.

        Called during sleep, weight update, or resume. Each secondary tier
        drains its in-flight transfers via drain_jobs() so no tier I/O is
        touching primary memory before the primary tier is reset. A stuck
        tier will block here visibly — preferable to silent corruption
        from reusing primary slots while a transfer is mid-copy.

        Secondary tiers are intentionally not reset: persistent stores
        (FS, network) keep their data across resets. Active request state is
        retained so those requests can continue after the reset; finished
        requests are finalized and removed.
        """
        for tier in self.secondary_tiers:
            tier.drain_jobs()
        # All tier I/O has stopped; consume their completion notifications
        # so manager bookkeeping is consistent before the primary reset.
        self._process_finished_jobs()

        # Deferred promotion submissions reserve primary slots that the
        # reset below invalidates; their submit_load() has not yet been
        # called so no tier I/O is touching that memory.
        self._pending_load_submissions.clear()

        finished_req_ids = []
        for req_id, state in self._req_state.items():
            state.pending_primary_stores = 0
            if not state.is_finished:
                continue
            for tier in self.secondary_tiers:
                tier.on_request_finished(state.req_context)
            self._maybe_observe_lookup_sync_delay(state)
            self._maybe_observe_lookup_async_delay(state)
            finished_req_ids.append(req_id)

        self.primary_tier.reset_cache()
        for wb in self._writeback.values():
            wb.n_lost += len(wb.dirty)  # unwritten blocks dropped with the cache
            wb.dirty.clear()
            wb.flushing.clear()
            wb.demote.clear()
            wb.touched.clear()
        self._writeback_jobs.clear()
        self._writeback_parent.clear()

        for req_id in finished_req_ids:
            del self._req_state[req_id]
        self._processed_jobs_this_step = False

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        if self._writeback:
            m = TieringOffloadingMetrics
            n_dirty = 0
            for wb in self._writeback.values():
                n_dirty += len(wb.dirty)
                if wb.n_flushed:
                    self._stats.increase_counter(m.WRITEBACK_FLUSHED, wb.n_flushed)
                    wb.n_flushed = 0
                if wb.n_lost:
                    self._stats.increase_counter(m.WRITEBACK_LOST, wb.n_lost)
                    wb.n_lost = 0
            self._stats.set_gauge(m.WRITEBACK_DIRTY, n_dirty)

        stats = self.primary_tier.get_stats()

        if stats is not None and stats.is_empty():
            stats = None

        for tier in self.secondary_tiers:
            tier_stats = tier.get_stats()
            if tier_stats is None or tier_stats.is_empty():
                continue
            if stats is None:
                stats = tier_stats
            else:
                stats.aggregate(tier_stats)

        if not self._stats.is_empty():
            if stats is None:
                stats = self._stats
            else:
                stats.aggregate(self._stats)
            self._stats = OffloadingConnectorStats()

        return stats

    @override
    def shutdown(self) -> None:
        """Shutdown all tiers and release resources."""
        for tier in self.secondary_tiers:
            tier.shutdown()
        self.primary_tier.shutdown()
