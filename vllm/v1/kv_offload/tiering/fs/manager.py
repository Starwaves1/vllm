# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)

Bounded mode (``max_bytes`` set):
    The tier keeps an LRU index of the block files it owns, in the scheduler
    thread only (worker threads just do I/O). Stores reserve space up front
    and evict least-recently-used files that no in-flight load is reading;
    lookups, loads and touches refresh recency. Lookups are answered from the
    index synchronously, so a disk hit starts its promotion in the same step.
    The directory ``<base_path>_r<rank>`` must be owned by one engine.
"""

import functools
import json
import os
import time
from collections import Counter, OrderedDict
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FsTierMetrics:
    """Metric names emitted by FileSystemTierManager."""

    CACHE_BYTES = "vllm:kv_offload_fs_cache_bytes"
    CACHE_BLOCKS = "vllm:kv_offload_fs_cache_blocks"
    LOAD_BYTES = "vllm:kv_offload_fs_load_bytes"
    LOAD_TIME = "vllm:kv_offload_fs_load_time"
    LOAD_FAILURES = "vllm:kv_offload_fs_load_failures"
    STORE_BYTES = "vllm:kv_offload_fs_store_bytes"
    STORE_TIME = "vllm:kv_offload_fs_store_time"
    EVICTED_BYTES = "vllm:kv_offload_fs_evicted_bytes"
    STORES_SKIPPED = "vllm:kv_offload_fs_stores_skipped"


def _parse_max_bytes(max_bytes: Any) -> int | None:
    if max_bytes is None:
        return None
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int | float | str):
        raise TypeError(f"max_bytes must be a non-negative integer, got {max_bytes!r}")
    value = int(float(max_bytes))
    if value < 0:
        raise ValueError(f"max_bytes must be a non-negative integer, got {max_bytes!r}")
    return value


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return (os.path.exists(p) for p in paths)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content. Bounded mode (``max_bytes``) requires one owner per
        directory, so it is not meant for sharing.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        m = FsTierMetrics
        return {
            m.CACHE_BYTES: OffloadingGaugeMetadata(
                documentation="Block-file bytes held by a bounded fs KV tier."
            ),
            m.CACHE_BLOCKS: OffloadingGaugeMetadata(
                documentation="Block files held by a bounded fs KV tier."
            ),
            m.LOAD_BYTES: OffloadingCounterMetadata(
                documentation="Bytes promoted from the fs tier to the CPU tier."
            ),
            m.LOAD_TIME: OffloadingCounterMetadata(
                documentation="Seconds spent in fs-tier load (promotion) jobs."
            ),
            m.LOAD_FAILURES: OffloadingCounterMetadata(
                documentation="fs-tier load jobs that failed (blocks recomputed)."
            ),
            m.STORE_BYTES: OffloadingCounterMetadata(
                documentation="Bytes submitted for writing to the fs tier."
            ),
            m.STORE_TIME: OffloadingCounterMetadata(
                documentation="Seconds spent in fs-tier store jobs."
            ),
            m.EVICTED_BYTES: OffloadingCounterMetadata(
                documentation="Bytes evicted from a bounded fs tier (LRU)."
            ),
            m.STORES_SKIPPED: OffloadingCounterMetadata(
                documentation=(
                    "Blocks a bounded fs tier did not write because no space "
                    "could be freed (every file pinned by an in-flight load)."
                )
            ),
        }

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        max_bytes: int | None = None,
    ):
        """
        Args:
            offloading_spec: Contains normalized offloading configuration and
                blocks_per_chunk.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
            max_bytes: Cap on block-file bytes (LRU eviction). None keeps the
                historical unbounded, write-through-forever behavior.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.locality = Locality(locality) if locality is not None else None
        self._max_bytes = _parse_max_bytes(max_bytes)

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Keys of in-flight load (promotion) jobs, so a failed load can mark
        # its cached lookup verdicts False (else the request livelocks).
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Bytes submitted per in-flight store job (for the store counter).
        self._store_job_bytes: dict[JobId, int] = {}
        # Per-job I/O time, written by the pool worker before the job is
        # published as finished and read on the scheduler thread afterwards.
        self._job_io_time: dict[JobId, float] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        # Counters, reported (and reset) by get_stats().
        self._n_load_bytes = 0
        self._n_load_time = 0.0
        self._n_load_failures = 0
        self._n_store_bytes = 0
        self._n_store_time = 0.0
        self._n_evicted_bytes = 0
        self._n_stores_skipped = 0

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

        # Bounded-mode accounting. Scheduler thread only.
        self._entries: OrderedDict[OffloadKey, int] = OrderedDict()  # LRU first
        self._cache_bytes = 0
        self._reserved_bytes = 0
        self._pinned: Counter[OffloadKey] = Counter()  # in-flight loads
        self._writing: set[OffloadKey] = set()  # in-flight stores
        self._store_job_writes: dict[JobId, list[OffloadKey]] = {}
        if self._max_bytes is not None:
            self._storage_dir = f"{self.file_mapper.base_path}_r{self.file_mapper.rank}"
            self._scan_existing()
            excess = self._cache_bytes - self._max_bytes
            if excess > 0:
                self._evict(excess)
            logger.info(
                "fs KV tier '%s' bounded to %.1f GB at %s: %d block files "
                "(%.1f GB) indexed at startup",
                tier_type,
                self._max_bytes / 1e9,
                self._storage_dir,
                len(self._entries),
                self._cache_bytes / 1e9,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

    # ------------------------------------------------------------------
    # Bounded-mode helpers (scheduler thread)
    # ------------------------------------------------------------------

    def _scan_existing(self) -> None:
        """Index block files left by a previous run, oldest mtime first, and
        delete temp files orphaned by a crash mid-write."""
        found: list[tuple[int, OffloadKey, int]] = []
        if not os.path.isdir(self._storage_dir):
            return
        for dirpath, _, filenames in os.walk(self._storage_dir):
            for name in filenames:
                path = os.path.join(dirpath, name)
                if name.endswith(".tmp"):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    continue
                if not name.endswith(".bin"):
                    continue
                try:
                    group_idx = int(os.path.basename(dirpath).rsplit("_g", 1)[1])
                    key = make_offload_key(bytes.fromhex(name[:-4]), group_idx)
                    st = os.stat(path)
                except (IndexError, ValueError, OSError):
                    continue
                if self.file_mapper.get_file_name(key) != path:
                    continue
                found.append((st.st_mtime_ns, key, st.st_size))
        found.sort(key=lambda t: t[0])
        for _, key, size in found:
            self._entries[key] = size
            self._cache_bytes += size

    def _evict(self, nbytes: int) -> int:
        """Delete least-recently-used files not pinned by an in-flight load
        until ``nbytes`` are freed (or nothing evictable is left)."""
        victims: list[OffloadKey] = []
        planned = 0
        for key, size in self._entries.items():
            if planned >= nbytes:
                break
            if key in self._pinned:
                continue
            victims.append(key)
            planned += size
        freed = 0
        evicted: list[OffloadKey] = []
        for key in victims:
            size = self._entries.pop(key)
            try:
                os.remove(self.file_mapper.get_file_name(key))
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("fs KV tier: failed to evict block file: %s", exc)
                self._entries[key] = size  # keep it accounted, now MRU
                continue
            self._cache_bytes -= size
            freed += size
            evicted.append(key)
        if evicted:
            self._n_evicted_bytes += freed
            # A request may hold a cached True probe verdict for these keys.
            self._lookup_manager.mark_miss(evicted)
            if self.events is not None:
                self.events.append(
                    OffloadingEvent(
                        keys=evicted,
                        medium=self.medium,
                        removed=True,
                        locality=self.locality,
                    )
                )
        return freed

    def _admit_store(
        self, job_id: JobId, keys: list[OffloadKey], block_ids: np.ndarray
    ) -> tuple[list[OffloadKey], list[int]]:
        """Filter a store job down to blocks not yet on disk and reserve space
        for them, evicting LRU files as needed."""
        assert self._max_bytes is not None
        new_keys: list[OffloadKey] = []
        new_bids: list[int] = []
        for key, bid in zip(keys, block_ids, strict=True):
            if key in self._entries:
                self._entries.move_to_end(key)
                continue
            if key in self._writing:
                continue
            new_keys.append(key)
            new_bids.append(int(bid))
        bs = self._block_size
        excess = self._cache_bytes + self._reserved_bytes + len(new_keys) * bs
        excess -= self._max_bytes
        if excess > 0:
            self._evict(excess)
            excess = self._cache_bytes + self._reserved_bytes + len(new_keys) * bs
            excess -= self._max_bytes
            if excess > 0:
                n_keep = max(0, len(new_keys) - (excess + bs - 1) // bs)
                self._n_stores_skipped += len(new_keys) - n_keep
                del new_keys[n_keep:]
                del new_bids[n_keep:]
        self._reserved_bytes += len(new_keys) * bs
        self._writing.update(new_keys)
        self._store_job_writes[job_id] = new_keys
        return new_keys, new_bids

    def _finish_store(self, job_id: JobId, success: bool) -> None:
        written = self._store_job_writes.pop(job_id, None)
        if written is None:
            return
        bs = self._block_size
        self._reserved_bytes -= len(written) * bs
        self._writing.difference_update(written)
        for key in written:
            # A failed batch stops at the first bad block; earlier ones landed.
            if not success and not os.path.exists(self.file_mapper.get_file_name(key)):
                continue
            if key not in self._entries:
                self._cache_bytes += bs
            self._entries[key] = bs
            self._entries.move_to_end(key)

    # ------------------------------------------------------------------
    # I/O tasks (pool worker threads)
    # ------------------------------------------------------------------

    def _run_io(self, job_id: JobId, fn, paths: list[str], offsets: list[int]):
        t0 = time.perf_counter()
        try:
            if paths:
                fn(
                    paths,
                    self._primary_kv_view,
                    offsets,
                    self._block_size,
                    self._use_o_direct,
                )
        finally:
            self._job_io_time[job_id] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # SecondaryTierManager API (scheduler thread)
    # ------------------------------------------------------------------

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if self._max_bytes is not None:
            # The index is authoritative in bounded mode: answer now instead
            # of a RETRY round trip through the async prober.
            if key in self._entries:
                self._entries.move_to_end(key)
                return LookupResult.HIT
            return LookupResult.MISS
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids: Collection[int] = job_metadata.block_ids
        if self._max_bytes is not None:
            keys, block_ids = self._admit_store(job_id, keys, job_metadata.block_ids)
        if self.events is not None:
            self._store_job_keys[job_id] = keys
        self._store_job_bytes[job_id] = len(keys) * self._block_size
        task = functools.partial(
            self._run_io,
            job_id,
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in keys],
            [int(bid) * self._block_size for bid in block_ids],
        )
        self._pool.enqueue_store(job_id, 1, [task])

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        if self._max_bytes is not None:
            for key in keys:
                self._pinned[key] += 1
                if key in self._entries:
                    self._entries.move_to_end(key)
        task = functools.partial(
            self._run_io,
            job_id,
            batch_load_block,
            [self.file_mapper.get_file_name(key) for key in keys],
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
        )
        self._pool.enqueue_load(job_id, 1, [task])

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        for job_id, success in self._pool.get_finished():
            io_time = self._job_io_time.pop(job_id, 0.0)
            load_keys = self._load_job_keys.pop(job_id, None)
            if load_keys is not None:
                if self._max_bytes is not None:
                    for key in load_keys:
                        self._pinned[key] -= 1
                        if self._pinned[key] <= 0:
                            del self._pinned[key]
                if success:
                    self._n_load_bytes += len(load_keys) * self._block_size
                    self._n_load_time += io_time
                else:
                    # The whole promotion job fails (0.27.1 has no partial
                    # keep). Force a miss so the scheduler recomputes instead
                    # of re-promoting an unreadable block forever.
                    self._n_load_failures += 1
                    self._lookup_manager.mark_miss(load_keys)
                    if self._max_bytes is not None:
                        for key in load_keys:
                            path = self.file_mapper.get_file_name(key)
                            if key in self._entries and not os.path.exists(path):
                                self._cache_bytes -= self._entries.pop(key)
            else:
                self._n_store_bytes += self._store_job_bytes.pop(job_id, 0)
                self._n_store_time += io_time
                if self._max_bytes is not None:
                    self._finish_store(job_id, success)
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        if self._max_bytes is None:
            return
        entries = self._entries
        for key in keys:
            if key in entries:
                entries.move_to_end(key)

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()
        m = FsTierMetrics
        if self._max_bytes is not None:
            stats.set_gauge(m.CACHE_BYTES, self._cache_bytes)
            stats.set_gauge(m.CACHE_BLOCKS, len(self._entries))
        for name, attr in (
            (m.LOAD_BYTES, "_n_load_bytes"),
            (m.LOAD_TIME, "_n_load_time"),
            (m.LOAD_FAILURES, "_n_load_failures"),
            (m.STORE_BYTES, "_n_store_bytes"),
            (m.STORE_TIME, "_n_store_time"),
            (m.EVICTED_BYTES, "_n_evicted_bytes"),
            (m.STORES_SKIPPED, "_n_stores_skipped"),
        ):
            value = getattr(self, attr)
            if value:
                stats.increase_counter(name, value)
                setattr(self, attr, type(value)(0))
        return None if stats.is_empty() else stats

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
