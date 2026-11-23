# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""DeepSeek V4.1 request-owned slots and worker storage for the single-pool experiment."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, fields

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import get_storage_block_size
from vllm_ascend.models.deepseek_v41.compact_cache import SinglePoolWindowSpec
from vllm_ascend.worker.v2 import attn_utils


@dataclass(eq=False)
class RequestStateLease:
    owner: str
    history: list
    window: int
    compressor: int
    pending: set = field(default_factory=set)
    retired: bool = False


class RequestStateLeases:
    """History IDs come from vLLM. Fixed slots live in two separate index spaces."""

    def __init__(self, history: BlockPool, windows: int, compressors: int):
        if min(windows, compressors) < 1:
            raise ValueError("state slot counts must be positive")
        self.history = history
        self.windows = deque(range(1, windows + 1))
        self.compressors = deque(range(1, compressors + 1))
        self.live = {}

    def admit(self, owner, pages):
        if owner in self.live:
            raise ValueError("duplicate request")
        if pages <= 0:
            raise ValueError("invalid history length")
        if not self.windows or not self.compressors or self.history.get_num_free_blocks() < pages:
            return None
        blocks = self.history.get_new_blocks(pages)
        lease = RequestStateLease(owner, blocks, self.windows.popleft(), self.compressors.popleft())
        self.live[owner] = lease
        return lease

    def check(self, lease, *, allow_retired=False):
        if self.live.get(lease.owner) is not lease or (lease.retired and not allow_retired):
            raise ValueError("stale or retired request")

    def extend(self, lease, extra):
        self.check(lease)
        if extra <= 0:
            raise ValueError("invalid growth")
        if self.history.get_num_free_blocks() < extra:
            return False
        lease.history.extend(self.history.get_new_blocks(extra))
        return True

    def begin_read(self, lease):
        self.check(lease)
        ticket = object()
        lease.pending.add(ticket)
        return ticket, ([b.block_id for b in lease.history], [lease.window], [lease.compressor])

    def complete(self, lease, ticket):
        self.check(lease, allow_retired=True)
        if ticket not in lease.pending:
            raise ValueError("unmatched completion")
        lease.pending.remove(ticket)
        self._release_if_ready(lease)

    def retire(self, lease):
        self.check(lease)
        lease.retired = True
        self._release_if_ready(lease)

    def _release_if_ready(self, lease):
        if lease.retired and not lease.pending:
            self.history.free_blocks(reversed(lease.history))
            self.windows.append(lease.window)
            self.compressors.append(lease.compressor)
            del self.live[lease.owner]


def layer_groups(specs):
    history, windows, compressors = {}, {}, {}
    for name, spec in sorted(specs.items()):
        if name.endswith(".swa_cache"):
            windows[name] = SinglePoolWindowSpec(**{f.name: getattr(spec, f.name) for f in fields(spec) if f.init})
        elif name.endswith(".compressor.state_cache"):
            compressors[name] = spec
        else:
            history[name] = spec
    if (len(history), len(windows), len(compressors)) != (8, 40, 3):
        raise ValueError("expected Flash history, SWA, and compressor layer counts of 8, 40, and 3")
    result = []
    for members in (history, windows, compressors):
        uniform = UniformTypeKVCacheSpecs.from_specs(members)
        if uniform is None:
            raise ValueError("incompatible specs in request group")
        result.append(KVCacheGroupSpec(layer_names=list(members), kv_cache_spec=uniform))
    return result


class RequestStateStorage:
    """Own one slab. Publish only leased group-local IDs to the registration views."""

    def __init__(self, specs, history: int, windows: int, compressors: int, allocate: Callable[[int], torch.Tensor]):
        if not 1 <= min(windows, compressors) <= max(windows, compressors) <= history:
            raise ValueError("probe requires positive state capacities within the history ID range")
        groups = layer_groups(specs)
        self.slots = (history + 1, windows + 1, compressors + 1)
        self.layout = {}
        size = 0
        # State first; existing physical history storage covers the unused tail
        # of the stock registration views. Each compute component owns its range.
        for group in (1, 2, 0):
            for name, spec in groups[group].kv_cache_spec.kv_cache_specs.items():
                row_bytes = spec.unpadded_page_size_bytes
                self.layout[name] = (group, spec, size, row_bytes)
                size += self.slots[group] * row_bytes
        if any(start + self.slots[0] * stride > size for _, _, start, stride in self.layout.values()):
            raise ValueError("a registration view would exceed the allocated slab")
        descriptors = [
            KVCacheTensor(size=size, layers=[name], offset=start, block_stride=stride, layer_stride=0)
            for name, (_, _, start, stride) in self.layout.items()
        ]
        self.cfg = KVCacheConfig(num_blocks=self.slots[0], kv_cache_groups=groups, kv_cache_tensors=descriptors)
        self.slab = allocate(size)
        if self.slab.dtype != torch.int8 or self.slab.shape != (size,) or not self.slab.is_contiguous():
            raise ValueError("allocator must return a contiguous int8 slab of the requested size")
        self.compute = {}
        self.registration = {}
        for name, (group, spec, start, stride) in self.layout.items():
            rows = get_storage_block_size(spec)
            shape = (self.slots[0], rows, spec.num_kv_heads, spec.head_size)
            shapes, dtypes = [shape], [spec.dtype]
            if name.endswith(".indexer.k_cache"):
                shapes.append((*shape[:-1], spec.scale_dim))
                dtypes.append(spec.scale_dtype)
            views = attn_utils._adjust_dsv4_kv_layout(self.slab, shapes, dtypes, stride, initial_offset_bytes=start)
            bounded = [v[: self.slots[group]] for v in views]
            self.compute[name] = tuple(bounded) if len(bounded) > 1 else bounded[0]
            self.registration[name] = tuple(views) if len(views) > 1 else views[0]

    def zero_slots(self, group: int, slot_ids: list[int]) -> None:
        """Clear newly leased slots in one group before publishing them to a worker.

        Callers supply host-side IDs after any preceding use of those slots is complete.
        The registration views overlap other layers and must never be used for zeroing.
        """
        if not 0 <= group < len(self.slots):
            raise ValueError("invalid request-state group")
        if not slot_ids:
            return
        if min(slot_ids) < 1 or max(slot_ids) >= self.slots[group]:
            raise ValueError("slot outside the group's usable capacity")
        indices = torch.tensor(slot_ids, device=self.slab.device, dtype=torch.long)
        for owner, _, start, stride in self.layout.values():
            if owner == group:
                length = self.slots[group] * stride
                self.slab.narrow(0, start, length).view(self.slots[group], stride).index_fill_(0, indices, 0)
