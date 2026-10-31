# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 history/request-state separation on the pinned runtime.

No MTP or prefix reuse. DSA CP stages local query windows from replicated
global KV; graph decode uses persistent lengths. Allocation hooks bridge the
pinned upstream API; the PD data protocol belongs to the connector adapter.
Replacing these hooks with the upstream builder/coordinator extension points
is a separate integration step, not a requirement to rewrite PD transport.
"""

import copy
import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec

WINDOW = 128
ROW_BYTES = 1024
_installed = False
_attention_installed = False
_workspace = {}
_direct_decode = os.environ.get("MENGZI_STATE_DECODE_DIRECT", "0") == "1"
_observed_paths = set()


def record_path(path):
    if path not in _observed_paths:
        _observed_paths.add(path)
        target = Path(os.environ["MENGZI_RESULTS_DIR"], f"state-paths-{os.getpid()}.json")
        target.write_text(json.dumps({"paths": sorted(_observed_paths), "single_token_direct_enabled": _direct_decode}))


@dataclass(frozen=True, kw_only=True)
class RequestWindowSpec(AscendSlidingWindowMLASpec):
    @property
    def uses_slot_mapping(self):
        return False

    @property
    def block_table_token_alignment(self):
        return None

    def max_num_blocks_per_req(self, vllm_config, max_len):
        return 1

    def max_admission_blocks_per_request(self, **kwargs):
        return 1

    def max_memory_usage_bytes(self, vllm_config):
        return self.page_size_bytes

    @property
    def prefix_cacheable(self):
        return False


def unwrap(spec):
    return spec.kv_cache_specs if isinstance(spec, UniformTypeKVCacheSpecs) else {"representative": spec}


def classify(spec):
    values = list(unwrap(spec).values())
    if all(isinstance(x, RequestWindowSpec) for x in values):
        return "swa"
    if all(isinstance(x, CircularBufferSpec) for x in values):
        return "ring"
    return "history"


def group_cache_specs(specs):
    groups = [dict(), dict(), dict()]
    for name, spec in specs.items():
        spec = replace(spec, page_size_padded=None)
        if isinstance(spec, AscendSlidingWindowMLASpec):
            assert spec.block_size == WINDOW and spec.sliding_window == WINDOW and spec.dtype == torch.bfloat16
            attrs = {f.name: getattr(spec, f.name) for f in fields(spec)}
            spec = RequestWindowSpec(**attrs)
            groups[1][name] = spec
        elif isinstance(spec, CircularBufferSpec):
            groups[2][name] = spec
        else:
            assert name.endswith((".long_kv_cache", ".indexer.k_cache")), name
            groups[0][name] = spec
    assert [len(g) for g in groups] == [8, 40, 3]
    result = [UniformTypeKVCacheSpecs.from_specs(g) for g in groups]
    assert all(g is not None for g in result)
    return result


def all_specs(groups):
    return {n: s for g in groups for n, s in unwrap(g.kv_cache_spec).items()}


def history_bytes(groups):
    return sum(
        spec.unpadded_page_size_bytes
        for group in groups
        if classify(group.kv_cache_spec) == "history"
        for spec in unwrap(group.kv_cache_spec).values()
    )


def make_config(vllm_config, groups, available_memory):
    assert vllm_config.speculative_config is None
    if vllm_config.kv_transfer_config is not None:
        from vllm import envs

        from vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state import validate_pd_config

        validate_pd_config(vllm_config)
        if envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("State PD initially requires Model Runner V1")
    assert not vllm_config.cache_config.enable_prefix_caching
    assert vllm_config.parallel_config.pipeline_parallel_size == 1
    profile_budget = available_memory
    # The framework/CLI-provided budget is authoritative. No historical
    # experiment file or diagnostic environment can change admission capacity.
    maximum = vllm_config.scheduler_config.max_num_seqs
    batch = vllm_config.scheduler_config.max_num_batched_tokens
    capacity = maximum + 1
    specs = all_specs(groups)
    page = history_bytes(groups)
    fixed = sum(
        capacity * spec.unpadded_page_size_bytes
        for spec in specs.values()
        if isinstance(spec, (RequestWindowSpec, CircularBufferSpec))
    )
    # One shared, main-stream SWA staging arena plus metadata allowance.
    staging_rows = ((batch + maximum * 256 + 127) // 128) * 128
    staging = staging_rows * ROW_BYTES + 2**20
    usable = available_memory - fixed - staging
    assert usable > page
    count = usable // page
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        assert override <= count, "Explicit override exceeds the state-separated byte budget"
        count = override
    tensors = []
    for name, spec in specs.items():
        n = capacity if isinstance(spec, (RequestWindowSpec, CircularBufferSpec)) else count
        stride = spec.unpadded_page_size_bytes
        tensors.append(KVCacheTensor(size=n * stride, layers=[name], offset=0, layer_stride=0, block_stride=stride))
    result = KVCacheConfig(
        num_blocks=count,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
    result.request_state_capacity = capacity
    result.state_staging_rows = staging_rows
    result.state_staging_allowance_bytes = staging
    result.request_state_bytes = fixed
    result.history_page_bytes = page
    assert sum(x.size for x in tensors) + staging <= available_memory
    folder = Path(os.environ.get("MENGZI_RESULTS_DIR", "/results/state"))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"state-plan-{os.getpid()}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "dp_rank": vllm_config.parallel_config.data_parallel_rank,
                "available_memory": available_memory,
                "history_blocks": count,
                "history_bytes_per_block": page,
                "profile_or_explicit_budget": profile_budget,
                "budget_source": "framework available_memory / explicit CLI budget",
                "state_capacity_including_null": capacity,
                "state_bytes": fixed,
                "staging_allowance_bytes": staging,
                "backing_bytes": sum(x.size for x in tensors),
                "state_group_counts": [len(g.layer_names) for g in groups],
                "extra_full_state_copy_pool": False,
                "allocation_contract": "configured_max_num_seqs_not_current_requests",
            },
            indent=2,
        )
    )
    return result


def cache_rows(config, name):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state import is_state_config

    if not is_state_config(config):
        return config.num_blocks
    if name.endswith((".swa_cache", ".compressor.state_cache")):
        for descriptor in config.kv_cache_tensors:
            if name in descriptor.layers:
                assert descriptor.size % descriptor.block_stride == 0
                return descriptor.size // descriptor.block_stride
        raise ValueError(f"Missing state allocation for {name}")
    return config.num_blocks


def index_cache_offset(config, source_spec):
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state import is_state_config

    return 0 if is_state_config(config) else source_spec.unpadded_page_size_bytes


class FixedStateManager(CircularBufferManager):
    def get_num_blocks_to_allocate(self, request_id, *args, **kwargs):
        if self.req_to_blocks.get(request_id):
            return 0
        # Fixed slots are checked explicitly by allocate_slots below. They
        # must not be counted as history pages in the scalar history gate.
        return 0


def bind_platform_state_pools(coordinator, config):
    # AscendHybridKVCacheCoordinator constructs managers directly and can bypass
    # the upstream base initializer. Identify the materialized state allocation,
    # then bind its pools after the actual platform factory has returned.
    state_descriptors = [
        d for d in config.kv_cache_tensors if len(d.layers) == 1 and d.layers[0].endswith(".swa_cache")
    ]
    if len(state_descriptors) != 40:
        return False
    before = [type(m).__name__ for m in coordinator.single_type_managers]
    managers = []
    for old, group in zip(coordinator.single_type_managers, config.kv_cache_groups):
        names = group.layer_names
        bounded = all(n.endswith(".swa_cache") for n in names) or all(
            n.endswith(".compressor.state_cache") for n in names
        )
        if not bounded or isinstance(old, FixedStateManager):
            managers.append(old)
            continue
        capacity = cache_rows(config, names[0])
        pool = BlockPool(capacity, False, coordinator.scheduler_block_size)
        spec = next(iter(unwrap(group.kv_cache_spec).values()))
        manager = FixedStateManager(
            spec,
            block_pool=pool,
            enable_caching=False,
            kv_cache_group_id=old.kv_cache_group_id,
            scheduler_block_size=coordinator.scheduler_block_size,
            dcp_world_size=1,
            pcp_world_size=1,
            needs_kv_cache_zeroing=False,
        )
        managers.append(manager)
    coordinator.single_type_managers = tuple(managers)
    coordinator.group_block_sizes = tuple(m.block_size for m in managers)
    assert len({id(m.block_pool) for m in managers}) == 3
    data = {
        "coordinator_class": type(coordinator).__name__,
        "manager_classes_before": before,
        "manager_classes_after": [type(m).__name__ for m in managers],
        "pool_free_blocks": [m.block_pool.get_num_free_blocks() for m in managers],
        "independent_pool_count": 3,
    }
    Path(os.environ["MENGZI_RESULTS_DIR"], f"state-coordinator-{os.getpid()}.json").write_text(
        json.dumps(data, indent=2)
    )
    return True


def install_coordinator():
    from vllm.v1.core import kv_cache_coordinator as mod

    original = mod.KVCacheCoordinator.__init__

    def initialize(self, *args, **kwargs):
        incoming = kwargs.get("kv_cache_config", args[0] if args else None)
        state_config = incoming is not None and any(
            classify(g.kv_cache_spec) == "swa" for g in incoming.kv_cache_groups
        )
        if state_config:
            # Managers need one representative retention spec per logical group;
            # worker-side descriptors retain every layer's actual geometry.
            incoming = copy.copy(incoming)
            incoming.kv_cache_groups = [
                replace(g, kv_cache_spec=next(iter(unwrap(g.kv_cache_spec).values()))) for g in incoming.kv_cache_groups
            ]
            if "kv_cache_config" in kwargs:
                kwargs["kv_cache_config"] = incoming
            else:
                args = (incoming, *args[1:])
        original(self, *args, **kwargs)
        config = self.kv_cache_config
        if not state_config:
            return
        assert not self.block_pool.enable_caching
        managers = []
        for old, group in zip(self.single_type_managers, config.kv_cache_groups):
            if classify(group.kv_cache_spec) == "history":
                managers.append(old)
                continue
            first_name = group.layer_names[0]
            capacity = cache_rows(config, first_name)
            pool = BlockPool(capacity, False, self.scheduler_block_size)
            manager = FixedStateManager(
                group.kv_cache_spec,
                block_pool=pool,
                enable_caching=False,
                kv_cache_group_id=old.kv_cache_group_id,
                scheduler_block_size=self.scheduler_block_size,
                dcp_world_size=1,
                pcp_world_size=1,
                needs_kv_cache_zeroing=False,
            )
            managers.append(manager)
        self.single_type_managers = tuple(managers)
        self.group_block_sizes = tuple(m.block_size for m in managers)

    mod.KVCacheCoordinator.__init__ = initialize
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_cache_init = KVCacheManager.__init__

    def initialize_cache(self, kv_cache_config, *args, **kwargs):
        original_cache_init(self, kv_cache_config, *args, **kwargs)
        bind_platform_state_pools(self.coordinator, kv_cache_config)

    KVCacheManager.__init__ = initialize_cache
    original_allocate = KVCacheManager.allocate_slots

    def allocate_slots(self, request, *args, **kwargs):
        from vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state import reserve_append_only

        managers = self.coordinator.single_type_managers
        states = [m for m in managers if isinstance(m, FixedStateManager)]
        if not states:
            return original_allocate(self, request, *args, **kwargs)
        # This is a single-threaded scheduler transaction. Existing history
        # admission still applies; either state pool can independently defer it.
        needs = [m for m in states if not m.req_to_blocks.get(request.request_id)]
        if not needs:
            return original_allocate(self, request, *args, **kwargs)
        if any(not m.block_pool.get_num_free_blocks() for m in needs):
            return None
        return reserve_append_only(
            managers, request.request_id, lambda: original_allocate(self, request, *args, **kwargs)
        )

    KVCacheManager.allocate_slots = allocate_slots


def state_swa_lengths(seq_lens, query_start):
    query = (query_start[1 : seq_lens.shape[0] + 1] - query_start[: seq_lens.shape[0]]).to(seq_lens.dtype)
    return torch.minimum(seq_lens, query + WINDOW - 1)


def _cpu_list(value, count):
    assert value.device.type == "cpu", "Do not synchronize NPU metadata into a hot CPU path"
    return value[:count].tolist()


def hydrate(metadata):
    state = getattr(metadata, "request_state", None)
    if state is not None and not hasattr(metadata, "state_ids"):
        for key, value in state.items():
            setattr(metadata, key, value)
    return metadata


def export_state(builder, result):
    # A declared dataclass field preserves this contract through CP replace().
    result.request_state = {key: value for key, value in vars(result).items() if key.startswith("state_")}
    builder._request_state_last_result = result
    return result


def global_swa(metadata):
    value = getattr(metadata.swa, "global_metadata", None)
    return hydrate(value if value is not None else metadata.swa)


def _prepare_state_metadata(builder, common, result):
    if builder._cache_kind != "swa" or not isinstance(builder.kv_cache_spec, RequestWindowSpec):
        return result
    n = result.num_reqs
    global_owner = getattr(getattr(builder, "_global_builder", None), "_request_state_last_result", None)
    global_query_max = global_owner.max_query_len if global_owner is not None else result.max_query_len
    if result.max_query_len <= 1 and global_query_max == 1:
        # The graph must retain addresses whose contents are updated per step.
        # The one-column state block table already belongs to the runner.
        assert common.block_table_tensor.shape[1] == 1
        if not hasattr(builder, "_state_decode_lengths"):
            builder._state_decode_lengths = torch.empty_like(builder._seq_lens)
        lengths = builder._state_decode_lengths[:n]
        torch.clamp(result.seq_lens, max=WINDOW, out=lengths)
        result.state_ids = common.block_table_tensor[:n, 0]
        result.state_stage_lengths = lengths
        if not _direct_decode:
            # Decode staging owns one fixed page per padded request row.
            # Graph replay must keep lengths, row offsets and table addresses.
            if not hasattr(builder, "_state_decode_rows"):
                rows = torch.arange(builder._seq_lens.numel(), dtype=torch.int32, device=builder._seq_lens.device)
                builder._state_decode_rows = rows * WINDOW
                builder._state_decode_table = rows.view(-1, 1)
            result.state_stage_row_starts = builder._state_decode_rows[:n]
            result.state_stage_block_table = builder._state_decode_table[:n]
            result.state_stage_rows_used = n * WINDOW
            result.state_stage_max_seq_len = WINDOW
            result.state_staging_capacity = (
                (
                    builder.vllm_config.scheduler_config.max_num_batched_tokens
                    + builder.vllm_config.scheduler_config.max_num_seqs * 256
                    + 127
                )
                // 128
            ) * 128
            assert result.state_stage_rows_used <= result.state_staging_capacity
        record_path("persistent_decode_metadata")
        return export_state(builder, result)
    qcpu = common.query_start_loc_cpu
    seqcpu = getattr(common, "seq_lens_cpu", None)
    if seqcpu is None:
        seqcpu = common._seq_lens_cpu
    q = _cpu_list(qcpu, n + 1)
    seq = _cpu_list(seqcpu, n)
    bases, lengths, cursor = [], [], 0
    for i in range(n):
        query = q[i + 1] - q[i]
        old = min(max(0, seq[i] - query), WINDOW - 1)
        length = old + query if query else 0
        bases.append(cursor)
        lengths.append(length)
        cursor += ((length + WINDOW - 1) // WINDOW) * WINDOW
    device = result.seq_lens.device
    max_pages = max(1, max(((n + 127) // 128 for n in lengths), default=0))
    table = [
        [base // 128 + j if j < (length + 127) // 128 else 0 for j in range(max_pages)]
        for base, length in zip(bases, lengths)
    ]
    result.state_ids = common.block_table_tensor[:n, 0].contiguous()
    result.state_stage_row_starts = torch.tensor(bases, dtype=torch.int32, device=device)
    result.state_stage_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    result.state_stage_block_table = torch.tensor(table, dtype=torch.int32, device=device)
    result.state_stage_rows_used = cursor
    result.state_stage_max_seq_len = max(lengths, default=0)
    result.state_staging_capacity = (
        (
            builder.vllm_config.scheduler_config.max_num_batched_tokens
            + builder.vllm_config.scheduler_config.max_num_seqs * 256
            + 127
        )
        // 128
    ) * 128
    return export_state(builder, result)


def _stage(attn, metadata):
    from vllm_ascend.models.deepseek_v41.state_kernels import assemble

    swa = hydrate(metadata.swa)
    owner = global_swa(metadata)
    ring = attn.dsa_attn.swa_cache_layer.kv_cache[0]
    values = attn._request_state_current_kv
    assert values is not None and values.shape[0] == owner.num_actual_tokens
    current = values.contiguous()
    key = str(ring.device)
    if key not in _workspace:
        _workspace[key] = torch.empty(
            (swa.state_staging_capacity // 128, 128, 1, 512), dtype=ring.dtype, device=ring.device
        )
    staging = _workspace[key]
    assert swa.state_stage_rows_used <= staging.shape[0] * 128
    if owner is swa:
        assemble(
            ring,
            current,
            swa.state_ids,
            swa.query_start_loc,
            swa.seq_lens,
            swa.state_stage_row_starts,
            staging,
            swa.max_query_len,
        )
    else:
        from vllm_ascend.models.deepseek_v41.state_kernels import assemble_cp

        assemble_cp(
            ring,
            current,
            swa.state_ids,
            swa.query_start_loc,
            swa.seq_lens,
            owner.query_start_loc,
            owner.seq_lens,
            swa.state_stage_row_starts,
            staging,
            swa.max_query_len,
        )
    return ring, current, staging


def install_attention():
    global _attention_installed
    if _attention_installed:
        return
    _attention_installed = True
    from vllm_ascend.attention import dsa_v41 as mod

    original_build = mod.AscendDSAV41MetadataBuilder.build

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        if isinstance(self.kv_cache_spec, RequestWindowSpec):
            common_attn_metadata = copy.copy(common_attn_metadata)
            size = int(getattr(common_attn_metadata, "num_input_tokens", common_attn_metadata.slot_mapping.shape[0]))
            common_attn_metadata.slot_mapping = self._flat_slot_mapping[:size]
            common_attn_metadata.slot_mapping.fill_(-1)
        value = original_build(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
        return _prepare_state_metadata(self, common_attn_metadata, value)

    mod.AscendDSAV41MetadataBuilder.build = build
    original_write = mod.AscendDSAV41Impl._write_swa_cache

    def defer_write(attn, metadata, value):
        hydrate(metadata)
        if hasattr(metadata, "state_ids"):
            # Existing projection code joins its auxiliary stream before the
            # attention call. Hold the projected rows until stage and commit.
            attn._request_state_current_kv = value
        else:
            original_write(attn, metadata, value)

    mod.AscendDSAV41Impl._write_swa_cache = staticmethod(defer_write)
    original_forward = mod.AscendDSAV41Impl._forward_attention

    def forward(self, attn, q, metadata, compressed_indices, **kwargs):
        hydrate(metadata.swa)
        if not hasattr(metadata.swa, "state_ids"):
            return original_forward(self, attn, q, metadata, compressed_indices, **kwargs)
        from vllm_ascend.models.deepseek_v41.state_kernels import commit

        owner = global_swa(metadata)
        if _direct_decode and metadata.swa.max_query_len == 1 and owner.max_query_len == 1:
            # A single query observes the entire final <=128-row window.
            # Once full, circular row order is immaterial to attention; overwrite
            # only the row that has just left that window before the read.
            record_path("single_token_direct_ring")
            current = attn._request_state_current_kv.contiguous()
            ring = attn.dsa_attn.swa_cache_layer.kv_cache[0]
            swa = copy.copy(metadata.swa)
            assert owner.num_actual_tokens == owner.num_actual_reqs
            commit(ring, current, owner.state_ids, owner.query_start_loc, owner.seq_lens)
            swa.block_table = swa.state_ids.view(-1, 1)
            swa.seq_lens = swa.state_stage_lengths
            replacement = replace(metadata, swa=swa)
            output = original_forward(self, attn, q, replacement, compressed_indices, **kwargs)
            current.record_stream(torch.npu.current_stream())
            attn._request_state_current_kv = None
            return output
        record_path("multi_token_staging")
        ring, current, staging = _stage(attn, metadata)
        layer = attn.dsa_attn.swa_cache_layer
        old_cache = layer.kv_cache
        replacement = replace(metadata, swa=copy.copy(metadata.swa))
        replacement.swa.block_table = metadata.swa.state_stage_block_table
        replacement.swa.seq_lens = metadata.swa.state_stage_lengths
        replacement.swa.replay_block_table = None
        replacement.swa.replay_cache_blocks = None
        replacement.swa.replay_cache_offsets = None
        layer.kv_cache = [staging]
        try:
            output = original_forward(self, attn, q, replacement, compressed_indices, **kwargs)
        finally:
            layer.kv_cache = old_cache
        commit(ring, current, owner.state_ids, owner.query_start_loc, owner.seq_lens)
        current.record_stream(torch.npu.current_stream())
        attn._request_state_current_kv = None
        return output

    mod.AscendDSAV41Impl._forward_attention = forward
    original_update = mod.AscendDSAV41Impl._update_caches

    def update_only(self, attn, hidden_states, metadata):
        original_update(self, attn, hidden_states, metadata)
        if hidden_states.shape[0] and hasattr(metadata.swa, "state_ids"):
            from vllm_ascend.models.deepseek_v41.state_kernels import commit

            value = attn._request_state_current_kv.contiguous()
            ring = attn.dsa_attn.swa_cache_layer.kv_cache[0]
            commit(ring, value, metadata.swa.state_ids, metadata.swa.query_start_loc, metadata.swa.seq_lens)
            value.record_stream(torch.npu.current_stream())
            attn._request_state_current_kv = None

    mod.AscendDSAV41Impl._update_caches = update_only


def install():
    global _installed
    if _installed:
        return
    _installed = True
    from vllm_ascend.models.deepseek_v41 import cache_config as mod

    mod.group_cache_specs = group_cache_specs
    mod.get_deepseek_v41_pool_bytes_per_block = history_bytes
    mod.get_deepseek_v41_kv_cache_config = make_config
    install_coordinator()


def install_worker(cls):
    original = cls._reshape_kv_cache_tensors

    def reshape(self, config, *args, **kwargs):
        views = original(self, config, *args, **kwargs)
        if not any(classify(g.kv_cache_spec) == "swa" for g in config.kv_cache_groups):
            return views
        storage, details = {}, {}
        for name, item in views.items():
            values = item if isinstance(item, tuple) else (item,)
            details[name] = []
            for value in values:
                pointer = value.untyped_storage().data_ptr()
                storage[pointer] = value.untyped_storage().nbytes()
                details[name].append(
                    {
                        "shape": list(value.shape),
                        "stride": list(value.stride()),
                        "dtype": str(value.dtype),
                        "storage_pointer": pointer,
                        "storage_bytes": storage[pointer],
                        "contiguous": value.is_contiguous(),
                    }
                )
        result = {
            "pid": os.getpid(),
            "dp_rank": self.vllm_config.parallel_config.data_parallel_rank,
            "device": str(self.device),
            "history_blocks": config.num_blocks,
            "physical_backings": len(storage),
            "unique_backing_bytes": sum(storage.values()),
            "views": details,
        }
        result["single_token_decode_direct_ring"] = _direct_decode
        Path(os.environ["MENGZI_RESULTS_DIR"], f"state-worker-{os.getpid()}.json").write_text(
            json.dumps(result, indent=2)
        )
        return views

    cls._reshape_kv_cache_tensors = reshape
