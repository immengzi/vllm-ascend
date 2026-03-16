#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Device allocator module for vllm-ascend.
# Provides memory management utilities for sleep mode and shared CPU memory pool.
#

from vllm_ascend.device_allocator.camem import (
    AllocationData,
    CaMemAllocator,
    create_and_map,
    find_loaded_library,
    get_pluggable_allocator,
    unmap_and_release,
    use_memory_pool_with_allocator,
)
from vllm_ascend.device_allocator.shared_cpu_pool import (
    SharedCPUMemoryPool,
    SharedMemoryBlock,
)

__all__ = [
    # CaMem allocator
    "AllocationData",
    "CaMemAllocator",
    "create_and_map",
    "find_loaded_library",
    "get_pluggable_allocator",
    "unmap_and_release",
    "use_memory_pool_with_allocator",
    # Shared CPU pool
    "SharedCPUMemoryPool",
    "SharedMemoryBlock",
]
