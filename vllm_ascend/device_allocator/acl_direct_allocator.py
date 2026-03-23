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
# Direct CANN ACL API Memory Allocator for vLLM-Ascend.
#
# This module provides a pluggable memory allocator that uses CANN's native
# aclrtMalloc/aclrtFree APIs directly, bypassing PyTorch's NPUCachingAllocator
# and mempool mechanism.
#
# Usage:
#   Set environment variable ENABLE_ACLAPI=1 to enable this allocator.
#   The allocator will be automatically initialized when vllm-ascend starts.
#

import ctypes
import os
from typing import Optional

import torch
import torch_npu
from vllm.logger import logger


# Environment variable to control whether to use direct CANN API for memory allocation
# Default is True (1) - use direct CANN API allocator
# Set to 0 to use the original NPUCachingAllocator
ENABLE_ACLAPI = bool(int(os.getenv("ENABLE_ACLAPI", "1")))


class ACLDirectAllocator:
    """
    A singleton class that provides a pluggable memory allocator using CANN's
    native aclrtMalloc/aclrtFree APIs.
    
    This allocator replaces PyTorch's default NPUCachingAllocator when
    ENABLE_ACLAPI=1 is set. It directly calls CANN's memory allocation
    functions without going through PyTorch's caching allocator or mempool.
    
    The allocator is designed to be used with torch.npu.memory.change_current_allocator()
    to replace the default allocator at runtime.
    
    Example:
        >>> from vllm_ascend.device_allocator.acl_direct_allocator import ACLDirectAllocator
        >>> allocator = ACLDirectAllocator.get_instance()
        >>> allocator.init_allocator()
        >>> allocator.change_to_acl_allocator()
    """
    
    _instance: Optional["ACLDirectAllocator"] = None
    _allocator_initialized: bool = False
    _original_allocator: Optional[torch.npu.memory._NPUAllocator] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "ACLDirectAllocator":
        """
        Get the singleton instance of ACLDirectAllocator.
        
        Returns:
            The singleton instance of ACLDirectAllocator.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if self._allocator_initialized:
            return
            
        self._lib_path: Optional[str] = None
        self._lib: Optional[ctypes.CDLL] = None
        self._alloc_fn_ptr: Optional[int] = None
        self._free_fn_ptr: Optional[int] = None
        self._pluggable_allocator: Optional[torch.npu.memory.NPUPluggableAllocator] = None
        
    def _find_acl_library(self) -> str:
        """
        Find the path to the CANN ACL runtime library.
        
        Returns:
            Path to libascendcl.so
            
        Raises:
            RuntimeError: If the library cannot be found.
        """
        # Try to find from loaded libraries first
        try:
            with open("/proc/self/maps") as f:
                for line in f:
                    if "libascendcl.so" in line:
                        start = line.index("/")
                        path = line[start:].strip()
                        if os.path.exists(path):
                            return path
        except Exception:
            pass
        
        # Try common paths
        possible_paths = [
            "/usr/local/Ascend/driver/lib64/libascendcl.so",
            "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
            "/usr/local/Ascend/ascend-toolkit/latest/lib64/stub/libascendcl.so",
            "libascendcl.so",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise RuntimeError(
            "Cannot find libascendcl.so. Please ensure CANN is properly installed."
        )
    
    def _load_acl_library(self) -> None:
        """
        Load the CANN ACL runtime library and get function pointers.
        
        Raises:
            RuntimeError: If the library or functions cannot be loaded.
        """
        if self._lib is not None:
            return
            
        self._lib_path = self._find_acl_library()
        logger.info(f"Loading CANN ACL library from: {self._lib_path}")
        
        try:
            self._lib = ctypes.CDLL(self._lib_path)
            
            # Get aclrtMalloc function pointer
            # Signature: aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
            aclrt_malloc = self._lib.aclrtMalloc
            aclrt_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
            aclrt_malloc.restype = ctypes.c_int
            self._alloc_fn_ptr = ctypes.cast(aclrt_malloc, ctypes.c_void_p).value
            
            # Get aclrtFree function pointer
            # Signature: aclError aclrtFree(void *devPtr)
            aclrt_free = self._lib.aclrtFree
            aclrt_free.argtypes = [ctypes.c_void_p]
            aclrt_free.restype = ctypes.c_int
            self._free_fn_ptr = ctypes.cast(aclrt_free, ctypes.c_void_p).value
            
            if self._alloc_fn_ptr is None or self._free_fn_ptr is None:
                raise RuntimeError("Failed to get function pointers from libascendcl.so")
                
            logger.info("Successfully loaded aclrtMalloc and aclrtFree function pointers")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CANN ACL library: {e}")
    
    def init_allocator(self) -> None:
        """
        Initialize the pluggable allocator with direct CANN API functions.
        
        This method loads the CANN library and creates a NPUPluggableAllocator
        that uses aclrtMalloc/aclrtFree directly.
        
        Raises:
            RuntimeError: If the allocator cannot be initialized.
        """
        if self._allocator_initialized:
            return
            
        try:
            self._load_acl_library()
            
            # Create the pluggable allocator using torch_npu's C API
            # This calls the internal function that creates an allocator from function pointers
            self._pluggable_allocator = torch_npu._C._npu_customAllocator(
                self._alloc_fn_ptr,
                self._free_fn_ptr
            )
            
            self._allocator_initialized = True
            logger.info("ACLDirectAllocator initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ACLDirectAllocator: {e}")
    
    def change_to_acl_allocator(self) -> None:
        """
        Change the current NPU memory allocator to use direct CANN API.
        
        This method replaces PyTorch's default NPUCachingAllocator with the
        custom allocator that uses aclrtMalloc/aclrtFree.
        
        Note:
            This method must be called before any NPU memory allocations are made.
            Calling it after allocations have been made will raise an error.
            
        Raises:
            RuntimeError: If the allocator cannot be changed.
        """
        if not self._allocator_initialized:
            self.init_allocator()
            
        try:
            # Save the original allocator for potential restore
            self._original_allocator = torch_npu._C._npu_getAllocator()
            
            # Change to our custom allocator
            torch_npu._C._npu_changeCurrentAllocator(self._pluggable_allocator)
            
            logger.info(
                "Successfully changed NPU memory allocator to use direct CANN API "
                "(aclrtMalloc/aclrtFree). NPUCachingAllocator and mempool are bypassed."
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to change to ACL direct allocator: {e}")
    
    def restore_original_allocator(self) -> None:
        """
        Restore the original NPU memory allocator.
        
        This method restores the default NPUCachingAllocator.
        
        Note:
            This method must be called before any NPU memory allocations are made
            after changing to the ACL allocator.
        """
        if self._original_allocator is None:
            logger.warning("No original allocator to restore")
            return
            
        try:
            torch_npu._C._npu_changeCurrentAllocator(self._original_allocator)
            logger.info("Restored original NPU memory allocator")
        except Exception as e:
            raise RuntimeError(f"Failed to restore original allocator: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the allocator has been initialized."""
        return self._allocator_initialized
    
    @property
    def is_acl_allocator_active(self) -> bool:
        """Check if the ACL direct allocator is currently active."""
        if not self._allocator_initialized:
            return False
        try:
            current = torch_npu._C._npu_getAllocator()
            return current == self._pluggable_allocator
        except Exception:
            return False


def maybe_init_acl_direct_allocator() -> bool:
    """
    Initialize and enable the ACL direct allocator if ENABLE_ACLAPI is set.
    
    By default, ENABLE_ACLAPI is set to 1, which means the direct CANN API allocator
    is used. Set ENABLE_ACLAPI=0 to use the original NPUCachingAllocator.
    
    Returns:
        True if the ACL direct allocator was enabled, False otherwise.
    """
    if not ENABLE_ACLAPI:
        logger.info(
            "ENABLE_ACLAPI=0: Using original NPUCachingAllocator. "
            "Set ENABLE_ACLAPI=1 (default) to use direct CANN API allocator."
        )
        return False
    
    try:
        allocator = ACLDirectAllocator.get_instance()
        allocator.change_to_acl_allocator()
        return True
    except Exception as e:
        logger.error(f"Failed to enable ACL direct allocator: {e}")
        logger.warning("Falling back to default NPUCachingAllocator")
        return False


# Export symbols
__all__ = [
    "ENABLE_ACLAPI",
    "ACLDirectAllocator",
    "maybe_init_acl_direct_allocator",
]
