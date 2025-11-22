# Copyright 2025 Geoffrey Taghon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Native MLX attention implementations that eliminate PyTorch conversion overhead.

This module implements the highest-priority optimization from profiling analysis:
keeping computations in MLX domain to eliminate the 700ms+ conversion bottleneck.
"""

import math
from typing import Any

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from openfold3.profiling import get_profiler
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

import torch


# Global cache for compiled functions to avoid recompilation overhead
_COMPILED_FUNCTION_CACHE: dict[str, Any] = {}

def get_or_compile_function(func_name: str, func, enable_compilation: bool = True):
    """
    Get cached compiled function or compile and cache it.

    This eliminates recompilation overhead while maintaining the performance
    benefits of mx.compile.
    """
    if not MLX_AVAILABLE or not enable_compilation:
        return func

    if func_name not in _COMPILED_FUNCTION_CACHE:
        with get_profiler().profile_operation(f"compile_{func_name}") if PROFILING_AVAILABLE else nullcontext():
            _COMPILED_FUNCTION_CACHE[func_name] = mx.compile(func)

    return _COMPILED_FUNCTION_CACHE[func_name]


def efficient_torch_to_mlx_batch(tensors: list[torch.Tensor]) -> list[mx.array]:
    """
    Batched, optimized conversion from PyTorch to MLX.

    Reduces conversion overhead by processing multiple tensors in a single call
    and using optimized memory transfer patterns.

    Note: Cannot be compiled as it handles PyTorch tensors.
    """
    mlx_arrays = []
    for tensor in tensors:
        # Use contiguous memory for faster transfer
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        mlx_arrays.append(mx.array(tensor.detach().cpu().numpy()))
    return mlx_arrays


def efficient_mlx_to_torch_batch(
    arrays: list[mx.array],
    devices: list[torch.device],
    dtypes: list[torch.dtype]
) -> list[torch.Tensor]:
    """
    Batched, optimized conversion from MLX to PyTorch.

    Note: Cannot be compiled as it produces PyTorch tensors.
    """
    import numpy as np
    torch_tensors = []
    for array, device, dtype in zip(arrays, devices, dtypes):
        numpy_array = np.array(array)
        torch_tensors.append(
            torch.from_numpy(numpy_array).to(device=device, dtype=dtype)
        )
    return torch_tensors


@mx.compile
def native_mlx_evoformer_attention_block(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    bias1: mx.array,
    bias2: mx.array | None,
    scale: float
) -> mx.array:
    """
    Complete evoformer attention block compiled as single computational graph.

    This eliminates all intermediate conversions and maximizes MLX optimization
    opportunities by keeping everything in the MLX computational domain.
    """
    # Q @ K.T with scaling - fully optimized einsum
    scores = mx.einsum("...qc,...kc->...qk", q, k) * scale

    # Add bias terms efficiently
    scores = scores + bias1
    if bias2 is not None:
        scores = scores + bias2

    # Softmax with MLX-native numerical stability
    attn_weights = mx.softmax(scores, axis=-1)

    # Final attention computation
    output = mx.einsum("...qk,...kc->...qc", attn_weights, v)

    return output


@mx.compile
def native_mlx_chunked_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    bias1: mx.array,
    bias2: mx.array | None,
    scale: float,
    chunk_size: int
) -> mx.array:
    """
    Memory-efficient chunked attention with full MLX compilation.

    Processes attention in chunks while maintaining compilation benefits
    and avoiding memory transfer overhead.
    """
    seq_len_q = q.shape[-2]

    if seq_len_q <= chunk_size:
        # Small enough for direct computation
        return native_mlx_evoformer_attention_block(q, k, v, bias1, bias2, scale)

    # Process in chunks
    output_chunks = []

    for q_start in range(0, seq_len_q, chunk_size):
        q_end = min(q_start + chunk_size, seq_len_q)

        q_chunk = q[..., q_start:q_end, :]
        b1_chunk = bias1[..., q_start:q_end, :] if bias1 is not None else None
        b2_chunk = bias2[..., q_start:q_end, :] if bias2 is not None else None

        chunk_output = native_mlx_evoformer_attention_block(
            q_chunk, k, v, b1_chunk, b2_chunk, scale
        )
        output_chunks.append(chunk_output)

    return mx.concatenate(output_chunks, axis=-2)


@mx.compile
def native_mlx_triangle_attention_block(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask_bias: mx.array,
    triangle_bias: mx.array,
    scale: float
) -> mx.array:
    """
    Native MLX triangle attention with full compilation optimization.

    Implements cuEquivariance-compatible triangle attention entirely in MLX
    domain for maximum performance.
    """
    # Compute attention scores
    scores = mx.einsum("...qc,...kc->...qk", q, k) * scale

    # Apply mask (convert boolean mask to additive form if needed)
    # Note: mask_bias should already be in additive form (-inf for masked positions)
    scores = scores + mask_bias + triangle_bias

    # Softmax and final computation
    attn_weights = mx.softmax(scores, axis=-1)
    output = mx.einsum("...qk,...kc->...qc", attn_weights, v)

    return output


class NativeMLXAttentionInterface:
    """
    High-performance MLX attention interface that eliminates conversion overhead.

    This class implements the core optimization strategy: minimize PyTorch-MLX
    conversions by batching them at module boundaries and keeping computation
    in MLX domain.
    """

    def __init__(self, enable_compilation: bool = True, enable_profiling: bool = True):
        self.enable_compilation = enable_compilation
        self.enable_profiling = enable_profiling

        # Cache compiled functions
        if enable_compilation:
            self._compiled_attention = get_or_compile_function(
                "native_evoformer_attention",
                native_mlx_evoformer_attention_block,
                enable_compilation
            )
            self._compiled_chunked = get_or_compile_function(
                "native_chunked_attention",
                native_mlx_chunked_attention,
                enable_compilation
            )
            self._compiled_triangle = get_or_compile_function(
                "native_triangle_attention",
                native_mlx_triangle_attention_block,
                enable_compilation
            )

    def evoformer_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor],
        use_chunked: bool = False,
        chunk_size: int = 1024
    ) -> torch.Tensor:
        """
        High-performance evoformer attention with minimized conversion overhead.
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX not available")

        # Store original properties for conversion back
        orig_device, orig_dtype = q.device, q.dtype

        # Single batched conversion to MLX
        profiler = get_profiler() if PROFILING_AVAILABLE else None

        with profiler.profile_operation("batch_torch_to_mlx") if profiler else nullcontext():
            torch_tensors = [q, k, v] + biases
            mlx_arrays = efficient_torch_to_mlx_batch(torch_tensors)

            q_mlx, k_mlx, v_mlx = mlx_arrays[:3]
            bias_arrays = mlx_arrays[3:]
            bias1_mlx = bias_arrays[0] if len(bias_arrays) > 0 else None
            bias2_mlx = bias_arrays[1] if len(bias_arrays) > 1 else None

        # Compute scale
        scale = 1.0 / math.sqrt(q_mlx.shape[-1])

        # Choose optimal computation path
        with profiler.profile_operation("native_mlx_computation") if profiler else nullcontext():
            if use_chunked:
                output_mlx = self._compiled_chunked(
                    q_mlx, k_mlx, v_mlx, bias1_mlx, bias2_mlx, scale, chunk_size
                ) if self.enable_compilation else native_mlx_chunked_attention(
                    q_mlx, k_mlx, v_mlx, bias1_mlx, bias2_mlx, scale, chunk_size
                )
            else:
                output_mlx = self._compiled_attention(
                    q_mlx, k_mlx, v_mlx, bias1_mlx, bias2_mlx, scale
                ) if self.enable_compilation else native_mlx_evoformer_attention_block(
                    q_mlx, k_mlx, v_mlx, bias1_mlx, bias2_mlx, scale
                )

        # Single conversion back to PyTorch
        with profiler.profile_operation("batch_mlx_to_torch") if profiler else nullcontext():
            torch_outputs = efficient_mlx_to_torch_batch(
                [output_mlx], [orig_device], [orig_dtype]
            )

        return torch_outputs[0]

    def triangle_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor],
        scale: float
    ) -> torch.Tensor:
        """
        High-performance triangle attention with minimized conversion overhead.
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX not available")

        if len(biases) != 2:
            raise ValueError("Triangle attention requires exactly 2 bias terms")

        orig_device, orig_dtype = q.device, q.dtype

        # Handle batched inputs (template module compatibility)
        is_batched = len(q.shape) > 5
        original_shape = None

        if is_batched:
            original_shape = q.shape
            batch, n_tmpl = q.shape[:2]
            q = q.view(batch * n_tmpl, *q.shape[2:])
            k = k.view(batch * n_tmpl, *k.shape[2:])
            v = v.view(batch * n_tmpl, *v.shape[2:])
            biases = [b.view(batch * n_tmpl, *b.shape[2:]) for b in biases]

        mask_bias, triangle_bias = biases

        # Convert mask format if needed
        if mask_bias.dtype != torch.bool and mask_bias.dtype != q.dtype:
            # Assume it's already in additive form or convert from binary
            if mask_bias.dtype == torch.bool:
                mask_bias = torch.where(
                    mask_bias,
                    torch.zeros_like(mask_bias, dtype=q.dtype),
                    torch.full_like(mask_bias, -float('inf'), dtype=q.dtype)
                )

        profiler = get_profiler() if PROFILING_AVAILABLE else None

        # Batched conversion to MLX
        with profiler.profile_operation("triangle_torch_to_mlx") if profiler else nullcontext():
            torch_tensors = [q, k, v, mask_bias, triangle_bias]
            mlx_arrays = efficient_torch_to_mlx_batch(torch_tensors)
            q_mlx, k_mlx, v_mlx, mask_mlx, tri_mlx = mlx_arrays

        # Native MLX triangle attention computation
        with profiler.profile_operation("native_triangle_computation") if profiler else nullcontext():
            output_mlx = self._compiled_triangle(
                q_mlx, k_mlx, v_mlx, mask_mlx, tri_mlx, scale
            ) if self.enable_compilation else native_mlx_triangle_attention_block(
                q_mlx, k_mlx, v_mlx, mask_mlx, tri_mlx, scale
            )

        # Convert back to PyTorch
        with profiler.profile_operation("triangle_mlx_to_torch") if profiler else nullcontext():
            torch_outputs = efficient_mlx_to_torch_batch(
                [output_mlx], [orig_device], [orig_dtype]
            )
        output = torch_outputs[0]

        # Restore original dimensions and apply transformations
        if len(q.shape) == 4 and output.shape[0] == 1:
            output = output.squeeze(0)

        if is_batched:
            output = output.view(original_shape[0], original_shape[1], *output.shape[1:])

        # Apply transpose (cuEquivariance compatibility)
        output = output.transpose(-2, -3)

        return output


# Global interface instance for easy access
_native_interface: NativeMLXAttentionInterface | None = None

def get_native_mlx_interface(enable_compilation: bool = True) -> NativeMLXAttentionInterface:
    """Get or create the global native MLX attention interface."""
    global _native_interface
    if _native_interface is None or _native_interface.enable_compilation != enable_compilation:
        _native_interface = NativeMLXAttentionInterface(enable_compilation=enable_compilation)
    return _native_interface


# Context manager for profiling (null context if not available)
try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __enter__(self): return self
        def __exit__(self, *args): pass


# High-level API functions for easy integration
def native_mlx_evoformer_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: list[torch.Tensor],
    use_chunked: bool = False,
    chunk_size: int = 1024,
    enable_compilation: bool = True
) -> torch.Tensor:
    """
    Drop-in replacement for MLX evoformer attention with optimized conversion handling.

    This function implements the primary optimization: minimizing PyTorch-MLX conversion
    overhead while maximizing compilation benefits.
    """
    interface = get_native_mlx_interface(enable_compilation)
    return interface.evoformer_attention(q, k, v, biases, use_chunked, chunk_size)


def native_mlx_triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: list[torch.Tensor],
    scale: float,
    enable_compilation: bool = True
) -> torch.Tensor:
    """
    Drop-in replacement for MLX triangle attention with optimized conversion handling.
    """
    interface = get_native_mlx_interface(enable_compilation)
    return interface.triangle_attention(q, k, v, biases, scale)


# Export public interface
__all__ = [
    "native_mlx_evoformer_attention",
    "native_mlx_triangle_attention",
    "NativeMLXAttentionInterface",
    "get_native_mlx_interface",
    "MLX_AVAILABLE"
]