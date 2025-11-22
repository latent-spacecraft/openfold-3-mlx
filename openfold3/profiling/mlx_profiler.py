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
MLX Performance Profiler for OpenFold3 Apple Silicon optimizations.

This module provides detailed profiling capabilities to identify
computational hotspots and guide mx.compile optimization strategies.
"""

import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import functools

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import torch


@dataclass
class ProfileEntry:
    """Single profiling measurement entry."""
    operation_name: str
    duration_ms: float
    input_shapes: List[tuple]
    output_shapes: List[tuple]
    memory_usage_mb: Optional[float] = None
    flops_estimate: Optional[int] = None
    compilation_candidate: bool = False


class MLXProfiler:
    """
    Comprehensive profiler for MLX operations in OpenFold3.

    Provides timing, memory usage, and FLOPS analysis to guide
    mx.compile optimization decisions.
    """

    def __init__(self, enable_detailed_profiling: bool = True):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.profile_data: List[ProfileEntry] = []
        self.operation_stack: List[str] = []
        self.total_time: float = 0.0

    def reset(self):
        """Reset all profiling data."""
        self.profile_data.clear()
        self.operation_stack.clear()
        self.total_time = 0.0

    @contextmanager
    def profile_operation(
        self,
        operation_name: str,
        input_tensors: Optional[List] = None,
        output_tensors: Optional[List] = None
    ):
        """
        Context manager for profiling individual operations.

        Args:
            operation_name: Name of the operation being profiled
            input_tensors: Input tensors (for shape analysis)
            output_tensors: Output tensors (for shape analysis)
        """
        if not self.enable_detailed_profiling:
            yield
            return

        self.operation_stack.append(operation_name)

        # Get input shapes
        input_shapes = []
        if input_tensors:
            for tensor in input_tensors:
                if hasattr(tensor, 'shape'):
                    input_shapes.append(tuple(tensor.shape))

        # Ensure MLX operations are synchronized
        if MLX_AVAILABLE:
            mx.eval(mx.array([1.0]))  # Sync point

        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Sync again after operation
            if MLX_AVAILABLE:
                mx.eval(mx.array([1.0]))

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Get output shapes
            output_shapes = []
            if output_tensors:
                for tensor in output_tensors:
                    if hasattr(tensor, 'shape'):
                        output_shapes.append(tuple(tensor.shape))

            # Estimate FLOPS for common operations
            flops_estimate = self._estimate_flops(
                operation_name, input_shapes, output_shapes
            )

            # Determine if this is a good compilation candidate
            compilation_candidate = self._is_compilation_candidate(
                operation_name, duration_ms, flops_estimate
            )

            # Store profile entry
            entry = ProfileEntry(
                operation_name=operation_name,
                duration_ms=duration_ms,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                flops_estimate=flops_estimate,
                compilation_candidate=compilation_candidate
            )

            self.profile_data.append(entry)
            self.total_time += duration_ms
            self.operation_stack.pop()

    def _estimate_flops(
        self,
        operation_name: str,
        input_shapes: List[tuple],
        output_shapes: List[tuple]
    ) -> Optional[int]:
        """
        Estimate FLOPS for common operations.

        This is crucial for identifying high-arithmetic-intensity operations
        that benefit most from compilation.
        """
        if not input_shapes:
            return None

        try:
            if "attention" in operation_name.lower():
                # For attention: Q@K.T + softmax + A@V
                # Assume first shape is [*, H, Q, C_hidden]
                if len(input_shapes) >= 3:
                    q_shape = input_shapes[0]
                    k_shape = input_shapes[1]
                    v_shape = input_shapes[2]

                    if len(q_shape) >= 3:
                        seq_len_q = q_shape[-2]
                        seq_len_k = k_shape[-2]
                        head_dim = q_shape[-1]

                        # Q@K.T: [Q, C] @ [C, K] = Q*K*C operations
                        qk_flops = seq_len_q * seq_len_k * head_dim
                        # A@V: [Q, K] @ [K, C] = Q*K*C operations
                        av_flops = seq_len_q * seq_len_k * head_dim
                        # Softmax: ~3 ops per element
                        softmax_flops = seq_len_q * seq_len_k * 3

                        return qk_flops + av_flops + softmax_flops

            elif "einsum" in operation_name.lower():
                # For einsum operations, estimate based on output size
                if output_shapes and len(output_shapes[0]) > 0:
                    output_elements = 1
                    for dim in output_shapes[0]:
                        output_elements *= dim
                    # Rough estimate: each output element requires ~input_dim operations
                    if input_shapes and len(input_shapes[0]) > 0:
                        input_dim = input_shapes[0][-1]
                        return output_elements * input_dim

            elif "matrix" in operation_name.lower() or "linear" in operation_name.lower():
                # Matrix multiplication: A[M,K] @ B[K,N] = M*N*K operations
                if len(input_shapes) >= 2 and len(input_shapes[0]) >= 2:
                    m = input_shapes[0][-2]
                    k = input_shapes[0][-1]
                    n = input_shapes[1][-1] if len(input_shapes[1]) >= 2 else k
                    return m * n * k * 2  # multiply-add

        except Exception:
            pass

        return None

    def _is_compilation_candidate(
        self,
        operation_name: str,
        duration_ms: float,
        flops_estimate: Optional[int]
    ) -> bool:
        """
        Determine if an operation is a good candidate for mx.compile.

        Good candidates:
        1. High computational intensity (high FLOPS)
        2. Significant runtime (>1ms typically)
        3. Attention/triangle operations (known hotspots)
        4. Pure computation (minimal host-device transfers)
        """
        # High-priority patterns for compilation
        high_priority_ops = [
            "attention", "triangle", "einsum", "matrix", "linear",
            "evoformer", "mlx_", "softmax"
        ]

        is_high_priority = any(
            pattern in operation_name.lower() for pattern in high_priority_ops
        )

        # Duration threshold (operations taking >0.5ms are worth compiling)
        significant_duration = duration_ms > 0.5

        # High computational intensity (>1M FLOPS)
        high_flops = flops_estimate is not None and flops_estimate > 1_000_000

        return (is_high_priority and significant_duration) or high_flops

    def get_compilation_candidates(self, min_duration_ms: float = 1.0) -> List[ProfileEntry]:
        """Get operations that are good candidates for compilation."""
        candidates = [
            entry for entry in self.profile_data
            if entry.compilation_candidate and entry.duration_ms >= min_duration_ms
        ]

        # Sort by duration (highest impact first)
        return sorted(candidates, key=lambda x: x.duration_ms, reverse=True)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get high-level profiling statistics."""
        if not self.profile_data:
            return {}

        total_time = sum(entry.duration_ms for entry in self.profile_data)
        total_flops = sum(
            entry.flops_estimate for entry in self.profile_data
            if entry.flops_estimate is not None
        )

        # Group by operation type
        op_groups = {}
        for entry in self.profile_data:
            op_type = entry.operation_name.split('_')[0]  # First part of name
            if op_type not in op_groups:
                op_groups[op_type] = {
                    'count': 0, 'total_time': 0, 'total_flops': 0
                }
            op_groups[op_type]['count'] += 1
            op_groups[op_type]['total_time'] += entry.duration_ms
            if entry.flops_estimate:
                op_groups[op_type]['total_flops'] += entry.flops_estimate

        return {
            'total_operations': len(self.profile_data),
            'total_time_ms': total_time,
            'total_flops': total_flops,
            'flops_per_second': total_flops / (total_time / 1000) if total_time > 0 else 0,
            'operation_groups': op_groups,
            'compilation_candidates': len(self.get_compilation_candidates())
        }

    def generate_report(self, output_dir: Path):
        """Generate comprehensive profiling report with visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary statistics
        stats = self.get_summary_stats()
        with open(output_dir / "profiling_summary.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Detailed data
        detailed_data = [asdict(entry) for entry in self.profile_data]
        with open(output_dir / "detailed_profiling.json", "w") as f:
            json.dump(detailed_data, f, indent=2)

        # Generate visualizations
        self._create_timing_chart(output_dir)
        self._create_flops_chart(output_dir)
        self._create_compilation_candidates_chart(output_dir)

    def _create_timing_chart(self, output_dir: Path):
        """Create timing breakdown chart."""
        if not self.profile_data:
            return

        # Group operations and sum times
        op_times = {}
        for entry in self.profile_data:
            op_type = entry.operation_name.split('_')[0]
            op_times[op_type] = op_times.get(op_type, 0) + entry.duration_ms

        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            op_times.values(),
            labels=op_times.keys(),
            autopct='%1.1f%%',
            startangle=90
        )

        ax.set_title('Computation Time Breakdown by Operation Type')
        plt.tight_layout()
        plt.savefig(output_dir / "timing_breakdown.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _create_flops_chart(self, output_dir: Path):
        """Create FLOPS analysis chart."""
        entries_with_flops = [
            entry for entry in self.profile_data
            if entry.flops_estimate is not None
        ]

        if not entries_with_flops:
            return

        # Calculate FLOPS/second for each operation
        flops_per_sec = [
            entry.flops_estimate / (entry.duration_ms / 1000)
            for entry in entries_with_flops
        ]
        operation_names = [entry.operation_name for entry in entries_with_flops]

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(range(len(flops_per_sec)), flops_per_sec)

        ax.set_xlabel('Operation')
        ax.set_ylabel('FLOPS/second')
        ax.set_title('Computational Intensity by Operation')
        ax.set_xticks(range(len(operation_names)))
        ax.set_xticklabels(operation_names, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(output_dir / "flops_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _create_compilation_candidates_chart(self, output_dir: Path):
        """Create compilation candidates analysis."""
        candidates = self.get_compilation_candidates()

        if not candidates:
            return

        # Create horizontal bar chart showing potential speedup
        fig, ax = plt.subplots(figsize=(12, 6))

        names = [entry.operation_name[:30] for entry in candidates[:10]]  # Top 10
        times = [entry.duration_ms for entry in candidates[:10]]

        bars = ax.barh(names, times, color='lightcoral')
        ax.set_xlabel('Duration (ms)')
        ax.set_title('Top Compilation Candidates by Runtime')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "compilation_candidates.png", dpi=150, bbox_inches='tight')
        plt.close()


# Decorator for easy profiling
def profile_mlx_operation(operation_name: str):
    """Decorator to automatically profile MLX operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the global profiler instance
            profiler = getattr(wrapper, '_profiler', None)
            if profiler is None:
                return func(*args, **kwargs)

            with profiler.profile_operation(operation_name,
                                          input_tensors=args,
                                          output_tensors=None):
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator


# Global profiler instance
_global_profiler: Optional[MLXProfiler] = None

def get_profiler() -> MLXProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = MLXProfiler()
    return _global_profiler

def enable_profiling():
    """Enable global profiling."""
    global _global_profiler
    _global_profiler = MLXProfiler(enable_detailed_profiling=True)

def disable_profiling():
    """Disable global profiling."""
    global _global_profiler
    if _global_profiler:
        _global_profiler.enable_detailed_profiling = False