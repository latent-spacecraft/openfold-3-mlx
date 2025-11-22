#!/usr/bin/env python3
"""
Benchmark the optimized native MLX attention against the original implementation
to validate the expected 2-3x performance improvements.
"""

import time
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from openfold3.profiling import enable_profiling, get_profiler, disable_profiling
from openfold3.core.model.primitives.attention_mlx import (
    mlx_evoformer_attention,
    mlx_triangle_attention,
    MLX_AVAILABLE
)
from openfold3.core.model.primitives.native_mlx_attention import (
    native_mlx_evoformer_attention,
    native_mlx_triangle_attention
)

def create_test_tensors(batch_size=1, seq_len=256, num_heads=8, head_dim=64):
    """Create test tensors similar to protein folding workloads."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # Bias terms
    mask_bias = torch.randn(batch_size, 1, 1, seq_len, seq_len)
    triangle_bias = torch.randn(batch_size, num_heads, seq_len, seq_len)

    return q, k, v, [mask_bias, triangle_bias]

def warmup_runs(func, inputs, num_warmup=3):
    """Perform warmup runs to ensure stable timing."""
    for _ in range(num_warmup):
        result = func(*inputs)
        # Force computation
        _ = result.mean().item()

def benchmark_function(func, inputs, num_runs=10, description=""):
    """Benchmark a function with multiple runs for statistical accuracy."""
    print(f"  Benchmarking {description}...")

    # Warmup
    warmup_runs(func, inputs)

    # Actual timing
    times = []
    for i in range(num_runs):
        start_time = time.perf_counter()
        result = func(*inputs)
        # Force computation
        _ = result.mean().item()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"    Average: {avg_time:.2f} ± {std_time:.2f} ms")
    return avg_time, std_time, result

def compare_evoformer_attention(seq_len, num_runs=5):
    """Compare original vs. optimized evoformer attention."""
    print(f"\n=== Evoformer Attention Comparison (seq_len={seq_len}) ===")

    q, k, v, biases = create_test_tensors(seq_len=seq_len)

    # Test direct attention (non-chunked)
    print("Direct Attention:")

    # Original implementation
    original_time, original_std, original_result = benchmark_function(
        lambda q, k, v, b: mlx_evoformer_attention(q, k, v, b, use_chunked=False, enable_compilation=True),
        (q, k, v, biases),
        num_runs,
        "Original MLX (with conversions)"
    )

    # Optimized implementation
    optimized_time, optimized_std, optimized_result = benchmark_function(
        lambda q, k, v, b: native_mlx_evoformer_attention(q, k, v, b, use_chunked=False, enable_compilation=True),
        (q, k, v, biases),
        num_runs,
        "Optimized Native MLX"
    )

    # Verify numerical correctness
    max_diff = torch.max(torch.abs(original_result - optimized_result)).item()
    print(f"    Max numerical difference: {max_diff:.2e}")

    speedup = original_time / optimized_time
    print(f"    Speedup: {speedup:.2f}x")

    return {
        'seq_len': seq_len,
        'original_time': original_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'max_diff': max_diff
    }

def compare_triangle_attention(seq_len, num_runs=5):
    """Compare original vs. optimized triangle attention."""
    print(f"\n=== Triangle Attention Comparison (seq_len={seq_len}) ===")

    q, k, v, biases = create_test_tensors(seq_len=seq_len)
    scale = 1.0 / np.sqrt(64)

    # Original implementation
    original_time, original_std, original_result = benchmark_function(
        lambda q, k, v, b: mlx_triangle_attention(q, k, v, b, scale, enable_compilation=True),
        (q, k, v, biases),
        num_runs,
        "Original Triangle MLX"
    )

    # Optimized implementation
    optimized_time, optimized_std, optimized_result = benchmark_function(
        lambda q, k, v, b: native_mlx_triangle_attention(q, k, v, b, scale, enable_compilation=True),
        (q, k, v, biases),
        num_runs,
        "Optimized Native Triangle MLX"
    )

    # Verify numerical correctness
    max_diff = torch.max(torch.abs(original_result - optimized_result)).item()
    print(f"    Max numerical difference: {max_diff:.2e}")

    speedup = original_time / optimized_time
    print(f"    Speedup: {speedup:.2f}x")

    return {
        'seq_len': seq_len,
        'original_time': original_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'max_diff': max_diff
    }

def benchmark_scaling_behavior():
    """Benchmark performance across different sequence lengths."""
    print("\n=== Scaling Behavior Analysis ===")

    test_sizes = [128, 256, 384, 512]
    evoformer_results = []
    triangle_results = []

    for seq_len in test_sizes:
        try:
            # Evoformer attention
            evo_result = compare_evoformer_attention(seq_len, num_runs=3)
            evoformer_results.append(evo_result)

            # Triangle attention (more memory intensive, so test smaller sizes)
            if seq_len <= 384:
                tri_result = compare_triangle_attention(seq_len, num_runs=3)
                triangle_results.append(tri_result)

        except Exception as e:
            print(f"  Error at seq_len={seq_len}: {e}")

    return evoformer_results, triangle_results

def create_performance_plots(evoformer_results, triangle_results):
    """Create performance visualization plots."""
    print("\n=== Creating Performance Visualization ===")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    if evoformer_results:
        seq_lens = [r['seq_len'] for r in evoformer_results]
        original_times = [r['original_time'] for r in evoformer_results]
        optimized_times = [r['optimized_time'] for r in evoformer_results]
        speedups = [r['speedup'] for r in evoformer_results]

        # Evoformer timing comparison
        ax1.plot(seq_lens, original_times, 'ro-', label='Original MLX', linewidth=2, markersize=8)
        ax1.plot(seq_lens, optimized_times, 'go-', label='Optimized Native MLX', linewidth=2, markersize=8)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Evoformer Attention Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Evoformer speedup
        ax2.plot(seq_lens, speedups, 'bo-', linewidth=2, markersize=8)
        ax2.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='2x Target')
        ax2.axhline(y=3.0, color='r', linestyle='--', alpha=0.7, label='3x Target')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Evoformer Attention Speedup')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    if triangle_results:
        tri_seq_lens = [r['seq_len'] for r in triangle_results]
        tri_original_times = [r['original_time'] for r in triangle_results]
        tri_optimized_times = [r['optimized_time'] for r in triangle_results]
        tri_speedups = [r['speedup'] for r in triangle_results]

        # Triangle timing comparison
        ax3.plot(tri_seq_lens, tri_original_times, 'ro-', label='Original Triangle MLX', linewidth=2, markersize=8)
        ax3.plot(tri_seq_lens, tri_optimized_times, 'go-', label='Optimized Triangle MLX', linewidth=2, markersize=8)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Time (ms)')
        ax3.set_title('Triangle Attention Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Triangle speedup
        ax4.plot(tri_seq_lens, tri_speedups, 'mo-', linewidth=2, markersize=8)
        ax4.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='2x Target')
        ax4.axhline(y=3.0, color='r', linestyle='--', alpha=0.7, label='3x Target')
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Speedup Factor')
        ax4.set_title('Triangle Attention Speedup')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimization_benchmarks.png', dpi=150, bbox_inches='tight')
    print("Performance plots saved to: optimization_benchmarks.png")

def profile_optimization_impact():
    """Profile the optimization impact using the profiling infrastructure."""
    print("\n=== Profiling Optimization Impact ===")

    enable_profiling()
    profiler = get_profiler()
    profiler.reset()

    try:
        # Test sequence
        q, k, v, biases = create_test_tensors(seq_len=256)

        # Run optimized version multiple times
        for i in range(5):
            result = native_mlx_evoformer_attention(
                q, k, v, biases,
                use_chunked=False,
                enable_compilation=True
            )
            _ = result.mean().item()

        # Analyze results
        stats = profiler.get_summary_stats()
        candidates = profiler.get_compilation_candidates()

        print(f"Optimized Implementation Stats:")
        print(f"  Total operations: {stats.get('total_operations', 0)}")
        print(f"  Total time: {stats.get('total_time_ms', 0):.2f} ms")
        print(f"  Compilation candidates: {len(candidates)}")

        # Check if conversion overhead is reduced
        conversion_ops = [
            entry for entry in profiler.profile_data
            if 'conversion' in entry.operation_name.lower()
        ]

        total_conversion_time = sum(op.duration_ms for op in conversion_ops)
        print(f"  Conversion overhead: {total_conversion_time:.2f} ms "
              f"({total_conversion_time / stats.get('total_time_ms', 1) * 100:.1f}% of total)")

    finally:
        disable_profiling()

def main():
    """Main benchmarking function."""
    print("=== MLX Optimization Validation Benchmark ===")
    print(f"MLX Available: {MLX_AVAILABLE}")

    if not MLX_AVAILABLE:
        print("ERROR: MLX not available. Please install with: pip install mlx")
        return

    try:
        # Run comprehensive benchmarks
        evoformer_results, triangle_results = benchmark_scaling_behavior()

        # Create performance visualizations
        create_performance_plots(evoformer_results, triangle_results)

        # Profile optimization impact
        profile_optimization_impact()

        # Summary
        print("\n=== Optimization Summary ===")
        if evoformer_results:
            avg_speedup = np.mean([r['speedup'] for r in evoformer_results])
            print(f"Average Evoformer Speedup: {avg_speedup:.2f}x")

            if avg_speedup >= 2.0:
                print("✅ SUCCESS: Achieved target 2x+ speedup!")
            else:
                print("⚠️  WARNING: Speedup below 2x target")

        if triangle_results:
            avg_tri_speedup = np.mean([r['speedup'] for r in triangle_results])
            print(f"Average Triangle Speedup: {avg_tri_speedup:.2f}x")

        # Numerical accuracy check
        max_errors = []
        if evoformer_results:
            max_errors.extend([r['max_diff'] for r in evoformer_results])
        if triangle_results:
            max_errors.extend([r['max_diff'] for r in triangle_results])

        if max_errors:
            max_error = max(max_errors)
            print(f"Maximum numerical error: {max_error:.2e}")
            if max_error < 1e-5:
                print("✅ Numerical accuracy maintained")
            else:
                print("⚠️  WARNING: High numerical error detected")

        print("\nOptimization validation complete!")

    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()