#!/usr/bin/env python3
"""
Profile MLX operations to identify computational hotspots and generate
the graphical analysis requested for mx.compile optimization.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import MLX profiling and attention modules
from openfold3.profiling import enable_profiling, get_profiler, disable_profiling
from openfold3.core.model.primitives.attention_mlx import (
    mlx_evoformer_attention,
    mlx_triangle_attention,
    analyze_compilation_opportunities,
    MLX_AVAILABLE
)

def create_test_tensors(batch_size=1, seq_len=256, num_heads=8, head_dim=64):
    """Create test tensors similar to what would be used in protein folding."""
    hidden_dim = num_heads * head_dim

    # Typical sizes for protein folding (sequence length varies, but 256 is common)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # Bias terms (attention mask and triangle bias for structural attention)
    mask_bias = torch.randn(batch_size, 1, 1, seq_len, seq_len)
    triangle_bias = torch.randn(batch_size, num_heads, seq_len, seq_len)

    return q, k, v, [mask_bias, triangle_bias]

def profile_evoformer_attention():
    """Profile evoformer attention operations."""
    print("Profiling evoformer attention operations...")

    # Test different sequence lengths to see scaling behavior
    test_sizes = [128, 256, 512, 1024]

    for seq_len in test_sizes:
        print(f"  Testing sequence length: {seq_len}")

        q, k, v, biases = create_test_tensors(seq_len=seq_len)

        # Test both chunked and non-chunked attention
        for use_chunked in [False, True]:
            chunk_label = "chunked" if use_chunked else "direct"
            print(f"    {chunk_label} attention...")

            try:
                # Run multiple iterations to get stable timing
                for i in range(3):
                    output = mlx_evoformer_attention(
                        q, k, v, biases,
                        use_chunked=use_chunked,
                        chunk_size=512,
                        enable_compilation=True
                    )
                    # Force evaluation
                    _ = output.mean().item()

            except Exception as e:
                print(f"    Error in {chunk_label} attention: {e}")

def profile_triangle_attention():
    """Profile triangle attention operations."""
    print("Profiling triangle attention operations...")

    test_sizes = [128, 256, 384]  # Triangle attention is more memory intensive

    for seq_len in test_sizes:
        print(f"  Testing sequence length: {seq_len}")

        q, k, v, biases = create_test_tensors(seq_len=seq_len)

        try:
            # Run multiple iterations to get stable timing
            for i in range(3):
                output = mlx_triangle_attention(
                    q, k, v, biases,
                    scale=1.0 / np.sqrt(64),  # head_dim = 64
                    enable_compilation=True
                )
                # Force evaluation
                _ = output.mean().item()

        except Exception as e:
            print(f"    Error in triangle attention: {e}")

def simulate_full_forward_pass():
    """Simulate a simplified forward pass with multiple attention operations."""
    print("Profiling simulated forward pass...")

    # Simulate multiple layers and attention types
    seq_len = 256
    num_layers = 4  # Simplified from the full 48 layers

    for layer in range(num_layers):
        print(f"  Layer {layer + 1}/{num_layers}")

        q, k, v, biases = create_test_tensors(seq_len=seq_len)

        # Simulate different attention operations in a typical evoformer block
        operations = [
            ("msa_row_attention", False),
            ("msa_col_attention", False),
            ("pair_triangle_attention_start", True),
            ("pair_triangle_attention_end", True)
        ]

        for op_name, is_triangle in operations:
            try:
                if is_triangle:
                    output = mlx_triangle_attention(
                        q, k, v, biases,
                        scale=1.0 / np.sqrt(64),
                        enable_compilation=True
                    )
                else:
                    output = mlx_evoformer_attention(
                        q, k, v, biases,
                        use_chunked=False,
                        enable_compilation=True
                    )
                # Force evaluation
                _ = output.mean().item()

            except Exception as e:
                print(f"    Error in {op_name}: {e}")

def main():
    """Main profiling function."""
    print("=== MLX Computational Hotspot Profiling ===")
    print(f"MLX Available: {MLX_AVAILABLE}")

    if not MLX_AVAILABLE:
        print("ERROR: MLX not available. Please install with: pip install mlx")
        return

    # Enable profiling
    print("Enabling detailed profiling...")
    enable_profiling()
    profiler = get_profiler()
    profiler.reset()

    try:
        # Run profiling tests
        profile_evoformer_attention()
        profile_triangle_attention()
        simulate_full_forward_pass()

        # Generate analysis
        print("\n=== Profiling Results ===")
        stats = profiler.get_summary_stats()

        print(f"Total operations profiled: {stats.get('total_operations', 0)}")
        print(f"Total time: {stats.get('total_time_ms', 0):.2f} ms")
        print(f"Total FLOPS: {stats.get('total_flops', 0):,}")

        if stats.get('total_flops', 0) > 0 and stats.get('total_time_ms', 0) > 0:
            flops_per_sec = stats['total_flops'] / (stats['total_time_ms'] / 1000)
            print(f"FLOPS/second: {flops_per_sec:,.0f}")

        print(f"Compilation candidates: {stats.get('compilation_candidates', 0)}")

        # Show operation groups
        if 'operation_groups' in stats:
            print(f"\nOperation breakdown:")
            for op_type, group_stats in stats['operation_groups'].items():
                print(f"  {op_type}: {group_stats['count']} ops, "
                      f"{group_stats['total_time']:.2f} ms, "
                      f"{group_stats['total_flops']:,} FLOPS")

        # Get compilation candidates
        candidates = profiler.get_compilation_candidates()
        print(f"\nTop compilation candidates:")
        for i, candidate in enumerate(candidates[:5]):
            print(f"  {i+1}. {candidate.operation_name}: "
                  f"{candidate.duration_ms:.2f} ms, "
                  f"{candidate.flops_estimate or 0:,} FLOPS")

        # Generate comprehensive report with visualizations
        output_dir = Path("profiling_output")
        print(f"\nGenerating detailed profiling report in {output_dir}...")
        profiler.generate_report(output_dir)

        # Generate compilation analysis
        compilation_analysis = analyze_compilation_opportunities()
        print(f"\nCompilation Analysis:")
        print(f"  Recommendation: {compilation_analysis.get('recommendation', 'N/A')}")
        print(f"  Estimated speedup: {compilation_analysis.get('estimated_speedup', 'N/A')}")

        print(f"\nDetailed reports and visualizations saved to: {output_dir.absolute()}")
        print("Files generated:")
        print("  - profiling_summary.json")
        print("  - detailed_profiling.json")
        print("  - timing_breakdown.png")
        print("  - flops_analysis.png")
        print("  - compilation_candidates.png")

    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()

    finally:
        disable_profiling()

if __name__ == "__main__":
    main()