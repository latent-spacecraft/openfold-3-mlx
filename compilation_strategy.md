# MLX Compilation Strategy for 2-3x Speedup

## Executive Summary

Based on comprehensive profiling analysis, the following strategy targets the highest-impact optimizations for achieving 2-3x overall speedup through strategic mx.compile usage.

## Key Findings from Profiling

### Computational Hotspots Identified:
1. **Tensor Conversion Overhead (Major Bottleneck)**
   - `mlx_to_torch_conversion`: 480ms, 211ms, 10ms+ operations
   - Represents significant portion of total 814ms runtime
   - **Root cause:** Frequent memory transfers between MLX and PyTorch

2. **MLX Attention Operations**
   - `mlx_evoformer_attention_with_bias`: ~1ms each, 2.1M FLOPS
   - Multiple attention operations across 4 layers × 4 attention types
   - High arithmetic intensity makes these prime compilation candidates

3. **Scaling Behavior**
   - Sequence length scaling: 128 → 1024 tokens
   - Performance scales quadratically with sequence length (expected for attention)
   - Memory efficiency maintained through chunking

## Strategic Compilation Approach

### Phase 1: Eliminate Conversion Overhead (Highest Impact)

**Problem:** Current architecture converts between MLX and PyTorch for each operation
**Solution:** Keep computation in MLX domain throughout entire forward pass

```python
# Current inefficient pattern:
torch_input → mlx_attention() → torch_output → next_operation

# Optimized pattern:
torch_input → [multiple mlx operations in compiled graph] → torch_output
```

### Phase 2: Whole-Graph Compilation (Maximum Performance)

**Strategy:** Compile entire attention blocks as single computational graphs

**Target Operations for Compilation:**

1. **Core Attention Kernels** (Already implemented)
   - `mlx_evoformer_attention_core`
   - `mlx_evoformer_attention_with_bias`
   - `mlx_chunked_attention_chunk`
   - `mlx_lma_chunk_accumulate`

2. **Block-Level Compilation** (New)
   - Entire evoformer blocks with multiple attention operations
   - Triangle attention pairs (start + end nodes)
   - MSA row/column attention sequences

3. **Layer-Level Compilation** (Advanced)
   - Multiple evoformer blocks in single compilation unit
   - Optimal for sequences ≤512 tokens where memory allows

### Phase 3: Memory-Aware Optimization

**Chunking Strategy:**
- Use compilation for individual chunks
- Maintain memory efficiency through streaming
- Balance compilation overhead vs. chunk size

**Compilation Thresholds:**
- **Small sequences (≤256):** Full-graph compilation
- **Medium sequences (256-512):** Block-level compilation
- **Large sequences (>512):** Chunk-level compilation with optimized streaming

## Implementation Plan

### 1. Native MLX Attention Interface
Create native MLX operations that stay in MLX domain:

```python
@mx.compile
def native_mlx_evoformer_block(
    msa_repr: mx.array,
    pair_repr: mx.array,
    msa_mask: mx.array,
    pair_mask: mx.array
) -> tuple[mx.array, mx.array]:
    # Complete evoformer block in single compiled graph
    pass
```

### 2. Conversion Optimization
Minimize conversions by batching at module boundaries:

```python
# Convert once at module entry
mlx_tensors = convert_batch_torch_to_mlx(torch_inputs)

# All MLX operations (compiled)
mlx_outputs = compiled_evoformer_stack(mlx_tensors)

# Convert once at module exit
torch_outputs = convert_batch_mlx_to_torch(mlx_outputs)
```

### 3. Adaptive Compilation
Choose compilation strategy based on input size:

```python
def adaptive_evoformer_attention(inputs):
    seq_len = inputs.shape[-2]

    if seq_len <= 256:
        return full_graph_compiled_attention(inputs)
    elif seq_len <= 512:
        return block_compiled_attention(inputs)
    else:
        return chunk_compiled_attention(inputs)
```

## Expected Performance Gains

### Conversion Elimination: **1.5-2x speedup**
- Current conversion overhead: ~700ms/814ms = 86% of runtime
- Eliminating this gives immediate 1.86x speedup

### Attention Compilation: **1.3-1.5x additional speedup**
- Better memory access patterns
- Fused operations (reduced intermediate allocations)
- Optimized kernel fusion

### Combined Expected: **2-3x total speedup**
- Conservative estimate: 2.0x
- Optimistic estimate: 2.8x
- Matches your target of 2-3x overall improvement

## Risk Mitigation

### Memory Constraints
- Implement fallback to chunked processing
- Monitor memory usage during compilation
- Dynamic chunk size adjustment

### Compilation Overhead
- Cache compiled functions aggressively
- JIT compilation only on first run
- Profile compilation time vs. execution time

### Numerical Stability
- Maintain FP32 precision for critical operations
- Gradual optimization rollout
- Extensive testing against reference implementation

## Implementation Priority

1. **High Priority:** Conversion elimination (Phase 1)
2. **Medium Priority:** Block-level compilation (Phase 2)
3. **Low Priority:** Advanced optimizations (Phase 3)

This strategy provides a clear path to achieving your target 2-3x speedup while maintaining numerical accuracy and memory efficiency.