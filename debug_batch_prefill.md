# Batch Prefill NPU Error Debugging Guide

## Problem
NPU error 507011 (D-cache UB OOB) occurs during batch prefill graph execution.

## How to Enable Debugging

### Option 1: Enable all debugging
```bash
export DEBUG_ROPE=1
export DEBUG_BATCH_PREFILL=1
export DEBUG_BUILD_PREFILL=1
```

### Option 2: Enable selective debugging

#### Check positions after right-align
```bash
export DEBUG_BATCH_PREFILL=1
```

#### Check RoPE indexing
```bash
export DEBUG_ROPE=1
```

#### Check build_prefill_metadata
```bash
export DEBUG_BUILD_PREFILL=1
```

## What to Look For

1. **positions min/max**: Should be 0 for padding tokens, valid positions for real tokens
2. **positions values**: Should not exceed `_cos_cache` size (usually model's max_position_embeddings)
3. **num_actual_tokens**: Should be the actual token count, not padded size
4. **input_positions shape**: Should match expected token count

## Expected Behavior

In batch prefill mode with right-alignment:
- Padding tokens have position=0
- Real tokens have their original positions (preserved from orig_pos)
- positions tensor shape = target_bs * target_seq_len
- positions tensor contains both padding and real token positions

## Run Benchmark with Debug

```bash
DEBUG_BATCH_PREFILL=1 DEBUG_ROPE=1 DEBUG_BUILD_PREFILL=1 \
bash examples/disaggregated_prefill_v1/run_laps_sharegpt_benchmark_deepseek_v4_single_node_prefill.sh
```

## Clean Up

To disable debugging:
```bash
unset DEBUG_ROPE
unset DEBUG_BATCH_PREFILL
unset DEBUG_BUILD_PREFILL
```
