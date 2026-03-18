#!/bin/bash
set -euo pipefail

# ========== 固定配置 ==========
MODEL_PATH="/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562/"
VLLM_SERVE_EXTRA_ARGS="--enforce-eager --max-model-len 8192"
QA_SCRIPT_PATH="/root/LMCache/benchmarks/multi_round_qa/multi-round-qa.py"
SUMMARY_CSV_PATH="/root/LMCache/benchmarks/multi_round_qa/summary.csv"
WAIT_AFTER_QA=60          # 等待 vllm serve 输出结束的秒数
COOLDOWN=30               # 每轮结束后冷却秒数
OUTPUT_DIR="/root/benchmark_results"

# ========== 测试用例数组（全自动遍历） ==========
GPU_MEM_UTILS=(0.1 0.85)
QA_ARGS_LIST=(
  "--answer-len 1000 --num-users 50 --num-rounds 5 --qps 0.5 --shared-system-prompt 1000 --user-history-prompt 2000 --time 300"
  "--answer-len 2000 --num-users 100 --num-rounds 3 --qps 1.0 --shared-system-prompt 500 --user-history-prompt 1000 --time 300"
)

# ========== 以下无需修改 ==========
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR/results.txt"
> "$RESULTS_FILE"

echo "========== Benchmark Results ==========" | tee -a "$RESULTS_FILE"

qa_idx=0
for qa_args in "${QA_ARGS_LIST[@]}"; do
  for gpu_mem in "${GPU_MEM_UTILS[@]}"; do
    for shmem in 0 1; do
      tag="gpu${gpu_mem}_qa${qa_idx}_shmem${shmem}"
      log_file="$OUTPUT_DIR/vllm_${tag}.log"
      csv_file="$OUTPUT_DIR/summary_${tag}.csv"

      echo ""
      echo ">>> Starting test: ENABLE_SHMEM=${shmem}, gpu_mem=${gpu_mem}, qa_idx=${qa_idx}"
      echo ">>> QA args: ${qa_args}"
      echo ""

      # 启动 vllm serve
      ENABLE_SHMEM=$shmem vllm serve "$MODEL_PATH" \
        --gpu-memory-utilization "$gpu_mem" \
        $VLLM_SERVE_EXTRA_ARGS > "$log_file" 2>&1 &
      VLLM_PID=$!
      echo ">>> vllm serve started, PID=${VLLM_PID}"

      # 等待 vllm serve 就绪
      echo ">>> Waiting for vllm serve to be ready..."
      elapsed=0
      while ! curl -sf http://localhost:8000/health > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
          echo ">>> ERROR: vllm serve exited unexpectedly. Check log: ${log_file}"
          echo "[${tag}] ERROR: vllm serve exited unexpectedly" | tee -a "$RESULTS_FILE"
          break 2  # skip this shmem value, try next
        fi
        if [ "$elapsed" -ge 300 ]; then
          echo ">>> ERROR: vllm serve did not become ready within 300s. Killing..."
          kill "$VLLM_PID" 2>/dev/null || true
          wait "$VLLM_PID" 2>/dev/null || true
          echo "[${tag}] ERROR: vllm serve timeout" | tee -a "$RESULTS_FILE"
          sleep "$COOLDOWN"
          continue 2  # skip to next iteration
        fi
        sleep 5
        elapsed=$((elapsed + 5))
      done
      echo ">>> vllm serve is ready (took ${elapsed}s)"

      # 运行 multi-round-qa.py
      echo ">>> Running multi-round-qa.py..."
      python3 "$QA_SCRIPT_PATH" \
        --model "$MODEL_PATH" \
        --base-url http://localhost:8000/v1 \
        $qa_args || true

      # 保留原始 summary.csv
      if [ -f "$SUMMARY_CSV_PATH" ]; then
        cp "$SUMMARY_CSV_PATH" "$csv_file"
        echo ">>> Saved summary CSV to ${csv_file}"
      else
        echo ">>> WARNING: summary.csv not found at ${SUMMARY_CSV_PATH}"
      fi

      # 解析 TTFT（第3列，跳过表头）
      ttft_avg="N/A"
      ttft_max="N/A"
      ttft_min="N/A"
      if [ -f "$csv_file" ]; then
        read -r ttft_avg ttft_max ttft_min <<< "$(awk -F',' 'NR>1 && $3!="" {
          sum += $3; count++;
          if (count == 1 || $3 > max) max = $3;
          if (count == 1 || $3 < min) min = $3;
        } END {
          if (count > 0) printf "%.4f %.4f %.4f", sum/count, max, min;
          else printf "N/A N/A N/A";
        }' "$csv_file")"
      fi

      # 等待 vllm serve 输出结束
      echo ">>> Waiting ${WAIT_AFTER_QA}s for vllm serve to finish logging..."
      sleep "$WAIT_AFTER_QA"

      # 解析 Prefix cache hit rate
      prefix_hit_rate="N/A"
      if [ -f "$log_file" ]; then
        hit_line=$(grep "Prefix cache hit rate" "$log_file" | tail -1 || true)
        if [ -n "$hit_line" ]; then
          prefix_hit_rate=$(echo "$hit_line" | grep -oP 'Prefix cache hit rate: \K[0-9.]+%' || echo "N/A")
        fi
      fi

      # 关闭 vllm serve
      echo ">>> Killing vllm serve (PID=${VLLM_PID})..."
      kill "$VLLM_PID" 2>/dev/null || true
      wait "$VLLM_PID" 2>/dev/null || true
      echo ">>> vllm serve stopped"

      # 确认端口 8000 已释放，防止下一轮健康检查误判
      echo ">>> Waiting for port 8000 to be released..."
      port_wait=0
      while curl -sf http://localhost:8000/health > /dev/null 2>&1; do
        if [ "$port_wait" -ge 60 ]; then
          echo ">>> WARNING: port 8000 still occupied after 60s, force killing any remaining process..."
          fuser -k 8000/tcp 2>/dev/null || true
          sleep 2
          break
        fi
        sleep 2
        port_wait=$((port_wait + 2))
      done
      echo ">>> Port 8000 released"

      # 记录结果
      {
        echo ""
        echo "[gpu_mem=${gpu_mem}, qa_args=\"${qa_args}\", ENABLE_SHMEM=${shmem}]"
        echo "  TTFT avg: ${ttft_avg}"
        echo "  TTFT max: ${ttft_max}"
        echo "  TTFT min: ${ttft_min}"
        echo "  Prefix cache hit rate: ${prefix_hit_rate}"
      } | tee -a "$RESULTS_FILE"

      # 冷却
      echo ""
      echo ">>> Cooling down ${COOLDOWN}s..."
      sleep "$COOLDOWN"
    done
  done
  qa_idx=$((qa_idx + 1))
done

echo "" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo ""
echo ">>> All tests completed. Results saved to ${RESULTS_FILE}"
