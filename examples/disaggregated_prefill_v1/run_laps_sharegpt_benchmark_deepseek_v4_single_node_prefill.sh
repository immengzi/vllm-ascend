#!/usr/bin/env bash
set -euo pipefail

# Run a ShareGPT benchmark for DeepSeek-V4 on a single-node Prefill-only deployment.
#
# This script runs ShareGPT multi-turn conversation benchmark against a local vLLM
# Prefill server for every LAPS variant. Output generation is capped to keep the
# measured path dominated by Prefill.
#
# Test Variants (CASE_VARIANTS):
#   Format: t<threshold>[:m<long_max_wait_ms>]
#
#   Examples:
#   off                    - No LAPS
#   t256                   - LAPS, strict short-priority (no aging)
#   t256:m2000             - LAPS, long requests aged in after 2000ms
#   t256:m500              - LAPS, long requests aged in after 500ms
#
# Default Variants:
#   off                    - Baseline (no LAPS)
#   t256                   - LAPS, no aging
#   t256:m2000             - LAPS, anti-starvation aging at 2000ms
#
# Environment Variables:
#   LAPS_LONG_MAX_WAIT_MS - Default long-request aging bound, ms (default: 0)
#   REQUEST_RATES         - Request rates for benchmark (default: "4 6 8 10")
#
# Example usage:
#   # Use default variants
#   bash run_laps_sharegpt_benchmark_deepseek_v4_single_node_prefill.sh
#
#   # Custom variants and request rates
#   CASE_VARIANTS="off t256 t256:m2000" REQUEST_RATES="4 8" bash script.sh

MODEL_PATH="${MODEL_PATH:-/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek_v4}"
RAW_DATASET_PATH="${RAW_DATASET_PATH:-/vllm-workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
MULTITURN_DIR="${MULTITURN_DIR:-${VLLM_DIR}/benchmarks/multi_turn}"
RESULT_DIR="${RESULT_DIR:-/vllm-workspace/bench_results/deepseek_v4_sharegpt_prefill_only_$(date +%Y%m%d_%H%M%S)}"
CONV_DATASET_PATH="${CONV_DATASET_PATH:-${RESULT_DIR}/sharegpt_conv.json}"
RESULTS_CSV="results.csv"
RAW_DATA_DIRNAME="${RAW_DATA_DIRNAME:-raw_data}"

CASE_VARIANTS="${CASE_VARIANTS:-off t256 t256:m2000}"
BENCH_REPEAT="${BENCH_REPEAT:-3}"
CASE_RETRY_LIMIT="${CASE_RETRY_LIMIT:-1}"

PREFILL_BIND_HOST="${PREFILL_BIND_HOST:-0.0.0.0}"
PREFILL_CONNECT_HOST="${PREFILL_CONNECT_HOST:-127.0.0.1}"
PREFILL_NODE_IP="${PREFILL_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
PREFILL_NIC_NAME="${PREFILL_NIC_NAME:-enp48s3u1u1c2}"
PREFILL_DEVICES="${PREFILL_DEVICES:-0,1,2,3,4,5,6,7}"
PREFILL_PORT="${PREFILL_PORT:-6800}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30100}"

PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-1}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-1}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
PREFILL_MAX_MODEL_LEN="${PREFILL_MAX_MODEL_LEN:-131072}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS:-4096}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-64}"
PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.95}"
PREFILL_BLOCK_SIZE="${PREFILL_BLOCK_SIZE:-128}"
PREFILL_API_SERVER_COUNT="${PREFILL_API_SERVER_COUNT:-1}"
PREFILL_MODEL_LOADER_NUM_THREADS="${PREFILL_MODEL_LOADER_NUM_THREADS:-16}"
PREFILL_ENABLE_PREFIX_CACHING="${PREFILL_ENABLE_PREFIX_CACHING:-0}"
PREFILL_ENABLE_CHUNKED_PREFILL="${PREFILL_ENABLE_CHUNKED_PREFILL:-0}"
PREFILL_SPECULATIVE_TOKENS="${PREFILL_SPECULATIVE_TOKENS:-1}"
PREFILL_SPECULATIVE_METHOD="${PREFILL_SPECULATIVE_METHOD:-deepseek_mtp}"

# ShareGPT / Multi-turn benchmark config
MAX_ITEMS="${MAX_ITEMS:-128}"
MIN_TURNS="${MIN_TURNS:-8}"
MAX_TURNS="${MAX_TURNS:-20}"
CONVERT_MAX_CONTENT_LEN="${CONVERT_MAX_CONTENT_LEN:-12000}"
NUM_CLIENTS="${NUM_CLIENTS:-24}"
MAX_ACTIVE_CONVERSATIONS="${MAX_ACTIVE_CONVERSATIONS:-96}"
REQUEST_RATES="${REQUEST_RATES:-4 6 8 10}"
WARMUP_STEP="${WARMUP_STEP:-1}"
LIMIT_MIN_TOKENS="${LIMIT_MIN_TOKENS:-1}"
LIMIT_MAX_TOKENS="${LIMIT_MAX_TOKENS:-1}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-300}"
MAX_RETRIES="${MAX_RETRIES:-0}"
CONVERSATION_SAMPLING="${CONVERSATION_SAMPLING:-round_robin}"
FORMAL_NO_EARLY_STOP="${FORMAL_NO_EARLY_STOP:-1}"
CONVERT_SAMPLE_FACTOR="${CONVERT_SAMPLE_FACTOR:-8}"

# LAPS Configuration
LAPS_LONG_MAX_WAIT_MS="${LAPS_LONG_MAX_WAIT_MS:-0}"
LAPS_STATS_LOG_INTERVAL_S="${LAPS_STATS_LOG_INTERVAL_S:-5}"

# Batch Prefill Graph Configuration
# ShareGPT conversations are typically shorter, use 512 as max
BATCH_PREFILL_GRAPH="${BATCH_PREFILL_GRAPH:-0}"
BATCH_PREFILL_MAX_SEQ_LEN="${BATCH_PREFILL_MAX_SEQ_LEN:-512}"
BATCH_PREFILL_BATCH_SIZES="${BATCH_PREFILL_BATCH_SIZES:-1,2,4,8}"
BATCH_PREFILL_SEQ_LENGTHS="${BATCH_PREFILL_SEQ_LENGTHS:-32,64,128,256,512}"

ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-30000}"
ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-60000}"
HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-17}"
HCCL_RDMA_RETRY_CNT="${HCCL_RDMA_RETRY_CNT:-7}"

STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-60}"
SLEEP_AFTER_STOP_S="${SLEEP_AFTER_STOP_S:-10}"
PORT_FREE_TIMEOUT_S="${PORT_FREE_TIMEOUT_S:-30}"

PREFILL_PID=""

mkdir -p "${RESULT_DIR}/logs"
mkdir -p "${RESULT_DIR}/${RAW_DATA_DIRNAME}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_cmd_presence() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    log "Command available: ${name} -> $(command -v "${name}")"
  else
    log "Command not available: ${name}"
  fi
}

log_local_tooling() {
  log "Checking local tooling"
  log_cmd_presence "curl"
  log_cmd_presence "pgrep"
  log_cmd_presence "ps"
  log_cmd_presence "ss"
  log_cmd_presence "lsof"
  log_cmd_presence "setsid"
  log_cmd_presence "python3"
}

source_env() {
  set +u
  [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  [ -f /usr/local/Ascend/cann-8.5.1/set_env.sh ] && source /usr/local/Ascend/cann-8.5.1/set_env.sh
  [ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
  [ -f /usr/local/Ascend/nnal/asdsip/set_env.sh ] && source /usr/local/Ascend/nnal/asdsip/set_env.sh
  set -euo pipefail
  export PYTHONPATH="${VLLM_DIR}:${PYTHONPATH:-}"
  export VLLM_USE_MODELSCOPE=False
  export HCCL_OP_EXPANSION_MODE=AIV
}

unset_proxy_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
}

kill_tree() {
  local pid="$1"
  [ -z "${pid}" ] && return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  local children
  children="$(pgrep -P "${pid}" 2>/dev/null || true)"
  for child in ${children}; do
    kill_tree "${child}"
  done

  kill "${pid}" 2>/dev/null || true
}

wait_gone() {
  local pid="$1"
  local deadline=$((SECONDS + STOP_TIMEOUT_S))
  [ -z "${pid}" ] && return 0

  while kill -0 "${pid}" 2>/dev/null; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Force killing pid=${pid}"
      kill -9 "${pid}" 2>/dev/null || true
      break
    fi
    sleep 1
  done
}

kill_matching_cmd() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "${pattern}" 2>/dev/null || true)"
  for pid in ${pids}; do
    [ "${pid}" = "$$" ] && continue
    [ "${pid}" = "${BASHPID}" ] && continue
    log "Stopping existing process pid=${pid}, pattern=${pattern}"
    kill_tree "${pid}"
  done
  for pid in ${pids}; do
    [ "${pid}" = "$$" ] && continue
    [ "${pid}" = "${BASHPID}" ] && continue
    wait_gone "${pid}"
  done
}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -Eq "[:.]${port}[[:space:]]"
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

print_port_users() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN || true
  fi
}

wait_port_free() {
  local port="$1"
  local name="$2"
  local deadline=$((SECONDS + PORT_FREE_TIMEOUT_S))

  while port_in_use "${port}"; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Port ${port} for ${name} is still in use after cleanup:"
      print_port_users "${port}"
      return 1
    fi
    sleep 1
  done
  log "Port ${port} for ${name} is free"
}

ensure_ports_free() {
  wait_port_free "${PREFILL_PORT}" "Prefill"
}

stop_existing_services() {
  kill_matching_cmd "vllm serve .*--port ${PREFILL_PORT}"
}

stop_services() {
  log "Stopping previous services"
  kill_tree "${PREFILL_PID}"
  stop_existing_services
  wait_gone "${PREFILL_PID}"
  PREFILL_PID=""
  sleep "${SLEEP_AFTER_STOP_S}"
  ensure_ports_free
}

cleanup() {
  stop_services || true
}
trap cleanup EXIT

wait_http() {
  local url="$1"
  local name="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  log "Waiting for ${name}: ${url}"
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Timed out waiting for ${name}"
      return 1
    fi
    sleep 2
  done
  log "${name} is ready"
}

wait_log() {
  local file="$1"
  local pattern="$2"
  local name="$3"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  log "Waiting for ${name} log pattern: ${pattern}"
  until grep -q "${pattern}" "${file}" 2>/dev/null; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Timed out waiting for ${name}; last 80 log lines:"
      tail -80 "${file}" || true
      return 1
    fi
    sleep 2
  done
  log "${name} is ready"
}

scrape_metrics_snapshot() {
  local service_url="$1"
  local output_file="$2"
  log "Scraping metrics from ${service_url}"
  if ! curl -fsS "${service_url}/metrics" > "${output_file}"; then
    log "Failed to scrape metrics from ${service_url}/metrics"
    return 1
  fi
}

append_case_status() {
  local case_name="$1"
  local status="$2"
  local detail="$3"
  python3 - "$case_name" "$status" "$detail" >> "${RESULT_DIR}/case_status.csv" <<'PY'
import csv
import sys
from datetime import datetime

csv.writer(sys.stdout, lineterminator="\n").writerow([
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
])
PY
}

case_variant_to_params() {
  local variant="$1"
  if [ "${variant}" = "off" ]; then
    printf 'off|||off|0\n'
    return
  fi

  # Parse LAPS variants with optional config: t256:m2000
  # t256  - threshold
  # m2000 - long_max_wait_ms anti-starvation aging bound (optional, default env)
  #
  # The emitted tuple keeps its historical 5-field shape so downstream CSV/argv
  # plumbing stays stable. Field 2 now carries long_max_wait_ms; fields 3 and 5
  # (formerly wait_max_batch and batch-prefill-graph) are retired and pinned 0.
  local threshold=""
  local long_max_wait_ms="${LAPS_LONG_MAX_WAIT_MS}"

  if [[ "${variant}" =~ ^t([0-9]+)(:.*)?$ ]]; then
    threshold="${BASH_REMATCH[1]}"
    local config="${BASH_REMATCH[2]:-}"

    if [[ "${config}" =~ :m([0-9]+) ]]; then
      long_max_wait_ms="${BASH_REMATCH[1]}"
    fi

    local suffix="laps_t${threshold}_m${long_max_wait_ms}"
    printf '%s|%s|%s|%s|%s\n' "${threshold}" "${long_max_wait_ms}" "0" "${suffix}" "0"
    return
  fi

  log "Unknown variant: ${variant}"
  return 1
}

build_conversation_dataset() {
  log "Building ShareGPT multi-turn conversation replay dataset"
  mkdir -p "$(dirname "${CONV_DATASET_PATH}")"
  python3 - \
    "${RAW_DATASET_PATH}" \
    "${CONV_DATASET_PATH}" \
    "${MAX_ITEMS}" \
    "${MIN_TURNS}" \
    "${MAX_TURNS}" \
    "${MAX_ACTIVE_CONVERSATIONS}" \
    "${CONVERT_MAX_CONTENT_LEN}" \
    "${CONVERT_SAMPLE_FACTOR}" <<'PY'
import json
import random
import sys
from pathlib import Path

raw_dataset_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
max_items = int(sys.argv[3])
min_turns = int(sys.argv[4])
max_turns = int(sys.argv[5])
max_active_conversations = int(sys.argv[6])
max_content_len = int(sys.argv[7])
convert_sample_factor = int(sys.argv[8])

seed = 99
random.seed(seed)

with raw_dataset_path.open("r", encoding="utf-8") as f:
    raw_data = json.load(f)

if not isinstance(raw_data, list):
    raise SystemExit("ERROR: RAW_DATASET_PATH should contain a list of ShareGPT items")

conversation_parts: dict[str, list[list[dict[str, object]]]] = {}
raw_items_seen = 0
for item in raw_data:
    if not isinstance(item, dict):
        continue
    raw_items_seen += 1
    item_id = item.get("id")
    conversations = item.get("conversations")
    if not isinstance(item_id, str) or not isinstance(conversations, list):
        continue

    conv_id, _, _ = item_id.partition("_")
    new_turns = [turn for turn in conversations if isinstance(turn, dict)]
    if conv_id not in conversation_parts:
        conversation_parts[conv_id] = []
    elif conversation_parts[conv_id] and new_turns:
        prev_turns = conversation_parts[conv_id][-1]
        if prev_turns and prev_turns[-1].get("from") == new_turns[0].get("from"):
            new_turns = new_turns[1:]
    if new_turns:
        conversation_parts[conv_id].append(new_turns)


def role_from_source(turn_from: object) -> str | None:
    if turn_from in {"human", "user"}:
        return "user"
    if turn_from in {"gpt", "bing", "chatgpt", "bard"}:
        return "assistant"
    return None


odd_tail_trimmed = 0
role_filtered = 0
content_filtered = 0
turn_count_filtered = 0
valid_conversations: list[dict[str, object]] = []

for conv_id, conv_parts in conversation_parts.items():
    merged_turns: list[dict[str, object]] = []
    for conv_part in conv_parts:
        merged_turns.extend(conv_part)

    if len(merged_turns) < min_turns:
        turn_count_filtered += 1
        continue

    normalized: list[dict[str, str]] = []
    failed_role = False
    failed_content = False

    for turn in merged_turns:
        role = role_from_source(turn.get("from"))
        if role is None:
            failed_role = True
            break

        content = turn.get("value")
        if not isinstance(content, str) or not content.strip() or len(content) > max_content_len:
            failed_content = True
            break

        normalized.append({"role": role, "content": content})
        if len(normalized) >= max_turns:
            break

    if failed_role:
        role_filtered += 1
        continue
    if failed_content:
        content_filtered += 1
        continue
    if not normalized:
        turn_count_filtered += 1
        continue

    while normalized and normalized[0]["role"] != "user":
        normalized.pop(0)

    if len(normalized) % 2 == 1:
        normalized = normalized[:-1]
        odd_tail_trimmed += 1

    if len(normalized) < 2:
        turn_count_filtered += 1
        continue
    if normalized[-1]["role"] != "assistant":
        role_filtered += 1
        continue

    expected = "user"
    alternating = True
    for msg in normalized:
        if msg["role"] != expected:
            alternating = False
            break
        expected = "assistant" if expected == "user" else "user"
    if not alternating:
        role_filtered += 1
        continue

    if len(normalized) < min_turns:
        turn_count_filtered += 1
        continue

    valid_conversations.append({"id": conv_id, "messages": normalized})

candidate_target = max(max_items, max_items * max(convert_sample_factor, 1))
candidate_pool = list(valid_conversations)
if len(candidate_pool) > candidate_target:
    candidate_pool = random.sample(candidate_pool, candidate_target)

final_dataset = list(candidate_pool)
if len(final_dataset) > max_items:
    final_dataset = random.sample(final_dataset, max_items)

final_count = len(final_dataset)
raw_conv_count = len(conversation_parts)
valid_count = len(valid_conversations)

print(f"dataset_builder=local_cleaning seed={seed}")
print(f"raw_items_seen={raw_items_seen}")
print(f"raw_conversations={raw_conv_count}")
print(f"valid_conversations={valid_count}")
print(f"odd_tail_trimmed={odd_tail_trimmed}")
print(f"role_filtered={role_filtered}")
print(f"content_filtered={content_filtered}")
print(f"turn_count_filtered={turn_count_filtered}")
print(f"candidate_target={candidate_target}")
print(f"candidate_pool={len(candidate_pool)}")
print(f"final_sampled_conversations={final_count}")

if final_count == 0:
    raise SystemExit(
        "ERROR: final cleaned conversation count is zero; "
        f"MAX_ITEMS={max_items}, MIN_TURNS={min_turns}, MAX_TURNS={max_turns}, "
        f"CONVERT_MAX_CONTENT_LEN={max_content_len}"
    )

if final_count < max_active_conversations:
    raise SystemExit(
        "ERROR: final cleaned conversation count is smaller than MAX_ACTIVE_CONVERSATIONS; "
        f"final_count={final_count}, MAX_ACTIVE_CONVERSATIONS={max_active_conversations}, "
        f"MAX_ITEMS={max_items}, MIN_TURNS={min_turns}, MAX_TURNS={max_turns}, "
        f"CONVERT_MAX_CONTENT_LEN={max_content_len}, CONVERT_SAMPLE_FACTOR={convert_sample_factor}"
    )

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)
PY
}

write_summary_header() {
  cat >"${RESULT_DIR}/${RESULTS_CSV}" <<'EOF'
case_name,repeat_id,variant,laps_enabled,laps_threshold,laps_wait_window_ms,laps_wait_max_batch,batch_prefill_graph,batch_prefill_max_seq_len,batch_prefill_batch_sizes,batch_prefill_seq_lengths,request_rate,completed,failed,ttft_mean_ms,ttft_median_ms,ttft_p99_ms,tpot_mean_ms,itl_mean_ms,itl_median_ms,e2el_mean_s,e2el_median_s,e2el_p99_s,total_input_tokens,total_output_tokens,request_throughput,total_token_throughput,prefill_queue_avg_ms,prefill_queue_median_ms,prefill_queue_p90_ms,prefill_queue_p99_ms,prefill_time_avg_ms,prefill_ttft_avg_ms,prefill_ttft_median_ms,prefill_ttft_p90_ms,prefill_ttft_p99_ms,summary_log
EOF
}

start_prefill() {
  local case_name="$1"
  local laps_threshold="$2"
  local laps_wait_window_ms="$3"
  local laps_wait_max_batch="$4"
  local use_batch_prefill_graph="${5:-0}"
  local log_file="${RESULT_DIR}/logs/${case_name}_prefill.log"
  local kv_transfer_config_json
  local prefill_extra_args=()

  kv_transfer_config_json="{\"kv_connector\":\"MooncakeHybridConnector\",\"kv_role\":\"kv_producer\",\"kv_port\":\"${PREFILL_KV_PORT}\",\"engine_id\":\"0\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":${PREFILL_DP_SIZE},\"tp_size\":${PREFILL_TP_SIZE}},\"decode\":{\"dp_size\":${DECODE_DP_SIZE},\"tp_size\":${DECODE_TP_SIZE}}}}"

  if [ "${PREFILL_ENABLE_CHUNKED_PREFILL}" = "1" ]; then
    prefill_extra_args+=(--enable-chunked-prefill)
  fi

  if [ "${PREFILL_ENABLE_PREFIX_CACHING}" = "1" ]; then
    prefill_extra_args+=(--enable-prefix-caching)
  else
    prefill_extra_args+=(--no-enable-prefix-caching)
  fi

  (
    source_env
    unset_proxy_env
    export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_IF_IP="${PREFILL_NODE_IP}"
    export GLOO_SOCKET_IFNAME="${PREFILL_NIC_NAME}"
    export TP_SOCKET_IFNAME="${PREFILL_NIC_NAME}"
    export HCCL_SOCKET_IFNAME="${PREFILL_NIC_NAME}"
    export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT}"
    export ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT}"
    export HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT}"
    export HCCL_RDMA_RETRY_CNT="${HCCL_RDMA_RETRY_CNT}"
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=1024
    export ACL_OP_INIT_MODE=1
    export ASCEND_A3_ENABLE=1
    export USE_MULTI_GROUPS_KV_CACHE=1
    export USE_MULTI_BLOCK_POOL=1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
    export ASCEND_RT_VISIBLE_DEVICES="${PREFILL_DEVICES}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib"

    if [ "${laps_threshold}" = "off" ]; then
      unset VLLM_ASCEND_LAPS_SCHEDULING VLLM_ASCEND_LAPS_THRESHOLD VLLM_ASCEND_LAPS_LONG_MAX_WAIT_MS VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S
    else
      export VLLM_ASCEND_LAPS_SCHEDULING=1
      export VLLM_ASCEND_LAPS_THRESHOLD="${laps_threshold}"
      # laps_wait_window_ms carries the long_max_wait_ms aging bound (tuple field 2).
      export VLLM_ASCEND_LAPS_LONG_MAX_WAIT_MS="${laps_wait_window_ms}"
      export VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S="${LAPS_STATS_LOG_INTERVAL_S}"
    fi

    # Batch Prefill Graph Configuration
    if [ "${use_batch_prefill_graph}" = "1" ]; then
      export VLLM_ASCEND_BATCH_PREFILL_GRAPH=1
      export VLLM_ASCEND_BATCH_PREFILL_MAX_SEQ_LEN="${BATCH_PREFILL_MAX_SEQ_LEN}"
      export VLLM_ASCEND_BATCH_PREFILL_BATCH_SIZES="${BATCH_PREFILL_BATCH_SIZES}"
      export VLLM_ASCEND_BATCH_PREFILL_SEQ_LENGTHS="${BATCH_PREFILL_SEQ_LENGTHS}"
    else
      unset VLLM_ASCEND_BATCH_PREFILL_GRAPH VLLM_ASCEND_BATCH_PREFILL_MAX_SEQ_LEN VLLM_ASCEND_BATCH_PREFILL_BATCH_SIZES VLLM_ASCEND_BATCH_PREFILL_SEQ_LENGTHS
    fi

    cd "${VLLM_ASCEND_DIR}"
    exec setsid vllm serve "${MODEL_PATH}" \
      --host "${PREFILL_BIND_HOST}" \
      --port "${PREFILL_PORT}" \
      --data-parallel-size "${PREFILL_DP_SIZE}" \
      --tensor-parallel-size "${PREFILL_TP_SIZE}" \
      --enable-expert-parallel \
      --seed 1024 \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --max-model-len "${PREFILL_MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${PREFILL_MAX_NUM_SEQS}" \
      --trust-remote-code \
      --gpu-memory-utilization "${PREFILL_GPU_MEMORY_UTILIZATION}" \
      --quantization ascend \
      --speculative-config "{\"num_speculative_tokens\": ${PREFILL_SPECULATIVE_TOKENS}, \"method\": \"${PREFILL_SPECULATIVE_METHOD}\"}" \
      --async-scheduling \
      --block-size "${PREFILL_BLOCK_SIZE}" \
      --enable-auto-tool-choice \
      --tokenizer-mode deepseek_v4 \
      --tool-call-parser deepseek_v4 \
      --reasoning-parser deepseek_v4 \
      --safetensors-load-strategy prefetch \
      "${prefill_extra_args[@]}" \
      --api-server-count "${PREFILL_API_SERVER_COUNT}" \
      --no-disable-hybrid-kv-cache-manager \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,4,6,8,10,12,14,16,18,20,22,24,32,36,40]}' \
      --model-loader-extra-config "{\"enable_multithread_load\":true,\"num_threads\":${PREFILL_MODEL_LOADER_NUM_THREADS}}" \
      --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":false,"multistream_dsa_preprocess":false,"recompute_scheduler_enable":true}' \
      --kv-transfer-config "${kv_transfer_config_json}"
  ) >"${log_file}" 2>&1 &
  PREFILL_PID=$!
  log "Started Prefill pid=${PREFILL_PID}, log=${log_file}"
}

append_summary() {
  local case_name="$1"
  local repeat_id="$2"
  local variant_name="$3"
  local threshold="$4"
  local laps_wait_window_ms="$5"
  local laps_wait_max_batch="$6"
  local use_batch_prefill_graph="$7"
  local batch_prefill_max_seq_len="$8"
  local batch_prefill_batch_sizes="$9"
  local batch_prefill_seq_lengths="${10}"
  local request_rate="${11}"
  local prefill_before="${12}"
  local prefill_after="${13}"
  local summary_log="${14}"
  python3 - "$case_name" "$repeat_id" "$variant_name" "$threshold" "$laps_wait_window_ms" "$laps_wait_max_batch" "$use_batch_prefill_graph" "$batch_prefill_max_seq_len" "$batch_prefill_batch_sizes" "$batch_prefill_seq_lengths" "$request_rate" "$prefill_before" "$prefill_after" "$summary_log" >> "${RESULT_DIR}/${RESULTS_CSV}" <<'PY'
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

(
    case_name,
    repeat_id,
    variant_name,
    threshold,
    laps_wait_window_ms,
    laps_wait_max_batch,
    use_batch_prefill_graph,
    batch_prefill_max_seq_len,
    batch_prefill_batch_sizes,
    batch_prefill_seq_lengths,
    request_rate,
    prefill_before,
    prefill_after,
    summary_log_path,
) = sys.argv[1:15]

bench_log = Path(summary_log_path)
text = bench_log.read_text(encoding="utf-8", errors="ignore") if bench_log.exists() else ""

HISTOGRAM_METRICS = [
    "vllm:time_to_first_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
]

def parse_prom(path: Path) -> dict[str, float | list[tuple[float, float]]]:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    out: dict[str, float | list[tuple[float, float]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_and_labels, value_text = line.rsplit(" ", 1)
        metric_name = metric_and_labels.split("{", 1)[0]
        value = float(value_text)
        if "_bucket{" in metric_and_labels:
            base = metric_and_labels.split("_bucket{", 1)[0]
            if base not in HISTOGRAM_METRICS:
                continue
            labels = metric_and_labels.split("{", 1)[1].rstrip("}")
            le_value = None
            for item in labels.split(","):
                if item.startswith('le='):
                    le_value = item.split("=", 1)[1].strip('"')
                    break
            if le_value is None:
                continue
            upper = math.inf if le_value == "+Inf" else float(le_value)
            out.setdefault(base + "_bucket", []).append((upper, value))
            continue
        for name in HISTOGRAM_METRICS:
            if metric_name == name + "_sum":
                out[name + "_sum"] = value
            elif metric_name == name + "_count":
                out[name + "_count"] = value
    return out

def avg_ms(before, after, metric_name):
    ds = after.get(metric_name + "_sum", 0.0) - before.get(metric_name + "_sum", 0.0)
    dc = after.get(metric_name + "_count", 0.0) - before.get(metric_name + "_count", 0.0)
    if dc <= 0:
        return ""
    return f"{(ds / dc) * 1000.0:.6f}"

def delta_buckets(before, after, metric_name):
    before_map = {upper: count for upper, count in before.get(metric_name + "_bucket", [])}
    after_map = {upper: count for upper, count in after.get(metric_name + "_bucket", [])}
    uppers = sorted(set(before_map) | set(after_map))
    return [(upper, after_map.get(upper, 0.0) - before_map.get(upper, 0.0)) for upper in uppers]

def histogram_quantile_ms(q, cumulative_buckets):
    buckets = list(cumulative_buckets)
    if not buckets:
        return ""
    buckets.sort(key=lambda item: item[0])
    total = buckets[-1][1]
    if total <= 0:
        return ""
    rank = q * total
    prev_upper = 0.0
    prev_count = 0.0
    for upper, count in buckets:
        if count >= rank:
            if math.isinf(upper):
                return f"{prev_upper * 1000.0:.6f}"
            bucket_count = count - prev_count
            if bucket_count <= 0:
                return f"{upper * 1000.0:.6f}"
            pos = (rank - prev_count) / bucket_count
            value = prev_upper + (upper - prev_upper) * pos
            return f"{value * 1000.0:.6f}"
        prev_upper = upper
        prev_count = count
    return ""

pb = parse_prom(Path(prefill_before))
pa = parse_prom(Path(prefill_after))
prefill_queue_avg = avg_ms(pb, pa, "vllm:request_queue_time_seconds")
prefill_queue_median = histogram_quantile_ms(0.50, delta_buckets(pb, pa, "vllm:request_queue_time_seconds"))
prefill_queue_p90 = histogram_quantile_ms(0.90, delta_buckets(pb, pa, "vllm:request_queue_time_seconds"))
prefill_queue_p99 = histogram_quantile_ms(0.99, delta_buckets(pb, pa, "vllm:request_queue_time_seconds"))
prefill_time_avg = avg_ms(pb, pa, "vllm:request_prefill_time_seconds")
prefill_ttft_avg = avg_ms(pb, pa, "vllm:time_to_first_token_seconds")
prefill_ttft_median = histogram_quantile_ms(0.50, delta_buckets(pb, pa, "vllm:time_to_first_token_seconds"))
prefill_ttft_p90 = histogram_quantile_ms(0.90, delta_buckets(pb, pa, "vllm:time_to_first_token_seconds"))
prefill_ttft_p99 = histogram_quantile_ms(0.99, delta_buckets(pb, pa, "vllm:time_to_first_token_seconds"))

# Parse benchmark output for statistics
def m(pat: str) -> str:
    match = re.search(pat, text, re.MULTILINE)
    return match.group(1) if match else ""

# Try to find JSON output in the log
completed = m(r"completed\s*=\s*([0-9]+)")
failed = m(r"failed\s*=\s*([0-9]+)")
ttft_mean = m(r"ttft_ms\s+.*?mean\s+([0-9.]+)")
ttft_median = m(r"ttft_ms\s+.*?median\s+([0-9.]+)")
ttft_p99 = m(r"ttft_ms\s+.*?p99\s+([0-9.]+)")
tpot_mean = m(r"tpot_ms\s+.*?mean\s+([0-9.]+)")
itl_mean = m(r"itl_ms\s+.*?mean\s+([0-9.]+)")
itl_median = m(r"itl_ms\s+.*?median\s+([0-9.]+)")
e2el_mean = m(r"e2el_s\s+.*?mean\s+([0-9.]+)")
e2el_median = m(r"e2el_s\s+.*?median\s+([0-9.]+)")
e2el_p99 = m(r"e2el_s\s+.*?p99\s+([0-9.]+)")
input_tokens = m(r"total_input_tokens\s*=\s*([0-9]+)")
output_tokens = m(r"total_output_tokens\s*=\s*([0-9]+)")
req_throughput = m(r"request_throughput\s*=\s*([0-9.]+)")
token_throughput = m(r"total_token_throughput\s*=\s*([0-9.]+)")

# Handle LAPS enabled flag
laps_enabled = 0 if threshold == "off" else 1
if threshold == "off":
    threshold = ""
    laps_wait_window_ms = ""
    laps_wait_max_batch = ""

row = [
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    case_name,
    repeat_id,
    variant_name,
    str(laps_enabled),
    str(threshold),
    str(laps_wait_window_ms),
    str(laps_wait_max_batch),
    str(use_batch_prefill_graph),
    str(batch_prefill_max_seq_len),
    str(batch_prefill_batch_sizes),
    str(batch_prefill_seq_lengths),
    str(request_rate),
    completed or "0",
    failed or "0",
    ttft_mean or "",
    ttft_median or "",
    ttft_p99 or "",
    tpot_mean or "",
    itl_mean or "",
    itl_median or "",
    e2el_mean or "",
    e2el_median or "",
    e2el_p99 or "",
    input_tokens or "0",
    output_tokens or "0",
    req_throughput or "0",
    token_throughput or "0",
    prefill_queue_avg,
    prefill_queue_median,
    prefill_queue_p90,
    prefill_queue_p99,
    prefill_time_avg,
    prefill_ttft_avg,
    prefill_ttft_median,
    prefill_ttft_p90,
    prefill_ttft_p99,
    summary_log_path,
]
csv.writer(sys.stdout, lineterminator="\n").writerow(row)
PY
}

run_multiturn_warmup() {
  local case_name="$1"
  local request_rate="$2"
  local summary_log="${RESULT_DIR}/logs/${case_name}_warmup.log"
  local output_file="${RESULT_DIR}/${case_name}_warmup_output.json"
  local runtime_file="${RESULT_DIR}/logs/${case_name}_warmup_runtime_sec.txt"
  local args=(
    "warmup"
    "${CONV_DATASET_PATH}"
    "${output_file}"
    "${runtime_file}"
    "${request_rate}"
    "${MODEL_PATH}"
    "${SERVED_MODEL_NAME}"
    "${PREFILL_CONNECT_HOST}"
    "${PREFILL_PORT}"
    "${NUM_CLIENTS}"
    "${MAX_ACTIVE_CONVERSATIONS}"
    "${LIMIT_MIN_TOKENS}"
    "${LIMIT_MAX_TOKENS}"
    "${MAX_RETRIES}"
    "${CONVERSATION_SAMPLING}"
    "${REQUEST_TIMEOUT_SEC}"
  )

  (
    source_env
    cd "${MULTITURN_DIR}"
    python3 - "${args[@]}" <<'PY'
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import benchmark_serving_multi_turn as bm

ConversationSampling = bm.ConversationSampling
BenchmarkArgs = bm.BenchmarkArgs
conversations_dict_to_list = bm.conversations_dict_to_list
conversations_list_to_dict = bm.conversations_list_to_dict
generate_conversations = bm.generate_conversations
get_client_config = bm.get_client_config
logger = bm.logger
main_mp = bm.main_mp
nanosec_to_sec = bm.nanosec_to_sec
parse_input_json_file = bm.parse_input_json_file

pd.set_option("display.precision", 2)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.expand_frame_repr", False)

mode = sys.argv[1]
input_file = Path(sys.argv[2])
output_file = Path(sys.argv[3])
runtime_file = Path(sys.argv[4])
request_rate = float(sys.argv[5])
model_path = sys.argv[6]
served_model_name = sys.argv[7]
prefill_connect_host = sys.argv[8]
prefill_port = sys.argv[9]
num_clients = int(sys.argv[10])
max_active_conversations = int(sys.argv[11])
limit_min_tokens = int(sys.argv[12])
limit_max_tokens = int(sys.argv[13])
max_retries = int(sys.argv[14])
conversation_sampling = sys.argv[15]
request_timeout_sec = int(sys.argv[16])


def build_args(rate: float, warmup_step: bool) -> SimpleNamespace:
    return SimpleNamespace(
        input_file=str(input_file),
        output_file=str(output_file),
        seed=0,
        model=model_path,
        served_model_name=served_model_name or None,
        url=f"http://{prefill_connect_host}:{prefill_port}",
        num_clients=num_clients,
        max_active_conversations=max_active_conversations,
        max_num_requests=None,
        warmup_step=warmup_step,
        max_turns=None,
        no_early_stop=False,
        limit_max_tokens=limit_max_tokens,
        limit_min_tokens=limit_min_tokens,
        request_rate=rate,
        max_retries=max_retries,
        conversation_sampling=ConversationSampling(conversation_sampling),
        verify_output=False,
        request_timeout_sec=request_timeout_sec,
        no_stream=False,
        excel_output=False,
        verbose=False,
        print_content=False,
        warmup_percentages="0%",
    )


async def main() -> None:
    if mode != "warmup":
        raise ValueError(f"Unsupported mode for warmup runner: {mode}")

    random.seed(0)
    np.random.seed(0)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with input_file.open() as f:
        input_data = json.load(f)

    gen_conv_args = None
    if isinstance(input_data, list):
        logger.info("Found %d items in the input file", len(input_data))
        conversations = conversations_list_to_dict(input_data)
    elif isinstance(input_data, dict):
        if "filetype" not in input_data:
            raise Exception(f"Input file {input_file} is invalid (missing 'filetype')")
        logger.info("Using input file with filetype: %s", input_data["filetype"])
        gen_conv_args = parse_input_json_file(input_data)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        conversations = generate_conversations(gen_conv_args, tokenizer)
    else:
        raise Exception(f"Input file {input_file} is invalid")

    args = build_args(request_rate, False)
    client_args, req_args = get_client_config(args, conversations)
    bench_args = BenchmarkArgs(url=args.url, num_clients=args.num_clients, early_stop=not args.no_early_stop)

    warmup_client_args = client_args._replace(skip_first_turn=False, max_turns=1, max_active_conversations=1)
    warmup_bench_args = bench_args._replace(early_stop=False)

    logger.info("Warmup start")
    warmup_start_ns = time.perf_counter_ns()
    conversations, _ = await main_mp(warmup_client_args, req_args, warmup_bench_args, tokenizer, conversations)
    warmup_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - warmup_start_ns)
    logger.info("Warmup runtime: %.3f sec", warmup_runtime_sec)
    logger.info("Warmup done")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(conversations_dict_to_list(conversations), f, indent=4)
    runtime_file.write_text(f"{warmup_runtime_sec:.6f}\n", encoding="utf-8")


asyncio.run(main())
PY
  ) 2>&1 | tee "${summary_log}"
}

run_multiturn_bench() {
  local case_name="$1"
  local request_rate="$2"
  local input_file="$3"
  local warmup_runtime_sec="${4:-}"
  local summary_log="${RESULT_DIR}/logs/${case_name}_bench.log"
  local output_file="${RESULT_DIR}/${case_name}_output.json"
  local raw_data_file="${RESULT_DIR}/${RAW_DATA_DIRNAME}/${case_name}_raw_data.csv"
  local args=(
    "formal"
    "${input_file}"
    "${output_file}"
    "${raw_data_file}"
    "${request_rate}"
    "${warmup_runtime_sec}"
    "${MODEL_PATH}"
    "${SERVED_MODEL_NAME}"
    "${PREFILL_CONNECT_HOST}"
    "${PREFILL_PORT}"
    "${NUM_CLIENTS}"
    "${MAX_ACTIVE_CONVERSATIONS}"
    "${LIMIT_MIN_TOKENS}"
    "${LIMIT_MAX_TOKENS}"
    "${MAX_RETRIES}"
    "${CONVERSATION_SAMPLING}"
    "${REQUEST_TIMEOUT_SEC}"
    "${WARMUP_STEP}"
    "${FORMAL_NO_EARLY_STOP}"
  )

  (
    source_env
    cd "${MULTITURN_DIR}"
    python3 - "${args[@]}" <<'PY'
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import benchmark_serving_multi_turn as bm

ConversationSampling = bm.ConversationSampling
BenchmarkArgs = bm.BenchmarkArgs
conversations_dict_to_list = bm.conversations_dict_to_list
conversations_list_to_dict = bm.conversations_list_to_dict
generate_conversations = bm.generate_conversations
get_client_config = bm.get_client_config
logger = bm.logger
main_mp = bm.main_mp
nanosec_to_sec = bm.nanosec_to_sec
parse_input_json_file = bm.parse_input_json_file
process_statistics = bm.process_statistics

pd.set_option("display.precision", 2)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.expand_frame_repr", False)

mode = sys.argv[1]
input_file = Path(sys.argv[2])
output_file = Path(sys.argv[3])
raw_data_file = Path(sys.argv[4])
request_rate = float(sys.argv[5])
warmup_runtime_str = sys.argv[6]
warmup_runtime_sec = float(warmup_runtime_str) if warmup_runtime_str else None
model_path = sys.argv[7]
served_model_name = sys.argv[8]
prefill_connect_host = sys.argv[9]
prefill_port = sys.argv[10]
num_clients = int(sys.argv[11])
max_active_conversations = int(sys.argv[12])
limit_min_tokens = int(sys.argv[13])
limit_max_tokens = int(sys.argv[14])
max_retries = int(sys.argv[15])
conversation_sampling = sys.argv[16]
request_timeout_sec = int(sys.argv[17])
warmup_step = sys.argv[18]
formal_no_early_stop = sys.argv[19] == "1"


def build_args(rate: float, warmup_step: bool) -> SimpleNamespace:
    return SimpleNamespace(
        input_file=str(input_file),
        output_file=str(output_file),
        seed=0,
        model=model_path,
        served_model_name=served_model_name or None,
        url=f"http://{prefill_connect_host}:{prefill_port}",
        num_clients=num_clients,
        max_active_conversations=max_active_conversations,
        max_num_requests=None,
        warmup_step=warmup_step,
        max_turns=None,
        no_early_stop=formal_no_early_stop,
        limit_max_tokens=limit_max_tokens,
        limit_min_tokens=limit_min_tokens,
        request_rate=rate,
        max_retries=max_retries,
        conversation_sampling=ConversationSampling(conversation_sampling),
        verify_output=False,
        request_timeout_sec=request_timeout_sec,
        no_stream=False,
        excel_output=False,
        verbose=False,
        print_content=False,
        warmup_percentages="0%",
    )


async def main() -> None:
    if mode != "formal":
        raise ValueError(f"Unsupported mode for formal runner: {mode}")

    random.seed(0)
    np.random.seed(0)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with input_file.open() as f:
        input_data = json.load(f)

    gen_conv_args = None
    if isinstance(input_data, list):
        logger.info("Found %d items in the input file", len(input_data))
        conversations = conversations_list_to_dict(input_data)
    elif isinstance(input_data, dict):
        if "filetype" not in input_data:
            raise Exception(f"Input file {input_file} is invalid (missing 'filetype')")
        logger.info("Using input file with filetype: %s", input_data["filetype"])
        gen_conv_args = parse_input_json_file(input_data)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        conversations = generate_conversations(gen_conv_args, tokenizer)
    else:
        raise Exception(f"Input file {input_file} is invalid")

    formal_uses_warmup = warmup_step == "1"
    args = build_args(request_rate, formal_uses_warmup)
    client_args, req_args = get_client_config(args, conversations)
    bench_args = BenchmarkArgs(url=args.url, num_clients=args.num_clients, early_stop=not args.no_early_stop)

    benchmark_start_ns = time.perf_counter_ns()
    client_convs, client_metrics = await main_mp(client_args, req_args, bench_args, tokenizer, conversations)
    benchmark_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - benchmark_start_ns)
    logger.info("Benchmark runtime: %.3f sec", benchmark_runtime_sec)

    process_statistics(
        client_metrics,
        test_params={},
        warmup_percentages=[0.0],
        verbose=args.verbose,
        gen_conv_args=gen_conv_args,
        excel_output=False,
        warmup_runtime_sec=warmup_runtime_sec,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(conversations_dict_to_list(client_convs), f, indent=4)


asyncio.run(main())
PY
  ) 2>&1 | tee "${summary_log}"
}

run_case() {
  local case_name="$1"
  local repeat_id="$2"
  local variant_name="$3"
  local request_rate="$4"
  local laps_threshold="$5"
  local laps_wait_window_ms="$6"
  local laps_wait_max_batch="$7"
  local use_batch_prefill_graph="$8"
  local case_key="${case_name}_run${repeat_id}"
  local warmup_runtime_file="${RESULT_DIR}/logs/${case_key}_warmup_runtime_sec.txt"
  local warmup_runtime_sec=""
  local prefill_before="${RESULT_DIR}/logs/${case_key}_prefill_metrics_before.prom"
  local prefill_after="${RESULT_DIR}/logs/${case_key}_prefill_metrics_after.prom"
  local summary_batch_prefill_max_seq_len=""
  local summary_batch_prefill_batch_sizes=""
  local summary_batch_prefill_seq_lengths=""

  mkdir -p "${RESULT_DIR}/${case_key}"

  log "========== CASE ${case_key} started =========="
  log "Variant: ${variant_name}, request_rate: ${request_rate}"
  log "Prepared dataset: ${CONV_DATASET_PATH}"
  if [ "${laps_threshold}" = "off" ]; then
    log "LAPS config: disabled"
  else
    log "LAPS config: threshold=${laps_threshold}, long_max_wait_ms=${laps_wait_window_ms}"
  fi
  if [ "${use_batch_prefill_graph}" = "1" ]; then
    log "Batch prefill graph: enabled, max_seq_len=${BATCH_PREFILL_MAX_SEQ_LEN}, batch_sizes=${BATCH_PREFILL_BATCH_SIZES}, seq_lengths=${BATCH_PREFILL_SEQ_LENGTHS}"
    summary_batch_prefill_max_seq_len="${BATCH_PREFILL_MAX_SEQ_LEN}"
    summary_batch_prefill_batch_sizes="${BATCH_PREFILL_BATCH_SIZES}"
    summary_batch_prefill_seq_lengths="${BATCH_PREFILL_SEQ_LENGTHS}"
  else
    log "Batch prefill graph: disabled"
  fi

  stop_services
  start_prefill "${case_key}" "${laps_threshold}" "${laps_wait_window_ms}" "${laps_wait_max_batch}" "${use_batch_prefill_graph}"

  wait_log "${RESULT_DIR}/logs/${case_key}_prefill.log" "Application startup complete" "Prefill startup"
  wait_http "http://${PREFILL_CONNECT_HOST}:${PREFILL_PORT}/health" "Prefill" || {
    tail -80 "${RESULT_DIR}/logs/${case_key}_prefill.log" || true
    return 1
  }

  if [ "${laps_threshold}" != "off" ]; then
    wait_log "${RESULT_DIR}/logs/${case_key}_prefill.log" "Ascend LAPS scheduler selected" "LAPS scheduler selection"
  fi

  if [ "${WARMUP_STEP}" = "1" ]; then
    log "Warmup start"
    run_multiturn_warmup "${case_key}" "${request_rate}"
    if [ -f "${warmup_runtime_file}" ]; then
      warmup_runtime_sec="$(tr -d '[:space:]' < "${warmup_runtime_file}")"
    fi
  fi

  log "Measured run start"
  scrape_metrics_snapshot "http://${PREFILL_CONNECT_HOST}:${PREFILL_PORT}" "${prefill_before}"
  run_multiturn_bench "${case_key}" "${request_rate}" "${CONV_DATASET_PATH}" "${warmup_runtime_sec}"
  scrape_metrics_snapshot "http://${PREFILL_CONNECT_HOST}:${PREFILL_PORT}" "${prefill_after}"

  append_summary \
    "${case_name}" \
    "${repeat_id}" \
    "${variant_name}" \
    "${laps_threshold}" \
    "${laps_wait_window_ms}" \
    "${laps_wait_max_batch}" \
    "${use_batch_prefill_graph}" \
    "${summary_batch_prefill_max_seq_len}" \
    "${summary_batch_prefill_batch_sizes}" \
    "${summary_batch_prefill_seq_lengths}" \
    "${request_rate}" \
    "${prefill_before}" \
    "${prefill_after}" \
    "${RESULT_DIR}/logs/${case_key}_bench.log"

  log "========== CASE ${case_key} completed =========="
}

run_case_with_recovery() {
  local case_name="$1"
  local repeat_id="$2"
  local variant_name="$3"
  local request_rate="$4"
  local laps_threshold="$5"
  local laps_wait_window_ms="$6"
  local laps_wait_max_batch="$7"
  local use_batch_prefill_graph="$8"
  local attempt=1
  local rc=0

  while [ "${attempt}" -le "${CASE_RETRY_LIMIT}" ]; do
    if [ "${attempt}" -gt 1 ]; then
      log "Retrying case ${case_name}: attempt ${attempt}/${CASE_RETRY_LIMIT}"
    fi

    if run_case "${case_name}" "${repeat_id}" "${variant_name}" "${request_rate}" "${laps_threshold}" "${laps_wait_window_ms}" "${laps_wait_max_batch}" "${use_batch_prefill_graph}"; then
      append_case_status "${case_name}_run${repeat_id}" "passed" "attempt=${attempt}"
      return 0
    fi

    rc=$?
    append_case_status "${case_name}_run${repeat_id}" "failed_attempt" "exit_code=${rc}; attempt=${attempt}"
    stop_services || true
    attempt=$((attempt + 1))
  done

  append_case_status "${case_name}_run${repeat_id}" "failed" "exit_code=${rc}; attempts=${CASE_RETRY_LIMIT}"
  return 0
}

main() {
  log_local_tooling
  verify_prereqs
  source_env
  unset_proxy_env
  write_summary_header
  build_conversation_dataset

  printf 'timestamp,case,status,detail\n' > "${RESULT_DIR}/case_status.csv"

  log "Running one-time local fallback cleanup before the first cold start"
  stop_existing_services || true
  sleep "${SLEEP_AFTER_STOP_S}"
  ensure_ports_free

  log "Results will be written to ${RESULT_DIR}"
  log "Model: ${MODEL_PATH}, served_name: ${SERVED_MODEL_NAME}"
  log "Dataset: ${RAW_DATASET_PATH} -> ${CONV_DATASET_PATH}"
  log "Case variants: ${CASE_VARIANTS}"
  log "Request rates: ${REQUEST_RATES}"
  log "Bench repeat: ${BENCH_REPEAT}"
  log "LAPS defaults: long_max_wait_ms=${LAPS_LONG_MAX_WAIT_MS}"
  log "Batch prefill graph: max_seq_len=${BATCH_PREFILL_MAX_SEQ_LEN}, batch_sizes=${BATCH_PREFILL_BATCH_SIZES}, seq_lengths=${BATCH_PREFILL_SEQ_LENGTHS}"

  local variant
  local case_spec
  local laps_threshold
  local laps_wait_window_ms
  local laps_wait_max_batch
  local case_suffix
  local use_batch_prefill_graph
  local repeat_idx
  local request_rate

  for request_rate in ${REQUEST_RATES}; do
    local rate_name="r${request_rate//./p}"
    for ((repeat_idx = 1; repeat_idx <= BENCH_REPEAT; repeat_idx++)); do
      for variant in ${CASE_VARIANTS}; do
        case_spec="$(case_variant_to_params "${variant}")"
        IFS='|' read -r laps_threshold laps_wait_window_ms laps_wait_max_batch case_suffix use_batch_prefill_graph <<< "${case_spec}"
        if [ "${laps_threshold}" = "off" ]; then
          log "  ${variant} -> case_suffix=${case_suffix}, laps=disabled, batch_prefill_graph=${use_batch_prefill_graph}"
        else
          log "  ${variant} -> case_suffix=${case_suffix}, threshold=${laps_threshold}, long_max_wait_ms=${laps_wait_window_ms}"
        fi
        run_case_with_recovery \
          "${rate_name}_${case_suffix}" \
          "${repeat_idx}" \
          "${variant}" \
          "${request_rate}" \
          "${laps_threshold}" \
          "${laps_wait_window_ms}" \
          "${laps_wait_max_batch}" \
          "${use_batch_prefill_graph}"
      done
    done
  done

  log "All benchmark cases completed. Results: ${RESULT_DIR}"
}

verify_prereqs() {
  [ -f "${RAW_DATASET_PATH}" ] || {
    log "RAW_DATASET_PATH not found: ${RAW_DATASET_PATH}"
    return 1
  }
  [ -d "${MULTITURN_DIR}" ] || {
    log "MULTITURN_DIR not found: ${MULTITURN_DIR}"
    return 1
  }
}

main "$@"
