#!/usr/bin/env bash
set -euo pipefail

# Run a ShareGPT multi-turn replay benchmark on a 1P1D vLLM-Ascend deployment.
# This wrapper keeps the existing PD/proxy launch flow, but switches the client
# workload from single-turn `vllm bench serve` to vLLM's official
# `benchmark_serving_multi_turn.py`.

MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ds-r1-1p5b}"
RAW_DATASET_PATH="${RAW_DATASET_PATH:-/vllm-workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
CONV_DATASET_PATH="${CONV_DATASET_PATH:-/vllm-workspace/datasets/sharegpt_conv_laps.json}"

VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
PROXY_DIR="${PROXY_DIR:-${VLLM_ASCEND_DIR}/examples/disaggregated_prefill_v1}"
MULTITURN_DIR="${MULTITURN_DIR:-${VLLM_DIR}/benchmarks/multi_turn}"
RESULT_DIR="${RESULT_DIR:-/vllm-workspace/bench_results/laps_sharegpt_multiturn_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_CSV="summary.csv"
METRICS_CSV="metrics_summary.csv"
METRICS_BRIEF_CSV="metrics_brief.csv"

PREFILL_DEVICES="${PREFILL_DEVICES:-0,1,2,3}"
DECODE_DEVICES="${DECODE_DEVICES:-4,5,6,7}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
DECODE_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"

PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-13700}"
DECODE_PORT="${DECODE_PORT:-13701}"
PROXY_PORT="${PROXY_PORT:-8080}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"
DECODE_KV_PORT="${DECODE_KV_PORT:-30100}"

MAX_ITEMS="${MAX_ITEMS:-256}"
MIN_TURNS="${MIN_TURNS:-6}"
MAX_TURNS="${MAX_TURNS:-12}"
CONVERT_MAX_CONTENT_LEN="${CONVERT_MAX_CONTENT_LEN:-12000}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
MAX_ACTIVE_CONVERSATIONS="${MAX_ACTIVE_CONVERSATIONS:-16}"
#
# Use positive open-loop rates by default. `0` means "no sleep" and is a
# saturation stress test, not the main paper-like configuration.
REQUEST_RATES="${REQUEST_RATES:-1 2 4}"
WARMUP_STEP="${WARMUP_STEP:-1}"
LIMIT_MAX_TOKENS="${LIMIT_MAX_TOKENS:-32}"
LIMIT_MIN_TOKENS="${LIMIT_MIN_TOKENS:-32}"
#
# Default threshold sweep: paper-reported transition band (roughly 150-512) plus
# one empirically useful higher point on this platform (1024).
CASE_VARIANTS="${CASE_VARIANTS:-off t256_w0 t384_w0 t512_w0 t1024_w0}"
BENCH_REPEAT="${BENCH_REPEAT:-3}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
MAX_RETRIES="${MAX_RETRIES:-0}"
CONVERSATION_SAMPLING="${CONVERSATION_SAMPLING:-round_robin}"

STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-600}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-60}"
SLEEP_AFTER_STOP_S="${SLEEP_AFTER_STOP_S:-5}"
PORT_FREE_TIMEOUT_S="${PORT_FREE_TIMEOUT_S:-30}"

PREFILL_PID=""
DECODE_PID=""
PROXY_PID=""

mkdir -p "${RESULT_DIR}/logs"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

write_summary_header() {
  cat >"${RESULT_DIR}/${SUMMARY_CSV}" <<'EOF'
case_name,repeat_id,variant,laps_enabled,laps_threshold,laps_wait_window_ms,laps_wait_max_batch,laps_long_prefill_cap,laps_short_reserved_ratio,request_rate,request_mode,num_clients,max_active_conversations,max_items,min_turns,max_turns,limit_min_tokens,limit_max_tokens,summary_log
EOF
cat >"${RESULT_DIR}/${METRICS_CSV}" <<'EOF'
case_name,repeat_id,variant,request_rate,request_mode,runtime_sec,requests_per_sec,warmup_runtime_sec,total_runtime_incl_warmup_sec,ttft_count,ttft_mean_ms,ttft_std_ms,ttft_min_ms,ttft_25_ms,ttft_50_ms,ttft_75_ms,ttft_90_ms,ttft_99_ms,ttft_max_ms,tpot_count,tpot_mean_ms,tpot_std_ms,tpot_min_ms,tpot_25_ms,tpot_50_ms,tpot_75_ms,tpot_90_ms,tpot_99_ms,tpot_max_ms,latency_count,latency_mean_ms,latency_std_ms,latency_min_ms,latency_25_ms,latency_50_ms,latency_75_ms,latency_90_ms,latency_99_ms,latency_max_ms,summary_log
EOF
cat >"${RESULT_DIR}/${METRICS_BRIEF_CSV}" <<'EOF'
case_name,repeat_id,variant,request_rate,request_mode,requests_per_sec,ttft_mean_ms,ttft_50_ms,ttft_90_ms,ttft_99_ms,tpot_mean_ms,tpot_50_ms,tpot_90_ms,tpot_99_ms,latency_mean_ms,latency_50_ms,latency_90_ms,latency_99_ms,summary_log
EOF
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

stop_existing_services() {
  kill_matching_cmd "vllm serve .*--port ${PREFILL_PORT}"
  kill_matching_cmd "vllm serve .*--port ${DECODE_PORT}"
  kill_matching_cmd "load_balance_proxy_server_example.py .*--port ${PROXY_PORT}"
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

wait_port_free() {
  local port="$1"
  local name="$2"
  local deadline=$((SECONDS + PORT_FREE_TIMEOUT_S))
  while port_in_use "${port}"; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Port ${port} for ${name} still in use"
      return 1
    fi
    sleep 1
  done
}

ensure_ports_free() {
  wait_port_free "${PREFILL_PORT}" "Prefill"
  wait_port_free "${DECODE_PORT}" "Decode"
  wait_port_free "${PROXY_PORT}" "Proxy"
}

stop_services() {
  log "Stopping previous services"
  kill_tree "${PROXY_PID}"
  kill_tree "${PREFILL_PID}"
  kill_tree "${DECODE_PID}"
  stop_existing_services
  wait_gone "${PROXY_PID}"
  wait_gone "${PREFILL_PID}"
  wait_gone "${DECODE_PID}"
  PROXY_PID=""
  PREFILL_PID=""
  DECODE_PID=""
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

variant_to_config() {
  local variant="$1"
  local threshold wait_window_ms wait_max_batch long_prefill_cap short_reserved_ratio

  if [ "${variant}" = "off" ]; then
    printf 'off|%s|%s|%s|%s\n' "" "" "" ""
    return 0
  fi

  threshold="${variant%%_*}"
  threshold="${threshold#t}"
  wait_window_ms="${LAPS_WAIT_WINDOW_MS:-0}"
  wait_max_batch="${LAPS_WAIT_MAX_BATCH:-4}"
  long_prefill_cap="${LAPS_LONG_PREFILL_CAP:-0}"
  short_reserved_ratio="${LAPS_SHORT_RESERVED_RATIO:-0}"

  IFS='_' read -r -a parts <<< "${variant}"
  for part in "${parts[@]:1}"; do
    case "${part}" in
      w*) wait_window_ms="${part#w}" ;;
      b*) wait_max_batch="${part#b}" ;;
      cap*) long_prefill_cap="${part#cap}" ;;
      res*)
        local raw_ratio="${part#res}"
        if [[ "${raw_ratio}" == *.* ]]; then
          short_reserved_ratio="${raw_ratio}"
        else
          short_reserved_ratio="$(awk "BEGIN { printf \"%.6f\", ${raw_ratio} / 100.0 }")"
        fi
        ;;
      *)
        log "Unsupported CASE_VARIANTS entry: ${variant}"
        return 1
        ;;
    esac
  done

  printf '%s|%s|%s|%s|%s\n' \
    "${threshold}" \
    "${wait_window_ms}" \
    "${wait_max_batch}" \
    "${long_prefill_cap}" \
    "${short_reserved_ratio}"
}

build_conversation_dataset() {
  log "Converting ShareGPT into multi-turn conversation replay dataset"
  mkdir -p "$(dirname "${CONV_DATASET_PATH}")"
  (
    source_env
    cd "${MULTITURN_DIR}"
    python3 convert_sharegpt_to_openai.py \
      "${RAW_DATASET_PATH}" \
      "${CONV_DATASET_PATH}" \
      --seed=99 \
      --max-items="${MAX_ITEMS}" \
      --min-turns="${MIN_TURNS}" \
      --max-turns="${MAX_TURNS}" \
      --max-content-len="${CONVERT_MAX_CONTENT_LEN}" \
      --model="${MODEL_PATH}"
  )

  python3 - "${CONV_DATASET_PATH}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

fixed = []
trimmed = 0
for item in data:
    messages = item.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        continue
    if len(messages) % 2 == 1:
        messages = messages[:-1]
        trimmed += 1
    if len(messages) < 2:
        continue
    if messages[0].get("role") != "user":
        continue
    if messages[-1].get("role") != "assistant":
        continue
    valid = True
    expected = "user"
    for msg in messages:
        if msg.get("role") != expected:
            valid = False
            break
        expected = "assistant" if expected == "user" else "user"
    if valid:
        item["messages"] = messages
        fixed.append(item)

with open(path, "w", encoding="utf-8") as f:
    json.dump(fixed, f, ensure_ascii=False, indent=2)

print(f"normalized_conversations={len(fixed)} trimmed_odd_tail={trimmed}")
PY
}

start_prefill() {
  local case_name="$1"
  local laps_threshold="$2"
  local wait_window_ms="$3"
  local wait_max_batch="$4"
  local long_prefill_cap="$5"
  local short_reserved_ratio="$6"
  local log_file="${RESULT_DIR}/logs/${case_name}_prefill.log"

  (
    source_env
    cd "${VLLM_ASCEND_DIR}"
    export ASCEND_RT_VISIBLE_DEVICES="${PREFILL_DEVICES}"
    if [ "${laps_threshold}" = "off" ]; then
      unset VLLM_ASCEND_LAPS_SCHEDULING VLLM_ASCEND_LAPS_THRESHOLD VLLM_ASCEND_LAPS_WAIT_WINDOW_MS VLLM_ASCEND_LAPS_WAIT_MAX_BATCH VLLM_ASCEND_LAPS_LONG_PREFILL_CAP VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S
    else
      export VLLM_ASCEND_LAPS_SCHEDULING=1
      export VLLM_ASCEND_LAPS_THRESHOLD="${laps_threshold}"
      export VLLM_ASCEND_LAPS_WAIT_WINDOW_MS="${wait_window_ms}"
      export VLLM_ASCEND_LAPS_WAIT_MAX_BATCH="${wait_max_batch}"
      export VLLM_ASCEND_LAPS_LONG_PREFILL_CAP="${long_prefill_cap}"
      export VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO="${short_reserved_ratio}"
      export VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S="${LAPS_STATS_LOG_INTERVAL_S:-0}"
    fi
    exec vllm serve "${MODEL_PATH}" \
      --host "${PREFILL_HOST}" \
      --port "${PREFILL_PORT}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --no-async-scheduling \
      --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"kv_producer\",\"kv_port\":\"${PREFILL_KV_PORT}\",\"engine_id\":\"0\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}},\"decode\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}}}}"
  ) >"${log_file}" 2>&1 &
  PREFILL_PID=$!
  log "Started Prefill pid=${PREFILL_PID}, log=${log_file}"
}

start_decode() {
  local case_name="$1"
  local log_file="${RESULT_DIR}/logs/${case_name}_decode.log"

  (
    source_env
    cd "${VLLM_ASCEND_DIR}"
    export ASCEND_RT_VISIBLE_DEVICES="${DECODE_DEVICES}"
    unset VLLM_ASCEND_LAPS_SCHEDULING VLLM_ASCEND_LAPS_THRESHOLD VLLM_ASCEND_LAPS_WAIT_WINDOW_MS VLLM_ASCEND_LAPS_WAIT_MAX_BATCH
    exec vllm serve "${MODEL_PATH}" \
      --host "${DECODE_HOST}" \
      --port "${DECODE_PORT}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${DECODE_MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --no-async-scheduling \
      --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"${DECODE_KV_PORT}\",\"engine_id\":\"1\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}},\"decode\":{\"dp_size\":1,\"tp_size\":${TP_SIZE}}}}"
  ) >"${log_file}" 2>&1 &
  DECODE_PID=$!
  log "Started Decode pid=${DECODE_PID}, log=${log_file}"
}

start_proxy() {
  local case_name="$1"
  local log_file="${RESULT_DIR}/logs/${case_name}_proxy.log"

  (
    source_env
    cd "${PROXY_DIR}"
    exec python3 load_balance_proxy_server_example.py \
      --host "${PROXY_HOST}" \
      --port "${PROXY_PORT}" \
      --prefiller-hosts "${PREFILL_HOST}" \
      --prefiller-ports "${PREFILL_PORT}" \
      --decoder-hosts "${DECODE_HOST}" \
      --decoder-ports "${DECODE_PORT}"
  ) >"${log_file}" 2>&1 &
  PROXY_PID=$!
  log "Started Proxy pid=${PROXY_PID}, log=${log_file}"
}

append_summary_row() {
  local case_name="$1"
  local repeat_id="$2"
  local variant_name="$3"
  local threshold="$4"
  local wait_window_ms="$5"
  local wait_max_batch="$6"
  local long_prefill_cap="$7"
  local short_reserved_ratio="$8"
  local request_rate="$9"
  local summary_log="${10}"
  local laps_enabled=1
  local request_mode="open_loop"

  if [ "${threshold}" = "off" ]; then
    laps_enabled=0
    threshold=""
    wait_window_ms=""
    wait_max_batch=""
    long_prefill_cap=""
    short_reserved_ratio=""
  fi
  if [ "${request_rate}" = "0" ] || [ "${request_rate}" = "0.0" ]; then
    request_mode="saturation_no_sleep"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${case_name}" \
    "${repeat_id}" \
    "${variant_name}" \
    "${laps_enabled}" \
    "${threshold}" \
    "${wait_window_ms}" \
    "${wait_max_batch}" \
    "${long_prefill_cap}" \
    "${short_reserved_ratio}" \
    "${request_rate}" \
    "${request_mode}" \
    "${NUM_CLIENTS}" \
    "${MAX_ACTIVE_CONVERSATIONS}" \
    "${MAX_ITEMS}" \
    "${MIN_TURNS}" \
    "${MAX_TURNS}" \
    "${LIMIT_MIN_TOKENS}" \
    "${LIMIT_MAX_TOKENS}" \
    "${summary_log}" >> "${RESULT_DIR}/${SUMMARY_CSV}"
}

parse_multiturn_metrics() {
  local case_name="$1"
  local repeat_id="$2"
  local variant_name="$3"
  local request_rate="$4"
  local summary_log="$5"
  local request_mode="open_loop"
  local metrics_line
  local runtime_sec requests_per_sec warmup_runtime_sec total_runtime_incl_warmup_sec
  local ttft_count ttft_mean_ms ttft_std_ms ttft_min_ms ttft_25_ms ttft_50_ms ttft_75_ms ttft_90_ms ttft_99_ms ttft_max_ms
  local tpot_count tpot_mean_ms tpot_std_ms tpot_min_ms tpot_25_ms tpot_50_ms tpot_75_ms tpot_90_ms tpot_99_ms tpot_max_ms
  local latency_count latency_mean_ms latency_std_ms latency_min_ms latency_25_ms latency_50_ms latency_75_ms latency_90_ms latency_99_ms latency_max_ms

  if [ "${request_rate}" = "0" ] || [ "${request_rate}" = "0.0" ]; then
    request_mode="saturation_no_sleep"
  fi

  metrics_line="$(
    python3 - "${summary_log}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""

def m(pat):
    match = re.search(pat, text, re.MULTILINE)
    return match.group(1) if match else ""

runtime_sec = m(r"runtime_sec\s*=\s*([0-9.]+)")
requests_per_sec = m(r"requests_per_sec\s*=\s*([0-9.]+)")
warmup_runtime_sec = m(r"warmup_runtime_sec\s*=\s*([0-9.]+)")
total_runtime = m(r"total_runtime_incl_warmup_sec\s*=\s*([0-9.]+)")

def grab_block(name):
    summary_pos = text.rfind("Statistics summary:")
    summary_text = text[summary_pos:] if summary_pos >= 0 else text

    match = re.search(rf"^\s*{name}\s+(.+)$", summary_text, re.MULTILINE)
    if not match:
        return [""] * 10

    tokens = [tok for tok in match.group(1).split() if tok != "..."]
    nums = [
        tok
        for tok in tokens
        if re.fullmatch(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", tok)
    ]
    while len(nums) < 10:
        nums.append("")
    return nums[:10]

ttft = grab_block("ttft_ms")
tpot = grab_block("tpot_ms")
latency = grab_block("latency_ms")

fields = [
    runtime_sec,
    requests_per_sec,
    warmup_runtime_sec,
    total_runtime,
    *ttft,
    *tpot,
    *latency,
]
print("|".join(fields))
PY
  )"

  IFS='|' read -r \
    runtime_sec requests_per_sec warmup_runtime_sec total_runtime_incl_warmup_sec \
    ttft_count ttft_mean_ms ttft_std_ms ttft_min_ms ttft_25_ms ttft_50_ms ttft_75_ms ttft_90_ms ttft_99_ms ttft_max_ms \
    tpot_count tpot_mean_ms tpot_std_ms tpot_min_ms tpot_25_ms tpot_50_ms tpot_75_ms tpot_90_ms tpot_99_ms tpot_max_ms \
    latency_count latency_mean_ms latency_std_ms latency_min_ms latency_25_ms latency_50_ms latency_75_ms latency_90_ms latency_99_ms latency_max_ms \
    <<< "${metrics_line}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${case_name}" \
    "${repeat_id}" \
    "${variant_name}" \
    "${request_rate}" \
    "${request_mode}" \
    "${runtime_sec}" \
    "${requests_per_sec}" \
    "${warmup_runtime_sec}" \
    "${total_runtime_incl_warmup_sec}" \
    "${ttft_count}" \
    "${ttft_mean_ms}" \
    "${ttft_std_ms}" \
    "${ttft_min_ms}" \
    "${ttft_25_ms}" \
    "${ttft_50_ms}" \
    "${ttft_75_ms}" \
    "${ttft_90_ms}" \
    "${ttft_99_ms}" \
    "${ttft_max_ms}" \
    "${tpot_count}" \
    "${tpot_mean_ms}" \
    "${tpot_std_ms}" \
    "${tpot_min_ms}" \
    "${tpot_25_ms}" \
    "${tpot_50_ms}" \
    "${tpot_75_ms}" \
    "${tpot_90_ms}" \
    "${tpot_99_ms}" \
    "${tpot_max_ms}" \
    "${latency_count}" \
    "${latency_mean_ms}" \
    "${latency_std_ms}" \
    "${latency_min_ms}" \
    "${latency_25_ms}" \
    "${latency_50_ms}" \
    "${latency_75_ms}" \
    "${latency_90_ms}" \
    "${latency_99_ms}" \
    "${latency_max_ms}" \
    "${summary_log}" >> "${RESULT_DIR}/${METRICS_CSV}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${case_name}" \
    "${repeat_id}" \
    "${variant_name}" \
    "${request_rate}" \
    "${request_mode}" \
    "${requests_per_sec}" \
    "${ttft_mean_ms}" \
    "${ttft_50_ms}" \
    "${ttft_90_ms}" \
    "${ttft_99_ms}" \
    "${tpot_mean_ms}" \
    "${tpot_50_ms}" \
    "${tpot_90_ms}" \
    "${tpot_99_ms}" \
    "${latency_mean_ms}" \
    "${latency_50_ms}" \
    "${latency_90_ms}" \
    "${latency_99_ms}" \
    "${summary_log}" >> "${RESULT_DIR}/${METRICS_BRIEF_CSV}"
}

run_multiturn_bench() {
  local case_name="$1"
  local request_rate="$2"
  local summary_log="${RESULT_DIR}/logs/${case_name}_bench.log"
  local output_file="${RESULT_DIR}/${case_name}_output.json"
  local args=(
    --input-file "${CONV_DATASET_PATH}"
    --output-file "${output_file}"
    --model "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --url "http://${PROXY_HOST}:${PROXY_PORT}"
    --num-clients "${NUM_CLIENTS}"
    --max-active-conversations "${MAX_ACTIVE_CONVERSATIONS}"
    --request-rate "${request_rate}"
    --conversation-sampling "${CONVERSATION_SAMPLING}"
    --request-timeout-sec "${REQUEST_TIMEOUT_SEC}"
    --max-retries "${MAX_RETRIES}"
    --limit-min-tokens "${LIMIT_MIN_TOKENS}"
    --limit-max-tokens "${LIMIT_MAX_TOKENS}"
  )

  if [ "${WARMUP_STEP}" = "1" ]; then
    args+=(--warmup-step)
  fi

  (
    source_env
    cd "${MULTITURN_DIR}"
    python3 - "${args[@]}" <<'PY'
import runpy
import sys

import pandas as pd

pd.set_option("display.precision", 2)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)
pd.set_option("display.expand_frame_repr", False)

sys.argv = ["benchmark_serving_multi_turn.py", *sys.argv[1:]]
runpy.run_path("benchmark_serving_multi_turn.py", run_name="__main__")
PY
  ) 2>&1 | tee "${summary_log}"
}

run_case() {
  local case_name="$1"
  local repeat_id="$2"
  local variant_name="$3"
  local request_rate="$4"
  local config threshold wait_window_ms wait_max_batch long_prefill_cap short_reserved_ratio

  config="$(variant_to_config "${variant_name}")"
  IFS='|' read -r threshold wait_window_ms wait_max_batch long_prefill_cap short_reserved_ratio <<< "${config}"
  if [ "${variant_name}" = "off" ]; then
    threshold="off"
  fi

  log "========== CASE ${case_name}_run${repeat_id} started =========="
  log "Case config: variant=${variant_name}, threshold=${threshold}, wait_window_ms=${wait_window_ms:-0}, wait_max_batch=${wait_max_batch:-0}, long_prefill_cap=${long_prefill_cap:-0}, short_reserved_ratio=${short_reserved_ratio:-0}, request_rate=${request_rate}"

  stop_services
  start_prefill "${case_name}_run${repeat_id}" "${threshold}" "${wait_window_ms}" "${wait_max_batch}" "${long_prefill_cap}" "${short_reserved_ratio}"
  start_decode "${case_name}_run${repeat_id}"

  wait_log "${RESULT_DIR}/logs/${case_name}_run${repeat_id}_prefill.log" "Application startup complete" "Prefill startup"
  wait_log "${RESULT_DIR}/logs/${case_name}_run${repeat_id}_decode.log" "Application startup complete" "Decode startup"
  wait_http "http://${PREFILL_HOST}:${PREFILL_PORT}/health" "Prefill"
  wait_http "http://${DECODE_HOST}:${DECODE_PORT}/health" "Decode"

  start_proxy "${case_name}_run${repeat_id}"
  wait_log "${RESULT_DIR}/logs/${case_name}_run${repeat_id}_proxy.log" "Application startup complete" "Proxy startup"
  wait_http "http://${PROXY_HOST}:${PROXY_PORT}/healthcheck" "Proxy"

  if [ "${threshold}" != "off" ]; then
    wait_log "${RESULT_DIR}/logs/${case_name}_run${repeat_id}_prefill.log" "Ascend LAPS scheduler selected" "LAPS scheduler selection"
  fi

  run_multiturn_bench "${case_name}_run${repeat_id}" "${request_rate}"
  append_summary_row \
    "${case_name}" \
    "${repeat_id}" \
    "${variant_name}" \
    "${threshold}" \
    "${wait_window_ms}" \
    "${wait_max_batch}" \
    "${long_prefill_cap}" \
    "${short_reserved_ratio}" \
    "${request_rate}" \
    "${RESULT_DIR}/logs/${case_name}_run${repeat_id}_bench.log"
  parse_multiturn_metrics \
    "${case_name}" \
    "${repeat_id}" \
    "${variant_name}" \
    "${request_rate}" \
    "${RESULT_DIR}/logs/${case_name}_run${repeat_id}_bench.log"

  log "========== CASE ${case_name}_run${repeat_id} completed =========="
}

main() {
  source_env
  write_summary_header
  build_conversation_dataset

  log "Results will be written to ${RESULT_DIR}"
  log "Using case variants: ${CASE_VARIANTS}"
  log "Using multi-turn request rates: ${REQUEST_RATES}"
  log "Conversation replay config: max_items=${MAX_ITEMS}, min_turns=${MIN_TURNS}, max_turns=${MAX_TURNS}, num_clients=${NUM_CLIENTS}, max_active_conversations=${MAX_ACTIVE_CONVERSATIONS}, limit_tokens=[${LIMIT_MIN_TOKENS}, ${LIMIT_MAX_TOKENS}]"

  for request_rate in ${REQUEST_RATES}; do
    local rate_name="${request_rate//./p}"
    for repeat_id in $(seq 1 "${BENCH_REPEAT}"); do
      for variant_name in ${CASE_VARIANTS}; do
        run_case "r${rate_name}_${variant_name}" "${repeat_id}" "${variant_name}" "${request_rate}"
      done
    done
  done

  log "All benchmark cases completed. Results: ${RESULT_DIR}"
  log "Summary CSV: ${RESULT_DIR}/${SUMMARY_CSV}"
  log "Full metrics CSV: ${RESULT_DIR}/${METRICS_CSV}"
  log "Brief metrics CSV: ${RESULT_DIR}/${METRICS_BRIEF_CSV}"
}

main "$@"
