#!/usr/bin/env bash
set -euo pipefail

# Run a ShareGPT multi-turn LAPS benchmark for GLM-5 on a dual-node 1P1D
# vLLM-Ascend deployment.
#
# Run this script on the Prefill node. It starts:
# - Prefill locally
# - Decode remotely over SSH (2 DP instances on the Decode node)
# - Proxy locally
# - ShareGPT multi-turn benchmark locally against the proxy
#
# Assumptions:
# - Passwordless SSH from the Prefill node to the Decode node is ready.
# - Both nodes already have the same software environment installed.
# - The model path and dataset path below are valid inside the container.

# ===========================================================================
# Configuration
# ===========================================================================

MODEL_PATH="${MODEL_PATH:-/workspace/models/GLM-5.1-w4a8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1}"
RAW_DATASET_PATH="${RAW_DATASET_PATH:-/vllm-workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
PROXY_DIR="${PROXY_DIR:-${VLLM_ASCEND_DIR}/examples/disaggregated_prefill_v1}"
MULTITURN_DIR="${MULTITURN_DIR:-${VLLM_DIR}/benchmarks/multi_turn}"
RESULT_DIR="${RESULT_DIR:-/vllm-workspace/bench_results/glm5_pd_2node_multiturn_$(date +%Y%m%d_%H%M%S)}"
CONV_DATASET_PATH="${CONV_DATASET_PATH:-${RESULT_DIR}/sharegpt_conv.json}"
RESULTS_CSV="results.csv"
RAW_DATA_DIRNAME="${RAW_DATA_DIRNAME:-raw_data}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/tmp/glm5_pd_2node_logs/$(basename "${RESULT_DIR}")}"
REMOTE_PID_DIR="${REMOTE_PID_DIR:-/tmp/glm5_pd_2node_pids}"

PREFILL_NODE_IP="${PREFILL_NODE_IP:-7.246.92.163}"
DECODE_NODE_IP="${DECODE_NODE_IP:-7.246.92.169}"
DECODE_SSH_HOST="${DECODE_SSH_HOST:-${DECODE_NODE_IP}}"
SSH_USER="${SSH_USER:-}"
SSH_BIN="${SSH_BIN:-ssh}"
SSH_OPTIONS="${SSH_OPTIONS:-}"
DECODE_CONTAINER_RUNTIME="${DECODE_CONTAINER_RUNTIME:-docker}"
DECODE_CONTAINER_NAME="${DECODE_CONTAINER_NAME:-vllm-a3-pd-lzm}"
ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-30000}"
ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-60000}"
HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-17}"
HCCL_RDMA_RETRY_CNT="${HCCL_RDMA_RETRY_CNT:-7}"

PREFILL_NIC_NAME="${PREFILL_NIC_NAME:-enp48s3u1u1c2}"
DECODE_NIC_NAME="${DECODE_NIC_NAME:-enp48s3u1u1}"

PREFILL_DEVICES="${PREFILL_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
DECODE0_DEVICES="${DECODE0_DEVICES:-0,1,2,3,4,5,6,7}"
DECODE1_DEVICES="${DECODE1_DEVICES:-8,9,10,11,12,13,14,15}"

PREFILL_PORT="${PREFILL_PORT:-6700}"
DECODE0_PORT="${DECODE0_PORT:-6721}"
DECODE1_PORT="${DECODE1_PORT:-6722}"
PROXY_PORT="${PROXY_PORT:-8000}"
PROXY_LISTEN_HOST="${PROXY_LISTEN_HOST:-0.0.0.0}"
PROXY_CONNECT_HOST="${PROXY_CONNECT_HOST:-127.0.0.1}"

PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"
DECODE_KV_PORT="${DECODE_KV_PORT:-30100}"
PREFILL_DP_RPC_PORT="${PREFILL_DP_RPC_PORT:-10521}"
DECODE_DP_RPC_PORT="${DECODE_DP_RPC_PORT:-10523}"

PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-1}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-16}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-2}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"

PREFILL_MAX_MODEL_LEN="${PREFILL_MAX_MODEL_LEN:-131072}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS:-4096}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-64}"
PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.95}"
PREFILL_ENABLE_CHUNKED_PREFILL="${PREFILL_ENABLE_CHUNKED_PREFILL:-1}"

DECODE_MAX_MODEL_LEN="${DECODE_MAX_MODEL_LEN:-200000}"
DECODE_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS:-32}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-8}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.92}"

MAX_ITEMS="${MAX_ITEMS:-128}"
MIN_TURNS="${MIN_TURNS:-8}"
MAX_TURNS="${MAX_TURNS:-20}"
CONVERT_MAX_CONTENT_LEN="${CONVERT_MAX_CONTENT_LEN:-12000}"
NUM_CLIENTS="${NUM_CLIENTS:-24}"
MAX_ACTIVE_CONVERSATIONS="${MAX_ACTIVE_CONVERSATIONS:-96}"
REQUEST_RATES="${REQUEST_RATES:-}"
WARMUP_STEP="${WARMUP_STEP:-1}"
LIMIT_MAX_TOKENS="${LIMIT_MAX_TOKENS:-32}"
LIMIT_MIN_TOKENS="${LIMIT_MIN_TOKENS:-32}"
CASE_VARIANTS="${CASE_VARIANTS:-}"
BENCH_REPEAT="${BENCH_REPEAT:-5}"
CASE_RETRY_LIMIT="${CASE_RETRY_LIMIT:-2}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-300}"
MAX_RETRIES="${MAX_RETRIES:-0}"
CONVERSATION_SAMPLING="${CONVERSATION_SAMPLING:-round_robin}"
CASE_PRESET="${CASE_PRESET:-multiturn}"
FORMAL_NO_EARLY_STOP="${FORMAL_NO_EARLY_STOP:-1}"
CONVERT_SAMPLE_FACTOR="${CONVERT_SAMPLE_FACTOR:-8}"

LAPS_WAIT_WINDOW_MS="${LAPS_WAIT_WINDOW_MS:-0}"
LAPS_WAIT_MAX_BATCH="${LAPS_WAIT_MAX_BATCH:-4}"
LAPS_LONG_PREFILL_CAP="${LAPS_LONG_PREFILL_CAP:-0}"
LAPS_SHORT_RESERVED_RATIO="${LAPS_SHORT_RESERVED_RATIO:-0}"
LAPS_STATS_LOG_INTERVAL_S="${LAPS_STATS_LOG_INTERVAL_S:-5}"

STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-60}"
SLEEP_AFTER_STOP_S="${SLEEP_AFTER_STOP_S:-10}"
PORT_FREE_TIMEOUT_S="${PORT_FREE_TIMEOUT_S:-30}"
PREFILL_KV_PORT_SPAN="${PREFILL_KV_PORT_SPAN:-16}"
GRACEFUL_STOP_TIMEOUT_S="${GRACEFUL_STOP_TIMEOUT_S:-15}"
KV_TIME_WAIT_TIMEOUT_S="${KV_TIME_WAIT_TIMEOUT_S:-70}"

# ===========================================================================
# Runtime state
# ===========================================================================

PREFILL_PID=""
PROXY_PID=""

mkdir -p "${RESULT_DIR}/logs"
mkdir -p "${RESULT_DIR}/${RAW_DATA_DIRNAME}"

# ===========================================================================
# Logging & environment
# ===========================================================================

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
  log_cmd_presence "ssh"
  log_cmd_presence "pgrep"
  log_cmd_presence "ps"
  log_cmd_presence "ss"
  log_cmd_presence "lsof"
  log_cmd_presence "netstat"
  log_cmd_presence "fuser"
  log_cmd_presence "setsid"
  log_cmd_presence "curl"
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

# ===========================================================================
# SSH helpers
# ===========================================================================

ssh_target() {
  if [ -n "${SSH_USER}" ]; then
    printf '%s@%s' "${SSH_USER}" "${DECODE_SSH_HOST}"
  else
    printf '%s' "${DECODE_SSH_HOST}"
  fi
}

ssh_run_cmd() {
  local remote_cmd="$1"
  local target
  target="$(ssh_target)"
  # shellcheck disable=SC2086
  ${SSH_BIN} ${SSH_OPTIONS} "${target}" "bash -lc ${remote_cmd@Q}"
}

ssh_run_in_decode_container() {
  local container_cmd="$1"
  local host_cmd
  printf -v host_cmd '%q exec %q bash -lc %q' \
    "${DECODE_CONTAINER_RUNTIME}" "${DECODE_CONTAINER_NAME}" "${container_cmd}"
  ssh_run_cmd "${host_cmd}"
}

ssh_check_decode_container_running() {
  local host_cmd
  printf -v host_cmd '%q inspect -f %q %q' \
    "${DECODE_CONTAINER_RUNTIME}" '{{.State.Running}}' "${DECODE_CONTAINER_NAME}"
  ssh_run_cmd "${host_cmd}"
}

# ===========================================================================
# Case presets & variant mapping
# ===========================================================================

resolve_case_preset() {
  case "${CASE_PRESET}" in
    multiturn)
      if [ -z "${REQUEST_RATES}" ]; then
        REQUEST_RATES="0 4 6 8"
      fi
      if [ -z "${CASE_VARIANTS}" ]; then
        CASE_VARIANTS="off t256_w0 t384_w0 t512_w0"
      fi
      ;;
    multiturn_budget)
      if [ -z "${REQUEST_RATES}" ]; then
        REQUEST_RATES="0 4"
      fi
      if [ -z "${CASE_VARIANTS}" ]; then
        CASE_VARIANTS="off t512_w0 t512_w0_cap2048_res30"
      fi
      ;;
    multiturn_threshold_sweep)
      if [ -z "${REQUEST_RATES}" ]; then
        REQUEST_RATES="0 4"
      fi
      if [ -z "${CASE_VARIANTS}" ]; then
        CASE_VARIANTS="off t256_w0 t384_w0 t512_w0 t1024_w0"
      fi
      ;;
    multiturn_budget_sweep)
      if [ -z "${REQUEST_RATES}" ]; then
        REQUEST_RATES="0 4"
      fi
      if [ -z "${CASE_VARIANTS}" ]; then
        CASE_VARIANTS="off t512_w0 t512_w0_cap1536_res20 t512_w0_cap2048_res30"
      fi
      ;;
    *)
      log "Unknown CASE_PRESET: ${CASE_PRESET}. Supported presets: 'multiturn', 'multiturn_budget', 'multiturn_threshold_sweep', 'multiturn_budget_sweep'."
      return 1
      ;;
  esac
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

# ===========================================================================
# Process management
# ===========================================================================

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

get_pgid() {
  local pid="$1"
  [ -z "${pid}" ] && return 1
  ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' '
}

kill_process_group() {
  local pid="$1"
  local pgid
  pgid="$(get_pgid "${pid}" || true)"
  [ -z "${pgid}" ] && return 1
  [ "${pgid}" = "$$" ] && return 1
  [ "${pgid}" = "${BASHPID}" ] && return 1
  log "Stopping process group pgid=${pgid} (anchor pid=${pid})"
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  return 0
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

wait_process_group_gone() {
  local pid="$1"
  local pgid
  local deadline=$((SECONDS + STOP_TIMEOUT_S))

  pgid="$(get_pgid "${pid}" || true)"
  [ -z "${pgid}" ] && return 0

  while pgrep -g "${pgid}" >/dev/null 2>&1; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Force killing process group pgid=${pgid}"
      kill -KILL -- "-${pgid}" 2>/dev/null || true
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
    kill_process_group "${pid}" || kill_tree "${pid}"
  done
  for pid in ${pids}; do
    [ "${pid}" = "$$" ] && continue
    [ "${pid}" = "${BASHPID}" ] && continue
    wait_process_group_gone "${pid}" || wait_gone "${pid}"
  done
}

# ===========================================================================
# Port management
# ===========================================================================

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
  if command -v netstat >/dev/null 2>&1; then
    netstat -ltnp 2>/dev/null | grep -Eq "[:.]${port}[[:space:]]"
    return $?
  fi
  return 1
}

port_listener_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnpH "( sport = :${port} )" 2>/dev/null \
      | sed -nE 's/.*pid=([0-9]+).*/\1/p' \
      | sort -u
    return 0
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${port}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u
    return 0
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -ltnp 2>/dev/null \
      | awk -v port=":${port}" '$4 ~ port"$" {print $7}' \
      | sed -nE 's#([0-9]+)/.*#\1#p' \
      | sort -u
    return 0
  fi
  return 0
}

print_port_users() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN || true
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltnp 2>/dev/null | grep -E "[:.]${port}[[:space:]]" || true
  fi
}

wait_port_free() {
  local port="$1"
  local name="$2"
  local deadline=$((SECONDS + PORT_FREE_TIMEOUT_S))

  while port_in_use "${port}"; do
    log "Waiting for port ${port} (${name}) to become free"
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Port ${port} for ${name} is still in use:"
      print_port_users "${port}"
      return 1
    fi
    sleep 1
  done
}

force_release_local_port() {
  local port="$1"
  local name="$2"
  local pids

  pids="$(port_listener_pids "${port}")"
  [ -z "${pids}" ] && return 0

  log "Port ${port} (${name}) is occupied; attempting to stop listener processes"
  print_port_users "${port}"
  for pid in ${pids}; do
    [ "${pid}" = "$$" ] && continue
    [ "${pid}" = "${BASHPID}" ] && continue
    if ! kill_process_group "${pid}"; then
      log "Stopping pid=${pid} holding port ${port} (${name})"
      kill_tree "${pid}"
    fi
  done
  for pid in ${pids}; do
    [ "${pid}" = "$$" ] && continue
    [ "${pid}" = "${BASHPID}" ] && continue
    wait_process_group_gone "${pid}" || wait_gone "${pid}"
  done
}

ensure_local_kv_ports_free() {
  local span="${PREFILL_KV_PORT_SPAN}"
  local end_port=$((PREFILL_KV_PORT + span - 1))
  local port
  log "Checking local prefill KV ports: ${PREFILL_KV_PORT}-${end_port}"
  for ((port = PREFILL_KV_PORT; port <= end_port; port++)); do
    wait_port_free "${port}" "Prefill KV"
  done
}

local_kv_ports_in_use() {
  local span="${PREFILL_KV_PORT_SPAN}"
  local end_port=$((PREFILL_KV_PORT + span - 1))
  local port

  for ((port = PREFILL_KV_PORT; port <= end_port; port++)); do
    if port_in_use "${port}"; then
      return 0
    fi
  done

  return 1
}

force_release_local_kv_ports() {
  local span="${PREFILL_KV_PORT_SPAN}"
  local end_port=$((PREFILL_KV_PORT + span - 1))
  local port

  log "Force releasing local prefill KV ports if occupied: ${PREFILL_KV_PORT}-${end_port}"
  for ((port = PREFILL_KV_PORT; port <= end_port; port++)); do
    if port_in_use "${port}"; then
      force_release_local_port "${port}" "Prefill KV"
    fi
  done
}

port_has_time_wait() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tanH state time-wait "( sport = :${port} )" 2>/dev/null | grep -q .
    return $?
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -tan 2>/dev/null | grep -E "[:.]${port}[[:space:]]" | grep -q "TIME_WAIT"
    return $?
  fi
  return 1
}

try_kill_time_wait_sockets() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -K state time-wait "( sport = :${port} )" >/dev/null 2>&1 || true
  fi
}

wait_kv_ports_time_wait_clear() {
  local span="${PREFILL_KV_PORT_SPAN}"
  local end_port=$((PREFILL_KV_PORT + span - 1))
  local timeout="${KV_TIME_WAIT_TIMEOUT_S}"
  local deadline=$((SECONDS + timeout))
  local port tw_ports

  tw_ports=""
  for ((port = PREFILL_KV_PORT; port <= end_port; port++)); do
    if port_has_time_wait "${port}"; then
      tw_ports="${tw_ports} ${port}"
    fi
  done
  [ -z "${tw_ports}" ] && return 0

  log "KV ports with TIME_WAIT sockets:${tw_ports}"
  for port in ${tw_ports}; do
    try_kill_time_wait_sockets "${port}"
  done
  sleep 1

  while [ "${SECONDS}" -lt "${deadline}" ]; do
    local still_tw=0
    for port in ${tw_ports}; do
      if port_has_time_wait "${port}"; then
        still_tw=1
        break
      fi
    done
    if [ "${still_tw}" -eq 0 ]; then
      log "KV ports TIME_WAIT cleared"
      return 0
    fi
    sleep 2
  done
  log "WARNING: some KV ports still have TIME_WAIT sockets after ${timeout}s"
}

local_fallback_processes_running() {
  pgrep -f "python .*launch_online_dp.py .*--vllm-start-port ${PREFILL_PORT}" >/dev/null 2>&1 && return 0
  pgrep -f "vllm serve .*--port ${PREFILL_PORT}" >/dev/null 2>&1 && return 0
  pgrep -f "EngineCore" >/dev/null 2>&1 && return 0
  pgrep -f "multiproc_executor" >/dev/null 2>&1 && return 0
  pgrep -f "Worker_TP" >/dev/null 2>&1 && return 0
  pgrep -f "DPEngineCoreProc" >/dev/null 2>&1 && return 0
  pgrep -f "benchmark_serving_multi_turn.py" >/dev/null 2>&1 && return 0
  pgrep -f "load_balance_proxy_server_example.py .*--port ${PROXY_PORT}" >/dev/null 2>&1 && return 0
  return 1
}

local_cleanup_needed() {
  if port_in_use "${PREFILL_PORT}" || port_in_use "${PROXY_PORT}" || local_kv_ports_in_use; then
    return 0
  fi
  local_fallback_processes_running
}

log_local_cleanup_remnants() {
  log "Checking local prefill/proxy remnants"
  pgrep -af "python .*launch_online_dp.py .*--vllm-start-port ${PREFILL_PORT}" || true
  pgrep -af "vllm serve .*--port ${PREFILL_PORT}" || true
  pgrep -af "EngineCore|multiproc_executor|Worker_TP|DPEngineCoreProc" || true
  pgrep -af "benchmark_serving_multi_turn.py" || true
  pgrep -af "load_balance_proxy_server_example.py .*--port ${PROXY_PORT}" || true
}

ensure_local_ports_free() {
  log "Checking local ports: Prefill=${PREFILL_PORT}, Proxy=${PROXY_PORT}"
  wait_port_free "${PREFILL_PORT}" "Prefill"
  wait_port_free "${PROXY_PORT}" "Proxy"
  ensure_local_kv_ports_free
}

# ===========================================================================
# Prerequisites verification
# ===========================================================================

verify_prereqs() {
  log "Verifying local prerequisites"
  command -v pgrep >/dev/null 2>&1 || {
    log "Required command not found: pgrep"
    return 1
  }
  command -v ps >/dev/null 2>&1 || {
    log "Required command not found: ps"
    return 1
  }
  command -v setsid >/dev/null 2>&1 || {
    log "Required command not found: setsid"
    return 1
  }
  if ! command -v ss >/dev/null 2>&1 \
    && ! command -v lsof >/dev/null 2>&1 \
    && ! command -v netstat >/dev/null 2>&1 \
    && ! command -v fuser >/dev/null 2>&1; then
    log "No local port inspection tool found. Need one of: ss, lsof, netstat, fuser"
    return 1
  fi
  [ -f "${RAW_DATASET_PATH}" ] || {
    log "Dataset not found: ${RAW_DATASET_PATH}"
    return 1
  }
  [ -d "${VLLM_ASCEND_DIR}" ] || {
    log "VLLM_ASCEND_DIR not found: ${VLLM_ASCEND_DIR}"
    return 1
  }
  [ -d "${VLLM_DIR}" ] || {
    log "VLLM_DIR not found: ${VLLM_DIR}"
    return 1
  }
  [ -d "${MULTITURN_DIR}" ] || {
    log "MULTITURN_DIR not found: ${MULTITURN_DIR}"
    return 1
  }
  [ -e "${MODEL_PATH}" ] || {
    log "Model path not found locally: ${MODEL_PATH}"
    return 1
  }
  log "Checking SSH connectivity to $(ssh_target)"
  ssh_run_cmd "true" >/dev/null
  log "Checking decode container status: runtime=${DECODE_CONTAINER_RUNTIME}, container=${DECODE_CONTAINER_NAME}"
  local container_running
  container_running="$(ssh_check_decode_container_running)"
  log "Decode container running state: ${container_running}"
  if [ "${container_running}" != "true" ]; then
    log "Decode container is not running on $(ssh_target): ${DECODE_CONTAINER_NAME}"
    return 1
  fi
  log "Checking decode container paths"
  local path_check_cmd
  path_check_cmd=$(cat <<EOF
set -euo pipefail
echo "[remote] checking path: ${VLLM_ASCEND_DIR}"
if [ -d ${VLLM_ASCEND_DIR@Q} ]; then
  ls -ld ${VLLM_ASCEND_DIR@Q}
else
  echo "[remote] missing directory: ${VLLM_ASCEND_DIR}"
  exit 1
fi
echo "[remote] checking path: ${VLLM_DIR}"
if [ -d ${VLLM_DIR@Q} ]; then
  ls -ld ${VLLM_DIR@Q}
else
  echo "[remote] missing directory: ${VLLM_DIR}"
  exit 1
fi
echo "[remote] checking path: ${MODEL_PATH}"
if [ -e ${MODEL_PATH@Q} ]; then
  ls -ld ${MODEL_PATH@Q}
else
  echo "[remote] missing model path: ${MODEL_PATH}"
  exit 1
fi
EOF
)
  if ! ssh_run_in_decode_container "${path_check_cmd}"; then
    log "Decode container path check failed"
    return 1
  fi
  log "Decode container paths verified"
}

# ===========================================================================
# Service lifecycle (stop / cleanup)
# ===========================================================================

stop_remote_services() {
  local container_cmd
  log "Stopping remote decode services in $(ssh_target):${DECODE_CONTAINER_NAME}"
  container_cmd=$(cat <<EOF
set -euo pipefail
mkdir -p ${REMOTE_LOG_DIR@Q} ${REMOTE_PID_DIR@Q}
stop_pidfile() {
  local pidfile="\$1"
  if [ ! -f "\$pidfile" ]; then
    return 0
  fi
  local pid
  pid="\$(cat "\$pidfile" 2>/dev/null || true)"
  if [ -z "\$pid" ]; then
    rm -f "\$pidfile"
    return 0
  fi
  if ! kill -0 "\$pid" 2>/dev/null; then
    echo "[remote] pid from \$pidfile already exited: \$pid"
    rm -f "\$pidfile"
    return 0
  fi
  local pgid
  pgid="\$(ps -o pgid= -p "\$pid" 2>/dev/null | tr -d ' ' || true)"
  if [ -n "\$pgid" ]; then
    echo "[remote] stopping process group pgid=\$pgid from \$pidfile"
    kill -TERM -- "-\$pgid" 2>/dev/null || true
    sleep 2
    pgrep -g "\$pgid" >/dev/null 2>&1 && kill -KILL -- "-\$pgid" 2>/dev/null || true
  else
    echo "[remote] stopping pid=\$pid from \$pidfile"
    kill -TERM "\$pid" 2>/dev/null || true
    sleep 2
    kill -KILL "\$pid" 2>/dev/null || true
  fi
  rm -f "\$pidfile"
}
echo "[remote] checking decode processes before stop"
pgrep -af "vllm serve .*--port ${DECODE0_PORT}|vllm serve .*--port ${DECODE1_PORT}" || true
stop_pidfile ${REMOTE_PID_DIR@Q}/decode0.pid
stop_pidfile ${REMOTE_PID_DIR@Q}/decode1.pid
pkill -f "vllm serve .*--port ${DECODE0_PORT}" 2>/dev/null || true
pkill -f "vllm serve .*--port ${DECODE1_PORT}" 2>/dev/null || true
pkill -f "EngineCore_DP0" 2>/dev/null || true
pkill -f "EngineCore_DP1" 2>/dev/null || true
pkill -f "DPEngineCoreProc" 2>/dev/null || true
echo "[remote] checking decode processes after stop"
pgrep -af "vllm serve .*--port ${DECODE0_PORT}|vllm serve .*--port ${DECODE1_PORT}" || true
echo "[remote] checking decode worker remnants after stop"
pgrep -af "EngineCore_DP0|EngineCore_DP1|DPEngineCoreProc|multiproc_executor" || true
sleep ${SLEEP_AFTER_STOP_S}
EOF
)
  ssh_run_in_decode_container "${container_cmd}" || true
  log "Remote decode stop command finished"
}

stop_tracked_local_services() {
  log "Stopping tracked local services (graceful → escalate)"
  # Send SIGTERM to just the main process (not the whole group) so that
  # vLLM's internal shutdown can cascade cleanly to EngineCore → Workers,
  # instead of every Worker_TP* logging death-pipe + shutdown noise at once.
  for pid in "${PROXY_PID}" "${PREFILL_PID}"; do
    [ -z "${pid}" ] && continue
    kill -0 "${pid}" 2>/dev/null || continue
    log "Sending SIGTERM to main process pid=${pid}"
    kill -TERM "${pid}" 2>/dev/null || true
  done
  local deadline=$((SECONDS + GRACEFUL_STOP_TIMEOUT_S))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    local any_alive=0
    for pid in "${PREFILL_PID}" "${PROXY_PID}"; do
      [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null && any_alive=1
    done
    [ "${any_alive}" -eq 0 ] && break
    sleep 1
  done
  for pid in "${PROXY_PID}" "${PREFILL_PID}"; do
    [ -z "${pid}" ] && continue
    kill -0 "${pid}" 2>/dev/null || continue
    log "Graceful stop timed out for pid=${pid}, killing process group"
    kill_process_group "${pid}" || kill_tree "${pid}"
  done
}

wait_tracked_local_services_gone() {
  wait_process_group_gone "${PROXY_PID}" || wait_gone "${PROXY_PID}"
  wait_process_group_gone "${PREFILL_PID}" || wait_gone "${PREFILL_PID}"
}

stop_existing_local_services_fallback() {
  log "Stopping local fallback processes if present"
  kill_matching_cmd "python .*launch_online_dp.py .*--vllm-start-port ${PREFILL_PORT}"
  kill_matching_cmd "vllm serve .*--port ${PREFILL_PORT}"
  kill_matching_cmd "EngineCore"
  kill_matching_cmd "multiproc_executor"
  kill_matching_cmd "Worker_TP"
  kill_matching_cmd "DPEngineCoreProc"
  kill_matching_cmd "benchmark_serving_multi_turn.py"
  kill_matching_cmd "load_balance_proxy_server_example.py .*--port ${PROXY_PORT}"
  log "Local fallback stop scan finished"
}

force_release_local_ports_as_last_resort() {
  log "Force releasing local service ports as last resort"
  if port_in_use "${PREFILL_PORT}"; then
    force_release_local_port "${PREFILL_PORT}" "Prefill"
  fi
  if port_in_use "${PROXY_PORT}"; then
    force_release_local_port "${PROXY_PORT}" "Proxy"
  fi
  force_release_local_kv_ports
}

stop_services() {
  log "Stopping previous services"
  stop_tracked_local_services || true
  log "Requested stop for tracked local child process groups"
  wait_tracked_local_services_gone
  PROXY_PID=""
  PREFILL_PID=""
  stop_remote_services || true
  if local_cleanup_needed; then
    log "Tracked process-group stop was not sufficient; running local fallback cleanup"
    log_local_cleanup_remnants
    stop_existing_local_services_fallback || true
  fi
  log "Waiting ${SLEEP_AFTER_STOP_S}s after stop"
  sleep "${SLEEP_AFTER_STOP_S}"
  if local_kv_ports_in_use; then
    log "KV ports still occupied after process cleanup; force releasing"
    force_release_local_kv_ports || true
    sleep 2
  fi
  if local_cleanup_needed; then
    log "Local fallback cleanup still left remnants; force releasing local ports"
    log_local_cleanup_remnants
    force_release_local_ports_as_last_resort || true
  fi
  wait_kv_ports_time_wait_clear || true
  if ! ensure_local_ports_free; then
    log "WARNING: some local ports may still be occupied after cleanup"
  fi
  log "Previous services cleanup complete"
}

cleanup() {
  stop_services || true
}
trap cleanup EXIT

# ===========================================================================
# Startup & health checks
# ===========================================================================

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

dump_local_prefill_log() {
  local case_name="$1"
  local log_file="${RESULT_DIR}/logs/${case_name}_prefill.log"
  if [ -f "${log_file}" ]; then
    log "Dumping local prefill log for case ${case_name}"
    tail -200 "${log_file}" || true
  else
    log "Local prefill log not found for case ${case_name}: ${log_file}"
  fi
}

wait_prefill_startup() {
  local case_name="$1"
  local pid="$2"
  local log_file="${RESULT_DIR}/logs/${case_name}_prefill.log"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  local pattern="Application startup complete"

  log "Waiting for Prefill startup log pattern: ${pattern}"
  while true; do
    if grep -q "${pattern}" "${log_file}" 2>/dev/null; then
      log "Prefill startup is ready"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      log "Prefill process exited before startup completed: pid=${pid}"
      dump_local_prefill_log "${case_name}"
      return 1
    fi
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      log "Timed out waiting for Prefill startup"
      dump_local_prefill_log "${case_name}"
      return 1
    fi
    sleep 2
  done
}

dump_remote_decode_logs() {
  local case_name="$1"
  local container_cmd
  log "Dumping remote decode logs for case ${case_name}"
  container_cmd=$(cat <<EOF
set -euo pipefail
for f in \
  ${REMOTE_LOG_DIR@Q}/${case_name}_decode0.log \
  ${REMOTE_LOG_DIR@Q}/${case_name}_decode1.log; do
  if [ -f "\$f" ]; then
    echo "===== \$f ====="
    tail -80 "\$f" || true
  fi
done
EOF
)
  ssh_run_in_decode_container "${container_cmd}" || true
}

write_summary_header() {
  cat >"${RESULT_DIR}/${RESULTS_CSV}" <<'EOF'
case_name,repeat_id,variant,laps_threshold,request_rate,request_mode,requests_per_sec,ttft_mean_ms,ttft_99_ms,tpot_mean_ms,latency_mean_ms,prefill_queue_avg_ms,prefill_queue_p95_ms,prefill_queue_p99_ms,prefill_time_avg_ms,prefill_ttft_avg_ms,prefill_ttft_p99_ms,decode_queue_avg_ms,decode_queue_p95_ms,decode_queue_p99_ms,decode_time_avg_ms,itl_avg_ms,e2e_avg_ms,summary_log
EOF
}

scrape_metrics_snapshot() {
  local service_role="$1"
  local service_name="$2"
  local service_url="$3"
  local sample_kind="$4"
  local output_file="$5"

  log "Scraping ${service_role}/${service_name} metrics (${sample_kind}) from ${service_url}"
  if ! curl -fsS "${service_url}/metrics" > "${output_file}"; then
    log "Failed to scrape metrics from ${service_url}/metrics"
    return 1
  fi
}

# ===========================================================================
# Dataset preparation
# ===========================================================================

build_conversation_dataset() {
  log "Building ShareGPT multi-turn conversation replay dataset with local cleaning"
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
        if not isinstance(content, str):
            failed_content = True
            break
        if not content.strip():
            failed_content = True
            break
        if len(content) > max_content_len:
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

# ===========================================================================
# Service start (prefill / decode / proxy)
# ===========================================================================

start_prefill() {
  local case_name="$1"
  local laps_threshold="$2"
  local wait_window_ms="$3"
  local wait_max_batch="$4"
  local long_prefill_cap="$5"
  local short_reserved_ratio="$6"
  local log_file="${RESULT_DIR}/logs/${case_name}_prefill.log"
  local kv_config
  local prefill_extra_args=()

  kv_config="{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"kv_producer\",\"kv_port\":\"${PREFILL_KV_PORT}\",\"engine_id\":\"0\",\"kv_connector_extra_config\":{\"use_ascend_direct\":true,\"prefill\":{\"dp_size\":${PREFILL_DP_SIZE},\"tp_size\":${PREFILL_TP_SIZE}},\"decode\":{\"dp_size\":${DECODE_DP_SIZE},\"tp_size\":${DECODE_TP_SIZE}}}}"

  if [ "${PREFILL_ENABLE_CHUNKED_PREFILL}" = "1" ]; then
    prefill_extra_args+=(--enable-chunked-prefill)
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
    export OMP_NUM_THREADS=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export HCCL_BUFFSIZE=256
    export ASCEND_AGGREGATE_ENABLE=1
    export ASCEND_TRANSPORT_PRINT=1
    export ACL_OP_INIT_MODE=1
    export ASCEND_A3_ENABLE=1
    export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
    export ASCEND_RT_VISIBLE_DEVICES="${PREFILL_DEVICES}"
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export HCCL_INTRA_ROCE_ENABLE=1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=0
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib"

    if [ "${laps_threshold}" = "off" ]; then
      unset VLLM_ASCEND_LAPS_SCHEDULING VLLM_ASCEND_LAPS_THRESHOLD VLLM_ASCEND_LAPS_WAIT_WINDOW_MS VLLM_ASCEND_LAPS_WAIT_MAX_BATCH VLLM_ASCEND_LAPS_LONG_PREFILL_CAP VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S
    else
      export VLLM_ASCEND_LAPS_SCHEDULING=1
      export VLLM_ASCEND_LAPS_THRESHOLD="${laps_threshold}"
      export VLLM_ASCEND_LAPS_WAIT_WINDOW_MS="${wait_window_ms}"
      export VLLM_ASCEND_LAPS_WAIT_MAX_BATCH="${wait_max_batch}"
      if [ -n "${long_prefill_cap}" ]; then
        export VLLM_ASCEND_LAPS_LONG_PREFILL_CAP="${long_prefill_cap}"
      else
        unset VLLM_ASCEND_LAPS_LONG_PREFILL_CAP
      fi
      if [ -n "${short_reserved_ratio}" ]; then
        export VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO="${short_reserved_ratio}"
      else
        unset VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO
      fi
      export VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S="${LAPS_STATS_LOG_INTERVAL_S}"
    fi

    cd "${VLLM_ASCEND_DIR}"
    exec setsid vllm serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${PREFILL_PORT}" \
      --tensor-parallel-size "${PREFILL_TP_SIZE}" \
      --enable-expert-parallel \
      --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
      --seed 1024 \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --max-model-len "${PREFILL_MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${PREFILL_MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${PREFILL_MAX_NUM_SEQS}" \
      --trust-remote-code \
      --gpu-memory-utilization "${PREFILL_GPU_MEMORY_UTILIZATION}" \
      --quantization ascend \
      --async-scheduling \
      --enforce-eager \
      --enable-auto-tool-choice \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --safetensors-load-strategy eager \
      "${prefill_extra_args[@]}" \
      --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
      --kv-transfer-config "${kv_config}"
  ) >"${log_file}" 2>&1 &
  PREFILL_PID=$!
  log "Started Prefill pid=${PREFILL_PID}, log=${log_file}"
}

start_decode_instance() {
  local case_name="$1"
  local instance_name="$2"
  local devices="$3"
  local port="$4"
  local dp_rank="$5"
  local remote_log="${REMOTE_LOG_DIR}/${case_name}_${instance_name}.log"
  local remote_pidfile="${REMOTE_PID_DIR}/${instance_name}.pid"
  local kv_config
  local remote_inner
  local container_cmd
  local remote_cmd

  kv_config="{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_port\":\"${DECODE_KV_PORT}\",\"engine_id\":\"1\",\"kv_connector_extra_config\":{\"use_ascend_direct\":true,\"prefill\":{\"dp_size\":${PREFILL_DP_SIZE},\"tp_size\":${PREFILL_TP_SIZE}},\"decode\":{\"dp_size\":${DECODE_DP_SIZE},\"tp_size\":${DECODE_TP_SIZE}}}}"

remote_inner=$(cat <<EOF
set -euo pipefail
$(declare -f source_env)
$(declare -f unset_proxy_env)
export VLLM_DIR=${VLLM_DIR@Q}
source_env
unset_proxy_env
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_IP=${DECODE_NODE_IP@Q}
export GLOO_SOCKET_IFNAME=${DECODE_NIC_NAME@Q}
export TP_SOCKET_IFNAME=${DECODE_NIC_NAME@Q}
export HCCL_SOCKET_IFNAME=${DECODE_NIC_NAME@Q}
export ASCEND_CONNECT_TIMEOUT=${ASCEND_CONNECT_TIMEOUT@Q}
export ASCEND_TRANSFER_TIMEOUT=${ASCEND_TRANSFER_TIMEOUT@Q}
export HCCL_RDMA_TIMEOUT=${HCCL_RDMA_TIMEOUT@Q}
export HCCL_RDMA_RETRY_CNT=${HCCL_RDMA_RETRY_CNT@Q}
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300000
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=${devices@Q}
export HCCL_INTRA_ROCE_ENABLE=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}:/usr/local/lib
unset VLLM_ASCEND_LAPS_SCHEDULING VLLM_ASCEND_LAPS_THRESHOLD VLLM_ASCEND_LAPS_WAIT_WINDOW_MS VLLM_ASCEND_LAPS_WAIT_MAX_BATCH VLLM_ASCEND_LAPS_LONG_PREFILL_CAP VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S
cd ${VLLM_ASCEND_DIR@Q}
exec vllm serve ${MODEL_PATH@Q} \
  --host 0.0.0.0 \
  --port ${port@Q} \
  --data-parallel-size ${DECODE_DP_SIZE@Q} \
  --data-parallel-rank ${dp_rank@Q} \
  --data-parallel-address ${DECODE_NODE_IP@Q} \
  --data-parallel-rpc-port ${DECODE_DP_RPC_PORT@Q} \
  --tensor-parallel-size ${DECODE_TP_SIZE@Q} \
  --enable-expert-parallel \
  --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
  --seed 1024 \
  --served-model-name ${SERVED_MODEL_NAME@Q} \
  --max-model-len ${DECODE_MAX_MODEL_LEN@Q} \
  --max-num-batched-tokens ${DECODE_MAX_NUM_BATCHED_TOKENS@Q} \
  --max-num-seqs ${DECODE_MAX_NUM_SEQS@Q} \
  --trust-remote-code \
  --gpu-memory-utilization ${DECODE_GPU_MEMORY_UTILIZATION@Q} \
  --quantization ascend \
  --async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --safetensors-load-strategy eager \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 12, 16, 20, 24, 28, 32]}' \
  --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
  --kv-transfer-config ${kv_config@Q}
EOF
)

  printf -v container_cmd 'set -euo pipefail\nmkdir -p %q %q\nrm -f %q\nnohup setsid bash -lc %q > %q 2>&1 < /dev/null & pid=$!\necho "$pid" | tee %q\n' \
    "${REMOTE_LOG_DIR}" "${REMOTE_PID_DIR}" "${remote_pidfile}" "${remote_inner}" "${remote_log}" "${remote_pidfile}"
  printf -v remote_cmd '%q exec %q bash -lc %q' \
    "${DECODE_CONTAINER_RUNTIME}" "${DECODE_CONTAINER_NAME}" "${container_cmd}"

  local remote_pid
  remote_pid="$(ssh_run_cmd "${remote_cmd}")"
  remote_pid="$(printf '%s\n' "${remote_pid}" | tail -n 1)"
  log "Started ${instance_name} in $(ssh_target):${DECODE_CONTAINER_NAME} pid=${remote_pid}, pidfile=${remote_pidfile}, log=${remote_log}"
}

start_decode() {
  local case_name="$1"
  start_decode_instance "${case_name}" "decode0" "${DECODE0_DEVICES}" "${DECODE0_PORT}" "0"
  start_decode_instance "${case_name}" "decode1" "${DECODE1_DEVICES}" "${DECODE1_PORT}" "1"
}

start_proxy() {
  local case_name="$1"
  local log_file="${RESULT_DIR}/logs/${case_name}_proxy.log"

  (
    source_env
    unset_proxy_env
    cd "${PROXY_DIR}"
    exec setsid python3 load_balance_proxy_server_example.py \
      --host "${PROXY_LISTEN_HOST}" \
      --port "${PROXY_PORT}" \
      --prefiller-hosts "${PREFILL_NODE_IP}" \
      --prefiller-ports "${PREFILL_PORT}" \
      --decoder-hosts "${DECODE_NODE_IP}" "${DECODE_NODE_IP}" \
      --decoder-ports "${DECODE0_PORT}" "${DECODE1_PORT}"
  ) >"${log_file}" 2>&1 &
  PROXY_PID=$!
  log "Started Proxy pid=${PROXY_PID}, log=${log_file}"
}

# ===========================================================================
# Metrics & reporting
# ===========================================================================

append_results_row() {
  local case_name="$1"
  local repeat_id="$2"
  local variant_name="$3"
  local threshold="$4"
  local request_rate="$5"
  local summary_log="$6"
  local prefill_before="$7"
  local prefill_after="$8"
  local decode0_before="$9"
  local decode0_after="${10}"
  local decode1_before="${11}"
  local decode1_after="${12}"
  local request_mode="open_loop"

  if [ "${threshold}" = "off" ]; then
    threshold=""
  fi
  if [ "${request_rate}" = "0" ] || [ "${request_rate}" = "0.0" ]; then
    request_mode="saturation_no_sleep"
  fi

  python3 - \
    "${case_name}" "${repeat_id}" "${variant_name}" "${threshold}" \
    "${request_rate}" "${request_mode}" "${summary_log}" \
    "${prefill_before}" "${prefill_after}" \
    "${decode0_before}" "${decode0_after}" \
    "${decode1_before}" "${decode1_after}" <<'PY' >> "${RESULT_DIR}/${RESULTS_CSV}"
import re
import sys
import math
from pathlib import Path

case_name, repeat_id, variant_name, threshold = sys.argv[1:5]
request_rate, request_mode, summary_log_path = sys.argv[5:8]
prefill_before_path, prefill_after_path = Path(sys.argv[8]), Path(sys.argv[9])
decode0_before_path, decode0_after_path = Path(sys.argv[10]), Path(sys.argv[11])
decode1_before_path, decode1_after_path = Path(sys.argv[12]), Path(sys.argv[13])

bench_log = Path(summary_log_path)
text = bench_log.read_text(encoding="utf-8", errors="ignore") if bench_log.exists() else ""

def m(pat):
    match = re.search(pat, text, re.MULTILINE)
    return match.group(1) if match else ""

requests_per_sec = m(r"requests_per_sec\s*=\s*([0-9.]+)")

def grab_block(name):
    summary_pos = text.rfind("Statistics summary:")
    summary_text = text[summary_pos:] if summary_pos >= 0 else text
    match = re.search(rf"^\s*{name}\s+(.+)$", summary_text, re.MULTILINE)
    if not match:
        return {}
    tokens = [tok for tok in match.group(1).split() if tok != "..."]
    nums = [tok for tok in tokens if re.fullmatch(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", tok)]
    keys = ["count", "mean", "std", "min", "p25", "p50", "p75", "p90", "p99", "max"]
    return dict(zip(keys, nums))

ttft = grab_block("ttft_ms")
tpot = grab_block("tpot_ms")
latency = grab_block("latency_ms")

# Client summary columns come from the benchmark log.
HISTOGRAM_METRICS = [
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:inter_token_latency_seconds",
]

def parse_prom(path):
    t = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    out = {}
    for line in t.splitlines():
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
        return 0.0, ""
    return dc, f"{(ds / dc) * 1000.0:.6f}"

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

pb, pa = parse_prom(prefill_before_path), parse_prom(prefill_after_path)
d0b, d0a = parse_prom(decode0_before_path), parse_prom(decode0_after_path)
d1b, d1a = parse_prom(decode1_before_path), parse_prom(decode1_after_path)

prefill_queue_delta = delta_buckets(pb, pa, "vllm:request_queue_time_seconds")
_, prefill_queue = avg_ms(pb, pa, "vllm:request_queue_time_seconds")
prefill_queue_p95 = histogram_quantile_ms(0.95, prefill_queue_delta)
prefill_queue_p99 = histogram_quantile_ms(0.99, prefill_queue_delta)
_, prefill_time = avg_ms(pb, pa, "vllm:request_prefill_time_seconds")
_, prefill_ttft = avg_ms(pb, pa, "vllm:time_to_first_token_seconds")
prefill_ttft_p99 = histogram_quantile_ms(0.99, delta_buckets(pb, pa, "vllm:time_to_first_token_seconds"))
decode_queue_delta = delta_buckets(d0b, d0a, "vllm:request_queue_time_seconds")
d1_decode_queue_delta = delta_buckets(d1b, d1a, "vllm:request_queue_time_seconds")
decode_queue_delta_map = {upper: count for upper, count in decode_queue_delta}
for upper, count in d1_decode_queue_delta:
    decode_queue_delta_map[upper] = decode_queue_delta_map.get(upper, 0.0) + count
decode_queue_p95 = histogram_quantile_ms(0.95, sorted(decode_queue_delta_map.items()))
decode_queue_p99 = histogram_quantile_ms(0.99, sorted(decode_queue_delta_map.items()))

def weighted_decode_avg(metric_name):
    c0, a0 = avg_ms(d0b, d0a, metric_name)
    c1, a1 = avg_ms(d1b, d1a, metric_name)
    num, den = 0.0, 0.0
    if c0 > 0 and a0:
        num += c0 * float(a0); den += c0
    if c1 > 0 and a1:
        num += c1 * float(a1); den += c1
    return f"{num / den:.6f}" if den > 0 else ""

row = [
    case_name, repeat_id, variant_name, threshold,
    request_rate, request_mode,
    requests_per_sec,
    ttft.get("mean", ""), ttft.get("p99", ""),
    tpot.get("mean", ""),
    latency.get("mean", ""),
    prefill_queue, prefill_queue_p95, prefill_queue_p99, prefill_time, prefill_ttft, prefill_ttft_p99,
    weighted_decode_avg("vllm:request_queue_time_seconds"),
    decode_queue_p95, decode_queue_p99,
    weighted_decode_avg("vllm:request_decode_time_seconds"),
    weighted_decode_avg("vllm:inter_token_latency_seconds"),
    weighted_decode_avg("vllm:e2e_request_latency_seconds"),
    summary_log_path,
]
print(",".join(row))
PY
}

# ===========================================================================
# Benchmark orchestration
# ===========================================================================

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
    "${PROXY_CONNECT_HOST}"
    "${PROXY_PORT}"
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
RequestStats = bm.RequestStats
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
proxy_connect_host = sys.argv[9]
proxy_port = sys.argv[10]
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
        url=f"http://{proxy_connect_host}:{proxy_port}",
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


def write_raw_data_csv(path: Path, client_metrics: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(RequestStats._fields)
    if client_metrics:
        raw_data = pd.DataFrame(client_metrics)
        raw_data = raw_data.sort_values(by=["start_time_ms"])
        raw_data["end_time_ms"] = raw_data["start_time_ms"] + raw_data["latency_ms"]
    else:
        raw_data = pd.DataFrame(columns=[*columns, "end_time_ms"])
    raw_data.to_csv(path, index=False)


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
    bench_args = BenchmarkArgs(
        url=args.url, num_clients=args.num_clients, early_stop=not args.no_early_stop
    )

    benchmark_start_ns = time.perf_counter_ns()
    client_convs, client_metrics = await main_mp(
        client_args, req_args, bench_args, tokenizer, conversations
    )
    benchmark_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - benchmark_start_ns)
    requests_per_sec = len(client_metrics) / benchmark_runtime_sec
    benchmark_runtime_ms = benchmark_runtime_sec * 1000.0
    logger.info(
        "All clients finished, benchmark runtime: %.3f sec (%.3f ms), requests per second: %.3f",
        benchmark_runtime_sec,
        benchmark_runtime_ms,
        requests_per_sec,
    )
    if warmup_runtime_sec is not None:
        total_runtime_sec = benchmark_runtime_sec + warmup_runtime_sec
        logger.info(
            "Warmup runtime: %.3f sec (%.3f ms)",
            warmup_runtime_sec,
            warmup_runtime_sec * 1000.0,
        )
        logger.info(
            "Total runtime (including warmup): %.3f sec (%.3f ms)",
            total_runtime_sec,
            total_runtime_sec * 1000.0,
        )

    params = {
        "model": args.model,
        "num_clients": args.num_clients,
        "num_conversations": len(conversations),
        "active_conversations": args.max_active_conversations,
        "seed": args.seed,
    }
    if args.limit_min_tokens > 0:
        params["min_tokens"] = args.limit_min_tokens
    if args.limit_max_tokens > 0:
        params["max_tokens"] = args.limit_max_tokens

    process_statistics(
        client_metrics,
        test_params=params,
        warmup_percentages=[0.0],
        verbose=args.verbose,
        gen_conv_args=gen_conv_args,
        excel_output=False,
        warmup_runtime_sec=warmup_runtime_sec,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(conversations_dict_to_list(client_convs), f, indent=4)

    write_raw_data_csv(raw_data_file, client_metrics)


asyncio.run(main())
PY
  ) 2>&1 | tee "${summary_log}"
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
    "${PROXY_CONNECT_HOST}"
    "${PROXY_PORT}"
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

Color = bm.Color
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
proxy_connect_host = sys.argv[8]
proxy_port = sys.argv[9]
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
        url=f"http://{proxy_connect_host}:{proxy_port}",
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

    logger.info("%sInput parameters:%s", Color.GREEN, Color.RESET)
    logger.info("url=%s", f"http://{proxy_connect_host}:{proxy_port}")
    logger.info("model=%s", model_path)
    logger.info("num_clients=%s", num_clients)

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
    bench_args = BenchmarkArgs(
        url=args.url, num_clients=args.num_clients, early_stop=not args.no_early_stop
    )

    warmup_client_args = client_args._replace(
        skip_first_turn=False, max_turns=1, max_active_conversations=1
    )
    warmup_bench_args = bench_args._replace(early_stop=False)

    logger.info("%sWarmup start%s", Color.PURPLE, Color.RESET)
    warmup_start_ns = time.perf_counter_ns()
    conversations, _ = await main_mp(
        warmup_client_args, req_args, warmup_bench_args, tokenizer, conversations
    )
    warmup_runtime_sec = nanosec_to_sec(time.perf_counter_ns() - warmup_start_ns)
    logger.info(
        "%sWarmup runtime: %.3f sec (%.3f ms)%s",
        Color.PURPLE,
        warmup_runtime_sec,
        warmup_runtime_sec * 1000.0,
        Color.RESET,
    )
    logger.info("%sWarmup done%s", Color.PURPLE, Color.RESET)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(conversations_dict_to_list(conversations), f, indent=4)
    runtime_file.write_text(f"{warmup_runtime_sec:.6f}\n", encoding="utf-8")


asyncio.run(main())
PY
  ) 2>&1 | tee "${summary_log}"
}

run_case() {
  local case_name="$1"
  local repeat_id="$2"
  local variant_name="$3"
  local request_rate="$4"
  local config threshold wait_window_ms wait_max_batch long_prefill_cap short_reserved_ratio
  local case_key="${case_name}_run${repeat_id}"
  local warmup_output_file="${RESULT_DIR}/${case_key}_warmup_output.json"
  local warmup_runtime_file="${RESULT_DIR}/logs/${case_key}_warmup_runtime_sec.txt"
  local warmup_runtime_sec=""
  local formal_input_file="${CONV_DATASET_PATH}"
  local prefill_before="${RESULT_DIR}/logs/${case_key}_prefill_metrics_before.prom"
  local prefill_after="${RESULT_DIR}/logs/${case_key}_prefill_metrics_after.prom"
  local decode0_before="${RESULT_DIR}/logs/${case_key}_decode0_metrics_before.prom"
  local decode0_after="${RESULT_DIR}/logs/${case_key}_decode0_metrics_after.prom"
  local decode1_before="${RESULT_DIR}/logs/${case_key}_decode1_metrics_before.prom"
  local decode1_after="${RESULT_DIR}/logs/${case_key}_decode1_metrics_after.prom"

  config="$(variant_to_config "${variant_name}")"
  IFS='|' read -r threshold wait_window_ms wait_max_batch long_prefill_cap short_reserved_ratio <<< "${config}"

  log "========== CASE ${case_key} started =========="
  log "Case config: variant=${variant_name}, threshold=${threshold}, wait_window_ms=${wait_window_ms:-0}, wait_max_batch=${wait_max_batch:-0}, long_prefill_cap=${long_prefill_cap:-0}, short_reserved_ratio=${short_reserved_ratio:-0}, request_rate=${request_rate}"

  stop_services
  start_prefill "${case_key}" "${threshold}" "${wait_window_ms}" "${wait_max_batch}" "${long_prefill_cap}" "${short_reserved_ratio}"
  start_decode "${case_key}"

  wait_prefill_startup "${case_key}" "${PREFILL_PID}"
  wait_http "http://${PREFILL_NODE_IP}:${PREFILL_PORT}/health" "Prefill" || {
    dump_local_prefill_log "${case_key}"
    return 1
  }
  wait_http "http://${DECODE_NODE_IP}:${DECODE0_PORT}/health" "Decode[0]" || {
    dump_local_prefill_log "${case_key}"
    dump_remote_decode_logs "${case_key}"
    return 1
  }
  wait_http "http://${DECODE_NODE_IP}:${DECODE1_PORT}/health" "Decode[1]" || {
    dump_local_prefill_log "${case_key}"
    dump_remote_decode_logs "${case_key}"
    return 1
  }

  start_proxy "${case_key}"
  wait_log "${RESULT_DIR}/logs/${case_key}_proxy.log" "Application startup complete" "Proxy startup"
  wait_http "http://${PROXY_CONNECT_HOST}:${PROXY_PORT}/healthcheck" "Proxy"

  if [ "${threshold}" != "off" ]; then
    wait_log "${RESULT_DIR}/logs/${case_key}_prefill.log" "Ascend LAPS scheduler selected" "LAPS scheduler selection"
  fi

  if [ "${WARMUP_STEP}" = "1" ]; then
    run_multiturn_warmup "${case_key}" "${request_rate}"
    if [ -f "${warmup_runtime_file}" ]; then
      warmup_runtime_sec="$(tr -d '[:space:]' < "${warmup_runtime_file}")"
    fi
    formal_input_file="${warmup_output_file}"
  fi

  scrape_metrics_snapshot "prefill" "prefill" "http://${PREFILL_NODE_IP}:${PREFILL_PORT}" "before" "${prefill_before}"
  scrape_metrics_snapshot "decode" "decode0" "http://${DECODE_NODE_IP}:${DECODE0_PORT}" "before" "${decode0_before}"
  scrape_metrics_snapshot "decode" "decode1" "http://${DECODE_NODE_IP}:${DECODE1_PORT}" "before" "${decode1_before}"
  run_multiturn_bench "${case_key}" "${request_rate}" "${formal_input_file}" "${warmup_runtime_sec}"
  scrape_metrics_snapshot "prefill" "prefill" "http://${PREFILL_NODE_IP}:${PREFILL_PORT}" "after" "${prefill_after}"
  scrape_metrics_snapshot "decode" "decode0" "http://${DECODE_NODE_IP}:${DECODE0_PORT}" "after" "${decode0_after}"
  scrape_metrics_snapshot "decode" "decode1" "http://${DECODE_NODE_IP}:${DECODE1_PORT}" "after" "${decode1_after}"
  append_results_row \
    "${case_name}" "${repeat_id}" "${variant_name}" "${threshold}" "${request_rate}" \
    "${RESULT_DIR}/logs/${case_key}_bench.log" \
    "${prefill_before}" "${prefill_after}" \
    "${decode0_before}" "${decode0_after}" \
    "${decode1_before}" "${decode1_after}"

  log "========== CASE ${case_key} completed =========="
}

run_case_with_recovery() {
  local case_name="$1"
  local repeat_id="$2"
  local variant_name="$3"
  local request_rate="$4"
  local attempt=1
  local rc=0

  while [ "${attempt}" -le "${CASE_RETRY_LIMIT}" ]; do
    if [ "${attempt}" -gt 1 ]; then
      log "Retrying case ${case_name}_run${repeat_id}: attempt ${attempt}/${CASE_RETRY_LIMIT}"
    fi

    run_case "${case_name}" "${repeat_id}" "${variant_name}" "${request_rate}" && return 0 || rc=$?
    log "Case ${case_name}_run${repeat_id} failed with exit code ${rc} on attempt ${attempt}/${CASE_RETRY_LIMIT}"
    stop_services || true
    attempt=$((attempt + 1))
  done

  log "Case ${case_name}_run${repeat_id} failed after ${CASE_RETRY_LIMIT} attempts"
  return "${rc}"
}

main() {
  log_local_tooling
  resolve_case_preset
  verify_prereqs
  source_env
  unset_proxy_env
  write_summary_header
  build_conversation_dataset

  log "Running one-time local fallback cleanup before the first cold start"
  stop_existing_local_services_fallback || true
  sleep "${SLEEP_AFTER_STOP_S}"
  ensure_local_ports_free

  log "Results will be written to ${RESULT_DIR}"
  log "Using case preset: ${CASE_PRESET}"
  log "Using case variants: ${CASE_VARIANTS}"
  log "Using multi-turn request rates: ${REQUEST_RATES}"
  log "Case retry limit: ${CASE_RETRY_LIMIT}"
  log "Conversation replay config: max_items=${MAX_ITEMS}, min_turns=${MIN_TURNS}, max_turns=${MAX_TURNS}, num_clients=${NUM_CLIENTS}, max_active_conversations=${MAX_ACTIVE_CONVERSATIONS}, limit_tokens=[${LIMIT_MIN_TOKENS}, ${LIMIT_MAX_TOKENS}], convert_sample_factor=${CONVERT_SAMPLE_FACTOR}"
  log "Benchmark intent: request_rate=0 is saturation-no-sleep; request_rate=4/6/8 are the main open-loop points"
  log "Benchmark intent: higher client concurrency and shorter outputs are used to reduce decode dominance without lowering prefill capacity"
  log "Model config: model_path=${MODEL_PATH}, served_model_name=${SERVED_MODEL_NAME}"
  log "Dual-node config: prefill=${PREFILL_NODE_IP}:${PREFILL_PORT}, decode=${DECODE_NODE_IP}:${DECODE0_PORT}/${DECODE1_PORT}, proxy=${PROXY_CONNECT_HOST}:${PROXY_PORT}"

  for request_rate in ${REQUEST_RATES}; do
    local rate_name="${request_rate//./p}"
    for repeat_id in $(seq 1 "${BENCH_REPEAT}"); do
      for variant_name in ${CASE_VARIANTS}; do
        run_case_with_recovery "r${rate_name}_${variant_name}" "${repeat_id}" "${variant_name}" "${request_rate}"
      done
    done
  done

  log "All benchmark cases completed. Results: ${RESULT_DIR}"
  log "Results CSV: ${RESULT_DIR}/${RESULTS_CSV}"
  log "Raw data: ${RESULT_DIR}/${RAW_DATA_DIRNAME}/"
  log "Logs & prometheus snapshots: ${RESULT_DIR}/logs/"
}

main "$@"
