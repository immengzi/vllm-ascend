#!/usr/bin/env bash
set -euo pipefail

# Run a real Claude-trace benchmark for GLM-5 on a single-node Prefill-only deployment.
#
# This script replays a prepared Claude dataset against a local vLLM Prefill
# server for every warmup / measured / repeat / LAPS variant.

MODEL_PATH="${MODEL_PATH:-/workspace/models/GLM-5.1-w4a8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1}"
PREPARED_TRACE_DIR="${PREPARED_TRACE_DIR:-}"

VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
VLLM_DIR="${VLLM_DIR:-/vllm-workspace/vllm}"
BENCH_DIR="${BENCH_DIR:-/vllm-workspace/vllm_bench_claude}"
BENCH_SCRIPT="${BENCH_SCRIPT:-${BENCH_DIR}/vllm_bench_claude.py}"
RESULT_DIR="${RESULT_DIR:-/vllm-workspace/bench_results/glm5_claude_prefill_only_$(date +%Y%m%d_%H%M%S)}"
CASE_VARIANTS="${CASE_VARIANTS:-off t4096 t8192}"
CASE_REPEATS="${CASE_REPEATS:-3}"
CASE_RETRY_LIMIT="${CASE_RETRY_LIMIT:-1}"

PREFILL_BIND_HOST="${PREFILL_BIND_HOST:-0.0.0.0}"
PREFILL_CONNECT_HOST="${PREFILL_CONNECT_HOST:-127.0.0.1}"
PREFILL_NODE_IP="${PREFILL_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
PREFILL_NIC_NAME="${PREFILL_NIC_NAME:-enp48s3u1u1c2}"
PREFILL_DEVICES="${PREFILL_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
PREFILL_PORT="${PREFILL_PORT:-6700}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"

PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-1}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-16}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-1}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-16}"
PREFILL_MAX_MODEL_LEN="${PREFILL_MAX_MODEL_LEN:-131072}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS:-4096}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-64}"
PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.95}"
PREFILL_ENABLE_CHUNKED_PREFILL="${PREFILL_ENABLE_CHUNKED_PREFILL:-1}"

RUN_WARMUP="${RUN_WARMUP:-1}"
POST_WARMUP_SLEEP_S="${POST_WARMUP_SLEEP_S:-10}"
WARMUP_TIMEOUT="${WARMUP_TIMEOUT:-1800}"
MEASURE_TIMEOUT="${MEASURE_TIMEOUT:-1800}"
REQUEST_MAX_TOKENS="${REQUEST_MAX_TOKENS:-1}"

LAPS_WAIT_WINDOW_MS="${LAPS_WAIT_WINDOW_MS:-0}"
LAPS_WAIT_MAX_BATCH="${LAPS_WAIT_MAX_BATCH:-4}"
LAPS_STATS_LOG_INTERVAL_S="${LAPS_STATS_LOG_INTERVAL_S:-5}"

ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-30000}"
ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-60000}"
HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-17}"
HCCL_RDMA_RETRY_CNT="${HCCL_RDMA_RETRY_CNT:-7}"

STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-60}"
SLEEP_AFTER_STOP_S="${SLEEP_AFTER_STOP_S:-10}"
PORT_FREE_TIMEOUT_S="${PORT_FREE_TIMEOUT_S:-30}"

PREFILL_PID=""
PREPARED_MANIFEST=""
PREPARED_SESSION_COUNT=""
PREPARED_REQUEST_COUNT=""

mkdir -p "${RESULT_DIR}/logs"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

append_case_status() {
  local case_name="$1"
  local status="$2"
  local detail="$3"
  printf '%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "${case_name}" "${status}" "${detail}" \
    >> "${RESULT_DIR}/case_status.tsv"
}

append_summary() {
  local case_name="$1"
  local variant_name="$2"
  local threshold="$3"
  local measured_json="$4"
  local prefill_before="$5"
  local prefill_after="$6"
  local prepared_dataset_dir="$7"
  local dataset_manifest="$8"
  local prepared_session_count="$9"
  local prepared_request_count="${10}"
  python3 - "$case_name" "$variant_name" "$threshold" "$measured_json" "$prefill_before" "$prefill_after" "$prepared_dataset_dir" "$dataset_manifest" "$prepared_session_count" "$prepared_request_count" >> "${RESULT_DIR}/summary.tsv" <<'PY'
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

(
    case_name,
    variant_name,
    threshold,
    measured_json,
    prefill_before,
    prefill_after,
    prepared_dataset_dir,
    dataset_manifest,
    prepared_session_count,
    prepared_request_count,
) = sys.argv[1:11]
with open(measured_json, "r", encoding="utf-8") as f:
    data = json.load(f)

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

row = [
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    case_name,
    variant_name,
    threshold,
    str(data.get("completed", 0)),
    str(data.get("failed", 0)),
    str(data.get("skipped_over_limit", 0)),
    str(data.get("duration", 0)),
    str((data.get("ttft_ms") or {}).get("mean", "")),
    str((data.get("ttft_ms") or {}).get("median", "")),
    str((data.get("ttft_ms") or {}).get("p99", "")),
    str((data.get("tpot_ms") or {}).get("mean", "")),
    str((data.get("tpot_ms") or {}).get("median", "")),
    str((data.get("itl_ms") or {}).get("mean", "")),
    str((data.get("itl_ms") or {}).get("median", "")),
    str((data.get("e2el_s") or {}).get("mean", "")),
    str((data.get("e2el_s") or {}).get("median", "")),
    str((data.get("e2el_s") or {}).get("p99", "")),
    str(data.get("total_input_tokens", 0)),
    str(data.get("total_output_tokens", 0)),
    str(data.get("request_throughput", 0)),
    str(data.get("total_token_throughput", 0)),
    prepared_dataset_dir,
    dataset_manifest,
    prepared_session_count,
    prepared_request_count,
    prefill_queue_avg,
    prefill_queue_median,
    prefill_queue_p90,
    prefill_queue_p99,
    prefill_time_avg,
    prefill_ttft_avg,
    prefill_ttft_median,
    prefill_ttft_p90,
    prefill_ttft_p99,
    measured_json,
]
print("\t".join(row))
PY
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

verify_prereqs() {
  [ -n "${PREPARED_TRACE_DIR}" ] || {
    log "PREPARED_TRACE_DIR is required"
    return 1
  }
  [ -d "${PREPARED_TRACE_DIR}" ] || {
    log "PREPARED_TRACE_DIR does not exist: ${PREPARED_TRACE_DIR}"
    return 1
  }
  PREPARED_MANIFEST="${PREPARED_TRACE_DIR}/manifest.json"
  [ -f "${PREPARED_MANIFEST}" ] || {
    log "Prepared dataset manifest not found: ${PREPARED_MANIFEST}"
    return 1
  }
  [ -f "${BENCH_SCRIPT}" ] || {
    log "BENCH_SCRIPT not found: ${BENCH_SCRIPT}"
    return 1
  }
}

load_prepared_dataset_metadata() {
  eval "$(
    python3 - "${PREPARED_MANIFEST}" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"PREPARED_SESSION_COUNT={shlex.quote(str(data.get('prepared_session_count', 0)))}")
print(f"PREPARED_REQUEST_COUNT={shlex.quote(str(data.get('prepared_request_count', 0)))}")
PY
  )"

  if [ -z "${PREPARED_SESSION_COUNT}" ] || [ "${PREPARED_SESSION_COUNT}" = "0" ]; then
    log "Prepared dataset has no usable session files: ${PREPARED_TRACE_DIR}"
    return 1
  fi
  if [ -z "${PREPARED_REQUEST_COUNT}" ] || [ "${PREPARED_REQUEST_COUNT}" = "0" ]; then
    log "Prepared dataset has no replayable requests: ${PREPARED_TRACE_DIR}"
    return 1
  fi
}

case_variant_to_params() {
  local variant="$1"
  if [ "${variant}" = "off" ]; then
    printf 'off|0|4|off\n'
    return
  fi
  if [[ "${variant}" =~ ^t[0-9]+$ ]]; then
    printf '%s|%s|%s|laps_%s\n' "${variant#t}" "${LAPS_WAIT_WINDOW_MS}" "${LAPS_WAIT_MAX_BATCH}" "${variant}"
    return
  fi

  log "Unknown variant: ${variant}"
  return 1
}

start_prefill() {
  local case_name="$1"
  local laps_threshold="$2"
  local laps_wait_window_ms="$3"
  local laps_wait_max_batch="$4"
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
      unset VLLM_ASCEND_LAPS_SCHEDULING VLLM_ASCEND_LAPS_THRESHOLD VLLM_ASCEND_LAPS_WAIT_WINDOW_MS VLLM_ASCEND_LAPS_WAIT_MAX_BATCH VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S
    else
      export VLLM_ASCEND_LAPS_SCHEDULING=1
      export VLLM_ASCEND_LAPS_THRESHOLD="${laps_threshold}"
      export VLLM_ASCEND_LAPS_WAIT_WINDOW_MS="${laps_wait_window_ms}"
      export VLLM_ASCEND_LAPS_WAIT_MAX_BATCH="${laps_wait_max_batch}"
      export VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S="${LAPS_STATS_LOG_INTERVAL_S}"
    fi

    cd "${VLLM_ASCEND_DIR}"
    exec setsid vllm serve "${MODEL_PATH}" \
      --host "${PREFILL_BIND_HOST}" \
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
      --no-enable-prefix-caching \
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

capture_latest_benchmark_json() {
  local output_dir="$1"
  local canonical_path="$2"
  local latest_file

  latest_file="$(find "${output_dir}" -maxdepth 1 -type f -name 'benchmark_*.json' | sort | tail -n 1)"
  if [ -z "${latest_file}" ]; then
    log "No benchmark JSON found in ${output_dir}"
    return 1
  fi

  cp "${latest_file}" "${canonical_path}"
}

run_claude_bench() {
  local case_name="$1"
  local label="$2"
  local timeout_s="$3"
  local output_dir="$4"
  local canonical_json="$5"
  local log_file="${RESULT_DIR}/logs/${case_name}_${label}.log"

  mkdir -p "${output_dir}"
  (
    source_env
    unset_proxy_env
    cd "${VLLM_ASCEND_DIR}"
    exec python3 "${BENCH_SCRIPT}" \
      -e "http://${PREFILL_CONNECT_HOST}:${PREFILL_PORT}" \
      --model "${SERVED_MODEL_NAME}" \
      --log-dir "${PREPARED_TRACE_DIR}" \
      --timeout "${timeout_s}" \
      --output-dir "${output_dir}" \
      --force-max-tokens "${REQUEST_MAX_TOKENS}" \
      --tokenizer "${MODEL_PATH}" \
      --max-model-len "${PREFILL_MAX_MODEL_LEN}" \
      --over-limit-policy skip \
      --summary-label "${label}"
  ) 2>&1 | tee "${log_file}"

  capture_latest_benchmark_json "${output_dir}" "${canonical_json}"
}

run_case() {
  local case_name="$1"
  local variant_name="$2"
  local laps_threshold="$3"
  local laps_wait_window_ms="$4"
  local laps_wait_max_batch="$5"
  local case_dir="${RESULT_DIR}/${case_name}"
  local warmup_output_dir="${case_dir}/warmup"
  local measured_output_dir="${case_dir}/measured"
  local warmup_json="${case_dir}/${case_name}_warmup.json"
  local measured_json="${case_dir}/${case_name}_measured.json"
  local prefill_before="${RESULT_DIR}/logs/${case_name}_prefill_metrics_before.prom"
  local prefill_after="${RESULT_DIR}/logs/${case_name}_prefill_metrics_after.prom"

  mkdir -p "${case_dir}"

  log "========== CASE ${case_name} started =========="
  log "Variant: ${variant_name}"
  log "Prepared trace dir: ${PREPARED_TRACE_DIR}"
  log "LAPS config: threshold=${laps_threshold}, wait_window_ms=${laps_wait_window_ms}, wait_max_batch=${laps_wait_max_batch}"

  stop_services
  start_prefill "${case_name}" "${laps_threshold}" "${laps_wait_window_ms}" "${laps_wait_max_batch}"
  wait_log "${RESULT_DIR}/logs/${case_name}_prefill.log" "Application startup complete" "Prefill startup"
  wait_http "http://${PREFILL_CONNECT_HOST}:${PREFILL_PORT}/health" "Prefill"

  if [ "${laps_threshold}" != "off" ]; then
    wait_log "${RESULT_DIR}/logs/${case_name}_prefill.log" "Ascend LAPS scheduler selected" "LAPS scheduler selection"
  fi

  if [ "${RUN_WARMUP}" = "1" ]; then
    log "Warmup start: ${PREPARED_TRACE_DIR}"
    run_claude_bench "${case_name}" "warmup" "${WARMUP_TIMEOUT}" "${warmup_output_dir}" "${warmup_json}"
    if [ "${POST_WARMUP_SLEEP_S}" -gt 0 ]; then
      log "Sleeping ${POST_WARMUP_SLEEP_S}s after warmup"
      sleep "${POST_WARMUP_SLEEP_S}"
    fi
  else
    log "Warmup disabled for ${case_name}"
  fi

  log "Measured run start: ${PREPARED_TRACE_DIR}"
  scrape_metrics_snapshot "http://${PREFILL_CONNECT_HOST}:${PREFILL_PORT}" "${prefill_before}"
  run_claude_bench "${case_name}" "measured" "${MEASURE_TIMEOUT}" "${measured_output_dir}" "${measured_json}"
  scrape_metrics_snapshot "http://${PREFILL_CONNECT_HOST}:${PREFILL_PORT}" "${prefill_after}"
  append_summary "${case_name}" "${variant_name}" "${laps_threshold}" "${measured_json}" "${prefill_before}" "${prefill_after}" "${PREPARED_TRACE_DIR}" "${PREPARED_MANIFEST}" "${PREPARED_SESSION_COUNT}" "${PREPARED_REQUEST_COUNT}"

  log "========== CASE ${case_name} completed =========="
}

run_case_with_recovery() {
  local case_name="$1"
  local variant_name="$2"
  local laps_threshold="$3"
  local laps_wait_window_ms="$4"
  local laps_wait_max_batch="$5"
  local attempt=1
  local rc=0

  while [ "${attempt}" -le "${CASE_RETRY_LIMIT}" ]; do
    if [ "${attempt}" -gt 1 ]; then
      log "Retrying case ${case_name}: attempt ${attempt}/${CASE_RETRY_LIMIT}"
    fi

    if run_case "${case_name}" "${variant_name}" "${laps_threshold}" "${laps_wait_window_ms}" "${laps_wait_max_batch}"; then
      append_case_status "${case_name}" "passed" "attempt=${attempt}"
      return 0
    fi

    rc=$?
    append_case_status "${case_name}" "failed_attempt" "exit_code=${rc}; attempt=${attempt}"
    stop_services || true
    attempt=$((attempt + 1))
  done

  append_case_status "${case_name}" "failed" "exit_code=${rc}; attempts=${CASE_RETRY_LIMIT}"
  return 0
}

main() {
  verify_prereqs
  source_env
  unset_proxy_env
  load_prepared_dataset_metadata

  printf 'timestamp\tcase\tstatus\tdetail\n' > "${RESULT_DIR}/case_status.tsv"
  printf 'timestamp\tcase\tvariant\tthreshold\tcompleted\tfailed\tskipped_over_limit\tduration_s\tttft_mean_ms\tttft_median_ms\tttft_p99_ms\ttpot_mean_ms\ttpot_median_ms\titl_mean_ms\titl_median_ms\te2el_mean_s\te2el_median_s\te2el_p99_s\ttotal_input_tokens\ttotal_output_tokens\trequest_throughput\ttotal_token_throughput\tprepared_dataset_dir\tdataset_manifest\tprepared_session_count\tprepared_request_count\tprefill_queue_avg_ms\tprefill_queue_median_ms\tprefill_queue_p90_ms\tprefill_queue_p99_ms\tprefill_time_avg_ms\tprefill_ttft_avg_ms\tprefill_ttft_median_ms\tprefill_ttft_p90_ms\tprefill_ttft_p99_ms\tmeasured_json\n' > "${RESULT_DIR}/summary.tsv"

  log "Results will be written to ${RESULT_DIR}"
  log "Prepared trace dir: ${PREPARED_TRACE_DIR}"
  log "Prepared manifest: ${PREPARED_MANIFEST}"
  log "Prepared session count: ${PREPARED_SESSION_COUNT}"
  log "Prepared request count: ${PREPARED_REQUEST_COUNT}"
  log "Case variants: ${CASE_VARIANTS}"
  log "Warmup enabled: ${RUN_WARMUP}"
  log "Warmup timeout: ${WARMUP_TIMEOUT}s"
  log "Measured timeout: ${MEASURE_TIMEOUT}s"
  log "LAPS defaults: wait_window_ms=${LAPS_WAIT_WINDOW_MS}, wait_max_batch=${LAPS_WAIT_MAX_BATCH}"

  local variant
  local case_spec
  local laps_threshold
  local laps_wait_window_ms
  local laps_wait_max_batch
  local case_suffix
  local repeat_idx

  for ((repeat_idx = 1; repeat_idx <= CASE_REPEATS; repeat_idx++)); do
    for variant in ${CASE_VARIANTS}; do
      case_spec="$(case_variant_to_params "${variant}")"
      IFS='|' read -r laps_threshold laps_wait_window_ms laps_wait_max_batch case_suffix <<< "${case_spec}"
      run_case_with_recovery \
        "claude_${case_suffix}_rep${repeat_idx}" \
        "${variant}" \
        "${laps_threshold}" \
        "${laps_wait_window_ms}" \
        "${laps_wait_max_batch}"
    done
  done

  log "All benchmark cases completed. Results: ${RESULT_DIR}"
}

main "$@"
