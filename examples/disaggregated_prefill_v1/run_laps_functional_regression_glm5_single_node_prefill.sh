#!/usr/bin/env bash
set -euo pipefail

# Run a small functional regression suite against the single-node Prefill-only
# GLM-5 deployment and compare outputs between LAPS disabled/enabled variants.
#
# The goal is not throughput measurement. This script is intended to compare
# plain text outputs between LAPS disabled/enabled variants using the
# OpenAI-compatible /v1/completions endpoint recommended by the local GLM-5
# deployment guide.

# ===========================================================================
# Configuration
# ===========================================================================

MODEL_PATH="${MODEL_PATH:-/workspace/models/GLM-5.1-w4a8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1}"

VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/vllm-workspace/vllm-ascend}"
RESULT_DIR="${RESULT_DIR:-/vllm-workspace/bench_results/glm5_functional_regression_$(date +%Y%m%d_%H%M%S)}"

PREFILL_BIND_HOST="${PREFILL_BIND_HOST:-0.0.0.0}"
PREFILL_CONNECT_HOST="${PREFILL_CONNECT_HOST:-127.0.0.1}"
PREFILL_NODE_IP="${PREFILL_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
PREFILL_NIC_NAME="${PREFILL_NIC_NAME:-enp48s3u1u1c2}"
PREFILL_DEVICES="${PREFILL_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
PREFILL_PORT="${PREFILL_PORT:-6700}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"

PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-16}"
PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-1}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-1}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-16}"
PREFILL_MAX_MODEL_LEN="${PREFILL_MAX_MODEL_LEN:-131072}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS:-4096}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-64}"
PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.95}"
PREFILL_ENABLE_CHUNKED_PREFILL="${PREFILL_ENABLE_CHUNKED_PREFILL:-1}"

REGRESSION_VARIANTS="${REGRESSION_VARIANTS:-off t256_w0}"
BASELINE_VARIANT="${BASELINE_VARIANT:-off}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-256}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
NUM_PARALLEL_REQUESTS="${NUM_PARALLEL_REQUESTS:-4}"

LAPS_LONG_MAX_WAIT_MS="${LAPS_LONG_MAX_WAIT_MS:-0}"
LAPS_LONG_PREFILL_CAP="${LAPS_LONG_PREFILL_CAP:-0}"
LAPS_SHORT_RESERVED_RATIO="${LAPS_SHORT_RESERVED_RATIO:-0}"
LAPS_STATS_LOG_INTERVAL_S="${LAPS_STATS_LOG_INTERVAL_S:-5}"

STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-1800}"
STOP_TIMEOUT_S="${STOP_TIMEOUT_S:-60}"
SLEEP_AFTER_STOP_S="${SLEEP_AFTER_STOP_S:-10}"
PORT_FREE_TIMEOUT_S="${PORT_FREE_TIMEOUT_S:-30}"

# ===========================================================================
# Runtime state
# ===========================================================================

PREFILL_PID=""

mkdir -p "${RESULT_DIR}/logs"
mkdir -p "${RESULT_DIR}/variant_results"

# ===========================================================================
# Logging & environment
# ===========================================================================

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

source_env() {
  set +u
  [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
  [ -f /usr/local/Ascend/cann-8.5.1/set_env.sh ] && source /usr/local/Ascend/cann-8.5.1/set_env.sh
  [ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
  [ -f /usr/local/Ascend/nnal/asdsip/set_env.sh ] && source /usr/local/Ascend/nnal/asdsip/set_env.sh
  set -euo pipefail
  export PYTHONPATH="${VLLM_ASCEND_DIR}:${PYTHONPATH:-}"
  export VLLM_USE_MODELSCOPE=False
  export HCCL_OP_EXPANSION_MODE=AIV
}

unset_proxy_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
}

# ===========================================================================
# Process helpers
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

# ===========================================================================
# Benchmark helpers
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

variant_to_config() {
  local variant="$1"
  # Field 2 carries long_max_wait_ms; field 3 (formerly wait_max_batch) is retired.
  local threshold long_max_wait_ms long_prefill_cap short_reserved_ratio

  if [ "${variant}" = "off" ]; then
    printf 'off|%s|%s|%s|%s\n' "" "" "" ""
    return
  fi

  threshold="${variant#t}"
  long_max_wait_ms="${LAPS_LONG_MAX_WAIT_MS:-0}"
  long_prefill_cap="${LAPS_LONG_PREFILL_CAP:-0}"
  short_reserved_ratio="${LAPS_SHORT_RESERVED_RATIO:-0}"

  IFS='_' read -ra parts <<< "${threshold}"
  threshold="${parts[0]}"
  for part in "${parts[@]:1}"; do
    case "${part}" in
      m*) long_max_wait_ms="${part#m}" ;;
      cap*) long_prefill_cap="${part#cap}" ;;
      res*) short_reserved_ratio="${part#res}" ;;
    esac
  done

  printf '%s|%s|%s|%s|%s\n' \
    "${threshold}" \
    "${long_max_wait_ms}" \
    "0" \
    "${long_prefill_cap}" \
    "${short_reserved_ratio}"
}

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
    export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-30000}"
    export ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-60000}"
    export HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-17}"
    export HCCL_RDMA_RETRY_CNT="${HCCL_RDMA_RETRY_CNT:-7}"
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

    # LAPS is configured via additional_config.laps_config (see below).
    # wait_window_ms carries the long_max_wait_ms aging bound (tuple field 2).
    # NOTE: long_prefill_cap / short_reserved_ratio are from a removed LAPS
    # design and are no longer applied; they are ignored here.
    if [ "${laps_threshold}" = "off" ]; then
      laps_config_json='"laps_config":{"enabled":false}'
    else
      laps_config_json="\"laps_config\":{\"enabled\":true,\"threshold\":${laps_threshold},\"long_max_wait_ms\":${wait_window_ms},\"stats_log_interval_s\":${LAPS_STATS_LOG_INTERVAL_S}}"
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
      --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "recompute_scheduler_enable": true, "ascend_compilation_config": {"enable_npugraph_ex": true},'"${laps_config_json}"'}' \
      --kv-transfer-config "${kv_config}"
  ) >"${log_file}" 2>&1 &
  PREFILL_PID=$!
  log "Started Prefill pid=${PREFILL_PID}, log=${log_file}"
}

run_variant_regression() {
  local variant_name="$1"
  local config threshold wait_window_ms wait_max_batch long_prefill_cap short_reserved_ratio
  local case_key="functional_${variant_name}"
  local runner_log="${RESULT_DIR}/logs/${case_key}_runner.log"

  config="$(variant_to_config "${variant_name}")"
  IFS='|' read -r threshold wait_window_ms wait_max_batch long_prefill_cap short_reserved_ratio <<< "${config}"

  log "========== VARIANT ${variant_name} started =========="
  log "Variant config: threshold=${threshold}, long_max_wait_ms=${wait_window_ms:-0}, long_prefill_cap=${long_prefill_cap:-0}, short_reserved_ratio=${short_reserved_ratio:-0}"

  stop_services
  start_prefill "${case_key}" "${threshold}" "${wait_window_ms}" "${wait_max_batch}" "${long_prefill_cap}" "${short_reserved_ratio}"

  wait_http "http://${PREFILL_CONNECT_HOST}:${PREFILL_PORT}/health" "Prefill" || {
    tail -80 "${RESULT_DIR}/logs/${case_key}_prefill.log" || true
    return 1
  }

  if [ "${threshold}" != "off" ]; then
    wait_log "${RESULT_DIR}/logs/${case_key}_prefill.log" "Ascend LAPS scheduler selected" "LAPS scheduler selection"
  fi

  python3 - \
    "${variant_name}" \
    "${RESULT_DIR}" \
    "${PROMPTS_FILE}" \
    "${SERVED_MODEL_NAME}" \
    "${PREFILL_CONNECT_HOST}" \
    "${PREFILL_PORT}" \
    "${PROMPT_MAX_TOKENS}" \
    "${REQUEST_TIMEOUT_SEC}" \
    "${NUM_PARALLEL_REQUESTS}" <<'PY' 2>&1 | tee "${runner_log}"
import collections
import concurrent.futures
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

variant_name = sys.argv[1]
result_dir = Path(sys.argv[2])
prompts_file = sys.argv[3]
served_model_name = sys.argv[4]
host = sys.argv[5]
port = int(sys.argv[6])
max_tokens = int(sys.argv[7])
timeout_sec = int(sys.argv[8])
num_parallel = int(sys.argv[9])

completions_url = f"http://{host}:{port}/v1/completions"
variant_file = result_dir / "variant_results" / f"{variant_name}.json"

DEFAULT_CASES = [
    {
        "id": "exact_echo_ascii_cn",
        "category": "exact_match",
        "prompt": (
            "逐字输出下面这行文本，不要添加任何字符，不要换行："
            "春晓Alpha-123/测试"
        ),
        "validator": "exact_match",
        "expected_text": "春晓Alpha-123/测试",
    },
    {
        "id": "exact_echo_special_chars",
        "category": "exact_match",
        "prompt": (
            "逐字输出下面这行文本，不要添加任何字符，不要换行："
            "Token_[]{}()<>+-=*/:_#%"
        ),
        "validator": "exact_match",
        "expected_text": "Token_[]{}()<>+-=*/:_#%",
    },
    {
        "id": "json_object",
        "category": "json",
        "prompt": (
            "只输出一个紧凑 JSON 对象，不要使用 Markdown，不要额外说明。"
            "字段必须包含：name(string，固定为\"Alice\")、years(number，固定为5)、"
            "skills(array，固定2项，依次为\"Python\"和\"SQL\")。"
        ),
        "validator": "json_object",
        "expected_json_keys": ["name", "years", "skills"],
        "expected_json_types": {
            "name": "string",
            "years": "number",
            "skills": "array",
        },
        "expected_json_values": {
            "name": "Alice",
            "years": 5,
            "skills": ["Python", "SQL"],
        },
        "expected_json_array_lengths": {"skills": 2},
    },
    {
        "id": "markdown_table_fixed",
        "category": "markdown_table",
        "prompt": (
            "只输出一个 Markdown 表格，不要输出表格以外的任何文字。"
            "表头固定为 Item | Value。"
            "数据固定为 3 行：GPU Utilization=80%，Memory Usage=70%，Batch Size=16。"
        ),
        "validator": "markdown_table",
        "expected_header": ["Item", "Value"],
        "expected_rows": [
            ["GPU Utilization", "80%"],
            ["Memory Usage", "70%"],
            ["Batch Size", "16"],
        ],
    },
    {
        "id": "python_code_block_fib",
        "category": "code",
        "prompt": (
            "只输出一个 ```python 代码块。代码块中定义 def fib(n):，使用迭代法，"
            "要求处理 n <= 0 返回 0，并且包含 for 循环。不要附加解释。"
        ),
        "validator": "python_code_block",
        "required_substrings": ["def fib(n):", "return 0"],
        "required_patterns": [r"\bfor\b"],
        "require_closed_code_fence": True,
    },
    {
        "id": "cn_bullets_fixed",
        "category": "chinese_bullets",
        "prompt": (
            "只输出 4 行中文 bullet，每行以 '- ' 开头，解释什么是批处理。"
            "每行不超过 10 个汉字，不要编号，不要额外文字。"
        ),
        "validator": "bullet_list",
        "expected_items": 4,
        "min_items": 4,
        "line_prefix": "- ",
        "max_chars_per_item": 10,
    },
    {
        "id": "mixed_language_fixed",
        "category": "mixed_language",
        "prompt": (
            "只输出 3 行介绍 vLLM。每行格式必须为：中文句子 (English sentence)。"
            "不要编号，不要额外文字。"
        ),
        "validator": "mixed_language_lines",
        "expected_items": 3,
        "min_items": 3,
    },
]

REQUIRED_CASE_FIELDS = ("id", "category", "prompt", "validator")
JSON_TYPE_CHECKERS = {
    "string": lambda value: isinstance(value, str),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def load_cases() -> list[dict]:
    cases = DEFAULT_CASES
    if prompts_file:
        path = Path(prompts_file)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit("PROMPTS_FILE must contain a JSON array")
        cases = data

    if not isinstance(cases, list):
        raise SystemExit("Case configuration must be a JSON array")

    seen_ids: set[str] = set()
    normalized_cases: list[dict] = []
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, dict):
            raise SystemExit(f"Case at index {index} must be a JSON object")
        missing_fields = [field for field in REQUIRED_CASE_FIELDS if field not in raw_case]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise SystemExit(
                f"Case {raw_case.get('id', index)!r} is missing required fields: {missing}. "
                "PROMPTS_FILE entries must provide id/category/prompt/validator."
            )
        case_id = str(raw_case["id"])
        if case_id in seen_ids:
            raise SystemExit(f"Duplicate case id detected: {case_id}")
        seen_ids.add(case_id)
        normalized_cases.append(raw_case)
    return normalized_cases


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n")


def normalize_display_text(text: str) -> str:
    return normalize_text(text).strip()


def shorten(text: str, width: int = 120) -> str:
    clean = " ".join(text.split())
    if len(clean) <= width:
        return clean
    return clean[: width - 3] + "..."


def stringify_optional(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def make_validation(
    *,
    validator: str,
    passed: bool,
    reason: str,
    details=None,
    failure_is_final: bool = True,
) -> dict:
    return {
        "validator": validator,
        "passed": passed,
        "reason": reason,
        "details": details,
        "failure_is_final": failure_is_final,
    }


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def contains_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def validate_exact_match(case: dict, raw_text: str) -> dict:
    expected = case["expected_text"]
    if raw_text == expected:
        return make_validation(
            validator="exact_match",
            passed=True,
            reason="exact text matched",
            details={"expected_text": expected},
        )
    if expected.startswith(raw_text):
        return make_validation(
            validator="exact_match",
            passed=False,
            reason="output is a strict prefix of expected text",
            details={"expected_text": expected},
            failure_is_final=False,
        )
    return make_validation(
        validator="exact_match",
        passed=False,
        reason="output does not exactly match expected text",
        details={"expected_text": expected},
    )


def validate_json_object(case: dict, display_text: str) -> dict:
    try:
        parsed = json.loads(display_text)
    except json.JSONDecodeError as exc:
        return make_validation(
            validator="json_object",
            passed=False,
            reason="response is not valid JSON",
            details={"error": str(exc)},
            failure_is_final=False,
        )

    if not isinstance(parsed, dict):
        return make_validation(
            validator="json_object",
            passed=False,
            reason="response JSON is not an object",
            details={"parsed_type": type(parsed).__name__},
        )

    expected_keys = case.get("expected_json_keys", [])
    missing_keys = [key for key in expected_keys if key not in parsed]
    if missing_keys:
        return make_validation(
            validator="json_object",
            passed=False,
            reason="response JSON is missing required keys",
            details={"missing_keys": missing_keys},
        )

    for key, expected_type in case.get("expected_json_types", {}).items():
        checker = JSON_TYPE_CHECKERS.get(expected_type)
        if checker is None:
            return make_validation(
                validator="json_object",
                passed=False,
                reason="unsupported expected_json_types entry",
                details={"key": key, "expected_type": expected_type},
            )
        if key not in parsed or not checker(parsed[key]):
            return make_validation(
                validator="json_object",
                passed=False,
                reason="response JSON field has unexpected type",
                details={
                    "key": key,
                    "expected_type": expected_type,
                    "actual_value": parsed.get(key),
                },
            )

    for key, expected_value in case.get("expected_json_values", {}).items():
        if parsed.get(key) != expected_value:
            return make_validation(
                validator="json_object",
                passed=False,
                reason="response JSON field has unexpected value",
                details={
                    "key": key,
                    "expected_value": expected_value,
                    "actual_value": parsed.get(key),
                },
            )

    for key, expected_length in case.get("expected_json_array_lengths", {}).items():
        value = parsed.get(key)
        if not isinstance(value, list) or len(value) != expected_length:
            return make_validation(
                validator="json_object",
                passed=False,
                reason="response JSON array has unexpected length",
                details={
                    "key": key,
                    "expected_length": expected_length,
                    "actual_length": len(value) if isinstance(value, list) else None,
                },
            )

    return make_validation(
        validator="json_object",
        passed=True,
        reason="response JSON passed schema validation",
    )


def parse_markdown_table(display_text: str) -> list[list[str]] | None:
    lines = [line.strip() for line in display_text.splitlines() if line.strip()]
    parsed_rows = []
    for line in lines:
        if "|" not in line:
            return None
        if not line.startswith("|") or not line.endswith("|"):
            return None
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        parsed_rows.append(cells)
    return parsed_rows


def validate_markdown_table(case: dict, display_text: str) -> dict:
    parsed_rows = parse_markdown_table(display_text)
    expected_header = case["expected_header"]
    expected_rows = case["expected_rows"]

    if parsed_rows is None:
        return make_validation(
            validator="markdown_table",
            passed=False,
            reason="response is not a Markdown table",
            failure_is_final=False,
        )

    expected_total_rows = 2 + len(expected_rows)
    if len(parsed_rows) < expected_total_rows:
        return make_validation(
            validator="markdown_table",
            passed=False,
            reason="response table is incomplete",
            details={"expected_rows": expected_total_rows, "actual_rows": len(parsed_rows)},
            failure_is_final=False,
        )

    if len(parsed_rows) > expected_total_rows:
        return make_validation(
            validator="markdown_table",
            passed=False,
            reason="response contains extra table rows",
            details={"expected_rows": expected_total_rows, "actual_rows": len(parsed_rows)},
        )

    header_row = parsed_rows[0]
    if header_row != expected_header:
        return make_validation(
            validator="markdown_table",
            passed=False,
            reason="response table header does not match expectation",
            details={"expected_header": expected_header, "actual_header": header_row},
        )

    separator_row = parsed_rows[1]
    if len(separator_row) != len(expected_header) or not all(set(cell) <= {"-"} and cell for cell in separator_row):
        return make_validation(
            validator="markdown_table",
            passed=False,
            reason="response table separator row is invalid",
            details={"separator_row": separator_row},
        )

    actual_rows = parsed_rows[2:]
    if actual_rows != expected_rows:
        return make_validation(
            validator="markdown_table",
            passed=False,
            reason="response table rows do not match expectation",
            details={"expected_rows": expected_rows, "actual_rows": actual_rows},
        )

    return make_validation(
        validator="markdown_table",
        passed=True,
        reason="response table passed validation",
    )


def validate_python_code_block(case: dict, display_text: str) -> dict:
    required_substrings = case.get("required_substrings", [])
    required_patterns = case.get("required_patterns", [])
    require_closed_code_fence = bool(case.get("require_closed_code_fence", True))

    if not display_text.startswith("```python"):
        return make_validation(
            validator="python_code_block",
            passed=False,
            reason="response does not start with a python fenced code block",
        )

    if require_closed_code_fence and not display_text.endswith("```"):
        return make_validation(
            validator="python_code_block",
            passed=False,
            reason="response code block is not closed",
            failure_is_final=False,
        )

    for substring in required_substrings:
        if substring not in display_text:
            return make_validation(
                validator="python_code_block",
                passed=False,
                reason="response code block is missing a required substring",
                details={"missing_substring": substring},
                failure_is_final=False,
            )

    for pattern in required_patterns:
        if not re.search(pattern, display_text):
            return make_validation(
                validator="python_code_block",
                passed=False,
                reason="response code block is missing a required pattern",
                details={"missing_pattern": pattern},
                failure_is_final=False,
            )

    return make_validation(
        validator="python_code_block",
        passed=True,
        reason="response code block passed validation",
    )


def validate_bullet_list(case: dict, display_text: str) -> dict:
    lines = [line.strip() for line in display_text.splitlines() if line.strip()]
    expected_items = int(case.get("expected_items", case.get("min_items", 0)))
    min_items = int(case.get("min_items", expected_items))
    line_prefix = case.get("line_prefix", "- ")
    max_chars_per_item = int(case.get("max_chars_per_item", 0))

    if len(lines) < min_items:
        return make_validation(
            validator="bullet_list",
            passed=False,
            reason="response does not contain enough bullet lines",
            details={"expected_items": expected_items, "actual_items": len(lines)},
            failure_is_final=False,
        )

    if expected_items and len(lines) != expected_items:
        return make_validation(
            validator="bullet_list",
            passed=False,
            reason="response contains unexpected number of bullet lines",
            details={"expected_items": expected_items, "actual_items": len(lines)},
        )

    for line in lines:
        if not line.startswith(line_prefix):
            return make_validation(
                validator="bullet_list",
                passed=False,
                reason="response line does not use the required bullet prefix",
                details={"line": line, "expected_prefix": line_prefix},
            )
        item_text = line[len(line_prefix):].strip()
        if not item_text:
            return make_validation(
                validator="bullet_list",
                passed=False,
                reason="response bullet item is empty",
                details={"line": line},
                failure_is_final=False,
            )
        if max_chars_per_item and len(item_text) > max_chars_per_item:
            return make_validation(
                validator="bullet_list",
                passed=False,
                reason="response bullet item exceeds max length",
                details={"line": line, "max_chars_per_item": max_chars_per_item},
            )

    return make_validation(
        validator="bullet_list",
        passed=True,
        reason="response bullet list passed validation",
    )


def validate_mixed_language_lines(case: dict, display_text: str) -> dict:
    lines = [line.strip() for line in display_text.splitlines() if line.strip()]
    expected_items = int(case.get("expected_items", case.get("min_items", 0)))
    min_items = int(case.get("min_items", expected_items))

    if len(lines) < min_items:
        return make_validation(
            validator="mixed_language_lines",
            passed=False,
            reason="response does not contain enough lines",
            details={"expected_items": expected_items, "actual_items": len(lines)},
            failure_is_final=False,
        )

    if expected_items and len(lines) != expected_items:
        return make_validation(
            validator="mixed_language_lines",
            passed=False,
            reason="response contains unexpected number of lines",
            details={"expected_items": expected_items, "actual_items": len(lines)},
        )

    for line in lines:
        if "(" not in line or ")" not in line:
            return make_validation(
                validator="mixed_language_lines",
                passed=False,
                reason="response line does not contain English text in parentheses",
                details={"line": line},
                failure_is_final=False,
            )
        chinese_part, english_part = line.split("(", 1)
        english_part = english_part.rsplit(")", 1)[0]
        if not contains_cjk(chinese_part):
            return make_validation(
                validator="mixed_language_lines",
                passed=False,
                reason="response line is missing Chinese content",
                details={"line": line},
            )
        if not contains_english(english_part):
            return make_validation(
                validator="mixed_language_lines",
                passed=False,
                reason="response line is missing English content inside parentheses",
                details={"line": line},
                failure_is_final=False,
            )

    return make_validation(
        validator="mixed_language_lines",
        passed=True,
        reason="response mixed-language lines passed validation",
    )


def validate_case_output(case: dict, raw_text: str, display_text: str) -> dict:
    validator = case["validator"]
    if validator == "exact_match":
        return validate_exact_match(case, raw_text)
    if validator == "json_object":
        return validate_json_object(case, display_text)
    if validator == "markdown_table":
        return validate_markdown_table(case, display_text)
    if validator == "python_code_block":
        return validate_python_code_block(case, display_text)
    if validator == "bullet_list":
        return validate_bullet_list(case, display_text)
    if validator == "mixed_language_lines":
        return validate_mixed_language_lines(case, display_text)
    raise SystemExit(f"Unsupported validator: {validator}")


def run_case(case: dict) -> dict:
    payload = {
        "model": served_model_name,
        "prompt": case["prompt"],
        "seed": 0,
        "temperature": 0.0,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        completions_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            response_text = response.read().decode("utf-8")
        latency_ms = (time.perf_counter() - start) * 1000.0
        response_json = json.loads(response_text)
        choice = response_json["choices"][0]
        text = stringify_optional(choice.get("text"))
        display_text = normalize_display_text(text)
        request_status = "ok" if display_text else "empty_response"
        finish_reason = choice.get("finish_reason")
        truncated = finish_reason == "length"
        if request_status != "ok":
            validation = make_validation(
                validator=case["validator"],
                passed=False,
                reason="empty response",
            )
            validation_status = "fail"
        else:
            validation = validate_case_output(case, normalize_text(text), display_text)
            if validation["passed"]:
                validation_status = "pass"
            elif truncated and not validation["failure_is_final"]:
                validation_status = "inconclusive"
            else:
                validation_status = "fail"
        return {
            "id": case["id"],
            "category": case.get("category", ""),
            "prompt": case["prompt"],
            "validator": case["validator"],
            "latency_ms": round(latency_ms, 3),
            "text": text,
            "display_text": display_text,
            "output_preview": shorten(display_text),
            "request_status": request_status,
            "validation_status": validation_status,
            "validation": validation,
            "finish_reason": finish_reason,
            "stop_reason": choice.get("stop_reason"),
            "truncated": truncated,
            "same_as_baseline": None,
            "raw_choice": choice,
        }
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return {
            "id": case["id"],
            "category": case.get("category", ""),
            "prompt": case["prompt"],
            "validator": case["validator"],
            "latency_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "text": "",
            "display_text": "",
            "output_preview": "",
            "request_status": "http_error",
            "validation_status": "fail",
            "validation": make_validation(
                validator=case["validator"],
                passed=False,
                reason="HTTP request failed",
                details={"http_status": exc.code},
            ),
            "finish_reason": None,
            "stop_reason": None,
            "truncated": False,
            "same_as_baseline": None,
            "error": {"http_status": exc.code, "error_body": error_body},
        }
    except Exception as exc:
        return {
            "id": case["id"],
            "category": case.get("category", ""),
            "prompt": case["prompt"],
            "validator": case["validator"],
            "latency_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "text": "",
            "display_text": "",
            "output_preview": "",
            "request_status": "request_failed",
            "validation_status": "fail",
            "validation": make_validation(
                validator=case["validator"],
                passed=False,
                reason="request execution failed",
                details={"error": repr(exc)},
            ),
            "finish_reason": None,
            "stop_reason": None,
            "truncated": False,
            "same_as_baseline": None,
            "error": {"error": repr(exc)},
        }


cases = load_cases()
results: list[dict] = []

with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_parallel)) as executor:
    futures = [executor.submit(run_case, case) for case in cases]
    for future in futures:
        results.append(future.result())

result_by_id = {item["id"]: item for item in results}
ordered_results = [result_by_id[case["id"]] for case in cases]
finish_reason_counts = collections.Counter(
    item["finish_reason"] or "none" for item in ordered_results
)
summary = {
    "variant": variant_name,
    "completions_url": completions_url,
    "model": served_model_name,
    "case_count": len(ordered_results),
    "request_ok_count": sum(1 for item in ordered_results if item["request_status"] == "ok"),
    "request_error_count": sum(1 for item in ordered_results if item["request_status"] != "ok"),
    "empty_response_count": sum(1 for item in ordered_results if item["request_status"] == "empty_response"),
    "http_error_count": sum(1 for item in ordered_results if item["request_status"] == "http_error"),
    "request_failed_count": sum(1 for item in ordered_results if item["request_status"] == "request_failed"),
    "validation_pass_count": sum(1 for item in ordered_results if item["validation_status"] == "pass"),
    "validation_fail_count": sum(1 for item in ordered_results if item["validation_status"] == "fail"),
    "inconclusive_count": sum(1 for item in ordered_results if item["validation_status"] == "inconclusive"),
    "same_as_baseline_count": None,
    "truncated_count": sum(1 for item in ordered_results if item["truncated"]),
    "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
}

payload = {
    "variant": variant_name,
    "generated_at": int(time.time()),
    "summary": summary,
    "cases": ordered_results,
}
variant_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

print(
    f"variant={variant_name} cases={summary['case_count']} "
    f"request_ok={summary['request_ok_count']} "
    f"pass={summary['validation_pass_count']} "
    f"fail={summary['validation_fail_count']} "
    f"inconclusive={summary['inconclusive_count']} "
    f"output={variant_file}"
)
for item in ordered_results:
    print(
        f"[{item['validation_status'].upper()}] {item['id']} "
        f"request_status={item['request_status']} "
        f"finish_reason={item['finish_reason']} "
        f"latency_ms={item['latency_ms']:.3f} "
        f"preview={item['output_preview']}"
    )
PY

  log "========== VARIANT ${variant_name} completed =========="
}

build_comparison_report() {
  python3 - \
    "${RESULT_DIR}" \
    "${BASELINE_VARIANT}" \
    "${REGRESSION_VARIANTS}" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
baseline_variant = sys.argv[2]
variants = sys.argv[3].split()

variant_payloads = {}
variant_paths = {}
for variant in variants:
    path = result_dir / "variant_results" / f"{variant}.json"
    if not path.exists():
        raise SystemExit(f"Missing variant result file: {path}")
    variant_payloads[variant] = json.loads(path.read_text(encoding="utf-8"))
    variant_paths[variant] = path

baseline_cases = {
    item["id"]: item for item in variant_payloads[baseline_variant]["cases"]
}


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n")


def case_conclusion(case: dict) -> str:
    if case["validation_status"] == "pass":
        return "功能通过且与 baseline 一致" if case["same_as_baseline"] else "功能通过但与 baseline 不一致"
    if case["validation_status"] == "inconclusive":
        return "结论性不足"
    return "功能失败"


def variant_conclusion(summary: dict, case_count: int, same_as_baseline_count: int | None, is_baseline: bool) -> str:
    if summary["request_error_count"] > 0 or summary["validation_fail_count"] > 0:
        return "功能失败"
    if summary["inconclusive_count"] > 0:
        return "结论性不足"
    if is_baseline:
        return "baseline 功能通过"
    if same_as_baseline_count == case_count:
        return "功能通过且与 baseline 一致"
    return "功能通过但与 baseline 不一致"


summary_rows = []
prompt_sections = []
overall_pass = True

for variant in variants:
    payload = variant_payloads[variant]
    same_as_baseline_count = 0
    for item in payload["cases"]:
        baseline_item = baseline_cases.get(item["id"])
        same_as_baseline = (
            True
            if variant == baseline_variant
            else baseline_item is not None and normalize(item.get("text", "")) == normalize(baseline_item.get("text", ""))
        )
        item["same_as_baseline"] = same_as_baseline
        item["conclusion"] = case_conclusion(item)
        if same_as_baseline:
            same_as_baseline_count += 1

    payload["summary"]["same_as_baseline_count"] = same_as_baseline_count
    payload["summary"]["conclusion"] = variant_conclusion(
        payload["summary"],
        len(payload["cases"]),
        same_as_baseline_count,
        variant == baseline_variant,
    )
    variant_paths[variant].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_rows.append(
        {
            "variant": variant,
            "case_count": len(payload["cases"]),
            "request_ok_count": payload["summary"]["request_ok_count"],
            "request_error_count": payload["summary"]["request_error_count"],
            "validation_pass_count": payload["summary"]["validation_pass_count"],
            "validation_fail_count": payload["summary"]["validation_fail_count"],
            "inconclusive_count": payload["summary"]["inconclusive_count"],
            "same_as_baseline_count": same_as_baseline_count,
            "truncated_count": payload["summary"]["truncated_count"],
            "conclusion": payload["summary"]["conclusion"],
        }
    )
    if (
        payload["summary"]["request_error_count"] > 0
        or payload["summary"]["validation_fail_count"] > 0
        or payload["summary"]["inconclusive_count"] > 0
    ):
        overall_pass = False

for prompt_id, baseline_item in baseline_cases.items():
    section = {
        "prompt_id": prompt_id,
        "category": baseline_item["category"],
        "prompt": baseline_item["prompt"],
        "validator": baseline_item["validator"],
        "variants": [],
    }
    for variant in variants:
        item = next(
            (case for case in variant_payloads[variant]["cases"] if case["id"] == prompt_id),
            None,
        )
        if item is None:
            continue
        section["variants"].append(
            {
                "variant": variant,
                "request_status": item["request_status"],
                "validation_status": item["validation_status"],
                "latency_ms": item["latency_ms"],
                "text": item.get("text", ""),
                "display_text": item.get("display_text", ""),
                "finish_reason": item.get("finish_reason"),
                "same_as_baseline": item.get("same_as_baseline"),
                "truncated": item.get("truncated", False),
                "validation": item.get("validation", {}),
                "conclusion": item.get("conclusion"),
            }
        )
    prompt_sections.append(section)

summary_json = {
    "baseline_variant": baseline_variant,
    "variants": variants,
    "summary_rows": summary_rows,
    "prompt_sections": prompt_sections,
    "overall_pass": overall_pass,
}
(result_dir / "functional_regression_summary.json").write_text(
    json.dumps(summary_json, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

lines = [
    "# Functional Regression Summary",
    "",
    f"- Baseline variant: `{baseline_variant}`",
    f"- Variants: `{' '.join(variants)}`",
    f"- Overall result: `{'PASS' if overall_pass else 'FAIL'}`",
    "",
    "## Variant Summary",
    "",
    "| Variant | Cases | Request OK | Request Error | Pass | Fail | Inconclusive | Same as Baseline | Truncated | Conclusion |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for row in summary_rows:
    lines.append(
        f"| {row['variant']} | {row['case_count']} | {row['request_ok_count']} | "
        f"{row['request_error_count']} | {row['validation_pass_count']} | "
        f"{row['validation_fail_count']} | {row['inconclusive_count']} | "
        f"{row['same_as_baseline_count']} | {row['truncated_count']} | "
        f"{row['conclusion']} |"
    )

lines.extend(["", "## Prompt Outputs", ""])
for section in prompt_sections:
    lines.append(f"### {section['prompt_id']}")
    lines.append("")
    lines.append(f"- Category: `{section['category']}`")
    lines.append(f"- Validator: `{section['validator']}`")
    if section["prompt"]:
        prompt_text = section["prompt"].replace("`", "'")
        lines.append(f"- Prompt: `{prompt_text}`")
    lines.append("")
    for item in section["variants"]:
        lines.append(
            f"#### {item['variant']} | request_status={item['request_status']} | "
            f"validation_status={item['validation_status']} | "
            f"latency_ms={item['latency_ms']:.3f} | "
            f"finish_reason={item['finish_reason']} | "
            f"same_as_baseline={'yes' if item['same_as_baseline'] else 'no'} | "
            f"truncated={'yes' if item['truncated'] else 'no'}"
        )
        lines.append("")
        lines.append(f"- Conclusion: {item['conclusion']}")
        lines.append(f"- Validation reason: {item['validation'].get('reason', '')}")
        if item["validation"].get("details") is not None:
            details = json.dumps(item["validation"]["details"], ensure_ascii=False)
            details_text = details.replace("`", "'")
            lines.append(f"- Validation details: `{details_text}`")
        lines.append("")
        lines.append("```text")
        lines.append(item["text"] if item["text"] else item["display_text"])
        lines.append("```")
        lines.append("")

(result_dir / "functional_regression_summary.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(f"comparison_summary={result_dir / 'functional_regression_summary.md'}")
raise SystemExit(0 if overall_pass else 1)
PY
}

main() {
  [ -z "${PREFILL_NODE_IP}" ] && PREFILL_NODE_IP="127.0.0.1"
  source_env
  unset_proxy_env

  log "Results will be written to ${RESULT_DIR}"
  log "Functional regression variants: ${REGRESSION_VARIANTS}"
  log "Baseline variant: ${BASELINE_VARIANT}"
  log "Prompt max tokens: ${PROMPT_MAX_TOKENS}"
  log "Parallel requests: ${NUM_PARALLEL_REQUESTS}"
  if [ -n "${PROMPTS_FILE}" ]; then
    log "Using prompts file: ${PROMPTS_FILE}"
  else
    log "Using built-in prompt set"
  fi

  log "Running one-time local fallback cleanup before the first cold start"
  stop_existing_services || true
  sleep "${SLEEP_AFTER_STOP_S}"
  ensure_ports_free

  for variant_name in ${REGRESSION_VARIANTS}; do
    run_variant_regression "${variant_name}"
  done

  if build_comparison_report; then
    log "Functional regression completed successfully"
  else
    log "Functional regression completed with failures"
    return 1
  fi
}

main "$@"