#!/usr/bin/env python3
"""Build a simplified prepared Claude trace dataset for replay benchmarks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from transformers import AutoTokenizer


DEFAULT_INPUT_DIR = "/vllm-workspace/vllm_bench_claude/logs"
REQUESTS_PER_SESSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prepared Claude dataset from litellm_stats session traces.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Input raw trace directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for prepared dataset files and manifest.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer/model path for prompt length precheck.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length for over-limit filtering.",
    )
    parser.add_argument(
        "--force-max-tokens",
        type=int,
        default=1,
        help="Forced max_tokens used during replay and precheck.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


class PreparedClaudeDatasetBuilder:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        tokenizer_path: str,
        max_model_len: Optional[int],
        force_max_tokens: int,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.tokenizer_path = tokenizer_path
        self.max_model_len = max_model_len
        self.force_max_tokens = force_max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

    @staticmethod
    def _extract_request_body(req: dict[str, Any]) -> dict[str, Any]:
        body = req.get("request", req)
        return body if isinstance(body, dict) else {}

    def _build_request_payload(self, request_data: dict[str, Any]) -> dict[str, Any]:
        request_payload: dict[str, Any] = {
            "messages": request_data.get("messages", []),
            "max_tokens": self.force_max_tokens,
        }

        for key in [
            "tools",
            "tool_choice",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "response_format",
            "parallel_tool_calls",
            "extra_body",
        ]:
            if key in request_data:
                if key == "extra_body" and isinstance(request_data[key], dict):
                    request_payload.update(request_data[key])
                else:
                    request_payload[key] = request_data[key]

        return request_payload

    def _estimate_prompt_tokens(
        self,
        request_data: dict[str, Any],
        request_payload: dict[str, Any],
    ) -> tuple[Optional[int], bool, Optional[str]]:
        messages = request_payload.get("messages") or request_data.get("messages") or []
        if not messages:
            return None, False, "missing_messages"

        tokenize_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        tool_aware = False
        estimate_reason = None

        try:
            if request_payload.get("tools") is not None:
                tokenize_kwargs["tools"] = request_payload["tools"]
                tool_aware = True
            token_ids = self.tokenizer.apply_chat_template(messages, **tokenize_kwargs)
            return len(token_ids), tool_aware, None
        except Exception as exc:
            return None, tool_aware, f"tokenize_failed:{type(exc).__name__}"

    @staticmethod
    def _safe_session_name(index: int, source_file: Path) -> str:
        rel_stem = "__".join(source_file.with_suffix("").parts)
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", rel_stem).strip("._") or "session"
        return f"{index:03d}__{safe_stem}.json"

    def build(self) -> dict[str, Any]:
        files_to_process = sorted(
            self.input_dir.rglob("*.json"),
            key=lambda path: path.relative_to(self.input_dir).as_posix(),
        )
        skipped_files: list[dict[str, str]] = []
        sessions: list[dict[str, Any]] = []
        total_retained_requests = 0
        total_filtered_over_limit = 0

        for session_index, json_file in enumerate(files_to_process, start=1):
            rel_path = json_file.relative_to(self.input_dir)
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                skipped_files.append({"file": rel_path.as_posix(), "reason": "invalid_json"})
                continue

            reqs = data.get("reqs", [])
            if not isinstance(reqs, list) or not reqs:
                skipped_files.append({"file": rel_path.as_posix(), "reason": "empty_or_missing_reqs"})
                continue

            reqs = sorted(reqs, key=lambda item: item.get("timestamp", ""))
            candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
            for idx, req in enumerate(reqs):
                request_data = self._extract_request_body(req)
                messages = request_data.get("messages", [])
                if not messages:
                    continue
                candidates.append((idx, req, request_data))
                if len(candidates) == REQUESTS_PER_SESSION:
                    break

            if not candidates:
                skipped_files.append({"file": rel_path.as_posix(), "reason": "no_valid_messages"})
                continue

            retained_reqs: list[dict[str, Any]] = []
            candidate_request_indices: list[int] = []
            retained_request_indices: list[int] = []
            filtered_over_limit_request_indices: list[int] = []

            for idx, req, request_data in candidates:
                request_index = idx + 1
                candidate_request_indices.append(request_index)
                request_payload = self._build_request_payload(request_data)
                estimated_prompt_tokens = None
                tool_aware_estimate = False
                estimate_reason = None
                over_limit = False

                if self.max_model_len is not None:
                    (
                        estimated_prompt_tokens,
                        tool_aware_estimate,
                        estimate_reason,
                    ) = self._estimate_prompt_tokens(request_data, request_payload)
                    if (
                        estimated_prompt_tokens is not None
                        and estimated_prompt_tokens + self.force_max_tokens > self.max_model_len
                    ):
                        over_limit = True

                if over_limit:
                    filtered_over_limit_request_indices.append(request_index)
                    total_filtered_over_limit += 1
                    continue

                retained_reqs.append(req)
                retained_request_indices.append(request_index)

            if not retained_reqs:
                skipped_files.append(
                    {"file": rel_path.as_posix(), "reason": "all_candidates_filtered"}
                )
                continue

            output_name = self._safe_session_name(len(sessions) + 1, rel_path)
            output_data = dict(data)
            output_data["reqs"] = retained_reqs
            output_path = self.output_dir / output_name
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
                f.write("\n")

            retained_count = len(retained_reqs)
            total_retained_requests += retained_count
            sessions.append(
                {
                    "session_id": output_path.stem,
                    "source_file": rel_path.as_posix(),
                    "output_file": output_name,
                    "source_request_count": len(reqs),
                    "candidate_count": len(candidates),
                    "retained_count": retained_count,
                    "candidate_request_indices": candidate_request_indices,
                    "retained_request_indices": retained_request_indices,
                    "filtered_over_limit_request_indices": filtered_over_limit_request_indices,
                }
            )

        prepared_session_count = len(sessions)
        manifest = {
            "source_root": str(self.input_dir),
            "tokenizer_path": self.tokenizer_path,
            "max_model_len": self.max_model_len,
            "force_max_tokens": self.force_max_tokens,
            "prepared_session_count": prepared_session_count,
            "prepared_request_count": total_retained_requests,
            "filtered_over_limit_request_count": total_filtered_over_limit,
            "requests_per_session_candidate_limit": REQUESTS_PER_SESSION,
            "skipped_files": skipped_files,
            "sessions": sessions,
        }

        if prepared_session_count == 0:
            raise ValueError("No valid sessions found in input directory.")
        if total_retained_requests == 0:
            raise ValueError("All candidate requests were filtered out; prepared dataset is empty.")

        with (self.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")

        return manifest


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    prepare_output_dir(output_dir, args.overwrite)

    builder = PreparedClaudeDatasetBuilder(
        input_dir=input_dir,
        output_dir=output_dir,
        tokenizer_path=args.tokenizer,
        max_model_len=args.max_model_len,
        force_max_tokens=args.force_max_tokens,
    )
    manifest = builder.build()

    print(f"Prepared dataset saved to: {output_dir}")
    print(f"Prepared sessions: {manifest['prepared_session_count']}")
    print(f"Prepared requests: {manifest['prepared_request_count']}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
