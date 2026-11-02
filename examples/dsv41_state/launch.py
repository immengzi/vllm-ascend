# SPDX-License-Identifier: Apache-2.0
"""Launch the pinned history/request-state implementation."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    example = Path(__file__).resolve().parent
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(example), env.get("PYTHONPATH")]))
    env["MENGZI_RESULTS_DIR"] = str(args.results_dir.resolve())
    env.setdefault("MENGZI_STATE_DECODE_DIRECT", "0")
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    forwarded = args.vllm_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if not forwarded:
        parser.error("Supply vllm serve arguments after --")
    raise SystemExit(subprocess.call(["vllm", "serve", *forwarded], env=env))


if __name__ == "__main__":
    main()
