# SPDX-License-Identifier: Apache-2.0
"""Launch the pinned history/request-state implementation."""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    env = os.environ.copy()
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    forwarded = args.vllm_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if not forwarded:
        parser.error("Supply vllm serve arguments after --")
    raise SystemExit(subprocess.call(["vllm", "serve", *forwarded], env=env))


if __name__ == "__main__":
    main()
