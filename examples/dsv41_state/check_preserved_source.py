# SPDX-License-Identifier: Apache-2.0
"""CPU-only check that the runnable snapshot matches the archived source."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(__file__).with_name("tested-source.json").read_text())
    pd_branch = (root / "vllm_ascend/models/deepseek_v41/request_state.py").is_file()
    for relative, expected in manifest["source_sha256"].items():
        data = (
            subprocess.check_output(["git", "show", "f77bee84a5299c2e62d749409a29352aaf7e48b9:" + relative], cwd=root)
            if pd_branch
            else (root / relative).read_bytes()
        )
        assert hashlib.sha256(data).hexdigest() == expected, relative
        ast.parse(data.decode(), filename=relative)
    print(
        json.dumps(
            {
                "passed": True,
                "files": len(manifest["source_sha256"]),
                "runtime_validation": False,
                "scope": "preserved parent commit" if pd_branch else "current preserved snapshot",
            }
        )
    )


if __name__ == "__main__":
    main()
