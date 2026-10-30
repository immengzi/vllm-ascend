# SPDX-License-Identifier: Apache-2.0
"""CPU-only check that the runnable snapshot matches the archived source."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(__file__).with_name("tested-source.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        path = root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
        ast.parse(path.read_text(), filename=relative)
    print(json.dumps({"passed": True, "files": len(manifest["source_sha256"]), "runtime_validation": False}))


if __name__ == "__main__":
    main()
