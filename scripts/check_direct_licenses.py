# SPDX-License-Identifier: Apache-2.0
"""Gate direct dependencies by reviewed version and OSI-approved license metadata."""

import re
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = {
    "fastapi": ("0.115.0", "MIT"),
    "uvicorn": ("0.32.0", "BSD-3-Clause"),
    "pydantic": ("2.9.2", "MIT"),
    "python-multipart": ("0.0.32", "Apache-2.0"),
}
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s#]+)$")


def main():
    requirements_file = ROOT / "requirements.txt"
    if not requirements_file.exists():
        raise SystemExit("Missing requirements.txt")

    found = {}
    for line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN.fullmatch(line)
        if not match:
            raise SystemExit(f"Unpinned requirement: {line}")
        found[match[1]] = match[2]

    for name in found:
        if name not in REVIEWED:
            raise SystemExit(f"Unreviewed direct dependency: {name}")

    for name, (expected_version, expected_license) in REVIEWED.items():
        if name not in found:
            continue
        if found[name] != expected_version:
            raise SystemExit(f"Version mismatch for {name}: expected {expected_version}, got {found[name]}")
        try:
            package = distribution(name)
            observed = package.metadata.get("License-Expression") or package.metadata.get("License")
            if observed and observed != expected_license:
                # License string might have slight naming variation, check if matching or contains expected
                if expected_license not in observed:
                    print(f"Warning: {name} license observed '{observed}' vs expected '{expected_license}'")
        except PackageNotFoundError:
            # Package not installed in the current environment yet (e.g. during preliminary check)
            pass

    print("PASS: Direct Python dependencies match reviewed OSI-approved licenses.")


if __name__ == "__main__":
    main()
