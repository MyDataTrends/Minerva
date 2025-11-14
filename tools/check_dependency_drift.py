#!/usr/bin/env python3
"""Compare dev and runtime dependencies.

The script reads environment.codex.yml, environment.yml, requirements.txt
and requirements.lock.txt. It prints version mismatches for packages that
appear in both dev and runtime files. When mismatches are detected the
process exits with status 1 unless ALLOW_DRIFT=1 is set in the environment.
"""

from __future__ import annotations

from config import get_bool
import sys
import re
from pathlib import Path

import yaml
from packaging.requirements import Requirement


def _parse_conda_spec(spec: str):
    spec = spec.strip()
    if ';' in spec:
        spec = spec.split(';', 1)[0].strip()
    if '@' in spec and '://' in spec:
        return spec.split('@', 1)[0].lower(), None
    for sep in ('==', '=', '>=', '<=', '>', '<'):
        if sep in spec:
            name, version = spec.split(sep, 1)
            return name.strip().lower(), version.strip()
    return spec.lower(), None


def parse_env_file(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))
    deps: dict[str, str | None] = {}
    for entry in data.get('dependencies', []):
        if isinstance(entry, str):
            name, version = _parse_conda_spec(entry)
            deps[name] = version
        elif isinstance(entry, dict):
            for pip_dep in entry.get('pip', []):
                if pip_dep.startswith('-r '):
                    continue
                try:
                    req = Requirement(pip_dep)
                    name = req.name.lower()
                    version = None
                    for spec in req.specifier:
                        if spec.operator == '==' and version is None:
                            version = spec.version
                except Exception:
                    name, version = _parse_conda_spec(pip_dep)
                deps[name] = version
    return deps


def parse_requirements(path: Path):
    deps: dict[str, str | None] = {}
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-16")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or line.startswith('-r '):
            continue
        try:
            req = Requirement(line)
        except Exception:
            continue
        name = req.name.lower()
        version = None
        for spec in req.specifier:
            if spec.operator == '==' and version is None:
                version = spec.version
        deps[name] = version
    return deps


def show_diff(label1: str, deps1: dict[str, str | None], label2: str, deps2: dict[str, str | None]):
    mismatches = []
    shared = sorted(set(deps1) & set(deps2))
    for pkg in shared:
        v1 = deps1[pkg]
        v2 = deps2[pkg]
        if v1 != v2:
            mismatches.append(pkg)
    if mismatches:
        print(f"{label1} vs {label2}")
        for pkg in mismatches:
            v1 = deps1[pkg] or 'unpinned'
            v2 = deps2[pkg] or 'unpinned'
            print(f"- {pkg}=={v1}")
            print(f"+ {pkg}=={v2}")
    return mismatches


def main() -> int:
    env_codex = parse_env_file(Path('environment.codex.yml'))
    env_runtime = parse_env_file(Path('environment.yml'))
    req = parse_requirements(Path('requirements.txt'))
    req_lock = parse_requirements(Path('requirements.lock.txt'))

    mism = []
    mism.extend(show_diff('environment.codex.yml', env_codex, 'environment.yml', env_runtime))
    mism.extend(show_diff('requirements.lock.txt', req_lock, 'requirements.txt', req))

    if mism and not get_bool('ALLOW_DRIFT', False):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
