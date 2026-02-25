from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART_ROOT = ROOT / os.environ.get("CI_ARTIFACTS_DIR", "ci_artifacts")
SNAPSHOT_ROOT = ART_ROOT / "snapshot" / "code"

INCLUDE_EXACT = [
    ".github/workflows/generate_video.yml",
    ".github/workflows/build_baseline_cache.yml",
    "config/settings.template.yaml",
    "requirements-ci.txt",
]
INCLUDE_PY_DIRS = ["src", "scripts"]
INCLUDE_SUFFIXES = {".py"}


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _iter_candidate_relpaths() -> list[str]:
    try:
        tracked = _run_git(["ls-files"])
        untracked = _run_git(["ls-files", "--others", "--exclude-standard"])
        rels = [
            ln.strip().replace("\\", "/")
            for ln in (tracked.splitlines() + untracked.splitlines())
            if ln.strip()
        ]
        return rels
    except Exception:
        rels: list[str] = []
        for base in INCLUDE_PY_DIRS:
            base_path = ROOT / base
            if not base_path.exists():
                continue
            for p in base_path.rglob("*.py"):
                if "__pycache__" in p.parts:
                    continue
                rels.append(p.relative_to(ROOT).as_posix())
        rels.extend(INCLUDE_EXACT)
        return sorted(set(rels))


def _is_included(rel: str) -> bool:
    rel_norm = rel.replace("\\", "/")
    if rel_norm in INCLUDE_EXACT:
        return True

    p = Path(rel_norm)
    if p.suffix.lower() not in INCLUDE_SUFFIXES:
        return False

    return any(rel_norm.startswith(f"{d}/") for d in INCLUDE_PY_DIRS)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, object]] = []
    seen: set[str] = set()

    for rel in _iter_candidate_relpaths():
        rel_norm = rel.replace("\\", "/")
        if rel_norm in seen:
            continue
        seen.add(rel_norm)

        if not _is_included(rel_norm):
            continue

        src = ROOT / rel_norm
        if not src.is_file():
            continue

        dst = SNAPSHOT_ROOT / rel_norm
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        copied.append(
            {
                "path": rel_norm,
                "bytes": src.stat().st_size,
                "md5": _md5(src),
            }
        )

    copied.sort(key=lambda x: str(x["path"]))

    meta = {
        "repo_root": str(ROOT),
        "snapshot_root": str(SNAPSHOT_ROOT),
        "count_files": len(copied),
        "git_head": None,
    }
    try:
        meta["git_head"] = _run_git(["rev-parse", "HEAD"])
    except Exception:
        pass

    manifest_path = ART_ROOT / "snapshot" / "code_snapshot_manifest.json"
    meta_path = ART_ROOT / "snapshot" / "code_snapshot_meta.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(copied, fh, indent=2, ensure_ascii=False)

    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"[OK] Code snapshot files: {len(copied)}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Meta    : {meta_path}")


if __name__ == "__main__":
    main()
