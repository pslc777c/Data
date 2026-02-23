from __future__ import annotations
import os
import sys
import site
from pathlib import Path
import json
import traceback

REQUIRED_DIRS = [
    "data",
    "data/bronze",
    "data/silver",
    "data/gold",
    "data/eval",
    "data/audit",
    "src/run_logs",
]

# Ajusta imports según lo que usa build_balanza_cosecha_raw.py
OPTIONAL_IMPORTS = [
    "pandas",
    "pyarrow",
    "yaml",
    "sqlalchemy",
    "pyodbc",      # si conectas a SQL Server via ODBC
    "pymssql",     # si aplica
]

def find_repo_root(start: Path) -> Path:
    """Busca raíz por presencia de .git o carpeta src/."""
    cur = start.resolve()
    for _ in range(10):
        if (cur / ".git").exists() or (cur / "src").exists():
            return cur
        cur = cur.parent
    return start.resolve()

def can_write(dirpath: Path) -> tuple[bool, str]:
    try:
        dirpath.mkdir(parents=True, exist_ok=True)
        probe = dirpath / "__write_probe__.txt"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def main() -> int:
    repo_root = find_repo_root(Path.cwd())
    os.chdir(repo_root)

    print("=== PREFLIGHT ===")
    print("exe:", sys.executable)
    print("py :", sys.version.replace("\n", " "))
    print("cwd:", Path.cwd())
    print("user_site:", site.getusersitepackages())
    print("RUN_MODE:", os.environ.get("RUN_MODE") or os.environ.get("DLH_ENV") or "<unset>")
    print("GITHUB_WORKSPACE:", os.environ.get("GITHUB_WORKSPACE", "<unset>"))

    settings_path = repo_root / "config" / "settings.yaml"
    print("settings_path:", settings_path)
    if not settings_path.exists():
        print("[FAIL] Missing config/settings.yaml")
        return 2

    # Crear dirs requeridas
    print("\n=== DIRS ===")
    for d in REQUIRED_DIRS:
        p = repo_root / d
        p.mkdir(parents=True, exist_ok=True)
        ok, msg = can_write(p)
        print(f"{d:15s} exists={p.exists()} write_ok={ok} ({msg})")
        if not ok:
            print("[FAIL] No write permission in:", p)
            return 3

    # Import checks
    print("\n=== IMPORTS ===")
    failed = []
    for m in OPTIONAL_IMPORTS:
        try:
            __import__(m)
            print(f"[OK] import {m}")
        except Exception as e:
            print(f"[WARN] import {m} failed -> {type(e).__name__}: {e}")
            failed.append(m)

    # Dump snapshot of env to file for artifacts
    env_snapshot = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cwd": str(Path.cwd()),
        "user_site": site.getusersitepackages(),
        "run_mode": os.environ.get("RUN_MODE") or os.environ.get("DLH_ENV"),
        "github_workspace": os.environ.get("GITHUB_WORKSPACE"),
        "missing_imports": failed,
    }
    out = repo_root / "src" / "run_logs" / "preflight_env.json"
    out.write_text(json.dumps(env_snapshot, indent=2), encoding="utf-8")
    print("\n[OK] wrote:", out)

    print("\n=== PREFLIGHT OK ===")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(99)