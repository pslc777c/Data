# autorun_pipeline_until_green.py
# Ejecuta en bucle TODOS los .py dentro de:
#   C:\Data-LakeHouse\src\gold
#   C:\Data-LakeHouse\src\eval
#   C:\Data-LakeHouse\src\models\ml2
#   C:\Data-LakeHouse\src\audit
#
# Objetivo: ir ejecutando (sin parar por errores) hasta que TODOS terminen OK,
# o hasta que se quede “atascado” (sin progreso) / llegue a MAX_PASSES.
#
# - Imprime log en vivo (stdout+stderr) y guarda TODO en logs.
# - Mantiene el cwd en C:\Data-LakeHouse (como tu forma de correr scripts).
# - Setea PYTHONPATH para incluir \src (para imports locales).
# - NO sobrescribe logs; crea carpeta run_logs\run_YYYYmmdd_HHMMSS\
#
# IMPORTANTE:
# - Si algún script requiere argumentos obligatorios, va a fallar siempre.
#   Usa SCRIPT_ARGS para pasarle args específicos por archivo.
# - Esto ejecuta código arbitrario de tu repo: úsalo solo en tu entorno controlado.

from __future__ import annotations

import os
import sys
import time
import json
import csv
import traceback
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# =========================
# CONFIG
# =========================
SRC_ROOT = Path(r"C:\Data-LakeHouse\src").resolve()
REPO_ROOT = SRC_ROOT.parent

TARGET_DIRS = [
    SRC_ROOT / "gold",
    SRC_ROOT / "eval",
    SRC_ROOT / "models" / "ml2",
    SRC_ROOT / "audit",
]

EXCLUDE_DIRS: Set[str] = {
    ".git", ".svn", ".hg",
    "__pycache__",
    ".venv", "venv", "env",
    "build", "dist",
    ".mypy_cache", ".pytest_cache",
    ".ruff_cache",
    ".idea", ".vscode",
}

# Máximo de pasadas del bucle completo
MAX_PASSES = 15

# Timeout por script (segundos). None = sin timeout.
# Sugerencia: 30 min = 1800
SCRIPT_TIMEOUT_SEC: Optional[int] = 1800

# Pausa entre pasadas (segundos)
SLEEP_BETWEEN_PASSES = 2

# Si True, reintenta scripts fallidos en la siguiente pasada aunque “no haya progreso”.
# Si False, corta cuando no hay progreso en una pasada.
ALLOW_STUCK_RETRY = False

# Args por script (ruta relativa desde \src)
# Ejemplo:
# SCRIPT_ARGS = {
#   r"models\ml2\train_ml2.py": ["--force", "1"],
# }
SCRIPT_ARGS: Dict[str, List[str]] = {}

# Si quieres correr SOLO un subconjunto por patrones, llena esto.
# Ejemplo: INCLUDE_SUBSTRINGS = ["build_", "apply_"]
INCLUDE_SUBSTRINGS: List[str] = []


# =========================
# Logging / resultados
# =========================
@dataclass
class RunResult:
    rel: str
    abs: str
    pass_no: int
    exit_code: int
    started_at: str
    ended_at: str
    seconds: float
    log_file: str
    last_lines: List[str]


def _now() -> datetime:
    return datetime.now()


def _ts() -> str:
    return _now().strftime("%Y-%m-%d %H:%M:%S")


def _iter_py_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.lower().endswith(".py") and fn.lower() != "__init__.py":
                yield (Path(dirpath) / fn).resolve()


def _rel_from_src(p: Path) -> str:
    return str(p.resolve().relative_to(SRC_ROOT)).replace("/", "\\")


def _should_include(rel: str) -> bool:
    if not INCLUDE_SUBSTRINGS:
        return True
    rl = rel.lower()
    return any(s.lower() in rl for s in INCLUDE_SUBSTRINGS)


def _priority_key(rel: str) -> Tuple[int, int, str]:
    """
    Orden heurístico para aumentar chance de “camino correcto”:
      - gold primero, luego models\ml2, luego eval, luego audit
      - dentro: build/etl -> train -> apply -> eval -> audit -> otros
    """
    r = rel.lower().replace("/", "\\")
    stage_rank = 99
    if r.startswith("gold\\"):
        stage_rank = 0
    elif r.startswith("models\\ml2\\"):
        stage_rank = 1
    elif r.startswith("eval\\"):
        stage_rank = 2
    elif r.startswith("audit\\"):
        stage_rank = 3

    name = Path(r).name
    fn_rank = 50
    prefixes = [
        ("build_", 0),
        ("etl_", 1),
        ("etl-", 1),
        ("train_", 2),
        ("fit_", 2),
        ("apply_", 3),
        ("run_", 4),
        ("eval_", 5),
        ("audit_", 6),
    ]
    for pref, rk in prefixes:
        if name.startswith(pref):
            fn_rank = rk
            break

    return (stage_rank, fn_rank, r)


def _make_run_dir() -> Path:
    run_dir = SRC_ROOT / "run_logs" / f"run_{_now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_env() -> Dict[str, str]:
    env = dict(os.environ)
    # PYTHONPATH incluye src para imports locales
    pp = env.get("PYTHONPATH", "")
    sep = ";"  # windows
    src_str = str(SRC_ROOT)
    if pp:
        if src_str.lower() not in [x.strip().lower() for x in pp.split(sep)]:
            env["PYTHONPATH"] = pp + sep + src_str
    else:
        env["PYTHONPATH"] = src_str
    return env


def _run_one(script_path: Path, pass_no: int, run_dir: Path) -> RunResult:
    rel = _rel_from_src(script_path)
    abs_path = str(script_path)

    # args extra
    extra = SCRIPT_ARGS.get(rel, [])

    # log file por script y pasada
    safe_name = rel.replace("\\", "__").replace(":", "")
    log_file = run_dir / f"pass{pass_no:02d}__{safe_name}.log"

    started = _now()
    last_lines: List[str] = []

    cmd = [sys.executable, abs_path, *extra]

    header = (
        f"\n{'='*110}\n"
        f"[{_ts()}] PASS {pass_no:02d} | RUN: {rel}\n"
        f"CMD: {' '.join(cmd)}\n"
        f"CWD: {REPO_ROOT}\n"
        f"{'='*110}\n"
    )

    print(header, end="")
    with log_file.open("w", encoding="utf-8", errors="replace") as lf:
        lf.write(header)

        try:
            p = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=_build_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # stream live
            start_t = time.time()
            while True:
                line = p.stdout.readline() if p.stdout else ""
                if line:
                    print(line, end="")
                    lf.write(line)

                    # mantener últimas 40 líneas para resumen
                    last_lines.append(line.rstrip("\n"))
                    if len(last_lines) > 40:
                        last_lines = last_lines[-40:]
                else:
                    # si terminó, sal
                    if p.poll() is not None:
                        break
                    # timeout check
                    if SCRIPT_TIMEOUT_SEC is not None and (time.time() - start_t) > SCRIPT_TIMEOUT_SEC:
                        p.kill()
                        msg = f"\n[TIMEOUT] {rel} excedió {SCRIPT_TIMEOUT_SEC}s. Proceso terminado.\n"
                        print(msg, end="")
                        lf.write(msg)
                        break
                    time.sleep(0.05)

            exit_code = p.wait(timeout=5) if p.poll() is None else (p.returncode or 0)

        except Exception as e:
            exit_code = 999
            err = (
                f"\n[EXCEPTION] Falló la ejecución de {rel}\n"
                f"{type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}\n"
            )
            print(err, end="")
            lf.write(err)
            last_lines.append(f"{type(e).__name__}: {e}")

    ended = _now()
    seconds = (ended - started).total_seconds()

    # footer
    footer = (
        f"\n--- RESULT ---\n"
        f"[{_ts()}] EXIT_CODE={exit_code} | SECONDS={seconds:.2f}\n"
        f"LOG: {log_file}\n"
        f"--------------\n\n"
    )
    print(footer, end="")
    with log_file.open("a", encoding="utf-8", errors="replace") as lf:
        lf.write(footer)

    return RunResult(
        rel=rel,
        abs=abs_path,
        pass_no=pass_no,
        exit_code=int(exit_code),
        started_at=started.isoformat(timespec="seconds"),
        ended_at=ended.isoformat(timespec="seconds"),
        seconds=float(seconds),
        log_file=str(log_file),
        last_lines=last_lines,
    )


def main() -> int:
    run_dir = _make_run_dir()

    # 1) recolectar scripts
    scripts: List[Path] = []
    for d in TARGET_DIRS:
        if d.exists():
            scripts.extend(list(_iter_py_files(d)))
        else:
            print(f"[WARN] No existe directorio objetivo: {d}")

    # dedup
    scripts = sorted(list(dict.fromkeys(scripts)), key=lambda p: _priority_key(_rel_from_src(p)))

    # aplicar include filter
    scripts = [p for p in scripts if _should_include(_rel_from_src(p))]

    if not scripts:
        print("[ERROR] No encontré scripts .py en los directorios objetivo.")
        return 2

    print("======================================================")
    print("AUTO-RUN PIPELINE SCRIPTS UNTIL GREEN (best-effort)")
    print(f"SRC_ROOT  : {SRC_ROOT}")
    print(f"REPO_ROOT : {REPO_ROOT}")
    print("TARGET_DIRS:")
    for d in TARGET_DIRS:
        print(" -", d)
    print(f"TOTAL SCRIPTS: {len(scripts)}")
    print(f"MAX_PASSES   : {MAX_PASSES}")
    print(f"TIMEOUT/py   : {SCRIPT_TIMEOUT_SEC}s")
    print(f"RUN_DIR      : {run_dir}")
    print("======================================================\n")

    # 2) bucle de ejecución
    ok: Set[str] = set()
    last_fail_set: Optional[Set[str]] = None
    all_results: List[RunResult] = []

    for pass_no in range(1, MAX_PASSES + 1):
        pending = [p for p in scripts if _rel_from_src(p) not in ok]
        if not pending:
            print("\n✅ TODO OK: todos los scripts terminaron en exit_code 0.\n")
            break

        print(f"\n########## PASS {pass_no:02d} | pending={len(pending)} ##########\n")
        before_ok = len(ok)

        fail_set: Set[str] = set()

        for sp in pending:
            res = _run_one(sp, pass_no, run_dir)
            all_results.append(res)
            if res.exit_code == 0:
                ok.add(res.rel)
            else:
                fail_set.add(res.rel)

        gained = len(ok) - before_ok
        print(f"\n[PASS {pass_no:02d}] nuevos OK: {gained} | OK acumulado: {len(ok)}/{len(scripts)}")

        if not fail_set:
            print("\n✅ PASS limpio: ya no hay fallos.\n")
            break

        # si no hubo progreso, decidir cortar / reintentar
        if gained == 0:
            print("\n⚠️  No hubo progreso en esta pasada (ningún script nuevo se puso OK).")
            if last_fail_set is not None and fail_set == last_fail_set and not ALLOW_STUCK_RETRY:
                print("⛔ Fallos repetidos (mismo set) -> corto para evitar bucle infinito.\n")
                break
            last_fail_set = set(fail_set)

        if SLEEP_BETWEEN_PASSES > 0:
            time.sleep(SLEEP_BETWEEN_PASSES)

    # 3) outputs de resumen
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    last_status_txt = run_dir / "final_status.txt"

    summary_obj = {
        "src_root": str(SRC_ROOT),
        "repo_root": str(REPO_ROOT),
        "target_dirs": [str(d) for d in TARGET_DIRS],
        "total_scripts": len(scripts),
        "scripts": [str(_rel_from_src(p)) for p in scripts],
        "ok_scripts": sorted(list(ok)),
        "pending_scripts": sorted([_rel_from_src(p) for p in scripts if _rel_from_src(p) not in ok]),
        "results": [asdict(r) for r in all_results],
        "generated_at": _ts(),
    }
    summary_json.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pass_no", "rel", "exit_code", "seconds", "log_file"])
        for r in all_results:
            w.writerow([r.pass_no, r.rel, r.exit_code, f"{r.seconds:.2f}", r.log_file])

    pending_final = [p for p in scripts if _rel_from_src(p) not in ok]
    with last_status_txt.open("w", encoding="utf-8") as f:
        f.write(f"OK {len(ok)}/{len(scripts)}\n\n")
        if pending_final:
            f.write("PENDIENTES (fallan / no llegaron a OK):\n")
            for p in pending_final:
                f.write(f"- {_rel_from_src(p)}\n")

    print("\n======================================================")
    print("RESUMEN FINAL")
    print(f"OK: {len(ok)}/{len(scripts)}")
    if pending_final:
        print("Pendientes (fallan):")
        for p in pending_final[:50]:
            print(" -", _rel_from_src(p))
        if len(pending_final) > 50:
            print(f" - ... ({len(pending_final)-50} más)")
    print("------------------------------------------------------")
    print("Archivos de salida:")
    print(" -", summary_json)
    print(" -", summary_csv)
    print(" -", last_status_txt)
    print("Logs por script en:")
    print(" -", run_dir)
    print("======================================================\n")

    return 0 if not pending_final else 1


if __name__ == "__main__":
    raise SystemExit(main())
