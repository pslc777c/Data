from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# =========================
# DATA STRUCTURES
# =========================

@dataclass(frozen=True)
class Step:
    """
    Step ejecutable del pipeline.

    - outputs_rel: lista de outputs relativos a artifacts_dir (data/.)
      Si es [] => nunca se hace skip por existencia (corre siempre)
      Si contiene wildcard/glob => skip si hay >=1 match
    - args: argumentos CLI para el script (ej: ["--mode","prod"])
    - extra_env: variables de entorno extra por step
    - timeout_seconds: si un step se cuelga, se corta
    """
    name: str
    layer: str
    script_relpath: str
    outputs_rel: list[str]

    args: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None


@dataclass
class RunResult:
    step: Step
    ok: bool
    seconds: float
    returncode: int
    stdout_path: Path
    stderr_path: Path
    skipped: bool = False


# =========================
# INTERNALS
# =========================

LAYER_ORDER = ["bronze", "silver", "preds", "features", "models", "gold", "ml2", "eval", "audit"]


def _fmt_cmd(cmd: list[str]) -> str:
    def q(s: str) -> str:
        return f'"{s}"' if (" " in s or "\t" in s) else s
    return " ".join(q(x) for x in cmd)


def _resolve_outputs(artifacts_dir: Path, outputs_rel: list[str]) -> list[Path]:
    """Devuelve paths reales a chequear. Si hay glob, expande."""
    out: list[Path] = []
    for rel in outputs_rel:
        p = (artifacts_dir / rel)
        # glob: si contiene wildcard, expandimos
        if any(ch in rel for ch in ["*", "?", "["]):
            matches = list(p.parent.glob(p.name))
            out.extend([m.resolve() for m in matches])
        else:
            out.append(p.resolve())
    return out


def _outputs_exist(artifacts_dir: Path, outputs_rel: list[str]) -> bool:
    """
    - outputs_rel vacía => no-skip
    - con rutas normales => all exist
    - con glob => existe si hay >=1 match (por cada patrón) y los normales existen
    """
    if not outputs_rel:
        return False

    for rel in outputs_rel:
        p = artifacts_dir / rel
        if any(ch in rel for ch in ["*", "?", "["]):
            matches = list(p.parent.glob(p.name))
            if len(matches) == 0:
                return False
        else:
            if not p.exists():
                return False
    return True


def run_step(
    step: Step,
    python_exe: str,
    repo_root: Path,
    artifacts_dir: Path,
    log_dir: Path,
    force: bool,
    dry_run: bool,
    extra_env: Optional[dict[str, str]] = None,
) -> RunResult:
    script_path = (repo_root / step.script_relpath).resolve()

    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{step.layer}__{step.name}.out.log"
    stderr_path = log_dir / f"{step.layer}__{step.name}.err.log"

    if not script_path.exists():
        stderr_path.write_text(f"Script not found: {script_path}\n", encoding="utf-8")
        return RunResult(step, False, 0.0, 2, stdout_path, stderr_path, skipped=False)

    if (not force) and _outputs_exist(artifacts_dir, step.outputs_rel):
        # dejamos un log mínimo para trazabilidad
        stdout_path.write_text(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SKIP {step.layer}:{step.name} (outputs exist)\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return RunResult(step, True, 0.0, 0, stdout_path, stderr_path, skipped=True)

    cmd = [python_exe, str(script_path), *step.args]

    env = dict(os.environ)

    # env global
    if extra_env:
        env.update(extra_env)

    # env por step
    if step.extra_env:
        env.update(step.extra_env)

    # ✅ incluir repo_root y repo_root/src en PYTHONPATH
    root_path = str(repo_root.resolve())
    src_path = str((repo_root / "src").resolve())

    prev = env.get("PYTHONPATH", "")
    parts = [root_path, src_path]
    if prev:
        parts.append(prev)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    # ✅ útil para scripts: saber dónde está data/logs sin hardcode
    env.setdefault("OPS_REPO_ROOT", root_path)
    env.setdefault("OPS_ARTIFACTS_DIR", str(artifacts_dir))
    env.setdefault("OPS_LOG_DIR", str(log_dir))

    # header
    outputs_abs = _resolve_outputs(artifacts_dir, step.outputs_rel)
    header = []
    header.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP {step.layer}:{step.name}")
    header.append(f"script={script_path}")
    header.append(f"cmd={_fmt_cmd(cmd)}")
    header.append(f"cwd={repo_root}")
    header.append(f"PYTHONPATH={env['PYTHONPATH']}")
    header.append(f"OPS_ARTIFACTS_DIR={env['OPS_ARTIFACTS_DIR']}")
    header.append(f"OPS_LOG_DIR={env['OPS_LOG_DIR']}")
    if step.timeout_seconds:
        header.append(f"timeout_seconds={step.timeout_seconds}")
    if step.outputs_rel:
        header.append("outputs(check):")
        for o in outputs_abs:
            header.append(f" - {o}")
    else:
        header.append("outputs: (none) => no-skip mode")
    header_txt = "\n".join(header) + "\n\n"

    if dry_run:
        stdout_path.write_text("[DRY RUN]\n" + header_txt, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return RunResult(step, True, 0.0, 0, stdout_path, stderr_path, skipped=False)

    t0 = time.time()
    rc = 0

    try:
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            out.write(header_txt)
            out.flush()

            p = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=out,
                stderr=err,
                env=env,
                timeout=step.timeout_seconds,
            )
            rc = p.returncode

    except subprocess.TimeoutExpired:
        rc = 124
        with stderr_path.open("a", encoding="utf-8") as err:
            err.write(f"\n[ERROR] TimeoutExpired after {step.timeout_seconds}s\n")

    except Exception as e:
        rc = 3
        with stderr_path.open("a", encoding="utf-8") as err:
            err.write(f"\n[ERROR] Exception running step: {e}\n")

    seconds = time.time() - t0
    ok = (rc == 0)
    return RunResult(step, ok, seconds, rc, stdout_path, stderr_path, skipped=False)


# =========================
# CLI
# =========================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ops.runner")
    p.add_argument("--list", action="store_true", help="List steps and exit")
    p.add_argument("--layer", default="", help="Comma-separated layers (bronze,silver,ml2,...)")
    p.add_argument("--step", default="", help="Comma-separated step names to run")
    p.add_argument("--from", dest="from_step", default="", help="Run from this step (inclusive)")
    p.add_argument("--to", dest="to_step", default="", help="Run to this step (inclusive)")

    # ✅ NEW: start-from layer
    p.add_argument(
        "--start-from",
        dest="start_from_layer",
        default="",
        help=f"Start pipeline from this layer (inclusive). Choices: {','.join(LAYER_ORDER)}",
    )

    p.add_argument("--force", action="store_true", help="Run even if outputs exist")
    p.add_argument("--dry-run", action="store_true", help="Print/run plan without executing scripts")
    p.add_argument("--python", default="", help="Override python executable (default: current)")
    p.add_argument("--artifacts-dir", default="data", help="Artifacts dir (default: data)")
    p.add_argument("--log-root", default="src/run_logs", help="Base log dir (default: src/run_logs)")
    return p.parse_args()


def _select_steps(
    all_steps: list[Step],
    layers: set[str],
    names: set[str],
    from_step: str,
    to_step: str,
    start_from_layer: str,
) -> list[Step]:
    selected = all_steps

    # ✅ Apply start-from layer FIRST (on full registry order)
    if start_from_layer:
        start = start_from_layer.strip().lower()
        if start not in LAYER_ORDER:
            raise SystemExit(
                f"[ERROR] --start-from '{start_from_layer}' invalid. "
                f"Use one of: {', '.join(LAYER_ORDER)}"
            )
        start_idx = LAYER_ORDER.index(start)
        allowed_layers = set(LAYER_ORDER[start_idx:])
        selected = [s for s in selected if s.layer.lower() in allowed_layers]

    if layers:
        selected = [s for s in selected if s.layer.lower() in layers]

    if names:
        selected = [s for s in selected if s.name in names]

    if from_step:
        idx = next((i for i, s in enumerate(selected) if s.name == from_step), None)
        if idx is None:
            raise SystemExit(f"[ERROR] --from '{from_step}' not found in selected steps")
        selected = selected[idx:]

    if to_step:
        idx = next((i for i, s in enumerate(selected) if s.name == to_step), None)
        if idx is None:
            raise SystemExit(f"[ERROR] --to '{to_step}' not found in selected steps")
        selected = selected[: idx + 1]

    return selected


def main() -> int:
    args = _parse_args()

    # import aquí para evitar circular import raro
    from src.ops.registry import build_registry  # <- IMPORT ABSOLUTO

    repo_root = Path(__file__).resolve().parents[2]  # .../Data-LakeHouse
    python_exe = args.python.strip() or os.sys.executable
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()

    all_steps = build_registry()

    if args.list:
        for s in all_steps:
            print(f"{s.layer:8s}  {s.name:45s}  {s.script_relpath}")
        return 0

    layers = {x.strip().lower() for x in args.layer.split(",") if x.strip()}
    names = {x.strip() for x in args.step.split(",") if x.strip()}

    plan = _select_steps(
        all_steps=all_steps,
        layers=layers,
        names=names,
        from_step=args.from_step,
        to_step=args.to_step,
        start_from_layer=args.start_from_layer,
    )

    if not plan:
        print("[WARN] No steps selected.")
        return 0

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = (repo_root / args.log_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PLAN] steps={len(plan)} artifacts_dir={artifacts_dir} logs={run_dir}")
    for i, s in enumerate(plan, 1):
        print(f"  {i:03d}. {s.layer}:{s.name}")

    results: list[RunResult] = []

    for s in plan:
        r = run_step(
            step=s,
            python_exe=python_exe,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
            log_dir=run_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
        results.append(r)

        if not r.ok:
            print(f"[FAIL] {s.layer}:{s.name} rc={r.returncode}")
            print(f"       stdout: {r.stdout_path}")
            print(f"       stderr: {r.stderr_path}")
            (run_dir / "final_status.txt").write_text("FAIL\n", encoding="utf-8")
            _write_summary(run_dir, results)
            return 1

        if r.skipped:
            print(f"[SKIP] {s.layer}:{s.name}")
        else:
            print(f"[OK  ] {s.layer}:{s.name} ({r.seconds:.1f}s)")

    (run_dir / "final_status.txt").write_text("OK\n", encoding="utf-8")
    _write_summary(run_dir, results)
    print("[OK] Pipeline finished.")
    return 0


def _write_summary(run_dir: Path, results: list[RunResult]) -> None:
    path = run_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "step", "ok", "skipped", "seconds", "returncode", "stdout", "stderr"])
        for r in results:
            w.writerow([
                r.step.layer,
                r.step.name,
                int(r.ok),
                int(r.skipped),
                f"{r.seconds:.3f}",
                r.returncode,
                str(r.stdout_path),
                str(r.stderr_path),
            ])


if __name__ == "__main__":
    raise SystemExit(main())