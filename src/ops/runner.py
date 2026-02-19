from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Step:
    """
    Step ejecutable del pipeline.

    - outputs_rel: lista de outputs relativos a artifacts_dir (data/.)
      Si es [] => nunca se hace skip por existencia (corre siempre)
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


def _outputs_exist(outputs: list[Path]) -> bool:
    # Si outputs list es vacía -> no-skip
    return bool(outputs) and all(p.exists() for p in outputs)


def _fmt_cmd(cmd: list[str]) -> str:
    def q(s: str) -> str:
        return f'"{s}"' if (" " in s or "\t" in s) else s
    return " ".join(q(x) for x in cmd)


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
    outputs_abs = [(artifacts_dir / o).resolve() for o in step.outputs_rel]

    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{step.layer}__{step.name}.out.log"
    stderr_path = log_dir / f"{step.layer}__{step.name}.err.log"

    if not script_path.exists():
        stderr_path.write_text(f"Script not found: {script_path}\n", encoding="utf-8")
        return RunResult(step, False, 0.0, 2, stdout_path, stderr_path)

    if (not force) and _outputs_exist(outputs_abs):
        return RunResult(step, True, 0.0, 0, stdout_path, stderr_path)

    cmd = [python_exe, str(script_path), *step.args]

    env = dict(os.environ)

    # env global
    if extra_env:
        env.update(extra_env)

    # env por step
    if step.extra_env:
        env.update(step.extra_env)

    # ✅ CLAVE: incluir repo_root y repo_root/src en PYTHONPATH
    root_path = str(repo_root.resolve())
    src_path = str((repo_root / "src").resolve())

    prev = env.get("PYTHONPATH", "")
    parts = [root_path, src_path]
    if prev:
        parts.append(prev)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    # ✅ Útil para scripts: saber dónde está data/logs sin hardcode
    env.setdefault("OPS_REPO_ROOT", root_path)
    env.setdefault("OPS_ARTIFACTS_DIR", str(artifacts_dir))
    env.setdefault("OPS_LOG_DIR", str(log_dir))

    header = []
    header.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STEP {step.layer}:{step.name}")
    header.append(f"script={script_path}")
    header.append(f"cmd={_fmt_cmd(cmd)}")
    header.append(f"cwd={repo_root}")
    header.append(f"PYTHONPATH={env['PYTHONPATH']}")
    header.append(f"OPS_ARTIFACTS_DIR={env['OPS_ARTIFACTS_DIR']}")
    if step.timeout_seconds:
        header.append(f"timeout_seconds={step.timeout_seconds}")
    if step.outputs_rel:
        header.append("outputs:")
        for o in outputs_abs:
            header.append(f" - {o}")
    else:
        header.append("outputs: (none) => no-skip mode")
    header_txt = "\n".join(header) + "\n\n"

    if dry_run:
        stdout_path.write_text("[DRY RUN]\n" + header_txt, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return RunResult(step, True, 0.0, 0, stdout_path, stderr_path)

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
    return RunResult(step, ok, seconds, rc, stdout_path, stderr_path)
