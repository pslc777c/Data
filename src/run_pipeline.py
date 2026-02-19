from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from ops.io import load_settings
from ops.registry import build_registry
from ops.runner import run_step
from ops.validators import validate_exists, validate_parquet_nonempty


LAYER_ORDER: list[str] = [
    "bronze",
    "silver",
    "preds",
    "features",
    "models",
    "gold",
    "ml2",
    "eval",
    "audit",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("PLANIFICACION MACRO - OPS (src layout)")

    ap.add_argument("--settings", default="config/settings.yaml")

    ap.add_argument("--only", choices=LAYER_ORDER, default=None)
    ap.add_argument("--from", dest="from_layer", choices=LAYER_ORDER, default=None)
    ap.add_argument("--until", dest="until_layer", choices=LAYER_ORDER, default=None)
    ap.add_argument("--step", default=None, help="Ejecutar un step por nombre exacto")

    ap.add_argument("--force", action="store_true", help="Re-ejecuta aunque existan outputs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--validate", action="store_true")

    return ap.parse_args()


def filter_steps(steps, only: str | None, from_layer: str | None, until_layer: str | None, step_name: str | None):
    if step_name:
        return [s for s in steps if s.name == step_name]

    if only:
        return [s for s in steps if s.layer == only]

    if from_layer or until_layer:
        i0 = LAYER_ORDER.index(from_layer) if from_layer else 0
        i1 = LAYER_ORDER.index(until_layer) if until_layer else (len(LAYER_ORDER) - 1)
        allowed = set(LAYER_ORDER[i0 : i1 + 1])
        return [s for s in steps if s.layer in allowed]

    return steps


def expand_outputs(artifacts_dir: Path, outputs_rel: List[str]) -> list[Path]:
    """
    Expande outputs_rel que pueden ser:
      - paths exactos (ej: gold/x.parquet)
      - patrones con glob (ej: models/ml2/harvest_start_ml2_*_meta.json)

    Devuelve Paths absolutos existentes (si hay glob, puede devolver múltiples).
    """
    expanded: list[Path] = []
    for rel in outputs_rel:
        # Si parece glob, expandimos
        if any(ch in rel for ch in ["*", "?", "["]):
            matches = list(artifacts_dir.glob(rel))
            expanded.extend([m.resolve() for m in matches])
        else:
            expanded.append((artifacts_dir / rel).resolve())
    return expanded


def only_parquets(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.suffix.lower() == ".parquet":
            out.append(p)
    return out


def main():
    args = parse_args()
    settings = load_settings(args.settings)

    # repo root = padre de /src
    repo_root = Path(__file__).resolve().parents[1]

    python_exe_cfg = settings.get("ops", {}).get("python", "venv/Scripts/python.exe")
    python_exe = str((repo_root / python_exe_cfg).resolve())

    artifacts_dir = (repo_root / settings.get("ops", {}).get("artifacts_dir", "data")).resolve()
    log_dir = (repo_root / settings.get("ops", {}).get("log_dir", "logs") / "ops").resolve()

    steps = filter_steps(build_registry(), args.only, args.from_layer, args.until_layer, args.step)
    if not steps:
        raise SystemExit("No steps selected.")

    print(f"repo_root={repo_root}")
    print(f"python={python_exe}")
    print(f"artifacts_dir={artifacts_dir}")
    print(f"log_dir={log_dir}")
    print(f"steps={len(steps)}")

    ok_all = True

    for s in steps:
        print(f"\n==> {s.layer}:{s.name}")

        res = run_step(
            step=s,
            python_exe=python_exe,
            repo_root=repo_root,
            artifacts_dir=artifacts_dir,
            log_dir=log_dir,
            force=args.force,
            dry_run=args.dry_run,
        )

        if not res.ok:
            ok_all = False
            print(f"   FAILED rc={res.returncode}")
            print(f"   logs:\n   - {res.stdout_path}\n   - {res.stderr_path}")
            if args.fail_fast:
                break
            continue

        if res.seconds > 0:
            print(f"   OK {res.seconds:.1f}s")
        else:
            print("   dry-run" if args.dry_run else "   skipped (outputs exist)")

        # Validate: ahora soporta outputs_rel con glob y valida nonempty solo en parquets
        if args.validate and getattr(s, "outputs_rel", None):
            outputs_abs = expand_outputs(artifacts_dir, list(s.outputs_rel))

            try:
                # Si hubo glob y no hizo match, esto debe fallar (es correcto)
                validate_exists(outputs_abs)

                pq = only_parquets(outputs_abs)
                if pq:
                    validate_parquet_nonempty(pq)

                print("   validated OK")
            except Exception as e:
                ok_all = False
                print(f"   VALIDATION FAILED: {e}")
                if args.fail_fast:
                    break

    if not ok_all:
        raise SystemExit(1)

    print("\nOPS finished OK")


if __name__ == "__main__":
    main()
