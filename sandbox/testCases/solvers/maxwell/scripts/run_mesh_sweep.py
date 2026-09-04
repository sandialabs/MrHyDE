#!/usr/bin/env python3
"""
Mesh/solver sweep: for each (nsteps, ref), update input_common_maxwell_hex3d.yaml (NX,NY,NZ)
and input_solver_common.yaml (number of steps) in the given subdir, run runtests.py -d <subdir>
from regression, then copy each test's mrhyde.log to mrhyde_t{i}_r{j}.log.
Usage: python run_mesh_sweep.py <dir>  e.g.  python run_mesh_sweep.py preconditioners
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MRHYDE = SCRIPT_DIR.parents[4]  # scripts -> maxwell -> solvers -> testCases -> sandbox -> MrHyDE
REGRESSION = MRHYDE / "regression"
DEFAULT_RESOLVER = SCRIPT_DIR / "resolve_yaml_imports.py"
DEFAULT_MRHYDE_EXE = MRHYDE.parent / "mrhyde.exe"


def replace_yaml_key(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*).*$", re.MULTILINE)
    return pat.sub(rf"\g<1>{value}", text)


def update_mesh_yaml(path: Path, nx: int, ny: int, nz: int) -> None:
    text = path.read_text()
    text = replace_yaml_key(text, "NX", str(nx))
    text = replace_yaml_key(text, "NY", str(ny))
    text = replace_yaml_key(text, "NZ", str(nz))
    path.write_text(text)


def update_solver_yaml(path: Path, nsteps: int) -> None:
    text = path.read_text()
    text = replace_yaml_key(text, "number of steps", str(nsteps))
    path.write_text(text)


def update_final_time_yaml(path: Path, final_time: float) -> None:
    text = path.read_text()
    text = replace_yaml_key(text, "final time", str(final_time))
    path.write_text(text)


def get_test_dirs(root: Path) -> list[Path]:
    return [d for d in root.iterdir() if d.is_dir() and (d / "mrhyde.tst").exists()]


def ensure_mrhyde_symlink(mrhyde_exe: Path) -> int:
    """Symlink regression/mrhyde to the given executable. Returns 0 on success."""
    src = mrhyde_exe.resolve()
    dest = REGRESSION / "mrhyde"
    if not src.is_file():
        print(f"mrhyde executable not found: {src}", file=sys.stderr)
        return 1
    if dest.exists():
        dest.unlink()
    dest.symlink_to(src)
    return 0


def run_runtests(root: Path) -> int:
    runtests_py = REGRESSION / "scripts" / "runtests.py"
    test_dir = os.path.relpath(root, REGRESSION)
    cmd = [sys.executable, str(runtests_py), "-d", test_dir]
    return subprocess.call(cmd, cwd=REGRESSION)


def copy_logs(i: int, j: int, root: Path) -> None:
    for d in get_test_dirs(root):
        log = d / "mrhyde.log"
        if log.exists():
            dest = d / f"mrhyde_t{i}_r{j}.log"
            shutil.copy2(log, dest)


def resolve_import_inputs(
    root: Path,
    resolver_script: Path,
    short_name: str = "input.short.yaml",
    output_name: str = "input.yaml",
) -> int:
    if not resolver_script.is_file():
        print(f"Resolver script not found: {resolver_script}", file=sys.stderr)
        return 1

    short_files = sorted(root.rglob(short_name))
    if not short_files:
        print(f"No {short_name} files found under {root}")
        return 0

    for short_file in short_files:
        out_file = short_file.with_name(output_name)
        cmd = [sys.executable, str(resolver_script), str(short_file), "-o", str(out_file)]
        ret = subprocess.call(cmd, cwd=SCRIPT_DIR)
        if ret != 0:
            print(f"Resolver failed ({ret}) for {short_file}", file=sys.stderr)
            return ret

    print(f"Resolved {len(short_files)} files from {short_name} -> {output_name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", type=Path, help="Subdirectory under maxwell (e.g. preconditioners)")
    ap.add_argument(
        "--nsteps",
        default="10, 20, 40, 80", #",20,40,80,120", #",40,80", #" ,300,500,600",
        help="Comma-separated number of steps",
    )
    ap.add_argument(
        "--ref",
        default="1,2,3", #",3",
        help="Comma-separated refinement levels; NX=NY=6*2^ref, NZ=3*2^ref (default: 0,1,2)",
    )
    ap.add_argument(
        "--final-time",
        type=float,
        default=2.0,
        help="Final time T for transient solve; if set, updates input_solver_common.yaml and is recorded in sweep_meta.json",
    )
    ap.add_argument(
        "--resolve-imports",
        action="store_true",
        help="Resolve input.short.yaml imports to input.yaml before running tests",
    )
    ap.add_argument(
        "--resolver-script",
        type=Path,
        default=DEFAULT_RESOLVER,
        help=f"Path to resolve_yaml_imports.py (default: {DEFAULT_RESOLVER})",
    )
    ap.add_argument(
        "--short-input-name",
        default="input.short.yaml",
        help="Source short input filename to resolve",
    )
    ap.add_argument(
        "--resolved-input-name",
        default="input.yaml",
        help="Resolved output filename produced by import resolver",
    )
    ap.add_argument(
        "--mrhyde-exe",
        type=Path,
        default=DEFAULT_MRHYDE_EXE,
        help="Path to mrhyde executable; symlinked to regression/mrhyde before run (default: %(default)s)",
    )
    args = ap.parse_args()
    root = (SCRIPT_DIR.parent / args.dir).resolve()

    ret = ensure_mrhyde_symlink(args.mrhyde_exe)
    if ret != 0:
        return ret
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    nsteps_list = [int(x.strip()) for x in args.nsteps.split(",")]
    ref_list = [int(x.strip()) for x in args.ref.split(",")]

    mesh_yaml = root / "input_common_maxwell_hex3d.yaml"
    solver_yaml = root / "input_solver_common.yaml"
    if not mesh_yaml.exists() or not solver_yaml.exists():
        print(f"Missing YAML: {mesh_yaml} or {solver_yaml}", file=sys.stderr)
        return 1

    if args.resolve_imports:
        ret = resolve_import_inputs(
            root=root,
            resolver_script=args.resolver_script.resolve(),
            short_name=args.short_input_name,
            output_name=args.resolved_input_name,
        )
        if ret != 0:
            return ret

    # Determine final time: prefer CLI override, otherwise read from solver YAML if present.
    final_time: float | None = None
    if args.final_time is not None:
        final_time = args.final_time
    else:
        text = solver_yaml.read_text()
        m = re.search(r"final time\s*:\s*([0-9.eE+-]+)", text)
        if m:
            try:
                final_time = float(m.group(1))
            except ValueError:
                final_time = None

    if final_time is not None:
        update_final_time_yaml(solver_yaml, final_time)

    # Write sweep metadata so post-processing scripts can label plots with dt and (Nx,Ny,Nz).
    sweep_meta_path = root / "sweep_meta.json"
    meta: dict[str, object] = {
        "final_time": final_time,
        "time": [],
        "spatial": [],
    }
    time_entries: list[dict[str, object]] = []
    for i, nsteps in enumerate(nsteps_list):
        entry: dict[str, object] = {"index": i, "nsteps": nsteps}
        if final_time is not None and nsteps != 0:
            entry["dt"] = final_time / nsteps
        time_entries.append(entry)
    spatial_entries: list[dict[str, object]] = []
    for j, ref in enumerate(ref_list):
        nx = ny = 8 * (2**ref)
        nz = 4 * (2**ref)
        spatial_entries.append(
            {
                "index": j,
                "ref": ref,
                "nx": nx,
                "ny": ny,
                "nz": nz,
            }
        )
    meta["time"] = time_entries
    meta["spatial"] = spatial_entries
    try:
        sweep_meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Wrote sweep metadata to {sweep_meta_path}")
    except OSError as exc:
        print(f"Warning: failed to write sweep_meta.json: {exc}", file=sys.stderr)

    for i, nsteps in enumerate(nsteps_list):
        for j, ref in enumerate(ref_list):
            nx = ny = 8 * (2**ref)
            nz = 4 * (2**ref)
            print(f"nsteps={nsteps} (i={i}) ref={ref} (j={j}) nx={nx} ny={ny} nz={nz}")
            update_mesh_yaml(mesh_yaml, nx, ny, nz)
            update_solver_yaml(solver_yaml, nsteps)
            ret = run_runtests(root)
            if ret != 0:
                print(f"runtests exited {ret}", file=sys.stderr)
            copy_logs(i, j, root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
