"""Compute the ready elliptic profile, save forces and a reproducible angle polar."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from .airfoil import export_csv
from .scenarios import PROFILE_SCENARIO
from .solver import Config, Solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=800)
    parser.add_argument("--angle", type=float, default=4.)
    parser.add_argument("--angles", type=float, nargs="+", default=[-4., 0., 4., 8.])
    parser.add_argument("--output", type=Path, default=Path("results/profile-validation.json"))
    args = parser.parse_args()
    config = Config(scenario=PROFILE_SCENARIO, mode="steady", profile_angle=args.angle,
        profile_cells=args.cells, profile_sweep=tuple(args.angles), speed=20., rho=1.225,
        nu=1.5e-5, intensity=.01, nx=320, ny=160)
    config.validate()
    started = time.perf_counter()
    result = Solver(config).run(progress=lambda i, n, a:
        print(f"alpha={a:+g} deg, iteration={i}/{n}", flush=True) if i % 100 == 0 else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output.parent/"profile.npz")
    export_csv(result, args.output.parent/"profile-history.csv")
    export_csv(result, args.output.parent/"profile-polar.csv", polar=True)
    aero = result.diagnostics["aerodynamics"]
    report = dict(config=asdict(config), seconds=time.perf_counter()-started,
        mesh_cells=result.diagnostics["mesh_cells"], converged=result.diagnostics["converged"],
        residuals=result.diagnostics["residuals"], mass_imbalance=result.diagnostics["mass_imbalance"],
        aerodynamics=aero,
        validation_scope="Numerical convergence and force-integration checks; no experimental or mesh-convergence accuracy claim")
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Saved {args.output}", flush=True)
    return 0 if all(p["converged"] and p["cl_cd"] is not None for p in aero["polar"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
