"""Reproducible channel-pressure check against Dean (1978), not a calibration."""
import argparse
import json
from pathlib import Path
import platform
import time
from dataclasses import asdict
import numpy as np
import scipy
from .solver import Config, Solver
from .scenarios import SCENARIOS


def channel_check(result):
    c = result.config
    if c.scenario != "channel" or c.top != "wall" or c.bottom != "wall":
        raise ValueError("Dean comparison applies only to the straight channel with two walls")
    # Same physical stations (nearest cell centres) on every mesh, beyond 9H.
    a, b = round(.65*c.nx-.5), round(.9*c.nx-.5)
    spacing = (b-a)*c.length/c.nx
    p = result.pressure[-1].mean(axis=0)
    measured = float(p[a]-p[b])
    re = c.speed*c.height/c.nu
    cf = .073*re**(-.25)  # skin friction coefficient: tau_w/(rho*Ub^2/2)
    reference = cf*c.rho*c.speed**2/c.height*spacing
    error = abs(measured/reference-1)
    return dict(stations_m=[(a+.5)*c.length/c.nx, (b+.5)*c.length/c.nx],
                pressure_drop_pa=measured, reference_pa=reference,
                relative_error=error, within_10_percent=bool(error <= .1),
                comparison="Dean smooth fully-developed plane-channel correlation, Cf=0.073 Re_H^-0.25",
                scope="Selected downstream stations only; excludes entrance losses. Not experimental validation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/validation.json")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--extended", action="store_true", help="Also run new geometries and large-grid single-step checks")
    parser.add_argument("--steady", action="store_true", help="Validate converged SIMPLE solutions for every scenario")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  channel=[], step=None)
    for nx, ny in ((64, 16), (96, 24), (144, 36)):
        c = Config(nx=nx, ny=ny)
        start = time.perf_counter()
        result = Solver(c).run()
        row = dict(grid=[nx, ny], seconds=time.perf_counter()-start,
                   diagnostics=result.diagnostics, pressure_check=channel_check(result))
        report["channel"].append(row)
        print(f"channel {nx}x{ny}: {row['seconds']:.2f}s, pressure error "
              f"{row['pressure_check']['relative_error']:.2%}", flush=True)
        if args.save_frames and nx == 96:
            result.save(output.parent/"channel.npz")
    start = time.perf_counter()
    result = Solver(Config(scenario="step")).run()
    report["step"] = dict(seconds=time.perf_counter()-start, diagnostics=result.diagnostics,
                          minimum_u=float(result.u[-1][result.fluid].min()),
                          pressure_accuracy="Not validated against experimental data")
    if args.save_frames:
        result.save(output.parent/"step.npz")
    if args.extended:
        report["additional_scenarios"] = []
        report["large_grid_single_step"] = []
        configs = [Config(scenario=name) for name in ("contraction", "expansion", "cylinder", "baffles")]
        large = [Config(nx=1024, ny=1024, duration=1e-6, frames=2)]
        large += [Config(scenario=name, nx=1024, ny=256, duration=1e-6, frames=2)
                  for name in ("step", "contraction", "expansion", "cylinder", "baffles")]
        for c in configs+large:
            start = time.perf_counter()
            result = Solver(c).run()
            row = dict(config=asdict(c), seconds=time.perf_counter()-start,
                       diagnostics=result.diagnostics,
                       scope="Numerical smoke check only; pressure accuracy not validated")
            key = "large_grid_single_step" if c.frames == 2 else "additional_scenarios"
            report[key].append(row)
            print(f"{c.scenario} {c.nx}x{c.ny}: {row['seconds']:.2f}s ({c.duration:g}s simulated)", flush=True)
    steady_ok = True
    if args.steady:
        report["steady"] = []
        for name in SCENARIOS:
            c = Config(mode="steady", scenario=name)
            start = time.perf_counter()
            result = Solver(c).run()
            row = dict(config=asdict(c), seconds=time.perf_counter()-start, diagnostics=result.diagnostics)
            if name == "channel":
                row["pressure_check"] = channel_check(result)
                steady_ok &= row["pressure_check"]["within_10_percent"]
            steady_ok &= result.diagnostics["converged"]
            report["steady"].append(row)
            print(f"SIMPLE {name}: converged={result.diagnostics['converged']}, "
                  f"iterations={result.diagnostics['iterations']}, {row['seconds']:.2f}s", flush=True)
            if args.save_frames:
                result.save(output.parent/f"{name}_steady.npz")
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Saved {output}", flush=True)
    return 0 if steady_ok and all(row["pressure_check"]["within_10_percent"] for row in report["channel"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
