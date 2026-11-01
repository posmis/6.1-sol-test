"""Reproducible channel-pressure check against Dean (1978), not a calibration."""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from .solver import Config, Solver


def channel_check(result):
    c = result.config
    if c.scenario != "channel":
        raise ValueError("Dean comparison applies only to the straight channel")
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
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Saved {output}", flush=True)
    return 0 if all(row["pressure_check"]["within_10_percent"] for row in report["channel"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
