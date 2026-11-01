"""Compare a source archive of the old engine with current transient and SIMPLE."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import numpy as np

CHILD = r'''
import sys,json,time,numpy as np
sys.path.insert(0, sys.argv[1])
from flowdesk.solver import Config,Solver
settings=json.loads(sys.argv[2]); repeats=int(sys.argv[3])
warm=dict(nx=16,ny=8,duration=1e-5,frames=2)
if 'mode' in settings:
 warm.update(mode=settings['mode'],max_iterations=10,coarse_start=False)
Solver(Config(**warm)).run()
timings=[]
for _ in range(repeats):
 start=time.perf_counter();result=Solver(Config(**settings)).run();timings.append(time.perf_counter()-start)
np.savez(sys.argv[4],u=result.u[-1],v=result.v[-1],pressure=result.pressure[-1])
print(json.dumps(dict(seconds=timings,median_seconds=float(np.median(timings)),diagnostics=result.diagnostics)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/performance.json"))
    args = parser.parse_args()
    baseline = args.baseline_source.resolve()
    current = Path(__file__).resolve().parent.parent
    if not (baseline/"flowdesk/solver.py").is_file() or args.repeats < 1:
        parser.error("Supply an extracted source archive and at least one repeat")
    report = dict(baseline_commit=args.baseline_commit, python=platform.python_version(),
                  repeats=args.repeats, jit_warmup_excluded=True, cases=[])
    with tempfile.TemporaryDirectory(prefix="flowdesk-bench-") as tmp:
        for scenario in ("channel", "contraction", "baffles"):
            settings = dict(scenario=scenario, nx=96, ny=24, duration=1.2, frames=80)
            row = dict(settings=settings)
            for mode in ("baseline", "transient", "steady"):
                source = baseline if mode == "baseline" else current
                config = settings if mode == "baseline" else dict(settings, mode=mode)
                path = Path(tmp)/f"{mode}.npz"
                completed = subprocess.run([sys.executable, "-c", CHILD, str(source),
                    json.dumps(config), str(args.repeats), str(path)], cwd=source,
                    check=True, capture_output=True, text=True)
                row[mode] = json.loads(completed.stdout)
                if mode != "baseline":
                    row[mode]["speedup"] = row["baseline"]["median_seconds"]/row[mode]["median_seconds"]
                    with np.load(Path(tmp)/"baseline.npz") as old, np.load(path) as new:
                        row[mode]["final_field_relative_l2_difference"] = {
                            key: float(np.linalg.norm(new[key]-old[key])/max(np.linalg.norm(old[key]), 1e-12))
                            for key in ("u", "v", "pressure")}
                print(f"{scenario} {mode}: {row[mode]['median_seconds']:.3f}s", flush=True)
            report["cases"].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    return 0 if all(row["steady"]["diagnostics"]["converged"] for row in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
