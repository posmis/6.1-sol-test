"""Reproduce DXF -> both native cell types -> converged flow and VTU export."""
import argparse
from pathlib import Path
import json
import time
from dataclasses import asdict
from .cad import load_dxf
from .meshing import MeshSettings, generate
from .solver import Config, Solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=Path(__file__).resolve().parent.parent/"examples")
    parser.add_argument("--output", type=Path, default=Path("results/custom-mesh-validation.json"))
    parser.add_argument("--with-obstacle", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for name in (["channel", "channel_with_obstacle"] if args.with_obstacle else ["channel"]):
        for cell_type in ("triangle", "quad"):
            geometry = load_dxf(args.examples/f"{name}.dxf")
            settings = MeshSettings(cell_type=cell_type, target_cells=100 if name == "channel" else 300)
            project = generate(geometry, settings)
            path = args.output.parent/f"{name}_{cell_type}.flowmesh"
            project.save(path)
            config = Config(scenario="custom", mesh_path=str(path.resolve()), mode="steady", max_iterations=1000,
                            nx=256, ny=max(8, min(256, round(256*(geometry.bounds[3]-geometry.bounds[1])/(geometry.bounds[2]-geometry.bounds[0])))))
            start = time.perf_counter()
            result = Solver(config).run()
            result.export_native(args.output.parent/f"{name}_{cell_type}.vtu")
            result.save(args.output.parent/f"{name}_{cell_type}.npz")
            cases.append(dict(source=name+".dxf", settings=asdict(settings), statistics=project.statistics,
                seconds=time.perf_counter()-start, diagnostics=result.diagnostics,
                pressure_reference_validation="Not performed; these checks establish numerical workflow and conservation"))
            print(f"{name} {cell_type}: {len(project.cells)} cells, converged={result.diagnostics['converged']}, "
                  f"{result.diagnostics['iterations']} iterations", flush=True)
    args.output.write_text(json.dumps(dict(cases=cases), indent=2)+"\n")
    return 0 if all(c["diagnostics"]["converged"] and c["diagnostics"]["mass_imbalance"] < 1e-6 for c in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
