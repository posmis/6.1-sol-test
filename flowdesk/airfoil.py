"""A ready asymmetric two-half-ellipse profile, native mesh and force integration."""
from dataclasses import replace
import csv
from pathlib import Path
import tempfile
import numpy as np
from .cad import Geometry
from .meshing import MeshSettings, generate


def profile_geometry(chord=.1, angle=4., upper=.10, lower=.06, segments=48):
    theta = np.linspace(0, np.pi, segments//2+1)
    x = .5*chord*(1-np.cos(theta))
    top = np.column_stack([x, chord*upper*np.sin(theta)])
    bottom = np.column_stack([x[-2:0:-1], -chord*lower*np.sin(theta[-2:0:-1])])
    body = np.vstack([top, bottom])
    a = np.radians(angle)
    rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    # Positive alpha pitches the nose up, with a horizontal flow from left to right.
    body = (body-[.25*chord, 0.])@rotation.T+[2.25*chord, 2*chord]
    outer = [[0., 0.], [8*chord, 0.], [8*chord, 4*chord], [0., 4*chord]]
    geometry = Geometry([outer, body.tolist()], [["open", "outlet", "open", "inlet"], ["wall"]*len(body)],
                        f"Asymmetric elliptic profile, alpha={angle:g} deg")
    geometry.validate(boundaries=True)
    return geometry


def coefficients(pressure, normals, lengths, wall_velocity, distances, rho, nu, speed, chord, turbulence=True):
    """Force on the body per unit span. Normals point from fluid into the body.

    Pressure force = integral(p*n_fluid ds), shear force = integral(rho*u_tau²*t ds).
    The shear model matches the solver's equilibrium wall function.
    """
    pressure = np.asarray(pressure)
    normals, lengths = np.asarray(normals), np.asarray(lengths)
    velocity, y = np.asarray(wall_velocity), np.maximum(distances, 1e-12)
    tangent = velocity-normals*np.sum(velocity*normals, axis=0)
    magnitude = np.linalg.norm(tangent, axis=0)
    ut = np.maximum(np.sqrt(nu*magnitude/y), 1e-9)
    if turbulence:
        for _ in range(8):
            log = np.log(np.maximum(9.8*y*ut/nu, 1.01))
            ut = np.maximum(ut-(ut*log/.41-magnitude)/((log+1)/.41), 1e-9)
        ut = np.where(y*ut/nu > 11.06, ut, np.sqrt(nu*magnitude/y))
    pressure_force = np.sum(normals*pressure*lengths, axis=1)
    shear_force = np.sum(rho*ut**2*lengths*tangent/np.maximum(magnitude, 1e-12), axis=1)
    force = pressure_force+shear_force
    normalization = .5*rho*speed**2*chord
    cd, cl = float(force[0]/normalization), float(force[1]/normalization)
    return dict(cl=cl, cd=cd, cl_cd=cl/cd if cd > 1e-8 else None,
                drag_n_per_m=float(force[0]), lift_n_per_m=float(force[1]),
                pressure_drag_n_per_m=float(pressure_force[0]), pressure_lift_n_per_m=float(pressure_force[1]),
                shear_drag_n_per_m=float(shear_force[0]), shear_lift_n_per_m=float(shear_force[1]))


def solve_profile(config, progress=None, cancel=None):
    from .unstructured import solve
    c = config
    angles = [c.profile_angle]
    for angle in c.profile_sweep:
        if not any(np.isclose(angle, a) for a in angles):
            angles.append(float(angle))
    points, base = [], None
    with tempfile.TemporaryDirectory(prefix="flowdesk-profile-") as directory:
        for index, angle in enumerate(angles):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Calculation cancelled")
            if progress:
                progress(0, c.max_iterations if c.mode == "steady" else c.frame_count-1, angle if c.mode == "steady" else 0.)
            geometry = profile_geometry(c.profile_chord, angle)
            chord = c.profile_chord
            h = chord*np.sqrt(4/c.profile_cells)
            zone = (1.7*chord, 1.6*chord, 3.4*chord, 2.4*chord, h)
            mesh = generate(geometry, MeshSettings(target_cells=c.profile_cells, zones=(zone,)))
            mesh.statistics["profile"] = dict(chord=chord, angle=angle, upper=.10, lower=.06)
            path = Path(directory)/f"profile-{index}.flowmesh"
            mesh.save(path)
            one = replace(c, profile_angle=angle, mesh_path=str(path), profile_sweep=())
            result = solve(one, (lambda i, n, t: progress(i, n, angle if c.mode == "steady" else t)) if progress else None, cancel)
            aero = result.diagnostics["aerodynamics"]
            points.append(dict(angle=angle, **aero["final"], converged=result.diagnostics["converged"],
                               iterations=result.diagnostics["iterations"],
                               max_residual=max(result.diagnostics["residuals"].values())))
            if base is None:
                base = result
    # Temporary meshing files are never required to reopen or plot the result.
    base.config = replace(base.config, mesh_path=None, profile_sweep=c.profile_sweep)
    base.diagnostics["aerodynamics"]["polar"] = sorted(points, key=lambda p: p["angle"]) if c.mode == "steady" else []
    base.diagnostics["aerodynamics"]["polar_requested"] = bool(c.profile_sweep)
    return base


def export_csv(result, path, polar=False):
    aero = result.diagnostics["aerodynamics"]
    rows = aero["polar"] if polar else aero["history"]
    columns = ["angle", "cl", "cd", "cl_cd", "converged", "iterations"] if polar else [
        "iteration", "cl", "cd", "cl_cd", "drag_n_per_m", "lift_n_per_m",
        "pressure_drag_n_per_m", "pressure_lift_n_per_m", "shear_drag_n_per_m", "shear_lift_n_per_m"]
    if not polar and result.config.mode == "transient":
        columns.insert(0, "time")
    with open(path, "w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
