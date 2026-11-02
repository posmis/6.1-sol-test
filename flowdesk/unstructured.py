"""Collocated finite-volume SIMPLE/Rhie–Chow on actual triangle/quad cells.

FiPy 4.0.3 supplies geometric operators and nonorthogonal diffusion assembly;
FlowDesk supplies SIMPLE, wall functions, k-epsilon and sparse linear solves.
"""
import os
import tempfile
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.sparse import diags
from threadpoolctl import threadpool_limits
from .meshing import MeshProject
from .steady import LinearSolves
from .scenarios import PROFILE_SCENARIO


def fipy_mesh(project):
    from fipy.meshes.mesh2D import Mesh2D
    faces, cells, tags = project.connectivity()
    mesh = Mesh2D(vertexCoords=project.points.T, faceVertexIDs=faces.T,
                  cellFaceIDs=np.ma.masked_less(cells.T, 0))
    return mesh, tags


def system(equation, variable, dt=None):
    """Pinned FiPy assembly adapter; no solve or hidden replacement of the mesh."""
    from fipy.solvers.scipy import LinearLUSolver
    prepared = equation._prepareLinearSystem(variable, LinearLUSolver(), (), dt)
    return prepared.matrix.matrix.tocsr(), np.asarray(prepared.RHSvector).copy()


def implicit_solve(linear, equation, var, key, alpha, scale, fixed=None, dt=None):
    a, b = system(equation, var, dt)
    if fixed is not None:
        mask, values = fixed
        a = a.tolil()
        for i in np.flatnonzero(mask):
            a.rows[i], a.data[i] = [int(i)], [1.]
        a = a.tocsr()
        b[mask] = values[mask]
    else:
        mask = np.zeros(len(b), bool)
    old = np.asarray(var.value).copy()
    # Volume-normalized equation defect, excluding prescribed wall k/epsilon cells.
    volume = np.asarray(var.mesh.cellVolumes)
    active = ~mask
    residual = float(np.linalg.norm(((a@old-b)/volume)[active]) /
        max(np.linalg.norm((b/volume)[active]), scale*np.sqrt(active.sum()), 1e-14))
    diagonal = a.diagonal()
    extra = np.where(mask, 0., diagonal*(1/alpha-1))
    answer = linear.solve(a+diags(extra), b+extra*old, old, "native_"+key)
    var.setValue(answer)
    return residual, diagonal+extra


def raster_sampler(project, nx, ny):
    # Piecewise-constant finite-volume display: every pixel refers to a real cell.
    from matplotlib.tri import Triangulation
    triangles, owners = [], []
    for i, nodes in enumerate(project.cells):
        triangles.append(nodes[:3])
        owners.append(i)
        if nodes[3] >= 0:
            triangles.append(nodes[[0, 2, 3]])
            owners.append(i)
    triangulation = Triangulation(project.points[:, 0], project.points[:, 1], triangles=np.array(triangles))
    xmin, ymin, xmax, ymax = project.geometry.bounds
    x = xmin+(np.arange(nx)+.5)*(xmax-xmin)/nx
    y = ymin+(np.arange(ny)+.5)*(ymax-ymin)/ny
    xx, yy = np.meshgrid(x, y)
    triangle_ids = triangulation.get_trifinder()(xx, yy)
    fluid = triangle_ids >= 0
    cell_ids = np.array(owners)[np.maximum(triangle_ids, 0)]
    return cell_ids, fluid


def rasterize(project, values, nx, ny):
    cell_ids, fluid = raster_sampler(project, nx, ny)
    return [np.repeat(np.where(fluid, q[cell_ids], 0)[None].astype(np.float32), 2, axis=0)
            for q in values], fluid


@threadpool_limits.wrap(limits=1, user_api="blas")
def solve(config, progress=None, cancel=None):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"flowdesk-mpl"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from fipy import CellVariable, FaceVariable, DiffusionTerm, UpwindConvectionTerm, ImplicitSourceTerm, TransientTerm
    from .fipy_compat import NamedDiffusionTermCorrection as DiffusionTermCorrection
    from fipy.variables.faceGradVariable import _FaceGradVariable
    from .solver import Result
    start = time.perf_counter()
    project = MeshProject.load(config.mesh_path)
    xmin, ymin, xmax, ymax = project.geometry.bounds
    c = replace(config, length=xmax-xmin, height=ymax-ymin)
    mesh, tags = fipy_mesh(project)
    nc, nf = mesh.numberOfCells, mesh.numberOfFaces
    area = np.asarray(mesh.cellVolumes)
    face_area = np.asarray(mesh._faceAreas)
    normals = np.asarray(mesh.faceNormals)
    owner = np.asarray(mesh.faceCellIDs[0], int)
    neighbour = np.ma.filled(mesh.faceCellIDs[1], -1).astype(int)
    interior = neighbour >= 0
    inlet, wall = tags == "inlet", tags == "wall"
    opened = (tags == "outlet") | (tags == "open")
    exterior = ~interior
    kin = 1.5*(c.speed*c.intensity)**2
    reference_length = c.profile_chord if c.scenario == PROFILE_SCENARIO else min(c.length, c.height)
    ein = .09**.75*kin**1.5/(.07*reference_length)
    p, pc = CellVariable(mesh=mesh, value=0.), CellVariable(mesh=mesh, value=0.)
    transient = c.mode == "transient"
    u = CellVariable(mesh=mesh, value=0. if transient and c.initial_condition == "rest" else float(c.speed), hasOld=transient)
    v = CellVariable(mesh=mesh, value=0., hasOld=transient)
    k = CellVariable(mesh=mesh, value=kin, hasOld=transient)
    e = CellVariable(mesh=mesh, value=ein, hasOld=transient)
    velocity = FaceVariable(mesh=mesh, rank=1, value=0.)
    inlet_velocity = -normals*c.speed
    u.constrain(inlet_velocity[0, inlet], inlet)
    v.constrain(inlet_velocity[1, inlet], inlet)
    k.constrain(kin, inlet)
    e.constrain(ein, inlet)
    # FiPy's unconstrained exterior has zero *total* flux. Explicit zero-gradient
    # outlet constraints are required to retain advective outflow in transport.
    for variable in (u, v, k, e):
        variable.faceGrad.constrain(0., opened)
    p.constrain(0., opened)
    pc.constrain(0., opened)
    nu = CellVariable(mesh=mesh, value=c.nu)
    rx, ry = CellVariable(mesh=mesh, value=0.), CellVariable(mesh=mesh, value=0.)
    sx, sy = CellVariable(mesh=mesh, value=0.), CellVariable(mesh=mesh, value=0.)
    ksink, esink = CellVariable(mesh=mesh, value=0.), CellVariable(mesh=mesh, value=0.)
    production, eproduction = CellVariable(mesh=mesh, value=0.), CellVariable(mesh=mesh, value=0.)
    nudiffe = CellVariable(mesh=mesh, value=c.nu)
    equations = [UpwindConvectionTerm(coeff=velocity)-DiffusionTermCorrection(coeff=nu.arithmeticFaceValue)
                  +ImplicitSourceTerm(coeff=r)-source+p.grad[i]
                 for i, r, source in ((0, rx, sx), (1, ry, sy))]
    keq = UpwindConvectionTerm(coeff=velocity)-DiffusionTermCorrection(coeff=nu.arithmeticFaceValue)+ImplicitSourceTerm(coeff=ksink)-production
    eeq = UpwindConvectionTerm(coeff=velocity)-DiffusionTermCorrection(coeff=nudiffe.arithmeticFaceValue)+ImplicitSourceTerm(coeff=esink)-eproduction
    if transient:
        equations = [TransientTerm()+eq for eq in equations]
        keq, eeq = TransientTerm()+keq, TransientTerm()+eeq
    d = CellVariable(mesh=mesh, value=1.)
    diffusion = DiffusionTerm(coeff=d.arithmeticFaceValue)
    pressure_eq = -diffusion+velocity.divergence
    wf = np.flatnonzero(wall)
    wc = owner[wf]
    distances = np.abs(np.sum((np.asarray(mesh.cellCenters)[:, wc]-np.asarray(mesh.faceCenters)[:, wf])*normals[:, wf], axis=0))
    distances = np.maximum(distances, 1e-12)
    wall_cells = np.bincount(wc, minlength=nc) > 0
    wall_weight = np.bincount(wc, weights=face_area[wf], minlength=nc)
    linear = LinearSolves(cancel)
    history, consecutive, converged = [], 0, False
    aero_history = []
    # Inward normal velocity at all inlet faces, not an assumed horizontal CAD orientation.
    velocity[0] = u.arithmeticFaceValue
    velocity[1] = v.arithmeticFaceValue
    velocity[:, wall] = 0.
    velocity[:, inlet] = inlet_velocity[:, inlet]
    if transient and c.initial_condition == "uniform":
        # Project the initial free stream onto impermeability and discrete
        # continuity. This is an initial condition, not a stationary RANS solve.
        # Unit d makes pc a velocity potential; do not treat it as pressure.
        ap, bp = system(pressure_eq, pc)
        pc.setValue(linear.solve(ap, bp, np.zeros(nc), "native_initial_p", positive=True))
        geom = np.asarray(diffusion._getGeomCoeff(pc))
        conductance = geom[0] if geom.ndim > 1 else geom
        difference = np.zeros(nf)
        difference[interior] = pc.value[neighbour[interior]]-pc.value[owner[interior]]
        difference[opened] = -pc.value[owner[opened]]
        velocity.setValue(velocity.value-normals*(conductance/face_area*difference))
        u.setValue(u.value-pc.grad[0].value)
        v.setValue(v.value-pc.grad[1].value)
        pc.setValue(0.)
    initial_divergence = float(np.max(abs(velocity.divergence.value)))
    times = np.linspace(0., c.duration, c.frame_count)
    cell_ids, fluid = raster_sampler(project, c.nx, c.ny)
    frames = [np.empty((len(times), c.ny, c.nx), np.float32) for _ in range(5)]
    def snapshot(index):
        ref = np.average(k.value[owner[opened]], weights=face_area[opened])
        static = (p.value-2*k.value/3+2*ref/3)*c.rho
        for field, value in zip(frames, (u.value, v.value, static, k.value, e.value)):
            field[index] = np.where(fluid, np.asarray(value)[cell_ids], 0.)
    snapshot(0)
    if transient:
        # Backward Euler, bounded upwind transport; time resolution is independent
        # of SIMPLE iterations. Subdivide long output intervals on convective scale.
        max_dt = .2*reference_length/c.speed
        if c.time_step:
            max_dt = min(max_dt, c.time_step)
        substeps = max(1, int(np.ceil((times[1]-times[0])/max_dt)))
        timeline = [(j, t, (times[j]-times[j-1])/substeps, sub == substeps)
                    for j in range(1, len(times)) for sub in range(1, substeps+1)
                    for t in [times[j-1]+sub*(times[j]-times[j-1])/substeps]]
    else:
        timeline = [(1, 0., None, True)]
    total_iterations = 0
    step_history = []
    relaxation = .7 if transient else .5
    for frame_index, physical_time, dt, save_frame in timeline:
        if transient:
            for variable in (u, v, k, e):
                variable.updateOld()
        consecutive, converged = 0, False
        for iteration in range(1, c.max_iterations+1):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Calculation cancelled")
            nut = np.minimum(.09*np.asarray(k.value)**2/np.maximum(e.value, 1e-12), .1*c.speed*reference_length)
            nu.setValue(c.nu+nut)
            nudiffe.setValue(c.nu+nut/1.3)
            uv = np.vstack([u.value[wc], v.value[wc]])
            nw = normals[:, wf]
            tangent = uv-nw*np.sum(nw*uv, axis=0)
            speed = np.linalg.norm(tangent, axis=0)
            ut = np.maximum(np.sqrt(c.nu*speed/distances), 1e-9)
            for _ in range(8):
                log = np.log(np.maximum(9.8*distances*ut/c.nu, 1.01))
                ut = np.maximum(ut-(ut*log/.41-speed)/((log+1)/.41), 1e-9)
            ut = np.where(distances*ut/c.nu > 11.06, ut, np.sqrt(c.nu*speed/distances))
            gamma = ut**2/np.maximum(speed, 1e-12)*face_area[wf]/area[wc]
            dragx = np.bincount(wc, weights=gamma*(1-nw[0]**2), minlength=nc)
            dragy = np.bincount(wc, weights=gamma*(1-nw[1]**2), minlength=nc)
            cross = np.bincount(wc, weights=gamma*nw[0]*nw[1], minlength=nc)
            rx.setValue(dragx)
            ry.setValue(dragy)
            ug, vg, ng = np.asarray(u.grad), np.asarray(v.grad), np.asarray(nu.grad)
            sx.setValue(cross*v.value+ng[0]*ug[0]+ng[1]*vg[0])
            sy.setValue(cross*u.value+ng[0]*ug[1]+ng[1]*vg[1])
            residual_u, au = implicit_solve(linear, equations[0], u, "u", relaxation, c.speed**2/c.length, dt=dt)
            residual_v, av = implicit_solve(linear, equations[1], v, "v", relaxation, c.speed**2/c.length, dt=dt)
            d.setValue(area/np.maximum((au+av)*.5, 1e-20))
            # Rhie–Chow prevents pressure checkerboarding on collocated polygon cells.
            face_gradient = _FaceGradVariable(p)
            for i, var in enumerate((u, v)):
                velocity[i] = var.arithmeticFaceValue+d.arithmeticFaceValue*(p.grad[i].arithmeticFaceValue-face_gradient[i])
            velocity[:, wall] = 0.
            velocity[:, inlet] = inlet_velocity[:, inlet]
            pc.setValue(0.)
            ap, bp = system(pressure_eq, pc)
            pc.setValue(linear.solve(ap, bp, np.zeros(nc), "native_p", positive=True))
            # Correct conservative face fluxes with exactly the conductance used in A_p.
            geom = np.asarray(diffusion._getGeomCoeff(pc))
            conductance = geom[0] if geom.ndim > 1 else geom
            difference = np.zeros(nf)
            difference[interior] = pc.value[neighbour[interior]]-pc.value[owner[interior]]
            difference[opened] = -pc.value[owner[opened]]
            velocity.setValue(velocity.value-normals*(conductance/face_area*difference))
            p.setValue(p.value+(.5 if transient else .25)*pc.value)
            u.setValue(u.value-d.value*pc.grad[0].value)
            v.setValue(v.value-d.value*pc.grad[1].value)
            kw = np.bincount(wc, weights=face_area[wf]*ut**2/.3, minlength=nc)/np.maximum(wall_weight, 1e-20)
            ew = np.bincount(wc, weights=face_area[wf]*ut**3/(.41*distances), minlength=nc)/np.maximum(wall_weight, 1e-20)
            prod = np.minimum(nut*(2*ug[0]**2+2*vg[1]**2+(ug[1]+vg[0])**2), 10*e.value)
            prod[wall_cells] = ew[wall_cells]
            normal_flux = np.sum(velocity.value*normals, axis=0)*face_area
            entering = opened & (normal_flux < 0)
            backflow_rate = np.bincount(owner[entering], weights=-normal_flux[entering], minlength=nc)/area
            oldk = np.maximum(k.value, 1e-10)
            # Replace zero-gradient incoming scalar flux F*q_cell by F*q_ambient.
            production.setValue(prod+backflow_rate*kin)
            eproduction.setValue(1.44*prod*e.value/oldk+backflow_rate*ein)
            ksink.setValue(e.value/oldk+backflow_rate)
            esink.setValue(1.92*e.value/oldk+backflow_rate)
            residual_k, _ = implicit_solve(linear, keq, k, "k", relaxation, kin*c.speed/c.length,
                                           (wall_cells, np.maximum(kw, 1e-10)), dt=dt)
            residual_e, _ = implicit_solve(linear, eeq, e, "e", relaxation, ein*c.speed/c.length,
                                           (wall_cells, np.maximum(ew, 1e-10)), dt=dt)
            k.setValue(np.maximum(k.value, 1e-10))
            e.setValue(np.maximum(e.value, 1e-10))
            continuity = float(np.max(abs(velocity.divergence.value))*c.length/c.speed)
            residual = dict(u=residual_u, v=residual_v, k=residual_k, e=residual_e, continuity=continuity)
            history.append(dict(iteration=total_iterations+iteration, time=physical_time, **residual))
            if c.scenario == PROFILE_SCENARIO and not transient:
                from .airfoil import coefficients
                forces = coefficients((p.value[wc]-2*k.value[wc]/3)*c.rho, nw, face_area[wf],
                                      np.vstack([u.value[wc], v.value[wc]]), distances,
                                      c.rho, c.nu, c.speed, c.profile_chord)
                aero_history.append(dict(iteration=iteration, **forces))
            consecutive = consecutive+1 if max(residual.values()) < c.tolerance else 0
            if progress and not transient:
                progress(iteration, c.max_iterations, 0.)
            if iteration >= (3 if transient else 10) and consecutive >= 3:
                converged = True
                break
        total_iterations += iteration
        if transient:
            if not converged:
                raise RuntimeError(f"Шаг времени t={physical_time:.6g} с не сошёлся: {residual}. "
                    "Увеличьте лимит итераций на шаг или увеличьте число кадров (уменьшите шаг времени).")
            flux_step = np.sum(velocity.value*normals, axis=0)*face_area
            step_history.append(dict(time=physical_time, dt=dt, iterations=iteration,
                residual=max(residual.values()), continuity=continuity,
                mass_imbalance=float(abs(flux_step[exterior].sum())/max(-flux_step[inlet].sum(), 1e-12)),
                max_courant=float(dt*np.max(np.bincount(owner, weights=abs(flux_step), minlength=nc)/area))))
            if c.scenario == PROFILE_SCENARIO:
                from .airfoil import coefficients
                forces = coefficients((p.value[wc]-2*k.value[wc]/3)*c.rho, normals[:, wf], face_area[wf],
                    np.vstack([u.value[wc], v.value[wc]]), distances, c.rho, c.nu, c.speed, c.profile_chord)
                aero_history.append(dict(iteration=total_iterations, time=physical_time, **forces))
            if save_frame:
                snapshot(frame_index)
                if progress:
                    progress(frame_index, len(times)-1, physical_time)
    flux = np.sum(velocity.value*normals, axis=0)*face_area
    inflow = float(-np.sum(flux[inlet]))
    reference_k = np.average(k.value[owner[opened]], weights=face_area[opened])
    static = (p.value-2*k.value/3+2*reference_k/3)*c.rho
    inlet_p = float(np.average(static[owner[inlet]], weights=face_area[inlet]))
    outlet_p = float(np.average(static[owner[opened]], weights=face_area[opened]))
    native_values = [np.asarray(q) for q in (u.value, v.value, static, k.value, e.value)]
    if transient:
        fields = frames
    else:
        fields, fluid = rasterize(project, native_values, c.nx, c.ny)
    diagnostics = dict(mode=c.mode, backend="FiPy polygon FVM / backward Euler URANS + SIMPLE Rhie-Chow" if transient else "FiPy polygon FVM / SIMPLE Rhie-Chow", converged=converged,
        iterations=total_iterations, residuals=residual, residual_history=history, coarse_start=None,
        wall_seconds=time.perf_counter()-start, steps=len(step_history), stored_frames=len(times), time_history=step_history,
        max_divergence=continuity*c.speed/c.length,
        mass_imbalance=float(abs(flux[exterior].sum())/max(inflow, 1e-12)),
        side_flow_m2_s=float(flux[tags == "open"].sum()), pressure_solver="native polygon SIMPLE / sparse-LU + AMG",
        final_frame_velocity_change=float(np.linalg.norm(fields[0][-1]-fields[0][-2])/max(np.linalg.norm(fields[0][-1]), 1e-12)), reynolds=c.speed*min(c.height, c.length)/c.nu,
        pressure_drop_pa=inlet_p-outlet_p, backflow=bool(np.any(flux[tags == "outlet"] < -1e-10)),
        initial_divergence=initial_divergence, initial_condition=c.initial_condition,
        mesh_cells=nc, mesh_nodes=len(project.points), mesh_statistics=project.statistics,
        note="Solved on native triangles/quads. Raster is display only. Pressure accuracy for imported geometries is not reference-validated.")
    native = dict(points=project.points, cells=project.cells, centres=np.asarray(mesh.cellCenters).T,
                  boundary_edges=project.boundary_edges, boundary_tags=project.boundary_tags,
                  **dict(zip(("u", "v", "pressure", "k", "epsilon"), native_values)))
    if aero_history:
        diagnostics["aerodynamics"] = dict(chord_m=c.profile_chord, angle_deg=c.profile_angle,
            reference_speed_m_s=c.speed, reference_density=c.rho,
            reynolds_chord=c.speed*c.profile_chord/c.nu, history=aero_history,
            final={key: value for key, value in aero_history[-1].items() if key != "iteration"},
            note="Forces on profile only, per unit span; pressure plus equilibrium wall shear. Not wind-tunnel validated.")
        diagnostics["reynolds"] = c.speed*c.profile_chord/c.nu
    return Result(c, times, *fields, fluid, diagnostics, native=native)
