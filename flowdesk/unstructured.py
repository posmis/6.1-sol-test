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


def fipy_mesh(project):
    from fipy.meshes.mesh2D import Mesh2D
    faces, cells, tags = project.connectivity()
    mesh = Mesh2D(vertexCoords=project.points.T, faceVertexIDs=faces.T,
                  cellFaceIDs=np.ma.masked_less(cells.T, 0))
    return mesh, tags


def system(equation, variable):
    """Pinned FiPy assembly adapter; no solve or hidden replacement of the mesh."""
    from fipy.solvers.scipy import LinearLUSolver
    prepared = equation._prepareLinearSystem(variable, LinearLUSolver(), (), None)
    return prepared.matrix.matrix.tocsr(), np.asarray(prepared.RHSvector).copy()


def implicit_solve(linear, equation, var, key, alpha, scale, fixed=None):
    a, b = system(equation, var)
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


def rasterize(project, values, nx, ny):
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
    fields = [np.repeat(np.where(fluid, q[cell_ids], 0)[None].astype(np.float32), 2, axis=0)
              for q in values]
    return fields, fluid


@threadpool_limits.wrap(limits=1, user_api="blas")
def solve(config, progress=None, cancel=None):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"flowdesk-mpl"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from fipy import CellVariable, FaceVariable, DiffusionTerm, DiffusionTermCorrection, UpwindConvectionTerm, ImplicitSourceTerm
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
    ein = .09**.75*kin**1.5/(.07*min(c.length, c.height))
    p, pc = CellVariable(mesh=mesh, value=0.), CellVariable(mesh=mesh, value=0.)
    u, v = CellVariable(mesh=mesh, value=c.speed), CellVariable(mesh=mesh, value=0.)
    k, e = CellVariable(mesh=mesh, value=kin), CellVariable(mesh=mesh, value=ein)
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
    # Inward normal velocity at all inlet faces, not an assumed horizontal CAD orientation.
    velocity[0] = u.arithmeticFaceValue
    velocity[1] = v.arithmeticFaceValue
    velocity[:, wall] = 0.
    velocity[:, inlet] = inlet_velocity[:, inlet]
    for iteration in range(1, c.max_iterations+1):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Calculation cancelled")
        nut = np.minimum(.09*np.asarray(k.value)**2/np.maximum(e.value, 1e-12), .1*c.speed*min(c.length, c.height))
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
        residual_u, au = implicit_solve(linear, equations[0], u, "u", .5, c.speed**2/c.length)
        residual_v, av = implicit_solve(linear, equations[1], v, "v", .5, c.speed**2/c.length)
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
        p.setValue(p.value+.25*pc.value)
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
        residual_k, _ = implicit_solve(linear, keq, k, "k", .5, kin*c.speed/c.length,
                                       (wall_cells, np.maximum(kw, 1e-10)))
        residual_e, _ = implicit_solve(linear, eeq, e, "e", .5, ein*c.speed/c.length,
                                       (wall_cells, np.maximum(ew, 1e-10)))
        k.setValue(np.maximum(k.value, 1e-10))
        e.setValue(np.maximum(e.value, 1e-10))
        continuity = float(np.max(abs(velocity.divergence.value))*c.length/c.speed)
        residual = dict(u=residual_u, v=residual_v, k=residual_k, e=residual_e, continuity=continuity)
        history.append(dict(iteration=iteration, **residual))
        consecutive = consecutive+1 if max(residual.values()) < c.tolerance else 0
        if progress:
            progress(iteration, c.max_iterations, 0.)
        if iteration >= 10 and consecutive >= 3:
            converged = True
            break
    flux = np.sum(velocity.value*normals, axis=0)*face_area
    inflow = float(-np.sum(flux[inlet]))
    reference_k = np.average(k.value[owner[opened]], weights=face_area[opened])
    static = (p.value-2*k.value/3+2*reference_k/3)*c.rho
    inlet_p = float(np.average(static[owner[inlet]], weights=face_area[inlet]))
    outlet_p = float(np.average(static[owner[opened]], weights=face_area[opened]))
    native_values = [np.asarray(q) for q in (u.value, v.value, static, k.value, e.value)]
    fields, fluid = rasterize(project, native_values, c.nx, c.ny)
    diagnostics = dict(mode="steady", backend="FiPy polygon FVM / SIMPLE Rhie-Chow", converged=converged,
        iterations=iteration, residuals=residual, residual_history=history, coarse_start=None,
        wall_seconds=time.perf_counter()-start, steps=0, stored_frames=2,
        max_divergence=continuity*c.speed/c.length,
        mass_imbalance=float(abs(flux[exterior].sum())/max(inflow, 1e-12)),
        side_flow_m2_s=0., pressure_solver="native polygon SIMPLE / sparse-LU + AMG",
        final_frame_velocity_change=0., reynolds=c.speed*min(c.height, c.length)/c.nu,
        pressure_drop_pa=inlet_p-outlet_p, backflow=bool(np.any(flux[opened] < -1e-10)),
        mesh_cells=nc, mesh_nodes=len(project.points), mesh_statistics=project.statistics,
        note="Solved on native triangles/quads. Raster is display only. Pressure accuracy for imported geometries is not reference-validated.")
    native = dict(points=project.points, cells=project.cells, centres=np.asarray(mesh.cellCenters).T,
                  boundary_edges=project.boundary_edges, boundary_tags=project.boundary_tags,
                  **dict(zip(("u", "v", "pressure", "k", "epsilon"), native_values)))
    return Result(c, np.array([0., c.duration]), *fields, fluid, diagnostics, native=native)
