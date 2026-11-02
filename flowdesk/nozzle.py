"""2D compressible ideal-gas Euler on body-fitted quadrilateral nozzle cells.

Finite-volume MUSCL/Rusanov + SSP RK2, conservative total energy, slip adiabatic
walls, reservoir inlet and characteristic pressure/supersonic outlet. No k-epsilon
or viscous wall law is applied to this inviscid model. The separate liquid mode
uses the existing incompressible Navier-Stokes solver.
"""
from dataclasses import replace
import time
import numpy as np
from numba import njit
from threadpoolctl import threadpool_limits


def normalize(c):
    changes = dict(length=c.nozzle_converging_length+c.nozzle_diverging_length,
                   height=max(c.nozzle_inlet_height, c.nozzle_exit_height), top='wall', bottom='wall', initial_condition='uniform')
    if c.nozzle_model == 'gas':
        changes.update(turbulence=False, energy=True, thermal_wall='adiabatic', heat_source=0.,
            initial_condition='rest', temperature_initial=c.nozzle_total_temperature, temperature_inlet=c.nozzle_total_temperature,
            heat_capacity=c.nozzle_gamma*c.nozzle_gas_constant/(c.nozzle_gamma-1),
            rho=c.nozzle_total_pressure/(c.nozzle_gas_constant*c.nozzle_total_temperature))
    return replace(c, **changes)


def height_at(c, x):
    x = np.asarray(x)
    left = c.nozzle_throat_height+.5*(c.nozzle_inlet_height-c.nozzle_throat_height)*(
        1+np.cos(np.pi*np.clip(x/c.nozzle_converging_length, 0, 1)))
    right = c.nozzle_throat_height+.5*(c.nozzle_exit_height-c.nozzle_throat_height)*(
        1-np.cos(np.pi*np.clip((x-c.nozzle_converging_length)/c.nozzle_diverging_length, 0, 1)))
    return np.where(x <= c.nozzle_converging_length, left, right)


def boundaries(c, count=257):
    x = np.linspace(0, c.length, count)
    h = height_at(c, x)
    lower, upper = (c.height-h)/2, (c.height+h)/2
    return dict(wall=np.vstack([np.column_stack([x, lower]), [np.nan, np.nan], np.column_stack([x, upper])]).astype(np.float32),
        inlet=np.array([[0., lower[0]], [0., upper[0]]], np.float32),
        outlet=np.array([[c.length, lower[-1]], [c.length, upper[-1]]], np.float32),
        open=np.empty((0, 2), np.float32))


def grid(c):
    x = np.linspace(0, c.length, c.nx+1)
    h = height_at(c, x)
    eta = np.linspace(-.5, .5, c.ny+1)
    xx = np.broadcast_to(x, (c.ny+1, c.nx+1))
    yy = c.height/2+eta[:, None]*h
    points = np.column_stack([xx.ravel(), yy.ravel()])
    ids = np.arange((c.ny+1)*(c.nx+1)).reshape(c.ny+1, c.nx+1)
    cells = np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]], axis=-1).reshape(-1, 4)
    edges = np.concatenate([np.column_stack([ids[0, :-1], ids[0, 1:]]),
        np.column_stack([ids[-1, :-1], ids[-1, 1:]]), np.column_stack([ids[:-1, 0], ids[1:, 0]]),
        np.column_stack([ids[:-1, -1], ids[1:, -1]])])
    tags = np.array(['wall']*(2*c.nx)+['inlet']*c.ny+['outlet']*c.ny)
    dx = c.length/c.nx
    area = np.broadcast_to(dx*(h[:-1]+h[1:])/(2*c.ny), (c.ny, c.nx)).copy()
    # Upward face normal multiplied by face length.
    syx = -eta[:, None]*np.diff(h)[None, :]
    return h, syx, area, points, cells, edges, tags


@njit(cache=True)
def primitive(w, gamma):
    q = np.empty_like(w)
    q[0] = w[0]
    q[1] = w[1]/w[0]
    q[2] = w[2]/w[0]
    q[3] = (gamma-1)*(w[3]-.5*(w[1]**2+w[2]**2)/w[0])
    return q


@njit(cache=True)
def limited(a, b):
    if a*b <= 0:
        return 0.
    return np.sign(a)*min(1.5*abs(a), .5*abs(a+b), 1.5*abs(b))


@njit(cache=True)
def slopes(q):
    ny, nx = q.shape[1:]
    sx, sy = np.zeros_like(q), np.zeros_like(q)
    for k in range(4):
        for i in range(ny):
            for j in range(nx):
                if 0 < j < nx-1:
                    sx[k, i, j] = limited(q[k, i, j]-q[k, i, j-1], q[k, i, j+1]-q[k, i, j])
                if 0 < i < ny-1:
                    sy[k, i, j] = limited(q[k, i, j]-q[k, i-1, j], q[k, i+1, j]-q[k, i, j])
    # Flatten reconstruction locally at strong pressure jumps to avoid
    # limiter chatter at a stationary shock; smooth flow remains second order.
    for i in range(ny):
        for j in range(nx):
            lo, hi = q[3, i, j], q[3, i, j]
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i+di, j+dj
                if 0 <= ii < ny and 0 <= jj < nx:
                    lo, hi = min(lo, q[3, ii, jj]), max(hi, q[3, ii, jj])
            if hi > 1.25*lo:
                sx[:, i, j] = 0.
                sy[:, i, j] = 0.
    return sx, sy


@njit(cache=True)
def state(q, slope, i, j, sign):
    r = q[0, i, j]+sign*.5*slope[0, i, j]
    p = q[3, i, j]+sign*.5*slope[3, i, j]
    if r <= 1e-10 or p <= 1e-6:
        return q[0, i, j], q[1, i, j], q[2, i, j], q[3, i, j]
    return r, q[1, i, j]+sign*.5*slope[1, i, j], q[2, i, j]+sign*.5*slope[2, i, j], p


@njit(cache=True)
def inlet(q, gamma, gas_r, total_p, total_t):
    rho, u, v, p = q
    a = np.sqrt(gamma*p/rho)
    b = (gamma-1)/2
    invariant = u-a/b
    a0sq = gamma*gas_r*total_t
    root = max(b*(1+b)*a0sq-b**3*invariant**2, 0.)
    ub = (b*b*invariant+np.sqrt(root))/(b*(1+b))
    ub = min(max(ub, 0.), .999*np.sqrt(2*a0sq/(gamma+1)))
    tb = total_t-.5*ub*ub/(gamma*gas_r/(gamma-1))
    pb = total_p*(tb/total_t)**(gamma/(gamma-1))
    return pb/(gas_r*tb), ub, 0., pb


@njit(cache=True)
def outlet(q, gamma, gas_r, back_p, total_t):
    rho, u, v, p = q
    a = np.sqrt(gamma*p/rho)
    if u >= a:
        return q
    rhob = rho*(back_p/p)**(1/gamma)
    ab = np.sqrt(gamma*back_p/rhob)
    ub = u+2*(a-ab)/(gamma-1)
    if ub < 0.:  # Ambient reservoir if the pressure boundary admits backflow.
        return back_p/(gas_r*total_t), ub, 0., back_p
    return rhob, ub, v, back_p


@njit(cache=True)
def reflect(q, sx, sy):
    r, u, v, p = q
    vn = (u*sx+v*sy)/(sx*sx+sy*sy)
    return r, u-2*vn*sx, v-2*vn*sy, p


@njit(cache=True)
def flux(left, right, sx, sy, gamma):
    rl, ul, vl, pl = left
    rr, ur, vr, pr = right
    length = np.sqrt(sx*sx+sy*sy)
    nl, nr = ul*sx+vl*sy, ur*sx+vr*sy
    el = pl/(gamma-1)+.5*rl*(ul*ul+vl*vl)
    er = pr/(gamma-1)+.5*rr*(ur*ur+vr*vr)
    speed = max(abs(nl)+np.sqrt(gamma*pl/rl)*length, abs(nr)+np.sqrt(gamma*pr/rr)*length)
    return (.5*(rl*nl+rr*nr-speed*(rr-rl)),
            .5*(rl*ul*nl+pl*sx+rr*ur*nr+pr*sx-speed*(rr*ur-rl*ul)),
            .5*(rl*vl*nl+pl*sy+rr*vr*nr+pr*sy-speed*(rr*vr-rl*vl)),
            .5*((el+pl)*nl+(er+pr)*nr-speed*(er-el)))


@njit(cache=True)
def rhs(w, h, syx, area, dx, gamma, gas_r, p0, t0, pb, second_order=True):
    q = primitive(w, gamma)
    ny, nx = q.shape[1:]
    if second_order:
        slope_x, slope_y = slopes(q)
    else:
        slope_x, slope_y = np.zeros_like(q), np.zeros_like(q)
    residual = np.zeros_like(w)
    # Integrated mass and total-energy flux at the two external cross sections.
    balance = np.zeros(4)
    for i in range(ny):
        for j in range(nx+1):
            if j == 0:
                qr = state(q, slope_x, i, 0, -1)
                ql = inlet(qr, gamma, gas_r, p0, t0)
            elif j == nx:
                ql = state(q, slope_x, i, nx-1, 1)
                qr = outlet(ql, gamma, gas_r, pb, t0)
            else:
                ql = state(q, slope_x, i, j-1, 1)
                qr = state(q, slope_x, i, j, -1)
            f = flux(ql, qr, h[j]/ny, 0., gamma)
            for k in range(4):
                if j > 0:
                    residual[k, i, j-1] -= f[k]
                if j < nx:
                    residual[k, i, j] += f[k]
            if j == 0:
                balance[0] += f[0]
                balance[2] += f[3]
            elif j == nx:
                balance[1] += f[0]
                balance[3] += f[3]
    for i in range(ny+1):
        for j in range(nx):
            if i == 0:
                qr = state(q, slope_y, 0, j, -1)
                ql = reflect(qr, syx[i, j], dx)
            elif i == ny:
                ql = state(q, slope_y, ny-1, j, 1)
                qr = reflect(ql, syx[i, j], dx)
            else:
                ql = state(q, slope_y, i-1, j, 1)
                qr = state(q, slope_y, i, j, -1)
            f = flux(ql, qr, syx[i, j], dx, gamma)
            for k in range(4):
                if i > 0:
                    residual[k, i-1, j] -= f[k]
                if i < ny:
                    residual[k, i, j] += f[k]
    return residual/area, balance


@njit(cache=True)
def stable_dt(w, h, syx, area, dx, gamma):
    q = primitive(w, gamma)
    ny, nx = q.shape[1:]
    dt = 1e30
    for i in range(ny):
        for j in range(nx):
            rho, u, v, p = q[:, i, j]
            a = np.sqrt(gamma*p/rho)
            spectral = ((abs(u)+a)*(h[j]+h[j+1])/ny+
                abs(u*syx[i, j]+v*dx)+a*np.hypot(syx[i, j], dx)+
                abs(u*syx[i+1, j]+v*dx)+a*np.hypot(syx[i+1, j], dx))*.5
            dt = min(dt, .4*area[i, j]/spectral)
    return dt


@njit(cache=True)
def positive(w, gamma):
    if not np.isfinite(w).all():
        return False
    for i in range(w.shape[1]):
        for j in range(w.shape[2]):
            r, mx, my, energy = w[:, i, j]
            if r <= 1e-10 or (gamma-1)*(energy-.5*(mx*mx+my*my)/r) <= 1e-6:
                return False
    return True


def area_mach(mach, gamma=1.4):
    return (2/(gamma+1)*(1+(gamma-1)*mach**2/2))**((gamma+1)/(2*(gamma-1)))/mach


def design(c):
    from scipy.optimize import brentq
    g, r = c.nozzle_gamma, c.nozzle_gas_constant
    ratio = c.nozzle_exit_height/c.nozzle_throat_height
    high = 2.
    while area_mach(high, g) < ratio:
        high *= 2
    m = brentq(lambda m: area_mach(m, g)-ratio, 1.+1e-8, high)
    factor = 1+(g-1)*m*m/2
    mass = c.nozzle_throat_height*c.nozzle_total_pressure*np.sqrt(g/(r*c.nozzle_total_temperature))*(2/(g+1))**((g+1)/(2*(g-1)))
    return dict(design_exit_mach=m, design_exit_pressure_pa=c.nozzle_total_pressure/factor**(g/(g-1)),
                choked_mass_flow_kg_s_per_m=mass)


@threadpool_limits.wrap(limits=1, user_api='blas')
def solve(c, progress=None, cancel=None, preview=None):
    from .solver import Result
    from .live import PreviewEmitter
    start = time.perf_counter()
    h, syx, area, points, cells, edges, tags = grid(c)
    dx, g = c.length/c.nx, c.nozzle_gamma
    p0, t0, pb, r = c.nozzle_total_pressure, c.nozzle_total_temperature, c.nozzle_back_pressure, c.nozzle_gas_constant
    w = np.zeros((4, c.ny, c.nx))
    w[0], w[3] = p0/(r*t0), p0/(g-1)
    # Every display pixel maps to an original curvilinear quadrilateral.
    centre_h = .5*(h[:-1]+h[1:])
    y = (np.arange(c.ny)+.5)*c.height/c.ny
    eta = (y[:, None]-c.height/2)/centre_h[None, :]+.5
    fluid = (eta >= 0) & (eta < 1)
    row = np.clip(np.floor(eta*c.ny).astype(int), 0, c.ny-1)
    cell_ids = row*c.nx+np.arange(c.nx)[None, :]
    live = PreviewEmitter(preview, c, fluid, cell_ids=cell_ids, boundaries=boundaries(c)) if preview else None
    def values():
        q = primitive(w, g)
        speed = np.hypot(q[1], q[2])
        temp = q[3]/(q[0]*r)
        mach = speed/np.sqrt(g*q[3]/q[0])
        return [a.ravel() for a in (q[1], q[2], q[3]-pb, temp, mach, q[0])]
    if live:
        live.emit(0, 0., values, force=True)
    times = np.linspace(0, c.duration, c.frame_count)
    fields = [np.empty((len(times), c.ny, c.nx), np.float32) for _ in range(6)]
    def snapshot(index):
        for field, value in zip(fields, values()):
            field[index] = np.where(fluid, value[cell_ids], 0.)
    snapshot(0)
    t, steps, residual, consecutive, converged = 0., 0, 1., 0, False
    residual_history = []
    mass_integral, energy_integral = 0., 0.
    reference_mass = design(c)["choked_mass_flow_kg_s_per_m"]
    initial_mass = float(np.sum(w[0]*area))
    initial_energy = float(np.sum(w[3]*area))
    scales = np.array([p0/(r*t0), p0/(r*t0)*np.sqrt(g*r*t0), p0/(r*t0)*np.sqrt(g*r*t0), p0/(g-1)])[:, None, None]
    frame_index = 1
    while (c.mode == 'transient' and frame_index < len(times)) or (c.mode == 'steady' and steps < c.max_iterations):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Calculation cancelled')
        dt = stable_dt(w, h, syx, area, dx, g)
        if c.time_step:
            dt = min(dt, c.time_step)
        if c.mode == 'transient':
            dt = min(dt, times[frame_index]-t)
        old = w
        for attempt in range(14):
            first, b1 = rhs(old, h, syx, area, dx, g, r, p0, t0, pb, attempt < 7)
            stage = old+dt*first
            if positive(stage, g):
                second, b2 = rhs(stage, h, syx, area, dx, g, r, p0, t0, pb, attempt < 7)
                candidate = .5*(old+stage+dt*second)
                if positive(candidate, g):
                    w = candidate
                    break
            dt *= .5
        else:
            raise FloatingPointError('Compressible nozzle step lost positivity. Reduce time step or improve nozzle proportions.')
        t += dt
        steps += 1
        mass_integral += .5*dt*(b1[0]-b1[1]+b2[0]-b2[1])
        energy_integral += .5*dt*(b1[2]-b1[3]+b2[2]-b2[3])
        residual = float(np.sqrt(np.mean(((w-old)/scales)**2))/dt*c.length/np.sqrt(g*r*t0))
        if live:
            live.emit(steps, t, values, phase='euler')
        if c.mode == 'steady':
            imbalance = abs(b2[0]-b2[1])/max(abs(b2[0]), abs(b2[1]), reference_mass, 1e-12)
            consecutive = consecutive+1 if residual < c.tolerance and imbalance < max(10*c.tolerance, 1e-5) else 0
            if steps % 10 == 0 or steps == 1:
                residual_history.append(dict(iteration=steps, u=residual, v=residual, k=0., e=0., continuity=imbalance))
                if progress:
                    progress(steps, c.max_iterations, t)
            if steps >= 50 and consecutive >= 3:
                converged = True
                break
        elif t >= times[frame_index]-1e-14:
            snapshot(frame_index)
            if progress:
                progress(frame_index, len(times)-1, t)
            frame_index += 1
    if c.mode == 'steady':
        snapshot(0)
        snapshot(1)
    q = primitive(w, g)
    _, balance = rhs(w, h, syx, area, dx, g, r, p0, t0, pb)
    inlet_mass, outlet_mass = balance[:2]
    native_values = values()
    native = dict(points=points, cells=cells, centres=points[cells].mean(axis=1),
        boundary_edges=edges, boundary_tags=tags, u=native_values[0], v=native_values[1],
        pressure=native_values[2], temperature=native_values[3], mach=native_values[4], density=native_values[5],
        k=np.zeros(c.nx*c.ny), epsilon=np.zeros(c.nx*c.ny),
        pressure_absolute=q[3].ravel().copy(), total_energy=w[3].ravel().copy())
    mass_defect = float(np.sum(w[0]*area))-initial_mass-mass_integral
    energy_defect = float(np.sum(w[3]*area))-initial_energy-energy_integral
    d = dict(mode=c.mode, backend='2D ideal-gas Euler / body-fitted MUSCL Rusanov SSP-RK2',
        turbulence_model='inviscid slip walls', converged=converged if c.mode == 'steady' else True,
        iterations=steps, steps=steps, residuals=dict(u=residual, v=residual, k=0., e=0., continuity=abs(inlet_mass-outlet_mass)/max(abs(inlet_mass), 1e-12)),
        residual_history=residual_history, coarse_start=None, wall_seconds=time.perf_counter()-start,
        stored_frames=len(times), max_divergence=float(np.max(abs(
            np.gradient(q[1], dx, axis=1)-(np.arange(c.ny)[:, None]+.5-c.ny/2)/c.ny*
            np.diff(h)[None, :]/dx/centre_h[None, :]*np.gradient(q[1], 1/c.ny, axis=0)+
            np.gradient(q[2], 1/c.ny, axis=0)/centre_h[None, :]))), mass_imbalance=float(abs(inlet_mass-outlet_mass)/max(abs(inlet_mass), 1e-12)),
        side_flow_m2_s=0., pressure_solver='Compressible conservative Euler fluxes', final_frame_velocity_change=float(np.linalg.norm(fields[0][-1]-fields[0][-2])/max(np.linalg.norm(fields[0][-1]), 1e-12)),
        reynolds=float(np.mean(q[0, :, 0])*np.mean(q[1, :, 0])*c.nozzle_inlet_height/1.85e-5),
        pressure_drop_pa=float(np.mean(q[3, :, 0])-np.mean(q[3, :, -1])), backflow=bool(np.any(q[1, :, -1] < 0)),
        mesh_cells=c.nx*c.ny, mesh_nodes=len(points),
        nozzle=dict(**design(c), mass_in_kg_s_per_m=float(inlet_mass), mass_out_kg_s_per_m=float(outlet_mass),
            throat_mach=float(np.mean(native_values[4].reshape(c.ny, c.nx)[:, np.argmin(centre_h)])),
            exit_mach=float(np.mean(native_values[4].reshape(c.ny, c.nx)[:, -1])),
            pressure_reference_pa=pb, physical_end_time=t,
            relative_mass_conservation_error=abs(mass_defect)/max(initial_mass, abs(mass_integral), 1e-12),
            relative_energy_conservation_error=abs(energy_defect)/max(initial_energy, abs(energy_integral), 1e-12)),
        note='True 2D inviscid ideal gas; slip adiabatic walls. No viscous boundary layer, k-epsilon or wall heating. Pressure relative to specified back pressure. Inlet/outlet flux difference includes compressible mass storage during startup.')
    if live:
        live.emit(steps, t, values, phase='euler', force=True)
    zeros = np.zeros_like(fields[0])
    return Result(c, times, fields[0], fields[1], fields[2], zeros, zeros.copy(), fluid, d,
                  native=native, temperature=fields[3], mach=fields[4], density=fields[5])
