"""Compiled finite-volume stencils. No fastmath: preserve IEEE arithmetic."""
import numpy as np
from numba import njit


@njit(cache=True)
def diffuse(q, nu, mask, dx, dy):
    out = np.zeros_like(q)
    ny, nx = q.shape
    for i in range(ny):
        for j in range(nx):
            if not mask[i, j]:
                continue
            if j+1 < nx and mask[i, j+1]:
                flux = (q[i, j+1]-q[i, j])*(nu[i, j]+nu[i, j+1])*.5/dx**2
                out[i, j] += flux
                out[i, j+1] -= flux
            if i+1 < ny and mask[i+1, j]:
                flux = (q[i+1, j]-q[i, j])*(nu[i, j]+nu[i+1, j])*.5/dy**2
                out[i, j] += flux
                out[i+1, j] -= flux
    return out


@njit(cache=True)
def advect(q, ax, ay, dx, dy):
    out = np.empty_like(q)
    ny, nx = q.shape
    for i in range(ny):
        for j in range(nx):
            x = (q[i, j]-q[i, max(j-1, 0)]) if ax[i, j] >= 0 else (q[i, min(j+1, nx-1)]-q[i, j])
            y = (q[i, j]-q[max(i-1, 0), j]) if ay[i, j] >= 0 else (q[min(i+1, ny-1), j]-q[i, j])
            out[i, j] = ax[i, j]*x/dx+ay[i, j]*y/dy
    return out


@njit(cache=True)
def scalar_transport(q, u, v, inlet, dx, dy):
    ny, nx = q.shape
    out = np.empty_like(q)
    for i in range(ny):
        for j in range(nx):
            ql = q[i, j-1] if j else inlet
            qr = q[i, j+1] if j+1 < nx else inlet if u[i, j+1] < 0 else q[i, j]
            qb = q[i-1, j] if i else inlet if v[i, j] > 0 else q[i, j]
            qt = q[i+1, j] if i+1 < ny else inlet if v[i+1, j] < 0 else q[i, j]
            left = u[i, j]*(ql if u[i, j] >= 0 else q[i, j])
            right = u[i, j+1]*(q[i, j] if u[i, j+1] >= 0 else qr)
            bottom = v[i, j]*(qb if v[i, j] >= 0 else q[i, j])
            top = v[i+1, j]*(q[i, j] if v[i+1, j] >= 0 else qt)
            out[i, j] = -(right-left)/dx-(top-bottom)/dy
    return out


@njit(cache=True)
def assemble(mask, nu, ax, ay, reaction, source, kind, fixed, dx, dy,
             conservative, uf, vf, inlet):
    """Five-point transport operator; kind: 1 fixed, 2/3/4/5 copy E/W/N/S."""
    ny, nx = mask.shape
    ids = np.full(mask.shape, -1, np.int32)
    n = 0
    for i in range(ny):
        for j in range(nx):
            if mask[i, j]:
                ids[i, j] = n
                n += 1
    rows, cols, data = np.empty(5*n, np.int32), np.empty(5*n, np.int32), np.empty(5*n)
    rhs = np.zeros(n)
    used = 0
    for i in range(ny):
        for j in range(nx):
            row = ids[i, j]
            if row < 0:
                continue
            if kind[i, j]:
                rows[used], cols[used], data[used] = row, row, 1.
                used += 1
                if kind[i, j] == 1:
                    rhs[row] = fixed[i, j]
                else:
                    di = 1 if kind[i, j] == 4 else -1 if kind[i, j] == 5 else 0
                    dj = 1 if kind[i, j] == 2 else -1 if kind[i, j] == 3 else 0
                    neighbour = ids[i+di, j+dj]
                    if neighbour >= 0:
                        rows[used], cols[used], data[used] = row, neighbour, -1.
                        used += 1
                continue
            diagonal = reaction[i, j]
            rhs[row] = source[i, j]
            for direction in range(4):
                di = -1 if direction == 2 else 1 if direction == 3 else 0
                dj = -1 if direction == 0 else 1 if direction == 1 else 0
                ii, jj = i+di, j+dj
                h = dx if direction < 2 else dy
                off = 0.
                inside = 0 <= ii < ny and 0 <= jj < nx
                if inside and mask[ii, jj]:
                    diffusion = (nu[i, j]+nu[ii, jj])*.5/h**2
                    diagonal += diffusion
                    off -= diffusion
                if conservative:
                    flux = (-uf[i, j] if direction == 0 else uf[i, j+1] if direction == 1
                            else -vf[i, j] if direction == 2 else vf[i+1, j]) / h
                    diagonal += max(flux, 0.)
                    if inside:
                        off += min(flux, 0.)
                    else:
                        rhs[row] -= min(flux, 0.)*inlet
                else:
                    vel = (-ax[i, j] if direction == 0 else ax[i, j] if direction == 1
                           else -ay[i, j] if direction == 2 else ay[i, j])
                    # Incoming neighbour in the advective form, zero-gradient at outer edge.
                    rate = max(-vel, 0.)/h
                    if inside:
                        diagonal += rate
                        off -= rate
                if inside and mask[ii, jj] and off != 0:
                    rows[used], cols[used], data[used] = row, ids[ii, jj], off
                    used += 1
            rows[used], cols[used], data[used] = row, row, max(diagonal, 1e-14)
            used += 1
    return rows[:used], cols[:used], data[:used], rhs


@njit(cache=True)
def pressure_system(fluid, du, dv, dx, dy, bottom_open, top_open):
    ny, nx = fluid.shape
    ids = np.full(fluid.shape, -1, np.int32)
    n = 0
    for i in range(ny):
        for j in range(nx):
            if fluid[i, j]:
                ids[i, j] = n
                n += 1
    rows, cols, data = np.empty(5*n, np.int32), np.empty(5*n, np.int32), np.empty(5*n)
    used = 0
    for i in range(ny):
        for j in range(nx):
            row = ids[i, j]
            if row < 0:
                continue
            diag = 0.
            for direction in range(4):
                ii = i+(-1 if direction == 2 else 1 if direction == 3 else 0)
                jj = j+(-1 if direction == 0 else 1 if direction == 1 else 0)
                weight = (du[i, j]/dx**2 if direction == 0 else du[i, j+1]/dx**2 if direction == 1
                          else dv[i, j]/dy**2 if direction == 2 else dv[i+1, j]/dy**2)
                if 0 <= ii < ny and 0 <= jj < nx:
                    if fluid[ii, jj]:
                        rows[used], cols[used], data[used] = row, ids[ii, jj], -weight
                        used += 1
                        diag += weight
                elif direction == 1 or (direction == 2 and bottom_open) or (direction == 3 and top_open):
                    diag += 2*weight
            rows[used], cols[used], data[used] = row, row, diag
            used += 1
    return rows[:used], cols[:used], data[:used]
