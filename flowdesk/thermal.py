"""Constant-property sensible energy, decoupled from incompressible momentum.

rho*cp*(dT/dt + div(U T)) = div((lambda + rho*cp*nu_t/Pr_t) grad(T)) + Q.
Solve theta=T-T_in to preserve uniform temperatures at machine precision.
Boundary conduction is evaluated at the physical face, not by fixing wall cells.
"""
import numpy as np
from .kernels import diffuse, scalar_transport


def temperature_scale(c):
    return max(abs(c.temperature_wall-c.temperature_inlet) if c.thermal_wall == 'isothermal' else 0.,
               abs(c.temperature_initial-c.temperature_inlet),
               abs(c.heat_source)*c.length/(c.rho*c.heat_capacity*c.speed), 1.)


def energy_budget(c, temperature, storage, net_advection, wall_heat, inlet_heat, source,
                  residual=0.):
    balance = storage+net_advection-wall_heat-inlet_heat-source
    scale = max(abs(storage)+abs(net_advection)+abs(wall_heat)+abs(inlet_heat)+abs(source), 1e-8)
    return dict(min_temperature_k=float(np.min(temperature)), max_temperature_k=float(np.max(temperature)),
        storage_w_per_m=float(storage), net_advective_heat_w_per_m=float(net_advection),
        wall_heat_w_per_m=float(wall_heat), inlet_conduction_w_per_m=float(inlet_heat),
        source_heat_w_per_m=float(source), imbalance_w_per_m=float(balance),
        relative_imbalance=float(abs(balance)/scale), residual=float(residual),
        molecular_diffusivity_m2_s=c.thermal_diffusivity,
        model='Constant-property sensible energy; no buoyancy, property feedback or viscous heating.')


def structured_boundary(s, alpha):
    c = s.c
    reaction = np.zeros(s.fluid.shape)
    source = np.full(s.fluid.shape, c.heat_source/(c.rho*c.heat_capacity))
    reaction[:, 0] += 2*alpha[:, 0]/s.dx**2
    if c.thermal_wall == 'isothermal':
        for _, h, mask in s.walls:
            rate = 2*alpha[mask]/h**2
            reaction[mask] += rate
            source[mask] += rate*(c.temperature_wall-c.temperature_inlet)
    return reaction, source


def structured_budget(s, theta, alpha, old=None, dt=None, residual=0.):
    c, f = s.c, s.fluid
    capacity = c.rho*c.heat_capacity
    volume = s.dx*s.dy
    adv = -capacity*volume*scalar_transport(theta, s.u, s.v, 0., s.dx, s.dy)[f].sum()
    wall = 0.
    if c.thermal_wall == 'isothermal':
        for _, h, mask in s.walls:
            wall += capacity*volume*np.sum(2*alpha[mask]/h**2*(c.temperature_wall-c.temperature_inlet-theta[mask]))
    inlet = -capacity*volume*np.sum((2*alpha[:, 0]/s.dx**2*theta[:, 0])[f[:, 0]])
    storage = 0. if dt is None else capacity*volume*np.sum((s.temperature-c.temperature_inlet-old)[f])/dt
    return energy_budget(c, s.temperature[f], storage, adv, wall, inlet,
                         c.heat_source*volume*f.sum(), residual)


def structured_step(s, nut, dt):
    c = s.c
    alpha = c.thermal_diffusivity+nut/c.turbulent_prandtl
    theta = s.temperature-c.temperature_inlet
    reaction, source = structured_boundary(s, alpha)
    rate = (scalar_transport(theta, s.u, s.v, 0., s.dx, s.dy)+
            diffuse(theta, alpha, s.fluid, s.dx, s.dy)+source-reaction*theta)
    s.temperature = c.temperature_inlet+theta+dt*rate
    if not np.isfinite(s.temperature[s.fluid]).all() or np.min(s.temperature[s.fluid]) <= 0:
        raise FloatingPointError('Energy equation produced invalid absolute temperature. Check heat source and time step.')
    # Forward Euler evaluates boundary heat and advection at the old temperature.
    s.energy_budget = structured_budget(s, theta, alpha, old=theta, dt=dt)


def structured_steady(s, linear):
    from .steady import operator, defect
    c, f = s.c, s.fluid
    nut = np.minimum(.09*s.k**2/np.maximum(s.e, 1e-12), .1*c.speed*c.height)
    alpha = c.thermal_diffusivity+nut/c.turbulent_prandtl
    reaction, source = structured_boundary(s, alpha)
    theta = s.temperature-c.temperature_inlet
    kind = np.zeros(f.shape, np.int32)
    a, b = operator(f, alpha, *s.centres(), reaction, source, kind, np.zeros(f.shape),
                    s.dx, s.dy, conservative=True, u=s.u, v=s.v, inlet=0.)
    theta[f] = linear.solve(a, b, theta[f], 'temperature')
    residual = defect(a, theta[f], b, temperature_scale(c)*c.speed/c.length)
    if residual > c.tolerance or not np.isfinite(theta[f]).all() or np.min(theta[f]+c.temperature_inlet) <= 0:
        raise RuntimeError(f'Energy equation failed to converge: residual={residual:.3e}')
    s.temperature = theta+c.temperature_inlet
    s.energy_budget = structured_budget(s, theta, alpha, residual=residual)


def native_diffusion(coeff):
    from .fipy_compat import NamedDiffusionTermCorrection

    class ThermalDiffusion(NamedDiffusionTermCorrection):
        def _calcAnisotropySource(self, coeff, var):
            if not hasattr(self, 'anisotropySource') and len(coeff) > 1:
                from fipy.variables.addOverFacesVariable import _AddOverFacesVariable
                unconstrained = var+0
                unconstrained.name = 'FlowDesk unconstrained temperature'
                gradients = unconstrained.grad.harmonicFaceValue.dot(self._getRotationTensor(var.mesh))
                # Natural zero heat flux at insulated/open faces must remain zero.
                # Use nonorthogonal deferred corrections only on shared faces;
                # constant Dirichlet boundaries use normal-distance conductances.
                flux = gradients[1:].dot(coeff[1:])*var.mesh.interiorFaces
                self.anisotropySource = _AddOverFacesVariable(flux)*var.mesh.cellVolumes
    return ThermalDiffusion(coeff=coeff)


class NativeThermal:
    """Energy transport on the same polygon cells and conservative face fluxes."""
    def __init__(self, c, mesh, tags, velocity, owner, normals, face_area, area):
        from fipy import CellVariable, UpwindConvectionTerm, ImplicitSourceTerm, TransientTerm
        self.c, self.mesh, self.tags = c, mesh, tags
        self.velocity, self.owner, self.normals = velocity, owner, normals
        self.face_area, self.area = face_area, area
        self.theta = CellVariable(mesh=mesh, value=c.temperature_initial-c.temperature_inlet,
                                  hasOld=c.mode == 'transient')
        inlet, wall = tags == 'inlet', tags == 'wall'
        opened = (tags == 'outlet') | (tags == 'open')
        self.theta.constrain(0., inlet)
        self.theta.faceGrad.constrain(0., opened)
        if c.thermal_wall == 'isothermal':
            self.theta.constrain(c.temperature_wall-c.temperature_inlet, wall)
        else:
            self.theta.faceGrad.constrain(0., wall)
        self.alpha = CellVariable(mesh=mesh, value=c.thermal_diffusivity)
        distances = np.asarray(mesh._cellDistances)
        delta = np.asarray(mesh.faceCenters)-np.asarray(mesh.cellCenters)[:, owner]
        normal_distance = np.maximum(abs(np.sum(delta*normals, axis=0)), 1e-12)
        fixed = inlet | (wall if c.thermal_wall == 'isothermal' else False)
        factor = np.where(fixed, distances/normal_distance, 1.)
        self.face_diffusivity = self.alpha.arithmeticFaceValue*factor
        self.backflow = CellVariable(mesh=mesh, value=0.)
        self.equation = (UpwindConvectionTerm(coeff=velocity)-
            native_diffusion(coeff=self.face_diffusivity)+
            ImplicitSourceTerm(coeff=self.backflow)-c.heat_source/(c.rho*c.heat_capacity))
        if c.mode == 'transient':
            self.equation = TransientTerm()+self.equation

    @property
    def temperature(self):
        return np.asarray(self.theta.value)+self.c.temperature_inlet

    def advance(self, linear, nut, dt, preview=None):
        from .unstructured import implicit_solve
        c = self.c
        self.alpha.setValue(c.thermal_diffusivity+nut/c.turbulent_prandtl)
        flux = np.sum(self.velocity.value*self.normals, axis=0)*self.face_area
        opened = (self.tags == 'outlet') | (self.tags == 'open')
        entering = opened & (flux < 0)
        self.backflow.setValue(np.bincount(self.owner[entering], weights=-flux[entering],
                                          minlength=len(self.area))/self.area)
        consecutive = 0
        for _ in range(c.max_iterations):
            residual, _ = implicit_solve(linear, self.equation, self.theta, 'temperature', 1.,
                                        temperature_scale(c)*c.speed/c.length, dt=dt)
            if preview:
                preview()
            consecutive = consecutive+1 if residual < c.tolerance else 0
            if consecutive >= 2:
                break
        else:
            raise RuntimeError(f'Energy equation failed to converge: residual={residual:.3e}. Refine the mesh or reduce time step.')
        if not np.isfinite(self.temperature).all() or np.min(self.temperature) <= 0:
            raise FloatingPointError('Energy equation produced invalid absolute temperature.')
        exterior = self.tags != 'interior'
        wall = self.tags == 'wall'
        inlet = self.tags == 'inlet'
        # At pressure openings an incoming flow carries the specified inlet T.
        adv = c.rho*c.heat_capacity*np.sum(flux[exterior]*np.where(flux[exterior] > 0,
                                                               self.theta.value[self.owner[exterior]], 0.))
        heat_in = np.zeros(len(self.tags))
        conductance = np.asarray(self.face_diffusivity)*self.face_area/np.asarray(self.mesh._cellDistances)
        heat_in[inlet] = -c.rho*c.heat_capacity*conductance[inlet]*self.theta.value[self.owner[inlet]]
        if c.thermal_wall == 'isothermal':
            heat_in[wall] = c.rho*c.heat_capacity*conductance[wall]*(
                c.temperature_wall-c.temperature_inlet-self.theta.value[self.owner[wall]])
        storage = 0. if dt is None else c.rho*c.heat_capacity*np.sum(
            self.area*(self.theta.value-self.theta.old.value))/dt
        return energy_budget(c, self.temperature, storage, adv, heat_in[wall].sum(),
                             heat_in[inlet].sum(), c.heat_source*self.area.sum(), residual)
