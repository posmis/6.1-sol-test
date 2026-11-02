"""Staggered-grid projection, donor-cell transport and high-Re k-epsilon.

SI units throughout. The pressure solve uses mechanical RANS pressure p/rho+2k/3;
saved results contain static pressure relative to the outlet mean.
Smooth equilibrium wall functions model wall stress.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np
from threadpoolctl import threadpool_limits
from .scenarios import SCENARIOS, PROFILE_SCENARIO, NOZZLE_SCENARIO, geometry
from .pressure import PressureSolver
from .kernels import diffuse as diffusion, advect as upwind, scalar_transport


@dataclass(frozen=True)
class Config:
    scenario: str = "channel"
    nx: int = 96
    ny: int = 24
    length: float = 0.6
    height: float = 0.04
    speed: float = 1.0
    rho: float = 998.2
    nu: float = 1.004e-6
    duration: float = 1.2
    initial_condition: str = "uniform"
    time_step: float = 0.  # zero: automatic; otherwise an upper bound in seconds
    frames: int = 80
    intensity: float = 0.05
    turbulence: bool = True
    energy: bool = False
    heat_capacity: float = 4182.  # J/(kg K)
    conductivity: float = .6  # W/(m K)
    turbulent_prandtl: float = .85
    temperature_initial: float = 293.15
    temperature_inlet: float = 293.15
    temperature_wall: float = 333.15
    thermal_wall: str = "adiabatic"
    heat_source: float = 0.  # W/m^3, uniform volumetric heating
    top: str = "wall"
    bottom: str = "wall"
    mode: str = "transient"
    max_iterations: int = 1000
    tolerance: float = 1e-4
    coarse_start: bool = True
    mesh_path: str | None = None
    profile_angle: float = 4.
    profile_chord: float = .1
    profile_cells: int = 800
    profile_sweep: tuple = ()
    nozzle_model: str = "gas"
    nozzle_inlet_height: float = .04
    nozzle_throat_height: float = .016
    nozzle_exit_height: float = .032
    nozzle_converging_length: float = .08
    nozzle_diverging_length: float = .12
    nozzle_total_pressure: float = 300000.
    nozzle_total_temperature: float = 300.
    nozzle_back_pressure: float = 30000.
    nozzle_gamma: float = 1.4
    nozzle_gas_constant: float = 287.05

    def __post_init__(self):
        object.__setattr__(self, "profile_sweep", tuple(self.profile_sweep))

    @property
    def is_compressible(self):
        return self.scenario == NOZZLE_SCENARIO and self.nozzle_model == "gas"

    @property
    def thermal_diffusivity(self):
        return self.conductivity/(self.rho*self.heat_capacity)

    @property
    def field_count(self):
        return 5+int(self.energy or self.is_compressible)+2*int(self.is_compressible)

    @property
    def frame_count(self):
        if self.mode == "steady":
            return 2
        # At most 256 MiB of stored float32 fields, independent of requested mesh.
        return min(self.frames, max(2, (256*1024**2)//(self.nx*self.ny*self.field_count*4)))

    def validate(self):
        if self.scenario == NOZZLE_SCENARIO:
            if self.nozzle_model not in ("gas", "liquid"):
                raise ValueError("Модель сопла: gas или liquid")
            names = ("nozzle_inlet_height", "nozzle_throat_height", "nozzle_exit_height",
                     "nozzle_converging_length", "nozzle_diverging_length")
            if self.nozzle_model == "gas":
                names += ("nozzle_total_pressure", "nozzle_total_temperature", "nozzle_back_pressure", "nozzle_gas_constant")
            for name in names:
                if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be finite and positive")
            if self.nozzle_throat_height >= min(self.nozzle_inlet_height, self.nozzle_exit_height):
                raise ValueError("Сопло: горловина должна быть уже входа и выхода")
            if self.nozzle_model == "gas" and (not 1.01 <= self.nozzle_gamma <= 1.67
                or self.nozzle_back_pressure >= self.nozzle_total_pressure):
                raise ValueError("Газовое сопло: 1.01≤γ≤1.67 и 0<pвыход<p0")
        if not isinstance(self.turbulence, bool) or not isinstance(self.energy, bool):
            raise ValueError("Physics switches must be boolean")
        if self.thermal_wall not in ("adiabatic", "isothermal"):
            raise ValueError("Thermal wall must be adiabatic or isothermal")
        if not np.isfinite(self.heat_source):
            raise ValueError("Heat source must be finite")
        for name in ("heat_capacity", "conductivity", "turbulent_prandtl", "temperature_initial",
                     "temperature_inlet", "temperature_wall"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive (temperature in K)")
        if self.scenario not in SCENARIOS and self.scenario not in ("custom", PROFILE_SCENARIO, NOZZLE_SCENARIO):
            raise ValueError("Unknown scenario")
        if self.mode == "transient" and self.profile_sweep:
            raise ValueError("Поляра по углам доступна в стационарном режиме; нестационарный расчёт — один угол")
        if (not np.isfinite(self.profile_angle) or not -30 <= self.profile_angle <= 30
            or not np.isfinite(self.profile_chord) or not .001 <= self.profile_chord <= 10
            or not isinstance(self.profile_cells, int) or not 400 <= self.profile_cells <= 30000
            or len(self.profile_sweep) > 21
            or any(not np.isfinite(a) or not -30 <= a <= 30 for a in self.profile_sweep)):
            raise ValueError("Некорректные параметры профиля: угол ±30°, хорда 0.001…10 м, сетка 400…30000")
        if self.scenario == "custom" and not self.mesh_path:
            raise ValueError("Для своей сетки нужен файл .flowmesh")
        if not (isinstance(self.nx, int) and isinstance(self.ny, int)
                and 16 <= self.nx <= 1024 and 8 <= self.ny <= 1024):
            raise ValueError("Grid must be 16..1024 × 8..1024")
        if self.top not in ("wall", "open") or self.bottom not in ("wall", "open"):
            raise ValueError("Top and bottom boundaries must be wall or open")
        if self.mode not in ("transient", "steady") or not 10 <= self.max_iterations <= (200000 if self.is_compressible else 10000):
            raise ValueError("Invalid solution mode or iteration limit")
        if not np.isfinite(self.tolerance) or not 1e-8 <= self.tolerance <= 1e-2:
            raise ValueError("Tolerance must be between 1e-8 and 1e-2")
        if self.scenario in SCENARIOS and self.initial_condition == "rest":
            raise ValueError("Пуск из покоя доступен для профиля и своей сетки")
        if self.initial_condition not in ("rest", "uniform"):
            raise ValueError("Начальное состояние: rest или uniform")
        if not np.isfinite(self.time_step) or self.time_step < 0:
            raise ValueError("Шаг времени должен быть неотрицательным (0 — авто)")
        for name in ("length", "height", "speed", "rho", "nu", "duration"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 2 <= self.frames <= 500 or not 0.001 <= self.intensity <= 0.2:
            raise ValueError("Invalid frames or turbulence intensity")


@dataclass
class Result:
    config: Config
    time: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    k: np.ndarray
    epsilon: np.ndarray
    fluid: np.ndarray
    diagnostics: dict
    native: dict | None = None
    temperature: np.ndarray | None = None
    mach: np.ndarray | None = None
    density: np.ndarray | None = None

    def save(self, path):
        extra = {"native_"+key: value for key, value in (self.native or {}).items()}
        for key in ("temperature", "mach", "density"):
            if getattr(self, key) is not None:
                extra[key] = getattr(self, key)
        np.savez_compressed(path, config=json.dumps(asdict(self.config)),
                            diagnostics=json.dumps(self.diagnostics), time=self.time,
                            u=self.u, v=self.v, pressure=self.pressure, k=self.k,
                            epsilon=self.epsilon, fluid=self.fluid, **extra)

    def export_native(self, path):
        if self.native is None:
            raise ValueError("В результате нет неструктурированной сетки")
        import meshio
        native = self.native
        cells, data = [], {key: [] for key in ("u", "v", "pressure", "k", "epsilon", "temperature", "mach", "density", "pressure_absolute", "total_energy") if key in native}
        for count, kind in ((3, "triangle"), (4, "quad")):
            mask = np.sum(native["cells"] >= 0, axis=1) == count
            if mask.any():
                cells.append((kind, native["cells"][mask, :count]))
                for key in data:
                    data[key].append(native[key][mask])
        points = np.column_stack([native["points"], np.zeros(len(native["points"]))])
        meshio.write(path, meshio.Mesh(points, cells, cell_data=data))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            cfg = Config(**json.loads(str(z["config"])))
            cfg.validate()
            arrays = {key: z[key].copy() for key in
                      ("time", "u", "v", "pressure", "k", "epsilon", "fluid")}
            shape = (len(arrays["time"]), cfg.ny, cfg.nx)
            if (shape[0] not in (cfg.frame_count, cfg.frames) or arrays["time"].ndim != 1
                or not np.isfinite(arrays["time"]).all()
                or not np.all(np.diff(arrays["time"]) > 0)
                or arrays["fluid"].dtype != bool
                or arrays["fluid"].shape != shape[1:]) or any(
                arrays[key].shape != shape or not np.isfinite(arrays[key]).all()
                for key in ("u", "v", "pressure", "k", "epsilon")
            ):
                raise ValueError("Invalid result dimensions or non-finite fields")
            temperature = z["temperature"].copy() if "temperature" in z.files else None
            if cfg.energy != (temperature is not None) or (temperature is not None and (
                temperature.shape != shape or not np.isfinite(temperature).all())):
                raise ValueError("Missing or invalid temperature field")
            gas_fields = {}
            for key in ("mach", "density"):
                value = z[key].copy() if key in z.files else None
                if cfg.is_compressible != (value is not None) or (value is not None and (
                    value.shape != shape or not np.isfinite(value).all())):
                    raise ValueError(f"Missing or invalid {key} field")
                gas_fields[key] = value
            native = {key[7:]: z[key].copy() for key in z.files if key.startswith("native_")}
            return cls(cfg, **arrays, diagnostics=json.loads(str(z["diagnostics"])), native=native or None, temperature=temperature, **gas_fields)


class Solver:
    def __init__(self, config=Config()):
        config.validate()
        if config.scenario == NOZZLE_SCENARIO:
            from .nozzle import normalize
            config = normalize(config)
        self.c = c = config
        if c.scenario in ("custom", PROFILE_SCENARIO) or c.is_compressible:
            return
        self.dx, self.dy = c.length/c.nx, c.height/c.ny
        self.fluid = geometry(c)
        f = self.fluid
        self.um = np.zeros((c.ny, c.nx+1), bool)
        self.vm = np.zeros((c.ny+1, c.nx), bool)
        self.um[:, 1:-1] = f[:, :-1] & f[:, 1:]
        self.um[:, 0], self.um[:, -1] = f[:, 0], f[:, -1]
        self.vm[1:-1] = f[:-1] & f[1:]
        if c.bottom == "open":
            self.vm[0] = f[0]
        if c.top == "open":
            self.vm[-1] = f[-1]
        self.u = self.um.astype(float)*c.speed
        self.v = np.zeros_like(self.vm, float)
        self.p = np.zeros_like(f, float)
        self.k = np.full(f.shape, 1.5*(c.speed*c.intensity)**2 if c.turbulence else 0.)
        self.e = 0.09**0.75*self.k**1.5/(0.07*c.height)
        self.kin, self.ein = float(self.k[0, 0]), float(self.e[0, 0])
        self.temperature = np.full(f.shape, c.temperature_initial) if c.energy else None
        self.energy_history = []
        self.walls = []
        for axis, sign, spacing in ((0, -1, self.dy), (0, 1, self.dy),
                                     (1, -1, self.dx), (1, 1, self.dx)):
            neighbour = np.roll(f, -sign, axis=axis)
            edge = [slice(None), slice(None)]
            edge[axis] = -1 if sign == 1 else 0
            neighbour[tuple(edge)] = (axis == 1 or
                (c.bottom if sign == -1 else c.top) == "open")
            self.walls.append((axis, spacing, f & ~neighbour))
        self.wall = np.logical_or.reduce([w[2] for w in self.walls])
        self._pressure_matrix()
        self.project(1.0)  # admissible divergence-free initial velocity, including the step
        self.p.fill(0)

    def _pressure_matrix(self):
        self.solve_p = PressureSolver(self.fluid, self.dx, self.dy, self.c.bottom, self.c.top)

    def centres(self):
        return (self.u[:, :-1]+self.u[:, 1:])*0.5, (self.v[:-1]+self.v[1:])*0.5

    def divergence(self):
        return np.diff(self.u, axis=1)/self.dx + np.diff(self.v, axis=0)/self.dy

    def project(self, dt):
        self.p[self.fluid] = self.solve_p(-self.divergence()[self.fluid]/dt)
        self.u[:, 1:-1] -= dt*np.diff(self.p, axis=1)/self.dx*self.um[:, 1:-1]
        self.u[:, -1] += dt*2*self.p[:, -1]/self.dx
        self.v[1:-1] -= dt*np.diff(self.p, axis=0)/self.dy*self.vm[1:-1]
        if self.c.bottom == "open":
            self.v[0] -= dt*2*self.p[0]/self.dy*self.vm[0]
        if self.c.top == "open":
            self.v[-1] += dt*2*self.p[-1]/self.dy*self.vm[-1]
        self.u *= self.um
        self.v *= self.vm

    def wall_terms(self, uc, vc):
        fx, fy = np.zeros_like(uc), np.zeros_like(vc)
        kw, ew, prod = np.zeros_like(uc), np.zeros_like(uc), np.zeros_like(uc)
        count = np.zeros_like(uc)
        for axis, h, mask in self.walls:
            if not mask.any():
                continue
            tangent = (uc if axis == 0 else vc)[mask]
            speed = np.abs(tangent)
            y = h*0.5
            # Solve smooth log law U/u_tau = log(E*y*u_tau/nu)/kappa.
            ut = np.maximum(np.sqrt(self.c.nu*speed/y), 1e-9)
            if self.c.turbulence:
                for _ in range(8):
                    log = np.log(np.maximum(9.8*y*ut/self.c.nu, 1.01))
                    residual = ut*log/0.41-speed
                    ut = np.maximum(ut-residual/((log+1)/0.41), 1e-9)
                ut = np.where(y*ut/self.c.nu > 11.06, ut, np.sqrt(self.c.nu*speed/y))
            force = -np.sign(tangent)*ut**2/h
            if axis == 0:
                fx[mask] += force
            else:
                fy[mask] += force
            kw[mask] += ut**2/0.3
            ew[mask] += ut**3/(0.41*y)
            prod[mask] += ut**3/(0.41*y)
            count[mask] += 1
        return fx, fy, kw/np.maximum(count, 1), ew/np.maximum(count, 1), prod

    def step(self, remaining):
        c, dx, dy = self.c, self.dx, self.dy
        uc, vc = self.centres()
        nut = np.minimum(0.09*self.k**2/np.maximum(self.e, 1e-12), 0.1*c.speed*c.height)
        viscosity = c.nu+nut
        max_diffusivity = float(np.max(viscosity))
        if c.energy:
            max_diffusivity = max(max_diffusivity, float(np.max(c.thermal_diffusivity+nut/c.turbulent_prandtl)))
        dt = min(0.35/(np.max(abs(self.u))/dx+np.max(abs(self.v))/dy+1e-9),
                 0.2/(max_diffusivity*(1/dx**2+1/dy**2)), remaining)
        fx, fy, kw, ew, wallprod = self.wall_terms(uc, vc)
        # Staggered velocities advect on their own control-volume locations.
        vu = np.pad((vc[:, :-1]+vc[:, 1:])*0.5, ((0, 0), (1, 1)), mode="edge")
        uv = np.pad((uc[:-1]+uc[1:])*0.5, ((1, 1), (0, 0)), mode="edge")
        nuu = np.pad((viscosity[:, :-1]+viscosity[:, 1:])*0.5, ((0, 0), (1, 1)), mode="edge")
        nuv = np.pad((viscosity[:-1]+viscosity[1:])*0.5, ((1, 1), (0, 0)), mode="edge")
        # Variable-viscosity symmetric strain stress: Laplacian term + transpose term.
        duy, dux = np.gradient(uc, dy, dx)
        dvy, dvx = np.gradient(vc, dy, dx)
        dnuty, dnutx = np.gradient(nut, dy, dx)
        fx += dnutx*dux+dnuty*dvx
        fy += dnutx*duy+dnuty*dvy
        fux = np.pad((fx[:, :-1]+fx[:, 1:])*0.5, ((0, 0), (1, 1)), mode="edge")
        fvy = np.pad((fy[:-1]+fy[1:])*0.5, ((1, 1), (0, 0)), mode="edge")
        newu = self.u+dt*(-upwind(self.u, self.u, vu, dx, dy)
                          +diffusion(self.u, nuu, self.um, dx, dy)+fux)
        newv = self.v+dt*(-upwind(self.v, uv, self.v, dx, dy)
                          +diffusion(self.v, nuv, self.vm, dx, dy)+fvy)
        newu[:, 0] = c.speed*self.um[:, 0]
        newu[:, -1] = newu[:, -2]
        if c.bottom == "open":
            newv[0] = newv[1]
        if c.top == "open":
            newv[-1] = newv[-2]
        self.u, self.v = newu*self.um, newv*self.vm
        self.project(dt)
        if c.turbulence:
            production = np.minimum(nut*(2*dux**2+2*dvy**2+(duy+dvx)**2), 10*self.e)
            production[self.wall] = wallprod[self.wall]
            # Conservative donor-cell scalar fluxes with specified inlet values.
            def transport(q, inlet):
                return scalar_transport(q, self.u, self.v, inlet, dx, dy)
            oldk = self.k.copy()
            self.k = np.maximum((self.k+dt*(transport(self.k, self.kin)
                        +diffusion(self.k, c.nu+nut, self.fluid, dx, dy)+production))
                        /(1+dt*self.e/np.maximum(oldk, 1e-12)), 1e-10)
            self.e = np.maximum((self.e+dt*(transport(self.e, self.ein)
                        +diffusion(self.e, c.nu+nut/1.3, self.fluid, dx, dy)
                        +1.44*production*self.e/np.maximum(oldk, 1e-12)))
                        /(1+dt*1.92*self.e/np.maximum(oldk, 1e-12)), 1e-10)
            self.k[self.wall] = np.maximum(kw[self.wall], 1e-10)
            self.e[self.wall] = np.maximum(ew[self.wall], 1e-10)
            self.k[~self.fluid], self.e[~self.fluid] = 0, 0
        if c.energy:
            from .thermal import structured_step
            structured_step(self, nut, dt)
        if not all(np.isfinite(q).all() for q in (self.u, self.v, self.k, self.e)):
            raise FloatingPointError("Solution diverged. Reduce speed or refine the grid.")
        return dt

    @threadpool_limits.wrap(limits=1, user_api="blas")
    def run(self, progress=None, cancel=None, preview=None):
        if self.c.is_compressible:
            from .nozzle import solve
            return solve(self.c, progress, cancel, preview=preview)
        if self.c.scenario == PROFILE_SCENARIO:
            from .airfoil import solve_profile
            return solve_profile(self.c, progress, cancel, preview=preview)
        if self.c.scenario == "custom":
            from .unstructured import solve
            return solve(self.c, progress, cancel, preview=preview)
        if self.c.mode == "steady":
            from .steady import solve
            return solve(self, progress, cancel, preview=preview)
        from .live import PreviewEmitter, structured_values
        live = PreviewEmitter(preview, self.c, self.fluid) if preview else None
        if live:
            live.emit(0, 0., lambda: structured_values(self), force=True)
        self.solve_p.cancel = cancel
        times = np.linspace(0, self.c.duration, self.c.frame_count)
        fields = [np.empty((len(times), self.c.ny, self.c.nx), np.float32) for _ in range(5)]
        temperature = np.empty_like(fields[0]) if self.c.energy else None
        t, steps, maxdiv = 0., 0, 0.
        last, change = None, 1.
        for frame, target in enumerate(times):
            while t < target-1e-12:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("Calculation cancelled")
                t += self.step(min(target-t, self.c.time_step) if self.c.time_step else target-t)
                steps += 1
                if live:
                    live.emit(steps, t, lambda: structured_values(self), phase="step")
                maxdiv = max(maxdiv, float(np.max(abs(self.divergence()[self.fluid]))))
            uc, vc = self.centres()
            if last is not None:
                change = float(np.linalg.norm((uc-last)[self.fluid]) /
                               max(np.linalg.norm(uc[self.fluid]), 1e-12))
            last = uc.copy()
            static = (self.p-2*self.k/3+2*np.mean(self.k[:, -1][self.fluid[:, -1]])/3)*self.c.rho
            for dest, q in zip(fields, (uc, vc, static, self.k, self.e)):
                dest[frame] = np.where(self.fluid, q, 0)
            if temperature is not None:
                temperature[frame] = np.where(self.fluid, self.temperature, 0.)
                if steps:
                    self.energy_history.append(dict(time=t, **self.energy_budget))
            if progress:
                progress(frame+1, len(times), t)
        inflow = float(self.u[:, 0].sum()*self.dy)
        outflow = float(self.u[:, -1].sum()*self.dy)
        sideflow = float((self.v[-1].sum()-self.v[0].sum())*self.dx)
        diagnostics = dict(steps=steps, max_divergence=maxdiv,
            mass_imbalance=abs(outflow+sideflow-inflow)/max(inflow, 1e-12),
            side_flow_m2_s=sideflow, stored_frames=len(times), pressure_solver=self.solve_p.method,
            final_frame_velocity_change=change,
            reynolds=self.c.speed*self.c.height/self.c.nu,
            pressure_drop_pa=float(np.mean(static[:, 0][self.fluid[:, 0]])
                                   -np.mean(static[:, -1][self.fluid[:, -1]])),
            backflow=bool(np.any(self.u[:, -1] < 0)),
            note="Finite-time flow result; small divergence alone does not establish steady convergence.")
        diagnostics["turbulence_model"] = "k-epsilon" if self.c.turbulence else "laminar"
        if self.c.energy:
            diagnostics["energy"] = dict(history=self.energy_history, **self.energy_budget)
        if live:
            live.emit(steps, t, lambda: structured_values(self), phase="step", force=True)
        return Result(self.c, times, *fields, self.fluid.copy(), diagnostics, temperature=temperature)
