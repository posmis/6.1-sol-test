"""Actual physical-time transport, conservative native fluxes and display traces."""
from dataclasses import replace
import numpy as np
import pytest
from flowdesk.solver import Config, Solver, Result
from flowdesk.airfoil import export_csv
from flowdesk.cad import Geometry
from flowdesk.meshing import generate, MeshSettings
from flowdesk.unstructured import implicit_solve
from flowdesk.steady import LinearSolves


def test_backward_euler_diffusion_temporal_refinement():
    # Exact heat eigenmode on an orthogonal grid. No nonlinear/pressure ambiguity:
    # checks that FiPy receives dt and each step uses the previous physical state.
    from fipy import Grid2D, CellVariable, TransientTerm, DiffusionTerm
    mesh = Grid2D(nx=32, ny=4, dx=1/32, dy=.25)
    x = np.asarray(mesh.cellCenters[0])
    exact = np.sin(np.pi*x)*np.exp(-np.pi**2*.04)
    errors = []
    for dt in (.01, .005):
        q = CellVariable(mesh=mesh, value=np.sin(np.pi*x), hasOld=True)
        q.constrain(0., mesh.facesLeft | mesh.facesRight)
        equation = TransientTerm()-DiffusionTerm()
        linear = LinearSolves(None)
        for _ in range(round(.04/dt)):
            q.updateOld()
            implicit_solve(linear, equation, q, 'heat', 1., 1., dt=dt)
        errors.append(np.linalg.norm(q.value-exact)/np.linalg.norm(exact))
    assert errors[1] < .6*errors[0]
    assert errors[1] < .011


@pytest.mark.parametrize('kind', ['triangle', 'quad'])
def test_native_obstacle_physical_frames_and_reload(tmp_path, kind):
    theta = np.linspace(2*np.pi, 0., 25)[:-1]
    body = np.column_stack([.04+.005*np.cos(theta), .02+.005*np.sin(theta)]).tolist()
    geometry = Geometry([[[0., 0.], [.12, 0.], [.12, .04], [0., .04]], body],
                        [["wall", "outlet", "wall", "inlet"], ["wall"]*len(body)])
    mesh = generate(geometry, MeshSettings(cell_type=kind, target_cells=250))
    path = tmp_path/'body.flowmesh'
    mesh.save(path)
    c = Config(scenario='custom', mesh_path=str(path), nx=96, ny=48,
               initial_condition='rest', duration=.003, frames=4, speed=20., nu=1.5e-5, rho=1.225, intensity=.01,
               max_iterations=300)
    progress = []
    r = Solver(c).run(lambda *args: progress.append(args))
    assert len(r.time) == 4 and r.time[-1] == c.duration
    assert np.all(r.u[0] == 0.)
    assert np.linalg.norm(r.u[1]-r.u[0]) > 1.
    assert np.linalg.norm(r.u[3]-r.u[1]) > .01
    assert np.linalg.norm(r.pressure[3]-r.pressure[1]) > .01
    assert np.all(np.isfinite(r.pressure))
    assert np.all(r.u[:, ~r.fluid] == 0.)
    steps = r.diagnostics['time_history']
    assert all(s['residual'] < c.tolerance for s in steps)
    assert max(s['mass_imbalance'] for s in steps) < 1e-7
    assert sum(s['dt'] for s in steps) == pytest.approx(c.duration)
    assert progress[-1] == (3, 3, c.duration)
    r.save(tmp_path/'transient.npz')
    path.unlink()
    restored = Result.load(tmp_path/'transient.npz')
    np.testing.assert_array_equal(restored.pressure, r.pressure)
    np.testing.assert_array_equal(restored.native['cells'], mesh.cells)


def test_profile_time_forces_and_csv(tmp_path):
    c = Config(scenario='elliptic_profile', profile_cells=400, nx=64, ny=32,
               duration=.002, frames=3, speed=20., nu=1.5e-5, rho=1.225, intensity=.01,
               max_iterations=300)
    r = Solver(c).run()
    history = r.diagnostics['aerodynamics']['history']
    assert all(b['time'] > a['time'] for a, b in zip(history, history[1:]))
    assert history[-1]['time'] == c.duration
    assert r.diagnostics['initial_divergence'] < 1e-6
    assert np.max(r.u[0]) > c.speed
    assert np.linalg.norm(r.u[-1]-r.u[0]) > .01
    assert not r.diagnostics['aerodynamics']['polar']
    assert history[0]['cl'] != history[-1]['cl']
    export_csv(r, tmp_path/'forces.csv')
    assert (tmp_path/'forces.csv').read_text(encoding='utf-8-sig').startswith('time,iteration,')
    r.save(tmp_path/'profile.npz')
    assert Result.load(tmp_path/'profile.npz').config.mesh_path is None
    with pytest.raises(RuntimeError, match='не сошёлся'):
        Solver(replace(c, max_iterations=10)).run()


def test_physical_tracer_displacement_and_scrubbing():
    from flowdesk.visualization import Tracers
    c = Config(nx=32, ny=16, length=1., height=1., speed=2., duration=1., frames=2)
    shape = (2, 16, 32)
    r = Result(c, np.array([0., 1.]), np.full(shape, 2.), np.zeros(shape),
               np.zeros(shape), np.ones(shape), np.ones(shape), np.ones(shape[1:], bool), {})
    tracers = Tracers(r, count=10)
    tracers.advance(0.)
    tracers.points[:] = [.3, .5]
    points, _ = tracers.advance(.02)
    np.testing.assert_allclose(points, np.tile([.34, .5], (10, 1)), atol=1e-12)
    tracers.advance(.03)
    assert len(tracers.trail) == 2
    tracers.advance(.01)
    assert tracers.trail == []
