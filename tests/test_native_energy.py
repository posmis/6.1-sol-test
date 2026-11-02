import numpy as np
import pytest
import meshio
from flowdesk.cad import example_geometry
from flowdesk.meshing import MeshSettings, generate
from flowdesk.solver import Config, Solver, Result


@pytest.mark.parametrize('kind,turbulence', [('triangle', False), ('quad', False), ('triangle', True)])
@pytest.mark.parametrize('mode', ['steady', 'transient'])
def test_native_energy_uses_real_cells_and_conserves_heat(tmp_path, kind, turbulence, mode):
    g = example_geometry()
    g.loops[0][1][0] = g.loops[0][2][0] = .12
    mesh = generate(g, MeshSettings(cell_type=kind, target_cells=100))
    path = tmp_path/'channel.flowmesh'
    mesh.save(path)
    c = Config(scenario='custom', mesh_path=str(path), mode=mode, nx=48, ny=16,
        turbulence=turbulence, energy=True, heat_capacity=1000., rho=1., conductivity=.02,
        speed=.1, nu=1e-4, thermal_wall='isothermal', frames=3, duration=.02, max_iterations=500)
    r = Solver(c).run()
    assert r.diagnostics['converged']
    assert r.temperature.shape == r.u.shape
    assert len(r.native['temperature']) == len(mesh.cells)
    assert r.native['temperature'].min() >= c.temperature_inlet-1e-3
    assert r.native['temperature'].max() < c.temperature_wall+1e-3
    assert r.native['temperature'].max() > c.temperature_inlet+.1
    assert r.diagnostics['energy']['relative_imbalance'] < 1e-6
    assert r.diagnostics['energy']['wall_heat_w_per_m'] > 0
    assert r.diagnostics['mass_imbalance'] < 1e-7
    if not turbulence:
        assert np.all(r.native['k'] == 0.) and np.all(r.native['epsilon'] == 0.)
    if mode == 'transient':
        assert np.linalg.norm(r.temperature[-1]-r.temperature[0]) > 1.
    r.save(tmp_path/'heated.npz')
    path.unlink()
    loaded = Result.load(tmp_path/'heated.npz')
    np.testing.assert_array_equal(loaded.temperature, r.temperature)
    loaded.export_native(tmp_path/'heated.vtu')
    exported = meshio.read(tmp_path/'heated.vtu')
    np.testing.assert_array_equal(exported.cell_data['temperature'][0], r.native['temperature'])


def test_laminar_force_uses_viscous_wall_gradient():
    from flowdesk.airfoil import coefficients
    r = coefficients([0.], np.array([[0.], [1.]]), np.array([.1]), np.array([[20.], [0.]]),
                     np.array([.01]), rho=1., nu=1e-6, speed=20., chord=.1, turbulence=False)
    assert r['shear_drag_n_per_m'] == pytest.approx(1e-6*20/.01*.1)


def test_native_open_backflow_carries_inlet_temperature_and_balances_energy():
    # A diagonal, divergence-free velocity enters through an inlet and an open
    # bottom boundary. This checks the actual incoming-energy boundary treatment.
    from fipy import Grid2D, FaceVariable
    from flowdesk.thermal import NativeThermal
    from flowdesk.steady import LinearSolves
    mesh = Grid2D(nx=12, ny=8, dx=.01, dy=.01)
    tags = np.full(mesh.numberOfFaces, 'interior', dtype='U8')
    tags[np.asarray(mesh.facesLeft)] = 'inlet'
    tags[np.asarray(mesh.facesRight)] = 'outlet'
    tags[np.asarray(mesh.facesTop) | np.asarray(mesh.facesBottom)] = 'open'
    velocity = FaceVariable(mesh=mesh, rank=1, value=((1.,), (.5,)))
    c = Config(energy=True, turbulence=False, length=.12, height=.08, rho=1.,
               heat_capacity=1000., conductivity=.001, temperature_initial=333.15)
    thermal = NativeThermal(c, mesh, tags, velocity, np.asarray(mesh.faceCellIDs[0], int),
        np.asarray(mesh.faceNormals), np.asarray(mesh._faceAreas), np.asarray(mesh.cellVolumes))
    thermal.theta.updateOld()
    budget = thermal.advance(LinearSolves(None), np.zeros(mesh.numberOfCells), .01)
    assert budget['relative_imbalance'] < 1e-8
    assert c.temperature_inlet <= thermal.temperature.min() < 330.
    assert thermal.temperature.max() <= c.temperature_initial+1e-8
    assert budget['wall_heat_w_per_m'] == 0.


def test_heated_profile_uses_wall_of_actual_hole():
    c = Config(scenario='elliptic_profile', profile_cells=400, nx=128, ny=64,
        speed=20., rho=1.225, nu=1.5e-5, intensity=.01, energy=True,
        heat_capacity=1005., conductivity=.0257, thermal_wall='isothermal',
        duration=.001, frames=3, max_iterations=300)
    r = Solver(c).run()
    assert len(r.native['temperature']) == r.diagnostics['mesh_cells']
    assert r.native['temperature'].min() >= c.temperature_inlet-1e-4
    assert c.temperature_inlet+1. < r.native['temperature'].max() < c.temperature_wall
    assert np.all(r.temperature[:, ~r.fluid] == 0.)
    assert np.linalg.norm(r.temperature[-1]-r.temperature[0]) > 1.
    assert r.diagnostics['energy']['relative_imbalance'] < 1e-6
