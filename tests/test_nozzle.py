from dataclasses import replace
from threading import Event
import numpy as np
import pytest
import meshio
from flowdesk.solver import Config, Solver, Result
from flowdesk.scenarios import NOZZLE_SCENARIO
from flowdesk.nozzle import normalize, height_at, grid, rhs


@pytest.fixture(scope='module')
def supersonic():
    c = Config(scenario=NOZZLE_SCENARIO, nx=64, ny=16, mode='steady', max_iterations=6000)
    frames = []
    r = Solver(c).run(preview=frames.append)
    return r, frames


def test_supersonic_flow_matches_isentropic_nozzle_reference(supersonic):
    r, _ = supersonic
    n = r.diagnostics['nozzle']
    assert r.diagnostics['converged'], r.diagnostics['residuals']
    assert abs(n['exit_mach']/n['design_exit_mach']-1) < .03
    assert abs(n['mass_out_kg_s_per_m']/n['choked_mass_flow_kg_s_per_m']-1) < .02
    assert abs(n['throat_mach']-1) < .1
    assert r.diagnostics['mass_imbalance'] < .001
    assert n['relative_mass_conservation_error'] < 1e-10
    assert n['relative_energy_conservation_error'] < 1e-10
    assert r.diagnostics['max_divergence'] > 1.
    assert r.config.energy and not r.config.turbulence
    assert np.all(r.k == 0.)


def test_two_dimensional_fields_and_total_temperature(supersonic):
    r, frames = supersonic
    c, native = r.config, r.native
    assert np.max(abs(native['v'])) > 5.  # Solved transverse momentum, not an extruded 1D solution.
    temp0 = native['temperature']+(native['u']**2+native['v']**2)/(2*c.heat_capacity)
    assert np.max(abs(temp0-c.nozzle_total_temperature))/c.nozzle_total_temperature < .02
    absolute = native['pressure']+c.nozzle_back_pressure
    np.testing.assert_allclose(absolute, native['density']*c.nozzle_gas_constant*native['temperature'], rtol=1e-10)
    for key in ('u', 'temperature', 'density', 'mach'):
        q = native[key].reshape(c.ny, c.nx)
        np.testing.assert_allclose(q, q[::-1], atol=1e-7)
    assert frames[-1].phase == 'euler'
    np.testing.assert_array_equal(frames[-1].mach, r.mach[-1])
    np.testing.assert_array_equal(frames[-1].density, r.density[-1])


def test_nozzle_result_roundtrip_and_native_quad_export(tmp_path, supersonic):
    r, _ = supersonic
    r.save(tmp_path/'nozzle.npz')
    restored = Result.load(tmp_path/'nozzle.npz')
    assert restored.config == r.config
    for key in ('mach', 'density', 'temperature', 'pressure'):
        np.testing.assert_array_equal(getattr(restored, key), getattr(r, key))
    restored.export_native(tmp_path/'nozzle.vtu')
    vtk = meshio.read(tmp_path/'nozzle.vtu')
    assert len(vtk.cells_dict['quad']) == r.config.nx*r.config.ny
    assert {'mach', 'density', 'temperature'} <= set(vtk.cell_data)


def test_geometry_settings_and_rest_state_are_conservative():
    c = normalize(Config(scenario=NOZZLE_SCENARIO, nx=64, ny=16, nozzle_converging_length=.1,
                        nozzle_diverging_length=.2, nozzle_exit_height=.05))
    assert c.length == pytest.approx(.3) and c.height == .05
    heights = height_at(c, np.array([0., .1, .3]))
    np.testing.assert_allclose(heights, [c.nozzle_inlet_height, c.nozzle_throat_height, c.nozzle_exit_height])
    h, syx, area, points, cells, edges, tags = grid(c)
    exact_area = .1*(c.nozzle_inlet_height+c.nozzle_throat_height)/2+.2*(c.nozzle_throat_height+c.nozzle_exit_height)/2
    assert area.sum() == pytest.approx(exact_area, rel=3e-5)
    p = points[cells]
    actual_area = .5*np.sum(p[:, :, 0]*np.roll(p[:, :, 1], -1, axis=1)-p[:, :, 1]*np.roll(p[:, :, 0], -1, axis=1), axis=1)
    np.testing.assert_allclose(actual_area, area.ravel(), atol=1e-16)
    w = np.zeros((4, c.ny, c.nx))
    w[0] = c.nozzle_total_pressure/(c.nozzle_gas_constant*c.nozzle_total_temperature)
    w[3] = c.nozzle_total_pressure/(c.nozzle_gamma-1)
    derivative, _ = rhs(w, h, syx, area, c.length/c.nx, c.nozzle_gamma, c.nozzle_gas_constant,
                       c.nozzle_total_pressure, c.nozzle_total_temperature, c.nozzle_total_pressure)
    assert np.max(abs(derivative)) < 1e-4


def test_transient_nozzle_records_real_pressure_temperature_mach_changes():
    c = Config(scenario=NOZZLE_SCENARIO, nx=48, ny=12, duration=.0015, frames=6)
    r = Solver(c).run()
    assert r.time[-1] == c.duration
    assert np.all(r.mach[0] == 0.)
    assert r.mach[-1].max() > 1.
    assert np.linalg.norm(r.pressure[-1]-r.pressure[1]) > 1000.
    assert r.temperature[-1, r.fluid].min() < c.nozzle_total_temperature-10.
    assert r.density[-1, r.fluid].min() > 0.
    assert r.diagnostics['nozzle']['relative_energy_conservation_error'] < 1e-10


@pytest.mark.parametrize('mode', ['steady', 'transient'])
def test_incompressible_liquid_mode_uses_adjustable_nozzle(mode):
    c = Config(scenario=NOZZLE_SCENARIO, nozzle_model='liquid', nx=32, ny=16,
        nozzle_throat_height=.012, nozzle_diverging_length=.16, mode=mode, max_iterations=300,
        turbulence=False, energy=True, heat_source=10000., speed=.01, nu=1e-4, duration=.01, frames=3)
    r = Solver(c).run()
    assert r.config.length == pytest.approx(.24)
    assert r.fluid[:, 0].sum() > r.fluid[:, 11].sum()
    assert r.diagnostics['mass_imbalance'] < 1e-8
    assert r.mach is None and r.density is None
    assert r.temperature is not None
    assert not r.config.is_compressible
    if mode == 'steady':
        assert r.diagnostics['converged']


def test_nozzle_validation_and_cancellation():
    for c in (Config(scenario=NOZZLE_SCENARIO, nozzle_throat_height=.04),
              Config(scenario=NOZZLE_SCENARIO, nozzle_back_pressure=400000),
              Config(scenario=NOZZLE_SCENARIO, nozzle_gamma=1),
              Config(scenario=NOZZLE_SCENARIO, nozzle_total_temperature=-1),
              Config(scenario=NOZZLE_SCENARIO, nozzle_model='other')):
        with pytest.raises(ValueError):
            Solver(c)
    event = Event()
    event.set()
    with pytest.raises(InterruptedError):
        Solver(Config(scenario=NOZZLE_SCENARIO, nx=16, ny=8)).run(cancel=event)
    assert Config(scenario=NOZZLE_SCENARIO, nx=1024, ny=1024).frame_count == 8


def test_high_back_pressure_is_subsonic_and_below_choked_mass_flow():
    c = Config(scenario=NOZZLE_SCENARIO, nx=48, ny=12, mode='steady',
               nozzle_back_pressure=291000., max_iterations=45000, tolerance=2e-4)
    r = Solver(c).run()
    n = r.diagnostics['nozzle']
    assert r.diagnostics['converged'], r.diagnostics['residuals']
    assert r.native['mach'].max() < 1.
    assert n['mass_out_kg_s_per_m'] < n['choked_mass_flow_kg_s_per_m']
    exact_m = np.sqrt(2/(c.nozzle_gamma-1)*((c.nozzle_total_pressure/c.nozzle_back_pressure)**((c.nozzle_gamma-1)/c.nozzle_gamma)-1))
    assert abs(n['exit_mach']/exact_m-1) < .08


def test_internal_shock_with_back_pressure_conserves_mass_and_energy():
    c = Config(scenario=NOZZLE_SCENARIO, nx=48, ny=12, duration=.008, frames=5,
               nozzle_back_pressure=225000.)
    r = Solver(c).run()
    m = r.native['mach'].reshape(c.ny, c.nx).mean(axis=0)
    p = r.native['pressure_absolute'].reshape(c.ny, c.nx).mean(axis=0)
    assert m.max() > 1.2 and m[-1] < .7
    assert p[-1] > 1.5*p[np.argmax(m)]
    assert abs(p[-1]/c.nozzle_back_pressure-1) < .03
    assert r.diagnostics['nozzle']['relative_mass_conservation_error'] < 1e-10
    assert r.diagnostics['nozzle']['relative_energy_conservation_error'] < 1e-10
