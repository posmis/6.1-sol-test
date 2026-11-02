"""Switchable flow models and sensible-energy conservation/analytic references."""
from dataclasses import replace, asdict
import json
import numpy as np
import pytest
from scipy.special import erfc
from flowdesk.solver import Config, Result, Solver
from flowdesk.steady import LinearSolves
from flowdesk.thermal import structured_steady, structured_step


def test_laminar_channel_matches_poiseuille():
    c = Config(mode='steady', nx=64, ny=20, turbulence=False, speed=.01, nu=1e-4,
               max_iterations=600, coarse_start=False)
    r = Solver(c).run()
    assert r.diagnostics['converged']
    assert np.all(r.k == 0.) and np.all(r.epsilon == 0.)
    assert r.diagnostics['turbulence_model'] == 'laminar'
    x = (np.arange(c.nx)+.5)*c.length/c.nx
    select = (x > .5*c.length) & (x < .85*c.length)
    gradient = np.polyfit(x[select], r.pressure[-1][:, select].mean(axis=0), 1)[0]
    reference = 12*c.rho*c.nu*c.speed/c.height**2
    assert abs(-gradient/reference-1) < .015
    y = (np.arange(c.ny)+.5)*c.height/c.ny
    exact = 6*c.speed*(y/c.height)*(1-y/c.height)
    np.testing.assert_allclose(r.u[-1, :, -6], exact, atol=.015*c.speed)


@pytest.mark.parametrize('turbulence', [False, True])
@pytest.mark.parametrize('mode', ['steady', 'transient'])
def test_independent_energy_switch_and_temperature_record(tmp_path, turbulence, mode):
    c = Config(mode=mode, turbulence=turbulence, energy=True, nx=32, ny=12,
        speed=.1, nu=1e-4, rho=1., heat_capacity=1000., conductivity=.02,
        frames=4, duration=.02, thermal_wall='isothermal', max_iterations=500)
    r = Solver(c).run()
    assert r.temperature.shape == r.u.shape
    assert np.all(r.temperature[:, ~r.fluid] == 0.)
    assert r.temperature[:, r.fluid].min() >= c.temperature_inlet-1e-4
    assert r.temperature[:, r.fluid].max() <= c.temperature_wall+1e-4
    assert r.temperature[-1, r.fluid].max() > c.temperature_inlet+.1
    if mode == 'transient':
        assert np.linalg.norm(r.temperature[-1]-r.temperature[0]) > .1
    else:
        assert r.diagnostics['converged']
    assert r.diagnostics['energy']['relative_imbalance'] < 1e-7
    # Sensible energy at fixed properties must not feed back into momentum.
    cold = Solver(replace(c, energy=False)).run()
    np.testing.assert_array_equal(r.u, cold.u)
    np.testing.assert_array_equal(r.pressure, cold.pressure)
    assert cold.temperature is None
    r.save(tmp_path/'heated.npz')
    loaded = Result.load(tmp_path/'heated.npz')
    np.testing.assert_array_equal(r.temperature, loaded.temperature)
    assert loaded.config == c


def test_energy_advection_diffusion_against_exact_solution():
    errors = []
    for nx in (32, 64):
        c = Config(nx=nx, ny=8, top='open', bottom='open', turbulence=False, energy=True,
            speed=.1, rho=1., heat_capacity=1000., conductivity=1., heat_source=1000.)
        s = Solver(c)
        structured_steady(s, LinearSolves(None))
        x = (np.arange(nx)+.5)*s.dx
        a = c.thermal_diffusivity
        exact = c.temperature_inlet+c.heat_source/(c.rho*c.heat_capacity*c.speed)*(
            x-a/c.speed*(np.exp((x-c.length)*c.speed/a)-np.exp(-c.length*c.speed/a)))
        errors.append(np.linalg.norm(s.temperature[4]-exact)/np.linalg.norm(exact-c.temperature_inlet))
        assert s.energy_budget['relative_imbalance'] < 1e-9
    assert errors[1] < .65*errors[0]
    assert errors[1] < .025


def test_transient_conduction_against_half_space_solution():
    c = Config(nx=16, ny=64, energy=True, turbulence=False, rho=1., heat_capacity=1000.,
               conductivity=.1, thermal_wall='isothermal')
    s = Solver(c)
    s.u.fill(0.)
    s.v.fill(0.)
    dt, duration = .00025, .02
    for _ in range(round(duration/dt)):
        structured_step(s, np.zeros(s.fluid.shape), dt)
        assert s.energy_budget['relative_imbalance'] < 1e-8
    y = (np.arange(c.ny)+.5)*s.dy
    exact = c.temperature_inlet+(c.temperature_wall-c.temperature_inlet)*(
        erfc(y/(2*np.sqrt(c.thermal_diffusivity*duration)))+
        erfc((c.height-y)/(2*np.sqrt(c.thermal_diffusivity*duration))))
    assert np.linalg.norm(s.temperature[:, 8]-exact)/np.linalg.norm(exact-c.temperature_inlet) < .015


@pytest.mark.parametrize('mode', ['steady', 'transient'])
def test_adiabatic_uniform_temperature_stays_uniform(mode):
    r = Solver(Config(energy=True, mode=mode, nx=32, ny=12, duration=.01, frames=3,
                      max_iterations=500)).run()
    np.testing.assert_allclose(r.temperature[:, r.fluid], 293.15, atol=3e-5)
    assert r.diagnostics['energy']['wall_heat_w_per_m'] == 0.
    assert r.diagnostics['energy']['relative_imbalance'] == 0.


def test_temperature_load_validation_and_legacy_defaults(tmp_path):
    r = Solver(Config(nx=16, ny=8, frames=2, duration=.001)).run()
    r.save(tmp_path/'old.npz')
    with np.load(tmp_path/'old.npz', allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files}
    config = asdict(r.config)
    for key in ('turbulence', 'energy', 'heat_capacity', 'conductivity', 'turbulent_prandtl',
                'temperature_initial', 'temperature_inlet', 'temperature_wall', 'thermal_wall', 'heat_source'):
        config.pop(key)
    payload['config'] = json.dumps(config)
    np.savez(tmp_path/'legacy.npz', **payload)
    legacy = Result.load(tmp_path/'legacy.npz')
    assert legacy.config.turbulence and not legacy.config.energy
    assert legacy.temperature is None
    config['energy'] = True
    payload['config'] = json.dumps(config)
    np.savez(tmp_path/'missing.npz', **payload)
    with pytest.raises(ValueError, match='temperature'):
        Result.load(tmp_path/'missing.npz')
    assert Config(nx=1024, ny=1024, energy=True).frame_count == 10
    for c in (Config(heat_capacity=0), Config(conductivity=-1), Config(turbulent_prandtl=0),
              Config(temperature_inlet=0), Config(heat_source=float('nan')), Config(thermal_wall='bad')):
        with pytest.raises(ValueError):
            c.validate()
