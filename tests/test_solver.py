from threading import Event
import numpy as np
import pytest
from flowdesk.solver import Config, Result, Solver, diffusion
from flowdesk.benchmark import channel_check
from flowdesk.visualization import streamlines


@pytest.mark.parametrize("scenario", ["channel", "step"])
def test_projection_conserves_mass_with_obstacles(scenario):
    s = Solver(Config(scenario=scenario, nx=32, ny=12))
    rng = np.random.default_rng(5)
    s.u[:, 1:] += rng.normal(0, .1, s.u[:, 1:].shape)
    s.v += rng.normal(0, .1, s.v.shape)
    s.u *= s.um
    s.v *= s.vm
    s.project(.001)
    assert np.max(abs(s.divergence()[s.fluid])) < 1e-8
    assert np.all(s.u[~s.um] == 0)
    assert np.all(s.v[~s.vm] == 0)
    assert s.u[:, -1].sum() == pytest.approx(s.u[:, 0].sum(), abs=1e-10)


def test_pressure_projection_manufactured_gradient():
    s = Solver(Config(nx=32, ny=12))
    x = (np.arange(32)+.5)*s.dx
    p = np.broadcast_to(np.cos(np.pi*x/(2*s.c.length)), s.fluid.shape)
    dt = .01
    s.u[:, 0] = 0
    s.u[:, 1:-1] = dt*np.diff(p, axis=1)/s.dx
    s.u[:, -1] = -dt*2*p[:, -1]/s.dx
    s.project(dt)
    np.testing.assert_allclose(s.p, p, atol=1e-11)
    assert np.max(abs(s.u)) < 1e-11


def test_diffusion_conserves_scalar_and_dissipates():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(12, 16))
    mask = np.ones_like(q, bool)
    mask[3:5, 4:6] = False
    d = diffusion(q, np.ones_like(q), mask, .1, .1)
    assert abs(d.sum()) < 1e-10
    assert np.sum(q*d) < 0
    assert np.all(d[~mask] == 0)


@pytest.mark.parametrize("nx,ny", [(64, 16), (96, 24)])
def test_channel_pressure_against_dean(nx, ny):
    r = Solver(Config(nx=nx, ny=ny, frames=30)).run()
    assert channel_check(r)["relative_error"] < .10
    assert r.diagnostics["mass_imbalance"] < 1e-9
    assert r.diagnostics["final_frame_velocity_change"] < 1e-4
    assert np.all(r.k[:, r.fluid] > 0)
    assert np.all(r.epsilon[:, r.fluid] > 0)


def test_step_recirculation_and_result_roundtrip(tmp_path):
    r = Solver(Config(scenario="step", nx=64, ny=16, frames=20)).run()
    assert r.u[-1][r.fluid].min() < -.05
    assert r.diagnostics["mass_imbalance"] < 1e-9
    assert not r.diagnostics["backflow"]
    assert np.all(r.u[:, ~r.fluid] == 0)
    path = tmp_path/"flow.npz"
    r.save(path)
    loaded = Result.load(path)
    np.testing.assert_array_equal(loaded.pressure, r.pressure)
    assert loaded.config == r.config


def test_cancel_and_invalid_inputs():
    event = Event()
    event.set()
    with pytest.raises(InterruptedError):
        Solver(Config(nx=32, ny=12)).run(cancel=event)
    for c in (Config(speed=-1), Config(ny=2), Config(duration=float("nan"))):
        with pytest.raises(ValueError):
            Solver(c)


def test_streamlines_follow_uniform_field():
    fluid = np.ones((12, 32), bool)
    paths = streamlines(fluid.astype(float), np.zeros_like(fluid, float), fluid, .6, .04)
    assert len(paths) >= 16
    for p in paths:
        assert np.all(np.diff(p[:, 0]) > 0)
        assert np.ptp(p[:, 1]) < 1e-12
