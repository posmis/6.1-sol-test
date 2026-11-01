import numpy as np
import pytest
import json
from flowdesk.solver import Solver, Config, Result
from flowdesk.pressure import PressureSolver
from flowdesk.scenarios import SCENARIOS
from flowdesk.benchmark import channel_check


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("bottom,top", [("wall", "wall"), ("open", "wall"),
                                        ("wall", "open"), ("open", "open")])
def test_all_geometries_and_boundary_combinations(scenario, bottom, top):
    s = Solver(Config(scenario=scenario, bottom=bottom, top=top, nx=64, ny=24,
                      duration=.01, frames=3))
    rng = np.random.default_rng(20)
    s.u[:, 1:] += rng.normal(0, .05, s.u[:, 1:].shape)
    s.v += rng.normal(0, .05, s.v.shape)
    s.u *= s.um
    s.v *= s.vm
    s.project(.001)
    assert np.max(abs(s.divergence()[s.fluid])) < 1e-7
    assert np.all(s.u[~s.um] == 0)
    assert np.all(s.v[~s.vm] == 0)
    for index, boundary in ((0, bottom), (-1, top)):
        if boundary == "wall":
            assert not s.vm[index].any()
        else:
            np.testing.assert_array_equal(s.vm[index], s.fluid[index])
    r = s.run()
    assert r.diagnostics["mass_imbalance"] < 1e-8
    assert np.isfinite(r.pressure).all()
    if scenario != "channel":
        assert (~r.fluid).any()


def test_open_sides_preserve_uniform_flow_and_remove_wall_stress(tmp_path):
    c = Config(top="open", bottom="open", nx=64, ny=16, duration=.1, frames=5)
    s = Solver(c)
    assert not s.wall.any()
    r = s.run()
    np.testing.assert_allclose(r.u, c.speed, atol=1e-6)
    np.testing.assert_allclose(r.v, 0, atol=1e-10)
    # Static pressure can vary because inlet turbulence is supplied while k decays.
    # The projected mechanical pressure p_static + 2 rho k/3 must be uniform.
    mechanical = r.pressure+2*c.rho*r.k/3
    np.testing.assert_allclose(mechanical, np.broadcast_to(mechanical[:, :1, :1], mechanical.shape), atol=1e-6)
    r.save(tmp_path/"open.npz")
    assert Result.load(tmp_path/"open.npz").config == c
    with pytest.raises(ValueError, match="two walls"):
        channel_check(r)


@pytest.mark.parametrize("bottom,top", [("wall", "wall"), ("open", "wall"),
                                        ("wall", "open"), ("open", "open")])
def test_iterative_projection_matches_sparse_lu(bottom, top, monkeypatch):
    c = Config(scenario="cylinder", nx=64, ny=24, bottom=bottom, top=top)
    direct = Solver(c)
    monkeypatch.setattr(PressureSolver, "DIRECT_LIMIT", 0)
    iterative = Solver(c)
    assert iterative.solve_p.method == "cg-spectral"
    np.testing.assert_allclose(iterative.u, direct.u, atol=1e-7)
    np.testing.assert_allclose(iterative.v, direct.v, atol=1e-7)
    assert np.max(abs(iterative.divergence()[iterative.fluid])) < 1e-7


@pytest.mark.parametrize("nx,ny", [(1024, 16), (32, 1024)])
def test_large_axis_runs_real_timestep(nx, ny):
    r = Solver(Config(nx=nx, ny=ny, duration=1e-6, frames=2)).run()
    assert r.u.shape == (2, ny, nx)
    assert r.diagnostics["pressure_solver"] == "spectral"
    assert r.diagnostics["steps"] >= 1
    assert r.diagnostics["mass_imbalance"] < 1e-8


def test_frame_memory_budget_and_limits():
    c = Config(nx=1024, ny=1024)
    c.validate()
    assert c.frame_count == 12
    assert c.frame_count*c.nx*c.ny*5*4 <= 256*1024**2
    for c in (Config(nx=1025), Config(ny=1025), Config(top="empty")):
        with pytest.raises(ValueError):
            c.validate()


def test_legacy_result_without_boundary_settings(tmp_path):
    r = Solver(Config(nx=32, ny=12, duration=.001, frames=2)).run()
    path = tmp_path/"legacy.npz"
    r.save(path)
    with np.load(path) as archive:
        data = dict(archive)
    config = json.loads(str(data["config"]))
    del config["top"], config["bottom"]
    data["config"] = json.dumps(config)
    np.savez_compressed(path, **data)
    restored = Result.load(path)
    assert restored.config.top == restored.config.bottom == "wall"
