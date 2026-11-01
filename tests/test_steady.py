from threading import Event
import numpy as np
import pytest
from scipy.sparse import diags
from flowdesk.solver import Solver, Config, Result
from flowdesk.benchmark import channel_check
from flowdesk.steady import operator, LinearSolves, prolongate
from flowdesk.kernels import scalar_transport, diffuse, advect


def test_implicit_scalar_operator_matches_conservative_balance():
    rng = np.random.default_rng(4)
    mask = np.ones((8, 16), bool)
    q = rng.uniform(.1, 1, mask.shape)
    nu = rng.uniform(.01, .02, mask.shape)
    u, v = rng.normal(size=(8, 17)), rng.normal(size=(9, 16))
    sink = np.full(mask.shape, .2)
    source = rng.uniform(0, .1, mask.shape)
    kind = np.zeros(mask.shape, np.int32)
    a, b = operator(mask, nu, q, q, sink, source, kind, q, .1, .2,
                    conservative=True, u=u, v=v, inlet=.3)
    expected = -scalar_transport(q, u, v, .3, .1, .2)-diffuse(q, nu, mask, .1, .2)+sink*q-source
    np.testing.assert_allclose((a@q.ravel()-b).reshape(q.shape), expected, atol=1e-12)


def test_implicit_momentum_operator_matches_transport():
    rng = np.random.default_rng(7)
    mask = np.ones((8, 16), bool)
    q, ax, ay = [rng.normal(size=mask.shape) for _ in range(3)]
    nu = np.full(mask.shape, .01)
    zero = np.zeros(mask.shape)
    a, b = operator(mask, nu, ax, ay, zero, zero, zero.astype(np.int32), q, .1, .2)
    expected = advect(q, ax, ay, .1, .2)-diffuse(q, nu, mask, .1, .2)
    np.testing.assert_allclose((a@q.ravel()-b).reshape(q.shape), expected, atol=1e-12)


@pytest.mark.parametrize("scenario", ["channel", "contraction"])
def test_simple_converges_and_matches_transient(scenario, tmp_path):
    c = Config(mode="steady", scenario=scenario, nx=64, ny=16, max_iterations=400, coarse_start=False)
    steady = Solver(c).run()
    transient = Solver(Config(scenario=scenario, nx=64, ny=16)).run()
    assert steady.diagnostics["converged"]
    assert max(steady.diagnostics["residuals"].values()) < c.tolerance
    assert steady.diagnostics["mass_imbalance"] < 1e-8
    assert steady.diagnostics["pressure_drop_pa"] == pytest.approx(transient.diagnostics["pressure_drop_pa"], rel=.005)
    np.testing.assert_allclose(steady.u[-1], transient.u[-1], atol=.003)
    np.testing.assert_array_equal(steady.u[0], steady.u[1])
    if scenario == "channel":
        assert channel_check(steady)["within_10_percent"]
    steady.save(tmp_path/"steady.npz")
    restored = Result.load(tmp_path/"steady.npz")
    assert restored.config.mode == "steady"
    assert restored.diagnostics["converged"]


@pytest.mark.parametrize("top,bottom", [("open", "wall"), ("wall", "open"), ("open", "open")])
def test_simple_open_boundaries(top, bottom):
    r = Solver(Config(mode="steady", nx=32, ny=12, top=top, bottom=bottom, max_iterations=400)).run()
    assert r.diagnostics["converged"]
    assert r.diagnostics["mass_imbalance"] < 1e-8
    assert np.isfinite(r.pressure).all()


def test_iteration_limit_is_not_success_and_cancel_works():
    r = Solver(Config(mode="steady", nx=32, ny=12, max_iterations=10)).run()
    assert not r.diagnostics["converged"]
    assert len(r.diagnostics["residual_history"]) == 10
    event = Event()
    event.set()
    with pytest.raises(InterruptedError):
        Solver(Config(mode="steady", nx=32, ny=12)).run(cancel=event)


def test_amg_solves_large_pressure_system():
    n = 21000
    a = diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1]).tocsr()
    expected = np.sin(np.linspace(0, np.pi, n+2)[1:-1])
    linear = LinearSolves(None)
    result = linear.solve(a, a@expected, np.zeros(n), "p", positive=True)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert linear.total_iterations > 0


def test_pressure_prolongation_preserves_outlet_reference():
    coarse, fine = Config(nx=32, ny=12), Config(nx=96, ny=24)
    old = np.broadcast_to(1-(np.arange(coarse.nx)+.5)/coarse.nx, (coarse.ny, coarse.nx)).copy()
    new = prolongate(old, "p", coarse, fine, (fine.ny, fine.nx))
    # Inlet is Neumann; away from its half coarse cell, a linear pressure is exact.
    expected = np.broadcast_to(1-(np.arange(fine.nx)+.5)/fine.nx, new.shape)
    np.testing.assert_allclose(new[:, 2:], expected[:, 2:], atol=1e-14)
