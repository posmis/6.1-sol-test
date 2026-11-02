import csv
from threading import Event
import numpy as np
import pytest
from flowdesk.airfoil import profile_geometry, coefficients
from flowdesk.solver import Config, Result, Solver
from flowdesk.scenarios import PROFILE_SCENARIO
from flowdesk.aero_gui import export_csv


def test_asymmetric_ellipse_and_positive_angle_definition():
    g = profile_geometry(chord=.1, angle=0)
    g.validate(boundaries=True)
    p = np.array(g.loops[1])
    assert p[:, 1].max()- .2 == pytest.approx(.01)
    assert .2-p[:, 1].min() == pytest.approx(.006)
    body_area = .8*.4-g.polygon.area
    assert body_area == pytest.approx(np.pi*.1**2*(.1+.06)/4, rel=.01)
    p = np.array(profile_geometry(angle=4).loops[1])
    assert p[0, 1] > p[len(p)//2, 1]  # nose up, flow to the right


def test_pressure_force_sign_normalization_and_gauge_invariance():
    theta = (np.arange(360)+.5)*2*np.pi/360
    radius = .5
    normals = -np.vstack([np.cos(theta), np.sin(theta)])
    ds = np.full(len(theta), radius*2*np.pi/len(theta))
    p = 10-3*radius*np.cos(theta)-4*radius*np.sin(theta)
    args = (normals, ds, np.zeros_like(normals), np.ones(len(theta))*.01, 2., 1e-5, 3., 1.)
    result = coefficients(p, *args)
    expected_drag, expected_lift = 3*np.pi*radius**2, 4*np.pi*radius**2
    assert result["drag_n_per_m"] == pytest.approx(expected_drag)
    assert result["lift_n_per_m"] == pytest.approx(expected_lift)
    assert result["cd"] == pytest.approx(expected_drag/(.5*2*3**2))
    assert result["cl_cd"] == pytest.approx(4/3)
    shifted = coefficients(p+5000, *args)
    assert shifted["cl"] == pytest.approx(result["cl"], abs=1e-10)
    assert shifted["cd"] == pytest.approx(result["cd"], abs=1e-10)


def test_wall_shear_force_and_undefined_efficiency():
    result = coefficients([0.], np.array([[0.], [-1.]]), [.5], np.array([[2.], [0.]]),
                          [.0001], 1., .001, 2., 1.)
    # Viscous wall-law branch: tau = rho nu U / y, force = tau * ds.
    assert result["shear_drag_n_per_m"] == pytest.approx(10.)
    assert result["pressure_drag_n_per_m"] == 0
    assert result["cd"] == pytest.approx(5.)
    zero = coefficients([0.], np.array([[0.], [-1.]]), [.5], np.zeros((2, 1)), [.01], 1., 1e-5, 2., 1.)
    assert zero["cl_cd"] is None


def test_profile_calculation_and_coefficient_roundtrip(tmp_path):
    # Integer speed also verifies that the FiPy velocity variable stays floating point.
    c = Config(scenario=PROFILE_SCENARIO, mode="steady", profile_cells=400, speed=20,
               rho=1.225, nu=1.5e-5, intensity=.01, nx=96, ny=48, max_iterations=600)
    r = Solver(c).run()
    assert r.diagnostics["converged"], r.diagnostics["residuals"]
    aero = r.diagnostics["aerodynamics"]
    assert len(aero["history"]) == r.diagnostics["iterations"]
    assert aero["final"]["cd"] > 0
    assert r.diagnostics["mass_imbalance"] < 1e-7
    assert len(aero["polar"]) == 1
    assert r.config.mesh_path is None
    r.save(tmp_path/"profile.npz")
    restored = Result.load(tmp_path/"profile.npz")
    assert restored.diagnostics["aerodynamics"] == aero
    export_csv(restored, tmp_path/"history.csv")
    export_csv(restored, tmp_path/"polar.csv", polar=True)
    with open(tmp_path/"history.csv", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == len(aero["history"])
    assert float(rows[-1]["cl"]) == pytest.approx(aero["final"]["cl"])


def test_profile_input_limits_and_cancel():
    for c in (Config(scenario=PROFILE_SCENARIO, profile_sweep=(0.,)), Config(profile_angle=40), Config(profile_chord=0),
              Config(profile_cells=100), Config(profile_sweep=(float("nan"),))):
        with pytest.raises(ValueError):
            c.validate()
    event = Event()
    event.set()
    with pytest.raises(InterruptedError):
        Solver(Config(scenario=PROFILE_SCENARIO, mode="steady")).run(cancel=event)


def test_polar_series_reuses_base_angle_and_preserves_nonconvergence():
    c = Config(scenario=PROFILE_SCENARIO, mode="steady", profile_cells=400, speed=20.,
               rho=1.225, nu=1.5e-5, intensity=.01, nx=32, ny=16, max_iterations=10,
               profile_sweep=(4., -4., 0., 0.))
    r = Solver(c).run()
    polar = r.diagnostics["aerodynamics"]["polar"]
    assert [p["angle"] for p in polar] == [-4., 0., 4.]
    assert all(not p["converged"] for p in polar)
    assert r.config.profile_angle == 4.
    assert r.config.profile_sweep == c.profile_sweep
