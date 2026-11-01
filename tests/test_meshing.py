from dataclasses import replace
import numpy as np
import pytest
import ezdxf
import meshio
from flowdesk.cad import Geometry, example_geometry, load_dxf
from flowdesk.meshing import MeshProject, MeshSettings, generate
from flowdesk.solver import Config, Result, Solver


def dxf_channel(path, units=4, hole=False, gap=False):
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = units
    points = [(0, 0), (120, 0), (120, 40), (0, 40)]
    for a, b, tag in zip(points, points[1:]+points[:1], ("WALL", "OUTLET", "WALL", "INLET")):
        if gap and tag == "INLET":
            continue
        doc.modelspace().add_line(a, b, dxfattribs={"layer": tag})
    if hole:
        doc.modelspace().add_circle((40, 20), 5, dxfattribs={"layer": "WALL"})
    doc.saveas(path)


def test_dxf_units_holes_and_named_boundaries(tmp_path):
    path = tmp_path/"model.dxf"
    dxf_channel(path, hole=True)
    g = load_dxf(path, curve_tolerance=1e-5)
    g.validate(boundaries=True)
    np.testing.assert_allclose(g.bounds, (0, 0, .12, .04), atol=1e-9)
    assert len(g.loops) == 2
    assert all(tag == "wall" for tag in g.tags[1])
    assert "inlet" in g.tags[0] and "outlet" in g.tags[0]
    assert g.polygon.area == pytest.approx(.12*.04-np.pi*.005**2, rel=1e-3)


def test_dxf_rejects_open_contours_and_unknown_units(tmp_path):
    path = tmp_path/"bad.dxf"
    dxf_channel(path, gap=True)
    with pytest.raises(ValueError, match="незамкнутые"):
        load_dxf(path)
    dxf_channel(path, units=0)
    with pytest.raises(ValueError, match="единицы"):
        load_dxf(path)
    assert load_dxf(path, units="mm").polygon.area == pytest.approx(.0048, rel=1e-6)


@pytest.mark.parametrize("kind,nodes", [("triangle", 3), ("quad", 4)])
def test_both_cell_types_are_real_and_exportable(tmp_path, kind, nodes):
    path = tmp_path/"model.dxf"
    dxf_channel(path, hole=True)
    g = load_dxf(path, curve_tolerance=.0001)
    m = generate(g, MeshSettings(cell_type=kind, target_cells=250))
    assert np.all(np.sum(m.cells >= 0, axis=1) == nodes)
    assert m.cell_areas().sum() == pytest.approx(g.polygon.area, rel=1e-6)
    assert m.statistics["min_quality"] > 0
    m.save(tmp_path/"model.flowmesh")
    restored = MeshProject.load(tmp_path/"model.flowmesh")
    np.testing.assert_array_equal(restored.cells, m.cells)
    restored.export(tmp_path/"model.msh")
    external = meshio.read(tmp_path/"model.msh")
    assert len(external.cells_dict[kind]) == len(m.cells)
    assert "inlet" in external.field_data and "outlet" in external.field_data


def test_local_refinement_changes_actual_cell_area():
    m = generate(example_geometry(), MeshSettings(cell_size=.008,
        zones=((.2, 0, .3, .02, .002),)))
    centres = m.points[m.cells[:, :3]].mean(axis=1)
    inside = (centres[:, 0] > .21) & (centres[:, 0] < .29) & (centres[:, 1] < .018)
    outside = centres[:, 0] > .45
    assert np.median(m.cell_areas()[inside]) < .4*np.median(m.cell_areas()[outside])


@pytest.mark.parametrize("kind", ["triangle", "quad"])
def test_dxf_to_native_mesh_to_converged_calculation(tmp_path, kind):
    path = tmp_path/"channel.dxf"
    dxf_channel(path)
    project = generate(load_dxf(path), MeshSettings(cell_type=kind, target_cells=100))
    model = tmp_path/"channel.flowmesh"
    project.save(model)
    config = Config(scenario="custom", mesh_path=str(model), mode="steady", nx=64, ny=24,
                    max_iterations=250)
    result = Solver(config).run()
    d = result.diagnostics
    assert d["converged"], d["residuals"]
    assert d["mass_imbalance"] < 1e-7
    assert d["pressure_drop_pa"] > 0
    assert len(result.native["u"]) == len(project.cells)
    np.testing.assert_array_equal(result.native["cells"], project.cells)
    result.save(tmp_path/"result.npz")
    model.unlink()  # Result must remain viewable without the original project.
    restored = Result.load(tmp_path/"result.npz")
    np.testing.assert_array_equal(restored.native["pressure"], result.native["pressure"])
    restored.export_native(tmp_path/"fields.vtu")
    external = meshio.read(tmp_path/"fields.vtu")
    assert len(external.cell_data["pressure"][0]) == len(project.cells)


def test_native_open_domain_preserves_uniform_flow(tmp_path):
    g = example_geometry()
    g.loops[0][1][0] = g.loops[0][2][0] = .12
    g.tags[0][0] = g.tags[0][2] = "open"
    m = generate(g, MeshSettings(target_cells=100))
    m.save(tmp_path/"open.flowmesh")
    r = Solver(Config(scenario="custom", mesh_path=str(tmp_path/"open.flowmesh"),
                      mode="steady", max_iterations=200)).run()
    assert r.diagnostics["converged"]
    np.testing.assert_allclose(r.native["u"], 1., atol=1e-7)
    np.testing.assert_allclose(r.native["v"], 0., atol=1e-7)
    assert r.diagnostics["mass_imbalance"] < 1e-7


def test_invalid_mesh_and_custom_transient_rejected(tmp_path):
    m = generate(example_geometry(), MeshSettings(target_cells=100))
    broken = replace(m, boundary_edges=m.boundary_edges[:-1], boundary_tags=m.boundary_tags[:-1])
    with pytest.raises(ValueError, match="грани"):
        broken.validate()
    with pytest.raises(ValueError, match="стационарный"):
        Config(scenario="custom", mesh_path="model.flowmesh").validate()


def test_target_search_keeps_best_attempt():
    g = example_geometry()
    g.loops[0][1][0] = g.loops[0][2][0] = .12
    m = generate(g, MeshSettings(cell_type="quad", target_cells=100))
    assert abs(len(m.cells)-100) <= 20
