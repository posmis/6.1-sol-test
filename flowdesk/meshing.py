"""Gmsh meshing, portable projects and connectivity for actual polygon cells."""
from dataclasses import dataclass, asdict
import json
import numpy as np
from shapely.geometry import box
from .cad import Geometry, BOUNDARIES


@dataclass
class MeshSettings:
    cell_type: str = "triangle"
    cell_size: float = .003
    target_cells: int = 0
    zones: tuple = ()  # xmin, ymin, xmax, ymax, cell size, all in metres
    max_cells: int = 1000000

    def validate(self, geometry):
        if self.cell_type not in ("triangle", "quad"):
            raise ValueError("Тип ячеек: triangle или quad")
        if not np.isfinite(self.cell_size) or self.cell_size <= 0 or not 0 <= self.target_cells <= self.max_cells:
            raise ValueError("Некорректный размер ячейки или число элементов")
        for z in self.zones:
            if len(z) != 5 or not np.isfinite(z).all() or z[2] <= z[0] or z[3] <= z[1] or z[4] <= 0:
                raise ValueError("Зона сгущения: xmin < xmax, ymin < ymax, размер > 0")
        estimate = geometry.polygon.area/self.cell_size**2
        if self.target_cells:
            estimate = self.target_cells
        for z in self.zones:
            estimate += geometry.polygon.intersection(box(*z[:4])).area/z[4]**2
        if estimate > self.max_cells*1.5:
            raise ValueError("Настройки создают слишком много ячеек. Увеличьте размер или уменьшите зоны сгущения.")


@dataclass
class MeshProject:
    geometry: Geometry
    settings: MeshSettings
    points: np.ndarray
    cells: np.ndarray  # padded with -1 for triangles
    boundary_edges: np.ndarray
    boundary_tags: np.ndarray
    statistics: dict

    def save(self, path):
        with open(path, "wb") as out:
            np.savez_compressed(out, geometry=json.dumps(self.geometry.to_dict()),
                settings=json.dumps(asdict(self.settings)), points=self.points, cells=self.cells,
                boundary_edges=self.boundary_edges, boundary_tags=self.boundary_tags,
                statistics=json.dumps(self.statistics))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            result = cls(Geometry(**json.loads(str(z["geometry"]))),
                         MeshSettings(**json.loads(str(z["settings"]))),
                         z["points"].copy(), z["cells"].copy(), z["boundary_edges"].copy(),
                         z["boundary_tags"].copy(), json.loads(str(z["statistics"])))
        result.validate()
        return result

    def validate(self):
        self.geometry.validate(boundaries=True)
        if (self.points.ndim != 2 or self.points.shape[1] != 2 or not np.isfinite(self.points).all()
            or self.cells.ndim != 2 or self.cells.shape[1] != 4 or len(self.cells) == 0
            or not np.issubdtype(self.cells.dtype, np.integer)
            or np.any(self.cells < -1) or np.any(self.cells >= len(self.points))
            or np.any(np.sum(self.cells >= 0, axis=1) < 3)):
            raise ValueError("Некорректные узлы или элементы сетки")
        if (self.boundary_edges.ndim != 2 or self.boundary_edges.shape[1] != 2
            or len(self.boundary_edges) != len(self.boundary_tags)
            or not np.issubdtype(self.boundary_edges.dtype, np.integer)
            or np.any(self.boundary_edges < 0) or np.any(self.boundary_edges >= len(self.points))
            or any(t not in BOUNDARIES for t in self.boundary_tags)):
            raise ValueError("Некорректные границы сетки")
        faces, cell_faces, _ = self.connectivity()
        count = np.bincount(cell_faces[cell_faces >= 0], minlength=len(faces))
        boundary = {tuple(edge) for edge in np.sort(self.boundary_edges, axis=1)}
        actual = {tuple(edge) for edge in faces[count == 1]}
        if np.any(count > 2) or boundary != actual or len(boundary) != len(self.boundary_edges):
            raise ValueError("Сетка имеет неоднозначные, неназначенные или неманифолдные грани")
        areas = self.cell_areas()
        if np.any(areas <= 1e-16):
            raise ValueError("Сетка содержит вырожденные / перевёрнутые элементы")
        if not np.isclose(areas.sum(), self.geometry.polygon.area, rtol=1e-6, atol=1e-14):
            raise ValueError("Площадь элементов не совпадает с областью жидкости")

    def cell_areas(self):
        areas = np.zeros(len(self.cells))
        for n in (3, 4):
            mask = np.sum(self.cells >= 0, axis=1) == n
            p = self.points[self.cells[mask, :n]]
            areas[mask] = .5*np.sum(p[:, :, 0]*np.roll(p[:, :, 1], -1, axis=1)
                -p[:, :, 1]*np.roll(p[:, :, 0], -1, axis=1), axis=1)
        return areas

    def connectivity(self):
        edges, owners = [], []
        for n in (3, 4):
            ids = np.flatnonzero(np.sum(self.cells >= 0, axis=1) == n)
            nodes = self.cells[ids, :n]
            for j in range(n):
                edges.append(np.sort(nodes[:, [j, (j+1) % n]], axis=1))
                owners.append(np.column_stack([ids, np.full(len(ids), j)]))
        edges, owners = np.vstack(edges), np.vstack(owners)
        faces, inverse = np.unique(edges, axis=0, return_inverse=True)
        cell_faces = np.full(self.cells.shape, -1, int)
        cell_faces[owners[:, 0], owners[:, 1]] = inverse
        tags = np.full(len(faces), "interior", dtype="U8")
        mapping = {tuple(edge): tag for edge, tag in zip(np.sort(self.boundary_edges, axis=1), self.boundary_tags)}
        for i, edge in enumerate(faces):
            tags[i] = mapping.get(tuple(edge), "interior")
        return faces, cell_faces, tags

    def export(self, path):
        import meshio
        blocks, physical = [], []
        for n, kind in ((3, "triangle"), (4, "quad")):
            selected = self.cells[np.sum(self.cells >= 0, axis=1) == n, :n]
            if len(selected):
                blocks.append((kind, selected))
                physical.append(np.full(len(selected), 10))
        blocks.append(("line", self.boundary_edges))
        physical.append(np.array([list(BOUNDARIES).index(t)+1 for t in self.boundary_tags]))
        fields = {tag: np.array([i+1, 1]) for i, tag in enumerate(BOUNDARIES)}
        fields["fluid"] = np.array([10, 2])
        points = np.column_stack([self.points, np.zeros(len(self.points))])
        meshio.write(path, meshio.Mesh(points, blocks, field_data=fields,
            cell_data={"gmsh:physical": physical, "gmsh:geometrical": physical}),
            file_format="gmsh22", binary=False)


def _capture(gmsh, geometry, settings, grouped, h, count):
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(coords).reshape(-1, 3)[:, :2]
    lookup = {int(tag): i for i, tag in enumerate(node_tags)}
    cells, ids_all = [], []
    for kind, ids, nodes in zip(*gmsh.model.mesh.getElements(2)):
        name, dim, order, nn, _, _ = gmsh.model.mesh.getElementProperties(kind)
        if dim != 2 or order != 1 or nn not in (3, 4):
            raise ValueError(f"Неподдерживаемый элемент Gmsh: {name}")
        data = np.array([lookup[int(t)] for t in nodes]).reshape(-1, nn)
        if settings.cell_type == "quad" and nn != 4:
            raise ValueError("Не удалось построить полностью четырёхугольную сетку")
        cells.append(np.pad(data, ((0, 0), (0, 4-nn)), constant_values=-1))
        ids_all.extend(ids)
    edges, tags = [], []
    for tag, curves in grouped.items():
        for curve in curves:
            for kind, _, nodes in zip(*gmsh.model.mesh.getElements(1, curve)):
                data = np.array([lookup[int(t)] for t in nodes]).reshape(-1, 2)
                edges.extend(data)
                tags.extend([tag]*len(data))
    q = gmsh.model.mesh.getElementQualities(ids_all, "minSICN")
    result = MeshProject(geometry, settings, points, np.vstack(cells), np.array(edges, int),
        np.array(tags, dtype="U8"), dict(cells=count, nodes=len(points),
            min_quality=float(min(q)), mean_quality=float(np.mean(q)),
            effective_size=float(h), requested_cells=settings.target_cells,
            note="Target count is approximate; local size constraints take precedence."))
    result.validate()
    return result



def generate(geometry, settings, progress=None):
    """Call in a spawned process: Gmsh global state never runs in a GUI thread."""
    import gmsh
    geometry.validate(boundaries=True)
    settings.validate(geometry)
    h = settings.cell_size
    if settings.cell_type == "quad":
        h *= 2  # final all-quad subdivision halves the characteristic edge length
    if settings.target_cells:
        h = np.sqrt(geometry.polygon.area/settings.target_cells)*(1.5 if settings.cell_type == "triangle" else 2.)
    gmsh.initialize(["flowdesk", "-nopopup"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("fluid")
        loops, grouped = [], {tag: [] for tag in BOUNDARIES}
        for loop, tags in zip(geometry.loops, geometry.tags):
            pts = [gmsh.model.geo.addPoint(x, y, 0, h) for x, y in loop]
            curves = []
            for j, tag in enumerate(tags):
                line = gmsh.model.geo.addLine(pts[j], pts[(j+1) % len(pts)])
                grouped[tag].append(line)
                curves.append(line)
            loops.append(gmsh.model.geo.addCurveLoop(curves))
        surface = gmsh.model.geo.addPlaneSurface(loops)
        gmsh.model.geo.synchronize()
        for tag, curves in grouped.items():
            if curves:
                gmsh.model.addPhysicalGroup(1, curves, list(BOUNDARIES).index(tag)+1, name=tag)
        gmsh.model.addPhysicalGroup(2, [surface], 10, name="fluid")
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        if settings.cell_type == "quad":
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            gmsh.model.mesh.setRecombine(2, surface)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        fields = []
        base = gmsh.model.mesh.field.add("MathEval")
        fields.append(base)
        for z in settings.zones:
            field = gmsh.model.mesh.field.add("Box")
            for key, value in zip(("XMin", "YMin", "XMax", "YMax", "VIn"), z):
                gmsh.model.mesh.field.setNumber(field, key, value*(2 if key == "VIn" and settings.cell_type == "quad" else 1))
            gmsh.model.mesh.field.setNumber(field, "VOut", 1e22)
            gmsh.model.mesh.field.setNumber(field, "Thickness", max(z[4]*3, h))
            fields.append(field)
        minimum = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minimum, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        best = None
        for attempt in range(4 if settings.target_cells else 1):
            gmsh.model.mesh.field.setString(base, "F", str(h))
            if progress:
                progress(attempt+1, 4 if settings.target_cells else 1)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize("Relocate2D", niter=5)
            types, element_tags, _ = gmsh.model.mesh.getElements(2)
            count = sum(len(ids) for ids in element_tags)
            if count > settings.max_cells:
                if settings.target_cells and attempt < 3:
                    gmsh.model.mesh.clear()
                    h *= np.sqrt(count/min(settings.target_cells, settings.max_cells*.95))
                    continue
                raise ValueError(f"Получено {count} элементов, лимит {settings.max_cells}. Увеличьте размер ячеек.")
            if best is None or abs(count-settings.target_cells) < abs(len(best.cells)-settings.target_cells):
                best = _capture(gmsh, geometry, settings, grouped, h, count)
            if not settings.target_cells or abs(count/settings.target_cells-1) < .1 or attempt == 3:
                break
            gmsh.model.mesh.clear()
            h *= np.sqrt(count/settings.target_cells)
        return best
    finally:
        gmsh.finalize()
