"""Planar DXF contours in metres, including curved edges and holes."""
from dataclasses import dataclass, asdict
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.polygon import orient
from shapely.ops import polygonize_full, unary_union
from shapely import STRtree

BOUNDARIES = {"inlet": "Вход", "outlet": "Выход (p=0)", "wall": "Стенка", "open": "Открытая (p=0)"}


@dataclass
class Geometry:
    loops: list
    tags: list
    source: str = ""
    origin: tuple = (0., 0.)

    @property
    def polygon(self):
        return Polygon(self.loops[0], self.loops[1:])

    @property
    def bounds(self):
        return self.polygon.bounds

    def segments(self):
        for li, loop in enumerate(self.loops):
            for j, point in enumerate(loop):
                yield li, j, np.array(point), np.array(loop[(j+1) % len(loop)]), self.tags[li][j]

    def validate(self, boundaries=False):
        if not self.loops or len(self.loops) != len(self.tags):
            raise ValueError("Нет замкнутого контура области жидкости")
        for loop, tags in zip(self.loops, self.tags):
            p = np.asarray(loop)
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
                raise ValueError("Некорректные координаты контура")
            if len(tags) != len(p) or any(t not in BOUNDARIES for t in tags):
                raise ValueError("Некорректные типы границ")
            if np.any(np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1) < 1e-12):
                raise ValueError("Контур содержит ребро нулевой длины")
        if not self.polygon.is_valid or self.polygon.area <= 1e-14:
            raise ValueError("Контур самопересекается, отверстия пересекаются или площадь равна нулю")
        if boundaries:
            tags = sum(self.tags, [])
            if "inlet" not in tags or not any(t in tags for t in ("outlet", "open")):
                raise ValueError("Назначьте хотя бы один вход и один выход / открытую границу")

    def to_dict(self):
        return asdict(self)


def load_dxf(path, units="auto", curve_tolerance=1e-4):
    import ezdxf
    from ezdxf import path as dxfpath
    from ezdxf.disassemble import recursive_decompose
    from ezdxf import units as du
    doc = ezdxf.readfile(path)
    if units == "auto":
        code = int(doc.header.get("$INSUNITS", 0))
        if not code:
            raise ValueError("В DXF не указаны единицы. Выберите мм, см или м и импортируйте снова.")
        scale = du.conversion_factor(code, du.M)
    else:
        scale = {"mm": .001, "cm": .01, "m": 1.}[units]
    if not np.isfinite(curve_tolerance) or curve_tolerance <= 0:
        raise ValueError("Допуск аппроксимации должен быть положительным")
    sources = []
    zvalues = []
    for entity in recursive_decompose(doc.modelspace()):
        if entity.dxftype() not in ("LINE", "ARC", "CIRCLE", "LWPOLYLINE", "POLYLINE", "ELLIPSE", "SPLINE"):
            continue
        p = dxfpath.make_path(entity)
        vertices = np.array([tuple(v) for v in p.flattening(distance=curve_tolerance/scale)])
        if len(vertices) < 2:
            continue
        zvalues.extend(vertices[:, 2]*scale)
        xy = vertices[:, :2]*scale
        tag = str(entity.dxf.layer).lower()
        tag = tag if tag in BOUNDARIES else "wall"
        for a, b in zip(xy[:-1], xy[1:]):
            if np.linalg.norm(b-a) > 1e-12:
                sources.append((a, b, tag))
    if not sources:
        raise ValueError("В DXF нет поддерживаемых 2D-линий или контуров")
    if np.ptp(zvalues) > max(curve_tolerance*.01, 1e-9):
        raise ValueError("Требуется плоский чертёж в плоскости XY, а не 3D-модель")
    points = np.array([p for a, b, _ in sources for p in (a, b)])
    origin = points.min(axis=0)
    tolerance = max(np.ptp(points, axis=0).max()*1e-9, 1e-12)
    lines = [LineString(np.round((np.array([a, b])-origin)/tolerance)*tolerance) for a, b, _ in sources]
    polygons, cuts, dangles, invalid = polygonize_full(unary_union(lines))
    if not cuts.is_empty or not dangles.is_empty or not invalid.is_empty:
        raise ValueError("DXF содержит незамкнутые, лишние или пересекающиеся линии. Замкните контуры и удалите вспомогательную геометрию.")
    polygons = list(polygons.geoms)
    if not polygons:
        raise ValueError("Не найден замкнутый контур области жидкости")
    largest = max(polygons, key=lambda p: Polygon(p.exterior).area)
    envelope = Polygon(largest.exterior)
    if not all(envelope.covers(p) for p in polygons):
        raise ValueError("Поддерживается одна связная область жидкости; отдельные несвязанные области импортируйте отдельно")
    if len(polygons) != 1+len(largest.interiors):
        raise ValueError("Вложенные острова внутри отверстий образуют отдельные области жидкости и не поддерживаются")
    largest = orient(largest, sign=1.)
    loops = [np.asarray(r.coords)[:-1].tolist() for r in (largest.exterior, *largest.interiors)]
    tags = []
    tree = STRtree(lines)
    for loop in loops:
        labels = []
        for a, b in zip(loop, np.roll(loop, -1, axis=0)):
            middle = (np.array(a)+b)/2
            labels.append(sources[int(tree.nearest(Point(middle)))][2])
        tags.append(labels)
    geometry = Geometry(loops, tags, str(path), tuple(float(v) for v in origin))
    geometry.validate()
    # Ordinary all-WALL drawings get useful defaults; explicit layer assignments win.
    if all(tag == "wall" for tag in sum(tags, [])):
        xmin, _, xmax, _ = geometry.bounds
        for li, j, a, b, _ in geometry.segments():
            if max(abs(a[0]-xmin), abs(b[0]-xmin)) < 10*tolerance:
                tags[li][j] = "inlet"
            elif max(abs(a[0]-xmax), abs(b[0]-xmax)) < 10*tolerance:
                tags[li][j] = "outlet"
    return geometry


def example_geometry():
    return Geometry([[[0., 0.], [.6, 0.], [.6, .04], [0., .04]]],
                    [["wall", "outlet", "wall", "inlet"]], "Прямой канал")
