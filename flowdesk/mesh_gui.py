"""Standalone DXF geometry and local mesh editor; also opened from FlowDesk."""
import multiprocessing as mp
import queue
import sys
import traceback
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from .cad import Geometry, BOUNDARIES, load_dxf, example_geometry
from .meshing import MeshSettings, MeshProject, generate

COLORS = dict(inlet="#5eead4", outlet="#60a5fa", wall="#fbbf24", open="#c4b5fd")


def build_worker(geometry, settings, messages):
    try:
        result = generate(Geometry(**geometry), MeshSettings(**settings),
                          lambda i, n: messages.put(("progress", (i, n))))
        messages.put(("result", result))
    except Exception as exc:
        from .error_report import error_payload
        text = traceback.format_exc()
        try:
            payload = error_payload(text, summary=f"{type(exc).__name__}: {exc}")
        except Exception:
            payload = text
        messages.put(("error", payload))


class MeshEditor(QtWidgets.QMainWindow):
    mesh_ready = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FlowDesk Mesh · DXF и сетка 2D")
        self.resize(1280, 850)
        self.geometry = example_geometry()
        self.project = None
        self.process = None
        self.zones = []
        self.curves = []
        self.syncing = False
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        panel = QtWidgets.QWidget()
        panel.setMaximumWidth(390)
        left = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("Геометрия → Сетка → Расчёт")
        title.setStyleSheet("font-size: 19px; font-weight: bold")
        left.addWidget(title)
        import_button = QtWidgets.QPushButton("Импортировать DXF…")
        import_button.clicked.connect(self.import_dxf)
        open_button = QtWidgets.QPushButton("Открыть проект сетки…")
        open_button.clicked.connect(self.open_project)
        left.addWidget(import_button)
        left.addWidget(open_button)
        self.units = QtWidgets.QComboBox()
        self.units.addItems(["Из DXF", "Миллиметры", "Сантиметры", "Метры"])
        self.tolerance = self.number(1e-7, .1, .0001)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["Треугольники", "Четырёхугольники"])
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["По размеру ячейки", "По числу элементов (приближённо)"])
        self.size = self.number(1e-7, 10000, .003)
        self.target = QtWidgets.QSpinBox()
        self.target.setRange(50, 1000000)
        self.target.setValue(2000)
        form = QtWidgets.QFormLayout()
        for text, widget in (("Единицы DXF", self.units), ("Допуск кривых, м", self.tolerance),
                              ("Тип элементов", self.kind), ("Плотность", self.method),
                              ("Размер, м", self.size), ("Элементов", self.target)):
            form.addRow(text, widget)
        left.addLayout(form)
        self.kind.currentIndexChanged.connect(self.invalidate)
        self.method.currentIndexChanged.connect(self.settings_changed)
        self.size.valueChanged.connect(self.invalidate)
        self.target.valueChanged.connect(self.invalidate)
        self.target.setEnabled(False)
        units_note = QtWidgets.QLabel("Все размеры ниже в метрах. Область — жидкость; внутренние контуры — отверстия / твёрдые тела.")
        units_note.setWordWrap(True)
        left.addWidget(units_note)
        self.add_zone_button = QtWidgets.QPushButton("Добавить зону сгущения")
        self.add_zone_button.setToolTip("Перетащите рамку и её угловой маркер, затем задайте локальный h в таблице")
        self.add_zone_button.clicked.connect(self.add_zone)
        self.remove_zone_button = QtWidgets.QPushButton("Удалить выбранную зону")
        self.remove_zone_button.clicked.connect(self.remove_zone)
        left.addWidget(self.add_zone_button)
        left.addWidget(self.remove_zone_button)
        self.zone_table = QtWidgets.QTableWidget(0, 5)
        self.zone_table.setHorizontalHeaderLabels(["Xmin", "Ymin", "Xmax", "Ymax", "h, м"])
        self.zone_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.zone_table.setMaximumHeight(150)
        self.zone_table.itemChanged.connect(self.zone_edited)
        left.addWidget(self.zone_table)
        self.build_button = QtWidgets.QPushButton("Построить сетку")
        self.build_button.clicked.connect(self.build)
        self.cancel_button = QtWidgets.QPushButton("Отменить построение")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        left.addWidget(self.build_button)
        left.addWidget(self.cancel_button)
        self.progress = QtWidgets.QProgressBar()
        left.addWidget(self.progress)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        left.addWidget(self.info)
        self.save_button = QtWidgets.QPushButton("Сохранить проект .flowmesh…")
        self.save_button.clicked.connect(self.save_project)
        self.export_button = QtWidgets.QPushButton("Экспортировать сетку .msh…")
        self.export_button.clicked.connect(self.export_mesh)
        self.use_button = QtWidgets.QPushButton("Передать модель в расчёт")
        self.use_button.clicked.connect(self.use)
        for button in (self.save_button, self.export_button, self.use_button):
            left.addWidget(button)
            button.setEnabled(False)
        left.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(390)
        scroll.setMaximumWidth(420)
        layout.addWidget(scroll)
        right = QtWidgets.QVBoxLayout()
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "x", units="m")
        self.plot.setLabel("left", "y", units="m")
        self.wire = pg.PlotCurveItem(pen=pg.mkPen((150, 180, 210, 150), width=.6), connect="finite")
        self.plot.addItem(self.wire)
        right.addWidget(self.plot, 1)
        right.addWidget(QtWidgets.QLabel("Нажмите ребро, чтобы выбрать его в таблице. Вход — зелёный, выход — голубой, стенка — жёлтый, открытая — фиолетовый."))
        self.boundaries = QtWidgets.QTableWidget(0, 3)
        self.boundaries.setHorizontalHeaderLabels(["Контур / ребро", "Длина, м", "Граница"])
        self.boundaries.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.boundaries.setMaximumHeight(210)
        right.addWidget(self.boundaries)
        layout.addLayout(right, 1)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)
        self.busy_controls = [import_button, open_button, self.units, self.tolerance,
            self.kind, self.method, self.size, self.target, self.add_zone_button,
            self.remove_zone_button, self.zone_table, self.build_button, self.boundaries,
            self.save_button, self.export_button, self.use_button, self.plot]
        self.update_geometry()

    @staticmethod
    def number(low, high, value):
        control = QtWidgets.QDoubleSpinBox()
        control.setDecimals(7)
        control.setRange(low, high)
        control.setValue(value)
        control.setSingleStep(max(value*.1, 1e-7))
        return control

    def invalidate(self, *_):
        self.project = None
        self.wire.setData([], [])
        for button in (self.save_button, self.export_button, self.use_button):
            button.setEnabled(False)
        self.info.setText("Настройки изменены. Постройте сетку заново.")

    def settings_changed(self, *_):
        self.size.setEnabled(self.method.currentIndex() == 0)
        self.target.setEnabled(self.method.currentIndex() == 1)
        self.invalidate()

    def update_geometry(self):
        for curve in self.curves:
            self.plot.removeItem(curve)
        self.curves = []
        segments = list(self.geometry.segments())
        for row in range(self.boundaries.rowCount()):
            combo = self.boundaries.cellWidget(row, 2)
            if combo is not None:
                combo.blockSignals(True)
        self.boundaries.clearContents()
        self.boundaries.setRowCount(len(segments))
        for row, (li, j, a, b, tag) in enumerate(segments):
            curve = pg.PlotCurveItem([a[0], b[0]], [a[1], b[1]], pen=pg.mkPen(COLORS[tag], width=3))
            curve.setClickable(True, width=10)
            curve.sigClicked.connect(lambda *_, row=row: self.select_edge(row))
            self.plot.addItem(curve)
            self.curves.append(curve)
            for col, text in enumerate((f"{li+1} / {j+1}", f"{np.linalg.norm(b-a):.6g}")):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.boundaries.setItem(row, col, item)
            combo = QtWidgets.QComboBox()
            combo.addItems(list(BOUNDARIES.values()))
            combo.setCurrentIndex(list(BOUNDARIES).index(tag))
            combo.currentIndexChanged.connect(lambda index, li=li, j=j, row=row: self.boundary_changed(li, j, row, index))
            self.boundaries.setCellWidget(row, 2, combo)
        self.plot.autoRange()
        self.info.setText(f"Площадь: {self.geometry.polygon.area:.6g} м² · отверстий: {len(self.geometry.loops)-1}\nНазначьте вход и выход, затем постройте сетку.")

    def select_edge(self, row):
        self.boundaries.selectRow(row)
        self.boundaries.scrollToItem(self.boundaries.item(row, 0))

    def boundary_changed(self, li, j, row, index):
        tag = list(BOUNDARIES)[index]
        self.geometry.tags[li][j] = tag
        self.curves[row].setPen(pg.mkPen(COLORS[tag], width=3))
        self.invalidate()

    def import_dxf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Импорт контуров", "", "DXF (*.dxf)")
        if not path:
            return
        try:
            geometry = load_dxf(path, ["auto", "mm", "cm", "m"][self.units.currentIndex()], self.tolerance.value())
            self.clear_zones()
            self.geometry = geometry
            self.invalidate()
            self.update_geometry()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Ошибка DXF", str(exc))

    def clear_zones(self):
        for roi in self.zones:
            self.plot.removeItem(roi)
        self.zones = []
        self.zone_table.setRowCount(0)

    def add_zone(self, checked=False, values=None):
        xmin, ymin, xmax, ymax = self.geometry.bounds
        if values is None:
            values = [xmin+(xmax-xmin)*.35, ymin+(ymax-ymin)*.25,
                      xmin+(xmax-xmin)*.65, ymin+(ymax-ymin)*.75, self.size.value()/3]
        row = len(self.zones)
        roi = pg.RectROI(values[:2], [values[2]-values[0], values[3]-values[1]], pen="#f472b6")
        self.zones.append(roi)
        self.plot.addItem(roi)
        self.zone_table.blockSignals(True)
        self.zone_table.insertRow(row)
        for col, value in enumerate(values):
            self.zone_table.setItem(row, col, QtWidgets.QTableWidgetItem(f"{value:.8g}"))
        self.zone_table.blockSignals(False)
        roi.sigRegionChangeFinished.connect(lambda _: self.roi_changed(roi))
        self.invalidate()

    def roi_changed(self, roi):
        row = self.zones.index(roi)
        p, size = roi.pos(), roi.size()
        self.zone_table.blockSignals(True)
        for col, value in enumerate((p.x(), p.y(), p.x()+size.x(), p.y()+size.y())):
            self.zone_table.item(row, col).setText(f"{value:.8g}")
        self.zone_table.blockSignals(False)
        self.invalidate()

    def zone_values(self):
        return [tuple(float(self.zone_table.item(row, col).text()) for col in range(5))
                for row in range(self.zone_table.rowCount())]

    def zone_edited(self, item):
        try:
            values = self.zone_values()[item.row()]
            roi = self.zones[item.row()]
            roi.blockSignals(True)
            roi.setPos(values[:2])
            roi.setSize([values[2]-values[0], values[3]-values[1]])
            roi.blockSignals(False)
            self.invalidate()
        except ValueError:
            self.info.setText("В таблице зоны нужны числовые координаты и размер в метрах")

    def remove_zone(self):
        row = self.zone_table.currentRow()
        if row >= 0:
            self.plot.removeItem(self.zones.pop(row))
            self.zone_table.removeRow(row)
            self.invalidate()

    def build(self):
        if self.process is not None:
            return
        from dataclasses import asdict
        try:
            settings = MeshSettings(cell_type=["triangle", "quad"][self.kind.currentIndex()],
                cell_size=self.size.value(), target_cells=self.target.value() if self.method.currentIndex() else 0,
                zones=self.zone_values())
            self.geometry.validate(boundaries=True)
            settings.validate(self.geometry)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Настройки сетки", str(exc))
            return
        self.invalidate()
        self.messages = mp.get_context("spawn").Queue()
        self.process = mp.get_context("spawn").Process(target=build_worker,
            args=(self.geometry.to_dict(), asdict(settings), self.messages))
        self.process.start()
        for control in self.busy_controls:
            control.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.info.setText("Gmsh строит сетку в отдельном процессе…")
        self.progress.setValue(0)

    def poll(self):
        if self.process is None:
            return
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "progress":
                    self.progress.setValue(round(value[0]/value[1]*90))
                else:
                    self.finish()
                    if kind == "result":
                        self.accept_mesh(value)
                    else:
                        from .error_report import ErrorDialog
                        ErrorDialog("Ошибка построения сетки", value, self).exec()
                    break
        except queue.Empty:
            if self.process.exitcode is not None:
                code = self.process.exitcode
                self.finish()
                self.info.setText(f"Построение завершилось без результата, код {code}")

    def finish(self):
        self.process.join(timeout=.1)
        self.process = None
        self.messages.close()
        for control in self.busy_controls:
            control.setEnabled(True)
        for button in (self.save_button, self.export_button, self.use_button):
            button.setEnabled(self.project is not None)
        self.cancel_button.setEnabled(False)
        self.size.setEnabled(self.method.currentIndex() == 0)
        self.target.setEnabled(self.method.currentIndex() == 1)

    def cancel(self):
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=2)
            self.finish()
            self.info.setText("Построение отменено")

    def accept_mesh(self, project):
        self.project = project
        faces, _, _ = project.connectivity()
        stride = max(1, int(np.ceil(len(faces)/20000)))
        points = project.points[faces[::stride]]
        lines = np.concatenate([points, np.full((len(points), 1, 2), np.nan)], axis=1).reshape(-1, 2)
        self.wire.setData(lines[:, 0], lines[:, 1])
        stats = project.statistics
        self.info.setText(f"Элементов: {len(project.cells)} · узлов: {len(project.points)}\n"
            f"Качество min/среднее: {stats['min_quality']:.3f} / {stats['mean_quality']:.3f}\n"
            + (f"Цель: {project.settings.target_cells}; точное число зависит от сгущения.\n" if project.settings.target_cells else "")
            + (f"Показана часть рёбер: каждое {stride}-е; в расчёт передаётся вся сетка." if stride > 1 else "В расчёт передаются эти элементы без замены на равномерную сетку."))
        self.progress.setValue(100)
        for button in (self.save_button, self.export_button, self.use_button):
            button.setEnabled(True)

    def save_project(self):
        if self.project is None:
            return None
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Проект геометрии и сетки", "model.flowmesh", "FlowDesk Mesh (*.flowmesh)")
        if path:
            if not path.endswith(".flowmesh"):
                path += ".flowmesh"
            try:
                self.project.save(path)
                return path
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Ошибка сохранения", str(exc))
        return None

    def open_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть проект", "", "FlowDesk Mesh (*.flowmesh)")
        if not path:
            return
        try:
            project = MeshProject.load(path)
            self.clear_zones()
            self.geometry = project.geometry
            self.update_geometry()
            self.kind.setCurrentIndex(0 if project.settings.cell_type == "triangle" else 1)
            self.method.setCurrentIndex(1 if project.settings.target_cells else 0)
            self.size.setValue(project.settings.cell_size)
            self.target.setValue(max(50, project.settings.target_cells))
            for zone in project.settings.zones:
                self.add_zone(values=zone)
            self.accept_mesh(project)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Ошибка проекта", str(exc))

    def export_mesh(self):
        if self.project is not None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт Gmsh", "model.msh", "Gmsh (*.msh)")
            if path:
                try:
                    self.project.export(path)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Ошибка экспорта", str(exc))

    def use(self):
        path = self.save_project()
        if path:
            self.mesh_ready.emit(str(Path(path).resolve()))

    def closeEvent(self, event):
        self.cancel()
        self.timer.stop()
        event.accept()


def main():
    mp.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(background="#101c2b", foreground="#dbeafe")
    editor = MeshEditor()
    windows = []
    def launch(path):
        from .gui import Window
        window = Window()
        window.use_custom_mesh(path)
        window.show()
        windows.append(window)
    editor.mesh_ready.connect(launch)
    editor.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
