"""Qt desktop frontend. The numerical solver runs in a spawned process."""
import multiprocessing as mp
import queue
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from .solver import Config, Result, Solver
from .live import PreviewMailbox
from .visualization import streamlines, Tracers
from .scenarios import SCENARIOS, PROFILE_SCENARIO, PROFILE_LABEL

PROFILE_INDEX = len(SCENARIOS)
CUSTOM_INDEX = len(SCENARIOS)+1

GRID_PRESETS = [(64, 16), (96, 24), (144, 36), (256, 64), (512, 128),
                (1024, 256), (1024, 1024)]


def calculate(config, messages, cancel, previews=None):
    try:
        started = time.perf_counter()
        result = Solver(config).run(
            lambda i, n, t: messages.put(("progress", (i, n, t))), cancel,
            preview=previews.publish if previews is not None else None)
        result.diagnostics["wall_seconds"] = time.perf_counter()-started
        messages.put(("result", result))
    except InterruptedError:
        messages.put(("cancelled", None))
    except Exception as exc:
        from .error_report import error_payload
        text = traceback.format_exc()
        try:
            payload = error_payload(text, config, f"{type(exc).__name__}: {exc}")
        except Exception:
            payload = text
        messages.put(("error", payload))


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlowDesk · 2D CFD / теплообмен")
        self.resize(1240, 780)
        self.tracers = None
        self.result = None
        self.preview = None
        self.preview_mailbox = None
        self.preview_domain = None
        self.calculation_id = 0
        self.process = None
        self.custom_mesh_path = None
        self.mesh_editors = []
        self.aero_dialogs = []
        self.last_scenario = 0
        self.playing = False
        self.position = 0.
        self.paths = []
        self.stream_key = None
        self.stream_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streamlines")
        self.stream_future = None
        self.context = mp.get_context("spawn")
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)
        sidebar = QtWidgets.QWidget()
        sidebar.setMaximumWidth(310)
        left = QtWidgets.QVBoxLayout(sidebar)
        title = QtWidgets.QLabel("FLOWDESK")
        title.setStyleSheet("font-size: 27px; font-weight: 700; color: #5eead4")
        left.addWidget(title)
        left.addWidget(QtWidgets.QLabel("Лаборатория течений · 2D CFD"))
        mesh_editor_button = QtWidgets.QPushButton("Геометрия и сетка (DXF)…")
        mesh_editor_button.clicked.connect(self.open_mesh_editor)
        mesh_load_button = QtWidgets.QPushButton("Загрузить свою сетку .flowmesh…")
        mesh_load_button.clicked.connect(self.load_mesh)
        left.addWidget(mesh_editor_button)
        left.addWidget(mesh_load_button)
        self.mesh_info = QtWidgets.QLabel()
        self.mesh_info.setWordWrap(True)
        left.addWidget(self.mesh_info)
        self.scenario = QtWidgets.QComboBox()
        self.scenario.addItems(list(SCENARIOS.values())+[PROFILE_LABEL, "Своя модель (DXF / .flowmesh)"])
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Стационарный · SIMPLE", "Нестационарный · анимация полей"])
        self.mode.setCurrentIndex(1)
        self.turbulence = QtWidgets.QCheckBox("Модель турбулентности k–ε")
        self.turbulence.setChecked(True)
        self.turbulence.setToolTip("Выключено: ламинарные уравнения Навье–Стокса, без турбулентной вязкости и переноса k/epsilon.")
        self.energy = QtWidgets.QCheckBox("Уравнение энергии · температура")
        self.energy.setToolTip("Перенос и теплопроводность при постоянных свойствах. Температура не меняет плотность и вязкость; без плавучести.")
        self.temperature_inlet = self.number(1., 5000., 293.15, " K", 2)
        self.temperature_initial = self.number(1., 5000., 293.15, " K", 2)
        self.temperature_wall = self.number(1., 5000., 333.15, " K", 2)
        self.thermal_wall = QtWidgets.QComboBox()
        self.thermal_wall.addItems(["Теплоизоляция", "Заданная температура"])
        self.heat_capacity = self.number(1., 1e6, 4182., " Дж/(кг·К)", 2)
        self.conductivity = self.number(1e-6, 1e4, .6, " Вт/(м·К)", 6)
        self.turbulent_prandtl = self.number(.1, 10., .85, "", 3)
        self.heat_source = self.number(-1e10, 1e10, 0., " Вт/м³", 1)
        self.iterations = QtWidgets.QSpinBox()
        self.iterations.setRange(10, 10000)
        self.iterations.setValue(1000)
        self.tolerance = self.number(1e-8, 1e-2, 1e-4, "", 8)
        self.coarse = QtWidgets.QCheckBox("Начать с грубой сетки")
        self.coarse.setChecked(True)
        self.grid = QtWidgets.QComboBox()
        self.grid.addItems([f"{nx} × {ny}" for nx, ny in GRID_PRESETS]+["Своя сетка"])
        self.grid.setCurrentIndex(1)
        self.nx = QtWidgets.QSpinBox()
        self.nx.setRange(16, 1024)
        self.nx.setValue(96)
        self.ny = QtWidgets.QSpinBox()
        self.ny.setRange(8, 1024)
        self.ny.setValue(24)
        self.frames = QtWidgets.QSpinBox()
        self.frames.setRange(2, 500)
        self.frames.setValue(80)
        self.top, self.bottom = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for boundary in (self.top, self.bottom):
            boundary.addItems(["Стенка", "Открытая (не стенка)"])
            boundary.setToolTip("Открытая: механическое давление 0; жидкость может входить и выходить. Не свободная поверхность.")
        self.speed = self.number(.1, 100, 1, " м/с")
        self.profile_angle = self.number(-30, 30, 4., " °", 1)
        self.profile_chord = self.number(.01, 2, .1, " м", 3)
        self.profile_cells = QtWidgets.QSpinBox()
        self.profile_cells.setRange(400, 30000)
        self.profile_cells.setValue(800)
        self.profile_polar = QtWidgets.QCheckBox("Поляра по углу атаки (серия расчётов)")
        self.profile_polar.setChecked(True)
        self.polar_min = self.number(-30, 30, -4., " °", 1)
        self.polar_max = self.number(-30, 30, 8., " °", 1)
        self.polar_step = self.number(.5, 30, 4., " °", 1)
        self.duration = self.number(.0001, 100, 1.2, " с", 4)
        self.initial_condition = QtWidgets.QComboBox()
        self.initial_condition.addItems(["Движущийся поток вокруг тела", "Пуск из покоя"])
        self.initial_condition.setToolTip("Для своей сетки и профиля. Движущийся поток предварительно проецируется на непроницаемые стенки; затем рассчитывается развитие следа.")
        self.time_step = self.number(0, 10, 0., " с", 6)
        self.time_step.setSpecialValueText("Авто")
        self.time_step.setToolTip("Верхняя граница физического шага. Уменьшайте для проверки временной сходимости. 0 — автоматический выбор.")
        self.length = self.number(.1, 3, .6, " м")
        self.height = self.number(.01, .2, .04, " м", 3)
        form = QtWidgets.QFormLayout()
        for label, widget in (("Сценарий", self.scenario), ("Режим", self.mode),
                              ("Макс. итераций", self.iterations), ("Невязка", self.tolerance),
                              ("Начальное поле", self.coarse), ("Сетка", self.grid),
                              ("Ячеек по X", self.nx), ("Ячеек по Y", self.ny),
                              ("Верхняя граница", self.top), ("Нижняя граница", self.bottom),
                              ("Кадров", self.frames),
                              ("Скорость входа", self.speed), ("Время расчёта", self.duration), ("Макс. шаг времени", self.time_step), ("Начальное состояние", self.initial_condition),
                              ("Длина", self.length), ("Высота области", self.height)):
            form.addRow(label, widget)
            if widget is self.mode:
                form.addRow(self.turbulence)
                form.addRow(self.energy)
        left.addLayout(form)
        self.thermal_controls = []
        for label, control in (("Температура входа", self.temperature_inlet),
            ("Начальная температура", self.temperature_initial), ("Стенки: теплообмен", self.thermal_wall),
            ("Температура стенок", self.temperature_wall), ("Теплоёмкость cp", self.heat_capacity),
            ("Теплопроводность λ", self.conductivity), ("Турбулентное Pr", self.turbulent_prandtl),
            ("Объёмный источник тепла", self.heat_source)):
            form.addRow(label, control)
            self.thermal_controls.append((form.labelForField(control), control))
        self.profile_controls = []
        for label, control in (("Угол атаки", self.profile_angle), ("Хорда", self.profile_chord),
            ("Элементов профиля", self.profile_cells), ("Cl/Cd(α)", self.profile_polar),
            ("Начальный α", self.polar_min), ("Конечный α", self.polar_max), ("Шаг α", self.polar_step)):
            form.addRow(label, control)
            self.profile_controls.append((form.labelForField(control), control))
        self.duration_caption = form.labelForField(self.duration)
        self.memory = QtWidgets.QLabel()
        self.memory.setWordWrap(True)
        left.addWidget(self.memory)
        self.grid.currentIndexChanged.connect(self.preset_changed)
        self.nx.valueChanged.connect(self.mesh_changed)
        self.ny.valueChanged.connect(self.mesh_changed)
        self.frames.valueChanged.connect(self.update_memory)
        self.energy.toggled.connect(self.mode_changed)
        self.turbulence.toggled.connect(self.mode_changed)
        self.thermal_wall.currentIndexChanged.connect(self.mode_changed)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.scenario.currentIndexChanged.connect(self.mode_changed)
        self.profile_polar.toggled.connect(self.mode_changed)
        self.profile_chord.valueChanged.connect(self.mode_changed)
        self.update_memory()
        self.settings = [control for _, control in self.profile_controls+self.thermal_controls]+[self.turbulence, self.energy, mesh_editor_button, mesh_load_button, self.scenario, self.mode, self.iterations, self.tolerance, self.coarse,
                         self.grid, self.nx, self.ny, self.top, self.bottom,
                         self.frames, self.speed, self.duration, self.time_step, self.initial_condition, self.length, self.height]
        self.run_button = QtWidgets.QPushButton("Рассчитать кадры")
        self.run_button.clicked.connect(self.start)
        self.cancel_button = QtWidgets.QPushButton("Отменить расчёт")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        left.addWidget(self.run_button)
        left.addWidget(self.cancel_button)
        self.progress = QtWidgets.QProgressBar()
        left.addWidget(self.progress)
        self.info = QtWidgets.QLabel("Выберите сценарий и выполните расчёт.\n\nРазмеры и поля — в системе СИ.")
        self.info.setWordWrap(True)
        left.addWidget(self.info)
        note = QtWidgets.QLabel("Несжимаемые Навье–Стокс; k–ε и энергия включаются отдельно.\nЦель ±10% проверена только для стандартного канала со стенками. Новые геометрии и открытые границы требуют отдельного эталона.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9aaec3; font-size: 12px")
        left.addWidget(note)
        left.addStretch()
        save = QtWidgets.QPushButton("Сохранить результат…")
        save.clicked.connect(self.save)
        load = QtWidgets.QPushButton("Открыть результат…")
        load.clicked.connect(self.load)
        self.save_button, self.load_button = save, load
        self.residual_button = QtWidgets.QPushButton("Невязки SIMPLE…")
        self.residual_button.setEnabled(False)
        self.residual_button.clicked.connect(self.show_residuals)
        left.addWidget(self.residual_button)
        self.aero_button = QtWidgets.QPushButton("Графики Cl / Cd / Cl/Cd…")
        self.aero_button.setEnabled(False)
        self.aero_button.clicked.connect(self.show_aerodynamics)
        left.addWidget(self.aero_button)
        left.addWidget(save)
        left.addWidget(load)
        self.native_export = QtWidgets.QPushButton("Поля на исходной сетке (.vtu)…")
        self.native_export.setEnabled(False)
        self.native_export.clicked.connect(self.export_native)
        left.addWidget(self.native_export)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMaximumWidth(340)
        scroll.setMinimumWidth(320)
        layout.addWidget(scroll)
        right = QtWidgets.QVBoxLayout()
        toolbar = QtWidgets.QHBoxLayout()
        self.field = QtWidgets.QComboBox()
        self.field.addItems(["Скорость · м/с", "Давление · Па", "Температура · K"])
        self.field.model().item(2).setEnabled(False)
        self.field.currentIndexChanged.connect(self.change_field)
        self.show_lines = QtWidgets.QCheckBox("Линии тока")
        self.show_lines.setChecked(True)
        self.show_lines.toggled.connect(self.render)
        equal_scale = QtWidgets.QCheckBox("Оси 1:1")
        equal_scale.setToolTip("По умолчанию оси масштабируются независимо для удобства просмотра")
        equal_scale.toggled.connect(lambda checked: self.plot.setAspectLocked(checked))
        self.rate = QtWidgets.QComboBox()
        self.rate.addItems(["0.25×", "0.5×", "1×", "2×"])
        self.rate.setToolTip("Скорость просмотра: при 1× весь нестационарный расчёт проигрывается за 8 секунд. Физическое время показано под полем.")
        self.rate.setCurrentIndex(2)
        toolbar.addWidget(self.field)
        toolbar.addWidget(self.show_lines)
        toolbar.addWidget(equal_scale)
        self.aero_toolbar = QtWidgets.QPushButton("Cl / Cd")
        self.aero_toolbar.setToolTip("История Cl, Cd, Cl/Cd и поляра по углам атаки")
        self.aero_toolbar.setEnabled(False)
        self.aero_toolbar.clicked.connect(self.show_aerodynamics)
        toolbar.addWidget(self.aero_toolbar)
        toolbar.addStretch()
        toolbar.addWidget(QtWidgets.QLabel("Воспроизведение"))
        toolbar.addWidget(self.rate)
        right.addLayout(toolbar)
        self.graphics = pg.GraphicsLayoutWidget()
        self.plot = self.graphics.addPlot()
        self.plot.setLabel("bottom", "x", units="m")
        self.plot.setLabel("left", "y", units="m")
        self.plot.setAspectLocked(False)
        self.plot.setMenuEnabled(False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image)
        cmap = pg.colormap.get("viridis")
        self.bar = pg.ColorBarItem(values=(0, 1), colorMap=cmap, interactive=False)
        self.bar.setImageItem(self.image, insert_in=self.plot)
        self.line = pg.PlotCurveItem(pen=pg.mkPen((235, 250, 255, 160), width=1), connect="finite")
        self.plot.addItem(self.line)
        self.particles = pg.ScatterPlotItem(size=4, pen=None, brush="#f8fafc")
        self.plot.addItem(self.particles)
        self.trails = pg.PlotCurveItem(pen=pg.mkPen((170, 235, 255, 110), width=1.3), connect="finite")
        self.plot.addItem(self.trails)
        self.boundaries = [pg.PlotCurveItem(), pg.PlotCurveItem()]
        for line in self.boundaries:
            self.plot.addItem(line)
        self.custom_lines = {tag: pg.PlotCurveItem(pen=pg.mkPen(color, width=2), connect="finite")
            for tag, color in dict(inlet="#5eead4", outlet="#60a5fa", wall="#fbbf24", open="#c4b5fd").items()}
        for line in self.custom_lines.values():
            self.plot.addItem(line)
        right.addWidget(self.graphics, 1)
        controls = QtWidgets.QHBoxLayout()
        self.play = QtWidgets.QPushButton("▶ Воспроизвести")
        self.play.clicked.connect(self.toggle)
        self.play.setEnabled(False)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 10000)
        self.slider.sliderMoved.connect(self.seek)
        self.time_label = QtWidgets.QLabel("t = 0.000 с")
        controls.addWidget(self.play)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time_label)
        right.addLayout(controls)
        self.caption = QtWidgets.QLabel("Линии — мгновенное поле; маркеры — направление, не трассеры. Масштаб осей независимый.")
        right.addWidget(self.caption)
        layout.addLayout(right, 1)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self.last_tick = time.perf_counter()
        self.mode_changed()
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #101c2b; color: #e1eaf5; font-size: 13px; }
            QPushButton { background: #20384d; border: 1px solid #35536a; border-radius: 6px; padding: 9px; }
            QPushButton:hover { background: #2d526b; } QPushButton:disabled { color: #60758a; }
            QComboBox, QDoubleSpinBox { background: #1b3044; padding: 6px; border: 1px solid #35536a; }
            QProgressBar { border: 1px solid #35536a; text-align: center; height: 18px; }
            QProgressBar::chunk { background: #167d8d; }
        """)

    @staticmethod
    def number(low, high, value, suffix, decimals=2):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setValue(value)
        widget.setSingleStep(10**(-decimals))
        widget.setSuffix(suffix)
        return widget

    def start(self):
        if self.process is not None:
            return
        custom = self.scenario.currentIndex() == CUSTOM_INDEX
        profile = self.scenario.currentIndex() == PROFILE_INDEX
        if custom and not self.custom_mesh_path:
            self.load_mesh()
            if not self.custom_mesh_path:
                return
        angles = ()
        if profile and self.mode.currentIndex() == 0 and self.profile_polar.isChecked():
            if self.polar_min.value() > self.polar_max.value():
                QtWidgets.QMessageBox.warning(self, "Углы поляры", "Начальный угол должен быть не больше конечного")
                return
            angles = tuple(float(a) for a in np.arange(self.polar_min.value(), self.polar_max.value()+1e-6, self.polar_step.value()))
            if len(angles) > 21:
                QtWidgets.QMessageBox.warning(self, "Углы поляры", "Максимум 21 угол. Увеличьте шаг.")
                return
        c = Config(scenario="custom" if custom else PROFILE_SCENARIO if profile else list(SCENARIOS)[self.scenario.currentIndex()],
                   mesh_path=self.custom_mesh_path if custom else None,
                   mode="steady" if self.mode.currentIndex() == 0 else "transient",
                   turbulence=self.turbulence.isChecked(), energy=self.energy.isChecked(),
                   heat_capacity=self.heat_capacity.value(), conductivity=self.conductivity.value(),
                   turbulent_prandtl=self.turbulent_prandtl.value(),
                   temperature_inlet=self.temperature_inlet.value(), temperature_initial=self.temperature_initial.value(),
                   temperature_wall=self.temperature_wall.value(), heat_source=self.heat_source.value(),
                   thermal_wall="adiabatic" if self.thermal_wall.currentIndex() == 0 else "isothermal",
                   max_iterations=self.iterations.value(), tolerance=self.tolerance.value(),
                   coarse_start=self.coarse.isChecked(),
                   nx=self.nx.value(), ny=self.ny.value(), frames=self.frames.value(),
                   top=["wall", "open"][self.top.currentIndex()],
                   bottom=["wall", "open"][self.bottom.currentIndex()],
                   speed=self.speed.value(), duration=self.duration.value(), time_step=self.time_step.value(),
                   initial_condition="rest" if (custom or profile) and self.initial_condition.currentIndex() == 1 else "uniform",
                   length=self.length.value(), height=self.height.value(),
                   rho=1.225 if profile else 998.2, nu=1.5e-5 if profile else 1.004e-6,
                   intensity=.01 if profile else .05,
                   profile_angle=self.profile_angle.value(), profile_chord=self.profile_chord.value(),
                   profile_cells=self.profile_cells.value(), profile_sweep=angles)
        self.playing = False
        self.play.setText("▶ Воспроизвести")
        self.preview = None
        self.preview_domain = None
        self.calculation_id += 1
        self.preview_mailbox = PreviewMailbox(self.context)
        self.play.setEnabled(False)
        self.slider.setEnabled(False)
        for control in (self.save_button, self.load_button, self.native_export, self.residual_button,
                        self.aero_button, self.aero_toolbar):
            control.setEnabled(False)
        self.tracers = None
        self.paths = []
        self.stream_key = None
        self.line.setData([], [])
        self.particles.setData([], [])
        self.trails.setData([], [])
        self.image.setImage(np.zeros((min(c.ny, 192), min(c.nx, 320))), autoLevels=False)
        self.image.setRect(QtCore.QRectF(0, 0, c.length, c.height))
        for line in self.boundaries+list(self.custom_lines.values()):
            line.setVisible(False)
        self.caption.setText("Идёт расчёт: ожидается первое промежуточное поле…")
        self.messages = self.context.Queue()
        self.calculation_mode = c.mode
        self.calculation_scenario = c.scenario
        self.cancel_event = self.context.Event()
        self.process = self.context.Process(target=calculate, args=(c, self.messages, self.cancel_event, self.preview_mailbox))
        self.process.start()
        for widget in self.settings+[self.run_button]:
            widget.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.info.setText(("Расчёт RANS / k–ε" if c.turbulence else "Расчёт ламинарного течения")+(" + энергия" if c.energy else "")+"…\nПервый запуск компилирует численные ядра.\nИнтерфейс остаётся доступен.")

    def mode_changed(self, *_):
        custom = self.scenario.currentIndex() == CUSTOM_INDEX
        profile = self.scenario.currentIndex() == PROFILE_INDEX
        steady = self.mode.currentIndex() == 0
        enabled = self.process is None
        self.mode.setEnabled(enabled)
        for label, control in self.thermal_controls:
            label.setVisible(self.energy.isChecked())
            control.setVisible(self.energy.isChecked())
            control.setEnabled(enabled and self.energy.isChecked())
        self.temperature_wall.setEnabled(enabled and self.energy.isChecked() and self.thermal_wall.currentIndex() == 1)
        self.temperature_initial.setEnabled(enabled and self.energy.isChecked() and not steady)
        self.turbulent_prandtl.setEnabled(enabled and self.energy.isChecked() and self.turbulence.isChecked())
        for control in (self.grid, self.nx, self.ny, self.length, self.height, self.top, self.bottom):
            control.setEnabled(enabled and not custom and not profile)
        for label, control in self.profile_controls:
            label.setVisible(profile)
            control.setVisible(profile)
            control.setEnabled(enabled and profile)
        for control in (self.polar_min, self.polar_max, self.polar_step):
            control.setEnabled(enabled and profile and steady and self.profile_polar.isChecked())
        self.profile_polar.setEnabled(enabled and profile and steady)
        self.coarse.setEnabled(enabled and steady and not custom and not profile)
        if not profile and self.last_scenario == PROFILE_INDEX:
            self.heat_capacity.setValue(4182.)
            self.conductivity.setValue(.6)
        if profile:
            if self.last_scenario != PROFILE_INDEX:
                self.speed.setValue(20.)
                self.heat_capacity.setValue(1005.)
                self.conductivity.setValue(.0257)
                if not steady:
                    self.duration.setValue(.025)
                self.set_mesh(320, 160)
            for control, value in ((self.length, 8*self.profile_chord.value()),
                                   (self.height, 4*self.profile_chord.value())):
                control.setDecimals(4)
                control.setRange(1e-7, 1e6)
                control.setValue(value)
            self.mesh_info.setText("Воздух: ρ=1.225 кг/м³, ν=1.5e−5 м²/с; турбулентность входа 1%.\n"
                "Верх/низ профиля: 0.10c / 0.06c. Nx/Ny — пиксели. Поляра требует отдельного расчёта каждого угла.")
        elif not custom:
            self.mesh_info.setText("")
        self.last_scenario = self.scenario.currentIndex()
        self.run_button.setText("Рассчитать SIMPLE" if steady else "Рассчитать кадры")
        self.duration_caption.setText("Цикл анимации" if steady else "Время расчёта")
        self.initial_condition.setEnabled(enabled and not steady and (custom or profile))
        self.time_step.setEnabled(enabled and not steady)
        self.frames.setEnabled(enabled and not steady)
        self.iterations.setToolTip("Лимит итераций SIMPLE на каждый физический шаг" if not steady and (custom or profile) else "Лимит итераций стационарного SIMPLE")
        for control in (self.iterations, self.tolerance):
            control.setEnabled(enabled and (steady or custom or profile))
        self.duration.setToolTip("В SIMPLE это длительность воспроизведения, не физического расчёта" if steady else "Длительность физического расчёта")
        self.update_memory()

    def open_mesh_editor(self):
        from .mesh_gui import MeshEditor
        editor = MeshEditor(self)
        editor.mesh_ready.connect(self.use_custom_mesh)
        editor.show()
        self.mesh_editors.append(editor)

    def load_mesh(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Своя модель", "", "FlowDesk Mesh (*.flowmesh)")
        if path:
            try:
                self.use_custom_mesh(path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Ошибка сетки", str(exc))

    def use_custom_mesh(self, path):
        from .meshing import MeshProject
        from pathlib import Path
        if self.process is not None:
            QtWidgets.QMessageBox.information(self, "Расчёт выполняется", "Дождитесь завершения или отмените текущий расчёт, затем загрузите сохранённый проект сетки.")
            return
        project = MeshProject.load(path)
        self.custom_mesh_path = str(Path(path).resolve())
        xmin, ymin, xmax, ymax = project.geometry.bounds
        for control in (self.length, self.height):
            control.setDecimals(7)
            control.setRange(1e-7, 1e6)
        self.length.setValue(xmax-xmin)
        self.height.setValue(ymax-ymin)
        self.set_mesh(256, max(8, min(256, round(256*(ymax-ymin)/(xmax-xmin)))))
        self.scenario.setCurrentIndex(CUSTOM_INDEX)
        self.mesh_info.setText(f"Своя модель: {len(project.cells)} реальных элементов.\n"
            "Границы и размер элементов заданы в утилите. Nx/Ny ниже — только пиксели отображения.")
        self.mode_changed()

    def set_mesh(self, nx, ny):
        for control, value in ((self.nx, nx), (self.ny, ny)):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self.grid.blockSignals(True)
        self.grid.setCurrentIndex(GRID_PRESETS.index((nx, ny)) if (nx, ny) in GRID_PRESETS else len(GRID_PRESETS))
        self.grid.blockSignals(False)
        self.update_memory()

    def preset_changed(self, index):
        if index < len(GRID_PRESETS):
            self.set_mesh(*GRID_PRESETS[index])

    def mesh_changed(self, *_):
        self.set_mesh(self.nx.value(), self.ny.value())

    def update_memory(self, *_):
        c = Config(nx=self.nx.value(), ny=self.ny.value(), frames=self.frames.value(),
                   mode="steady" if self.mode.currentIndex() == 0 else "transient", energy=self.energy.isChecked())
        mib = c.nx*c.ny*c.frame_count*c.field_count*4/1024**2
        self.memory.setText(f"Сохранится {c.frame_count} кадров · поля {mib:.1f} МиБ."
            + ("\nSIMPLE: одно итоговое поле, анимация маркеров." if c.mode == "steady" else "\nКадры сокращены до бюджета 256 МиБ." if c.frame_count < c.frames else "")
            + ("\nКрупная сетка: расчёт существенно дольше; мелкая сетка у стенки не гарантирует точность k–ε." if c.nx*c.ny > 65536 else ""))

    def finish_worker(self):
        self.process.join(timeout=.1)
        self.process = None
        self.messages.close()
        self.preview_mailbox = None
        self.save_button.setEnabled(True)
        self.load_button.setEnabled(True)
        for widget in self.settings+[self.run_button]:
            widget.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.mode_changed()

    def cancel(self):
        if self.process is not None:
            self.cancel_event.set()
            self.cancel_button.setEnabled(False)

    def tick(self):
        now = time.perf_counter()
        elapsed, self.last_tick = now-self.last_tick, now
        if self.process is not None:
            if self.preview_mailbox is not None:
                frame = self.preview_mailbox.take_latest()
                if frame is not None:
                    self.accept_preview(frame)
            try:
                while True:
                    kind, value = self.messages.get_nowait()
                    if kind == "progress":
                        i, n, t = value
                        self.progress.setValue(round(i/n*100))
                        self.info.setText(f"Профиль α={t:+.1f}° · итерация {i}/{n}" if self.calculation_scenario == PROFILE_SCENARIO and self.calculation_mode == "steady" else f"SIMPLE · {'грубая сетка · ' if t < 0 else ''}итерация {i}/{n}" if self.calculation_mode == "steady"
                                          else f"Кадр {i}/{n} · t = {t:.3f} с")
                    else:
                        self.finish_worker()
                        if kind == "result":
                            self.accept_result(value)
                            if value.config.mode == "transient":
                                self.position = value.config.duration
                                self.render()
                        elif kind == "error":
                            self.discard_preview()
                            self.info.setText("Ошибка расчёта")
                            from .error_report import ErrorDialog
                            ErrorDialog("Ошибка расчёта — полный отчёт", value, self).exec()
                        else:
                            self.discard_preview()
                            self.info.setText("Расчёт отменён")
                        break
            except queue.Empty:
                # A crashed child cannot report through the queue.
                if self.process is not None and self.process.exitcode is not None:
                    code = self.process.exitcode
                    self.finish_worker()
                    self.discard_preview()
                    self.info.setText(f"Процесс завершился без результата (код {code}).")
        if self.process is None and self.playing and self.result is not None:
            speed = [.25, .5, 1, 2][self.rate.currentIndex()]
            self.position = (self.position+elapsed*speed*(self.result.config.duration/8 if self.result.config.mode == "transient" else 1.)) % self.result.config.duration
            self.render()
        elif self.show_lines.isChecked() and self.stream_future is not None and self.stream_future[2].done():
            self.render()

    def discard_preview(self):
        self.preview = None
        self.preview_domain = None
        if self.result is not None:
            self.accept_result(self.result)
        else:
            self.line.setData([], [])
            self.particles.setData([], [])
            self.trails.setData([], [])
            self.image.setImage(np.zeros((8, 16)), autoLevels=False)
            for line in self.custom_lines.values():
                line.setVisible(False)
            for line in self.boundaries:
                line.setVisible(False)
            self.play.setEnabled(False)
            self.slider.setEnabled(False)
            self.field.model().item(2).setEnabled(False)
            if self.field.currentIndex() == 2:
                self.field.setCurrentIndex(0)
            self.caption.setText("Нет завершённого результата.")
            self.time_label.setText("Расчёт остановлен")

    def accept_preview(self, frame):
        self.preview = frame
        c = frame.config
        self.field.model().item(2).setEnabled(frame.temperature is not None)
        if self.field.currentIndex() == 2 and frame.temperature is None:
            self.field.setCurrentIndex(0)
        domain = (c.scenario, c.length, c.height, c.profile_angle, c.nx, c.ny, frame.phase == "coarse")
        if domain != self.preview_domain:
            self.preview_domain = domain
            self.paths = []
            self.stream_key = None
            self.line.setData([], [])
            self.image.setImage(np.zeros(frame.fluid.shape), autoLevels=False)
            self.image.setRect(QtCore.QRectF(0, 0, c.length, c.height))
            self.plot.setRange(xRange=(0, c.length), yRange=(0, c.height), padding=.05)
            for tag, line in self.custom_lines.items():
                line.setVisible(frame.boundaries is not None)
                if frame.boundaries is not None:
                    coords = frame.boundaries[tag]
                    line.setData(coords[:, 0], coords[:, 1])
            for line, boundary, row, y in zip(self.boundaries, (c.bottom, c.top),
                                             (frame.fluid[0], frame.fluid[-1]), (0., c.height)):
                x = (np.arange(c.nx)+.5)*c.length/c.nx
                line.setData(np.where(row, x, np.nan), np.full(c.nx, y), connect="finite")
                line.setPen(pg.mkPen("#60d5ed" if boundary == "open" else "#fbbf24", width=2))
                line.setVisible(c.scenario not in ("custom", PROFILE_SCENARIO))
        self.caption.setText("ПРЕДВАРИТЕЛЬНОЕ ПОЛЕ · расчёт продолжается. Итерации SIMPLE не являются физическим временем.")
        self.change_field()

    def accept_result(self, result):
        self.preview = None
        self.preview_domain = None
        self.slider.setEnabled(True)
        self.tracers = Tracers(result) if result.config.mode == "transient" else None
        self.trails.setData([], [])
        self.caption.setText("Поля скорости и давления меняются во времени. Линии — мгновенные линии тока; частицы движутся по рассчитанной скорости." if self.tracers else "Стационарное поле. Маркеры показывают направление; физическое время не рассчитывается.")
        self.result = result
        self.field.model().item(2).setEnabled(result.temperature is not None)
        if self.field.currentIndex() == 2 and result.temperature is None:
            self.field.setCurrentIndex(0)
        self.position = 0.
        self.stream_key = None
        self.paths = []
        self.line.setData([], [])
        self.particles.setData([], [])
        self.play.setEnabled(True)
        d = result.diagnostics
        self.native_export.setEnabled(result.native is not None)
        self.residual_button.setEnabled(bool(d.get("residual_history")))
        self.aero_button.setEnabled(bool(d.get("aerodynamics")))
        self.aero_toolbar.setEnabled(bool(d.get("aerodynamics")))
        state = (f"SIMPLE: {'сошлось' if d['converged'] else 'НЕ СОШЛОСЬ — достигнут лимит'}\n"
                 f"Итераций: {d['iterations']}; макс. невязка: {max(d['residuals'].values()):.2e}\n"
                 if result.config.mode == "steady" else "")
        self.info.setText(state+f"Расчёт завершён · {d.get('wall_seconds', 0):.2f} с\n"
            f"Re = {d['reynolds']:.0f}\nΔp вход–выход = {d['pressure_drop_pa']:.2f} Па\n"
            f"Небаланс расхода: {d['mass_imbalance']:.2e}\n"
            + (f"Изменение u между последними кадрами: {d['final_frame_velocity_change']:.2e}\n" if result.config.mode == "transient" else "")
            + ("Обратный поток на выходе: увеличьте длину канала.\n" if d['backflow'] else "")
            + (f"Температура: {d['energy']['min_temperature_k']:.2f}…{d['energy']['max_temperature_k']:.2f} K\n"
               f"Небаланс энергии: {d['energy']['relative_imbalance']:.2e}\n" if "energy" in d else "")
            + ("Модель: k–ε.\n" if result.config.turbulence else "Модель: ламинарная.\n")
            + "Сходимость и точность см. в отчёте проверки.")
        c = result.config
        self.turbulence.setChecked(c.turbulence)
        self.energy.setChecked(c.energy)
        self.initial_condition.setCurrentIndex(0 if c.initial_condition == "uniform" else 1)
        self.mode.setCurrentIndex(0 if c.mode == "steady" else 1)
        self.iterations.setValue(c.max_iterations)
        self.tolerance.setValue(c.tolerance)
        self.coarse.setChecked(c.coarse_start)
        if c.scenario in ("custom", PROFILE_SCENARIO):
            for control in (self.length, self.height):
                control.setDecimals(7)
                control.setRange(1e-7, 1e6)
            self.custom_mesh_path = c.mesh_path
            self.scenario.setCurrentIndex(CUSTOM_INDEX if c.scenario == "custom" else PROFILE_INDEX)
            if c.scenario == PROFILE_SCENARIO:
                self.profile_angle.setValue(c.profile_angle)
                self.profile_chord.setValue(c.profile_chord)
                self.profile_cells.setValue(c.profile_cells)
                self.profile_polar.setChecked(bool(c.profile_sweep))
                if c.profile_sweep:
                    self.polar_min.setValue(min(c.profile_sweep))
                    self.polar_max.setValue(max(c.profile_sweep))
                    unique_angles = sorted(set(c.profile_sweep))
                    if len(unique_angles) > 1:
                        self.polar_step.setValue(min(np.diff(unique_angles)))
            self.mesh_info.setText(f"Своя модель: {d.get('mesh_cells', 0)} реальных элементов.\n"
                "Результат содержит поля на исходной сетке. Nx/Ny — пиксели отображения.")
            if c.scenario == PROFILE_SCENARIO:
                aero = d["aerodynamics"]
                last = aero["final"]
                ratio = "—" if last["cl_cd"] is None else f"{last['cl_cd']:.3f}"
                self.mesh_info.setText(f"Профиль · α={c.profile_angle:+.1f}° · {d['mesh_cells']} элементов\n"
                    f"Cl={last['cl']:.4f} · Cd={last['cd']:.4f} · Cl/Cd={ratio}\n"
                    f"Поляра: {len(aero.get('polar', []))} углов. ρ={c.rho:g}; силы на единичный размах.")
        else:
            self.scenario.setCurrentIndex(list(SCENARIOS).index(c.scenario))
            self.mesh_info.setText("")
        result_mesh_info = self.mesh_info.text()
        self.top.setCurrentIndex(0 if c.top == "wall" else 1)
        self.bottom.setCurrentIndex(0 if c.bottom == "wall" else 1)
        self.frames.setValue(c.frames)
        for control, value in ((self.speed, c.speed), (self.duration, c.duration), (self.time_step, c.time_step),
                               (self.length, c.length), (self.height, c.height)):
            control.setValue(value)
        for name in ("heat_capacity", "conductivity", "turbulent_prandtl", "temperature_initial",
                     "temperature_inlet", "temperature_wall", "heat_source"):
            getattr(self, name).setValue(getattr(c, name))
        self.thermal_wall.setCurrentIndex(0 if c.thermal_wall == "adiabatic" else 1)
        self.mode_changed()
        self.mesh_info.setText(result_mesh_info)
        self.set_mesh(c.nx, c.ny)
        self.image.setImage(np.zeros((c.ny, c.nx)), autoLevels=False)
        self.image.setRect(QtCore.QRectF(0, 0, c.length, c.height))
        for tag, line in self.custom_lines.items():
            line.setVisible(result.native is not None)
            if result.native is not None:
                native = result.native
                points = native["points"]-native["points"].min(axis=0)
                edges = points[native["boundary_edges"][native["boundary_tags"] == tag]]
                joined = np.concatenate([edges, np.full((len(edges), 1, 2), np.nan)], axis=1).reshape(-1, 2)
                line.setData(joined[:, 0], joined[:, 1])
        for line, boundary, row, y in zip(self.boundaries, (c.bottom, c.top),
                                         (result.fluid[0], result.fluid[-1]), (0, c.height)):
            # Leave geometric solids intact even when an outer edge is open.
            x = (np.arange(c.nx)+.5)*c.length/c.nx
            line.setData(np.where(row, x, np.nan), np.full(c.nx, y), connect="finite")
            line.setPen(pg.mkPen("#60d5ed" if boundary == "open" else "#fbbf24", width=2,
                style=QtCore.Qt.PenStyle.DashLine if boundary == "open" else QtCore.Qt.PenStyle.SolidLine))
            line.setVisible(result.native is None)
        self.caption.setToolTip("Жёлтая граница — стенка. Голубая пунктирная — открытая граница давления.")
        self.plot.setRange(xRange=(0, c.length), yRange=(0, c.height), padding=.05)
        self.change_field()
        self.progress.setValue(100)

    def change_field(self, *_):
        if self.result is None and self.preview is None:
            return
        r = self.preview if self.preview is not None else self.result
        values = np.hypot(r.u, r.v) if self.field.currentIndex() == 0 else r.pressure if self.field.currentIndex() == 1 else r.temperature
        values = values[r.fluid] if self.preview is not None else values[:, r.fluid]
        low, high = float(values.min()), float(values.max())
        if self.field.currentIndex() == 1:
            bound = max(abs(low), abs(high), 1e-6)
            low, high = -bound, bound
        self.bar.setColorMap(pg.colormap.get("CET-D1" if self.field.currentIndex() == 1 else "inferno" if self.field.currentIndex() == 2 else "viridis"))
        self.bar.setLevels((low, max(high, low+1e-6)))
        self.render()

    def render(self, *_):
        live = self.preview is not None
        if (self.result is None and not live) or (self.process is not None and not live):
            return
        r = self.preview if live else self.result
        owner_token = ("preview", self.calculation_id, self.preview_domain) if live else ("result", id(r))
        t = r.physical_time if live else self.position
        if live:
            u, v = r.u, r.v
            field = np.hypot(u, v) if self.field.currentIndex() == 0 else r.pressure if self.field.currentIndex() == 1 else r.temperature
            stage = {"flow": "SIMPLE", "step": "шаг", "energy": "энергия", "coarse": "грубая сетка"}[r.phase]
            angle = f" · α={r.config.profile_angle:+.1f}°" if r.config.scenario == PROFILE_SCENARIO else ""
            self.time_label.setText(f"Предпросмотр · {stage} · итерация {r.iteration}"+
                (f" · t={t:.6g} с" if r.config.mode == "transient" else "")+angle)
        else:
            index = int(np.clip(np.searchsorted(r.time, t, side="right")-1, 0, len(r.time)-2))
            alpha = np.clip((t-r.time[index])/(r.time[index+1]-r.time[index]), 0, 1)
            def blend(q):
                return q[index]*(1-alpha)+q[index+1]*alpha
            u, v = blend(r.u), blend(r.v)
            field = np.hypot(u, v) if self.field.currentIndex() == 0 else blend(r.pressure) if self.field.currentIndex() == 1 else blend(r.temperature)
            self.slider.setValue(round(t/r.config.duration*10000))
            self.time_label.setText(f"{'Анимация' if r.config.mode == 'steady' else 't'} = {t:.3f} с")
        self.image.setImage(np.where(r.fluid, field, np.nan), autoLevels=False)
        self.line.setVisible(self.show_lines.isChecked())
        self.particles.setVisible(self.show_lines.isChecked() and not live)
        self.trails.setVisible(self.show_lines.isChecked() and self.tracers is not None and not live)
        if self.show_lines.isChecked():
            key = (r.iteration, r.config.profile_angle, r.phase == "coarse") if live else int(t/r.config.duration*80)  # integrate only eight frozen fields per simulated second
            if self.stream_future is not None and self.stream_future[2].done():
                owner, sampled_key, future = self.stream_future
                self.stream_future = None
                if owner == owner_token:
                    self.paths = future.result()
                    self.stream_key = sampled_key
                    joined = np.concatenate([np.vstack([p, [np.nan, np.nan]]) for p in self.paths]) if self.paths else np.empty((0, 2))
                    self.line.setData(joined[:, 0], joined[:, 1])
            if self.stream_key != key and self.stream_future is None:
                # Integration never blocks the playback timer; one pending field at most.
                self.stream_future = (owner_token, key, self.stream_pool.submit(
                    streamlines, u.copy(), v.copy(), r.fluid, r.config.length, r.config.height))
            if live:
                return
            if self.tracers is not None:
                dots, trails = self.tracers.advance(t)
                self.particles.setData(dots[:, 0], dots[:, 1])
                self.trails.setData(trails[:, 0], trails[:, 1])
            else:
                # Direction markers are used only for explicitly stationary results.
                ds = max(.45*min(r.config.length/r.config.nx, r.config.height/r.config.ny), r.config.length/1800)
                dots = np.array([p[int((t*.35*r.config.speed/ds+i*31) % len(p))] for i, p in enumerate(self.paths)])
                self.particles.setData(dots[:, 0], dots[:, 1]) if len(dots) else self.particles.setData([], [])

    def toggle(self):
        if self.process is not None:
            return
        self.playing = not self.playing
        self.play.setText("⏸ Пауза" if self.playing else "▶ Воспроизвести")

    def seek(self, value):
        if self.process is not None:
            return
        if self.result is not None:
            self.position = value/10000*self.result.config.duration
            self.render()

    def save(self):
        if self.process is not None or self.preview is not None or self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить расчёт", "flow.npz", "FlowDesk (*.npz)")
        if path:
            try:
                self.result.save(path if path.endswith(".npz") else path+".npz")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def show_residuals(self):
        if self.result is None or not self.result.diagnostics.get("residual_history"):
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Невязки стационарного SIMPLE")
        dialog.resize(760, 480)
        layout = QtWidgets.QVBoxLayout(dialog)
        plot = pg.PlotWidget()
        plot.setLogMode(y=True)
        plot.addLegend()
        plot.setLabel("bottom", "Итерация")
        plot.setLabel("left", "Нормированная невязка")
        history = self.result.diagnostics["residual_history"]
        x = [row["iteration"] for row in history]
        for key, color in zip(("u", "v", "k", "e", "continuity"),
                              ("#5eead4", "#60a5fa", "#fbbf24", "#f472b6", "#c4b5fd")):
            plot.plot(x, [max(row[key], 1e-16) for row in history], pen=color, name=key)
        plot.addItem(pg.InfiniteLine(pos=np.log10(self.result.config.tolerance), angle=0,
                                    pen=pg.mkPen("#ffffff", style=QtCore.Qt.PenStyle.DashLine)))
        layout.addWidget(plot)
        layout.addWidget(QtWidgets.QLabel("Сходимость требует всех невязок ниже порога три итерации подряд."))
        dialog.exec()

    def export_native(self):
        if self.process is not None or self.preview is not None or self.result is None or self.result.native is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Поля на расчётных элементах", "flow.vtu", "VTK unstructured (*.vtu)")
        if path:
            try:
                self.result.export_native(path if path.endswith(".vtu") else path+".vtu")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Ошибка экспорта", str(exc))

    def show_aerodynamics(self):
        if self.result is None or not self.result.diagnostics.get("aerodynamics"):
            return
        from .aero_gui import AeroDialog
        dialog = AeroDialog(self.result, self)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.aero_dialogs.append(dialog)
        dialog.destroyed.connect(lambda *_, dialog=dialog: self.aero_dialogs.remove(dialog) if dialog in self.aero_dialogs else None)
        dialog.show()

    def load(self):
        if self.process is not None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть расчёт", "", "FlowDesk (*.npz)")
        if path:
            try:
                self.playing = False
                self.play.setText("▶ Воспроизвести")
                self.accept_result(Result.load(path))
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Ошибка чтения", str(exc))

    def closeEvent(self, event):
        for dialog in list(self.aero_dialogs):
            dialog.close()
        for editor in self.mesh_editors:
            editor.close()
        self.timer.stop()
        self.stream_pool.shutdown(wait=True, cancel_futures=True)
        if self.process is not None:
            self.cancel_event.set()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        event.accept()


def main():
    mp.freeze_support()
    pg.setConfigOptions(antialias=False, background="#101c2b", foreground="#cbd5e1")
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
