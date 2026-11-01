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
from .visualization import streamlines


def calculate(config, messages, cancel):
    try:
        started = time.perf_counter()
        result = Solver(config).run(
            lambda i, n, t: messages.put(("progress", (i, n, t))), cancel)
        result.diagnostics["wall_seconds"] = time.perf_counter()-started
        messages.put(("result", result))
    except InterruptedError:
        messages.put(("cancelled", None))
    except Exception:
        messages.put(("error", traceback.format_exc()))


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlowDesk · 2D RANS / k–ε")
        self.resize(1240, 780)
        self.result = None
        self.process = None
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
        left.addWidget(QtWidgets.QLabel("Лаборатория течений · вода · 2D"))
        self.scenario = QtWidgets.QComboBox()
        self.scenario.addItems(["Прямой канал", "Обратная ступенька"])
        self.grid = QtWidgets.QComboBox()
        self.grid.addItems(["Быстрая · 64 × 16", "Стандартная · 96 × 24", "Точная · 144 × 36"])
        self.grid.setCurrentIndex(1)
        self.speed = self.number(.1, 5, 1, " м/с")
        self.duration = self.number(.2, 10, 1.2, " с")
        self.length = self.number(.1, 3, .6, " м")
        self.height = self.number(.01, .2, .04, " м", 3)
        form = QtWidgets.QFormLayout()
        for label, widget in (("Сценарий", self.scenario), ("Сетка", self.grid),
                              ("Скорость входа", self.speed), ("Время расчёта", self.duration),
                              ("Длина", self.length), ("Высота выхода", self.height)):
            form.addRow(label, widget)
        left.addLayout(form)
        self.settings = [self.scenario, self.grid, self.speed, self.duration, self.length, self.height]
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
        note = QtWidgets.QLabel("Модель: несжимаемые RANS + k–ε.\nГладкие стенки, пристеночные функции.\n\nЦель ±10% относится к проверяемому перепаду давления. Для ступеньки точность пока не подтверждена.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9aaec3; font-size: 12px")
        left.addWidget(note)
        left.addStretch()
        save = QtWidgets.QPushButton("Сохранить результат…")
        save.clicked.connect(self.save)
        load = QtWidgets.QPushButton("Открыть результат…")
        load.clicked.connect(self.load)
        left.addWidget(save)
        left.addWidget(load)
        layout.addWidget(sidebar)
        right = QtWidgets.QVBoxLayout()
        toolbar = QtWidgets.QHBoxLayout()
        self.field = QtWidgets.QComboBox()
        self.field.addItems(["Скорость · м/с", "Давление · Па"])
        self.field.currentIndexChanged.connect(self.change_field)
        self.show_lines = QtWidgets.QCheckBox("Линии тока")
        self.show_lines.setChecked(True)
        self.show_lines.toggled.connect(self.render)
        equal_scale = QtWidgets.QCheckBox("Оси 1:1")
        equal_scale.setToolTip("По умолчанию оси масштабируются независимо для удобства просмотра")
        equal_scale.toggled.connect(lambda checked: self.plot.setAspectLocked(checked))
        self.rate = QtWidgets.QComboBox()
        self.rate.addItems(["0.25×", "0.5×", "1×", "2×"])
        self.rate.setCurrentIndex(2)
        toolbar.addWidget(self.field)
        toolbar.addWidget(self.show_lines)
        toolbar.addWidget(equal_scale)
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
        nx, ny = [(64, 16), (96, 24), (144, 36)][self.grid.currentIndex()]
        c = Config(scenario=["channel", "step"][self.scenario.currentIndex()], nx=nx, ny=ny,
                   speed=self.speed.value(), duration=self.duration.value(),
                   length=self.length.value(), height=self.height.value())
        self.playing = False
        self.play.setText("▶ Воспроизвести")
        self.messages = self.context.Queue()
        self.cancel_event = self.context.Event()
        self.process = self.context.Process(target=calculate, args=(c, self.messages, self.cancel_event))
        self.process.start()
        for widget in self.settings+[self.run_button]:
            widget.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.info.setText("Расчёт RANS / k–ε…\nИнтерфейс остаётся доступен.")

    def finish_worker(self):
        self.process.join(timeout=.1)
        self.process = None
        self.messages.close()
        for widget in self.settings+[self.run_button]:
            widget.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def cancel(self):
        if self.process is not None:
            self.cancel_event.set()
            self.cancel_button.setEnabled(False)

    def tick(self):
        now = time.perf_counter()
        elapsed, self.last_tick = now-self.last_tick, now
        if self.process is not None:
            try:
                while True:
                    kind, value = self.messages.get_nowait()
                    if kind == "progress":
                        i, n, t = value
                        self.progress.setValue(round(i/n*100))
                        self.info.setText(f"Кадр {i}/{n} · t = {t:.3f} с")
                    else:
                        self.finish_worker()
                        if kind == "result":
                            self.accept_result(value)
                        elif kind == "error":
                            self.info.setText("Ошибка расчёта")
                            QtWidgets.QMessageBox.critical(self, "Ошибка", value)
                        else:
                            self.info.setText("Расчёт отменён")
                        break
            except queue.Empty:
                # A crashed child cannot report through the queue.
                if self.process is not None and self.process.exitcode is not None:
                    code = self.process.exitcode
                    self.finish_worker()
                    self.info.setText(f"Процесс завершился без результата (код {code}).")
        if self.playing and self.result is not None:
            speed = [.25, .5, 1, 2][self.rate.currentIndex()]
            self.position = (self.position+elapsed*speed) % self.result.config.duration
            self.render()
        elif self.show_lines.isChecked() and self.stream_future is not None and self.stream_future[2].done():
            self.render()

    def accept_result(self, result):
        self.result = result
        self.position = 0.
        self.stream_key = None
        self.paths = []
        self.line.setData([], [])
        self.particles.setData([], [])
        self.play.setEnabled(True)
        d = result.diagnostics
        self.info.setText(f"Расчёт завершён · {d.get('wall_seconds', 0):.2f} с\n"
            f"Re = {d['reynolds']:.0f}\nΔp вход–выход = {d['pressure_drop_pa']:.2f} Па\n"
            f"Небаланс расхода: {d['mass_imbalance']:.2e}\n"
            f"Изменение u между последними кадрами: {d['final_frame_velocity_change']:.2e}\n"
            + ("Обратный поток на выходе: увеличьте длину канала.\n" if d['backflow'] else "")
            + "Сходимость и точность см. в отчёте проверки.")
        c = result.config
        self.scenario.setCurrentIndex(0 if c.scenario == "channel" else 1)
        for control, value in ((self.speed, c.speed), (self.duration, c.duration),
                               (self.length, c.length), (self.height, c.height)):
            control.setValue(value)
        sizes = [(64, 16), (96, 24), (144, 36)]
        if (c.nx, c.ny) in sizes:
            self.grid.setCurrentIndex(sizes.index((c.nx, c.ny)))
        self.image.setImage(np.zeros((c.ny, c.nx)), autoLevels=False)
        self.image.setRect(QtCore.QRectF(0, 0, c.length, c.height))
        self.plot.setRange(xRange=(0, c.length), yRange=(0, c.height), padding=.05)
        self.change_field()
        self.progress.setValue(100)

    def change_field(self, *_):
        if self.result is None:
            return
        r = self.result
        values = np.hypot(r.u, r.v) if self.field.currentIndex() == 0 else r.pressure
        values = values[:, r.fluid]
        low, high = float(values.min()), float(values.max())
        self.bar.setLevels((low, max(high, low+1e-6)))
        self.render()

    def render(self, *_):
        if self.result is None:
            return
        r, t = self.result, self.position
        index = int(np.clip(np.searchsorted(r.time, t, side="right")-1, 0, len(r.time)-2))
        alpha = np.clip((t-r.time[index])/(r.time[index+1]-r.time[index]), 0, 1)
        def blend(q):
            return q[index]*(1-alpha)+q[index+1]*alpha
        u, v = blend(r.u), blend(r.v)
        field = np.hypot(u, v) if self.field.currentIndex() == 0 else blend(r.pressure)
        self.image.setImage(np.where(r.fluid, field, np.nan), autoLevels=False)
        self.slider.setValue(round(t/r.config.duration*10000))
        self.time_label.setText(f"t = {t:.3f} с")
        self.line.setVisible(self.show_lines.isChecked())
        self.particles.setVisible(self.show_lines.isChecked())
        if self.show_lines.isChecked():
            key = int(t*8)  # integrate only eight frozen fields per simulated second
            if self.stream_future is not None and self.stream_future[2].done():
                owner, sampled_key, future = self.stream_future
                self.stream_future = None
                if owner == id(r):
                    self.paths = future.result()
                    self.stream_key = sampled_key
                    joined = np.concatenate([np.vstack([p, [np.nan, np.nan]]) for p in self.paths]) if self.paths else np.empty((0, 2))
                    self.line.setData(joined[:, 0], joined[:, 1])
            if self.stream_key != key and self.stream_future is None:
                # Integration never blocks the playback timer; one pending field at most.
                self.stream_future = (id(r), key, self.stream_pool.submit(
                    streamlines, u.copy(), v.copy(), r.fluid, r.config.length, r.config.height))
            # Markers illustrate flow direction; they are not physical tracers.
            dots = [p[int((t*.35*r.config.speed/(.45*min(r.config.length/r.config.nx,
                        r.config.height/r.config.ny))+i*31) % len(p))] for i, p in enumerate(self.paths)]
            if dots:
                dots = np.array(dots)
                self.particles.setData(dots[:, 0], dots[:, 1])
            else:
                self.particles.setData([], [])

    def toggle(self):
        self.playing = not self.playing
        self.play.setText("⏸ Пауза" if self.playing else "▶ Воспроизвести")

    def seek(self, value):
        if self.result is not None:
            self.position = value/10000*self.result.config.duration
            self.render()

    def save(self):
        if self.result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить расчёт", "flow.npz", "FlowDesk (*.npz)")
        if path:
            try:
                self.result.save(path if path.endswith(".npz") else path+".npz")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть расчёт", "", "FlowDesk (*.npz)")
        if path:
            try:
                self.playing = False
                self.play.setText("▶ Воспроизвести")
                self.accept_result(Result.load(path))
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Ошибка чтения", str(exc))

    def closeEvent(self, event):
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
