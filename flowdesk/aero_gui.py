"""Postprocessing of solved profile forces; iteration histories and angle polars."""
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from .airfoil import export_csv

class AeroDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("Профиль · Cl / Cd / Cl/Cd")
        self.resize(980, 820)
        layout = QtWidgets.QVBoxLayout(self)
        d, aero = result.diagnostics, result.diagnostics["aerodynamics"]
        last = aero["final"]
        ratio = "не определено" if last["cl_cd"] is None else f"{last['cl_cd']:.4f}"
        summary = QtWidgets.QLabel(f"α = {aero['angle_deg']:+.1f}° · Re(c) = {aero['reynolds_chord']:.0f} · "
            f"Cl = {last['cl']:.5f} · Cd = {last['cd']:.5f} · Cl/Cd = {ratio}\n"
            + ("Нестационарный расчёт: коэффициенты показаны во времени." if result.config.mode == "transient" else "Решение сошлось." if d["converged"] else "НЕ СОШЛОСЬ: коэффициенты пока не являются установившимся результатом."))
        summary.setWordWrap(True)
        layout.addWidget(summary)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 1)
        history = pg.GraphicsLayoutWidget()
        self.history_plots = []
        for i, (key, label, color) in enumerate((("cl", "Cl", "#5eead4"), ("cd", "Cd", "#60a5fa"), ("cl_cd", "Cl/Cd", "#fbbf24"))):
            plot = history.addPlot(row=i, col=0)
            plot.setLabel("left", label)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.setLabel("bottom", "Время, с" if result.config.mode == "transient" else "Итерация SIMPLE")
            plot.showGrid(x=True, y=True, alpha=.2)
            plot.plot([p["time" if result.config.mode == "transient" else "iteration"] for p in aero["history"]],
                      [np.nan if p[key] is None else p[key] for p in aero["history"]], pen=color)
            self.history_plots.append(plot)
        tabs.addTab(history, "История Cl, Cd и Cl/Cd")
        polar_widget = QtWidgets.QWidget()
        polar_layout = QtWidgets.QVBoxLayout(polar_widget)
        polar = aero.get("polar", [])
        self.polar_plots = []
        if len(polar) > 1:
            graphics = pg.GraphicsLayoutWidget()
            for i, (key, label, color) in enumerate((("cl", "Cl", "#5eead4"), ("cd", "Cd", "#60a5fa"), ("cl_cd", "Cl/Cd", "#fbbf24"))):
                plot = graphics.addPlot(row=i, col=0)
                plot.setLabel("left", label)
                plot.getAxis("left").enableAutoSIPrefix(False)
                plot.setLabel("bottom", "Угол атаки α", units="°")
                plot.showGrid(x=True, y=True, alpha=.2)
                plot.plot([p["angle"] for p in polar],
                    [np.nan if not p["converged"] or p[key] is None else p[key] for p in polar],
                    pen=color, symbol="o", symbolSize=7, symbolBrush=color)
                bad = [p for p in polar if not p["converged"] and p[key] is not None]
                if bad:
                    plot.plot([p["angle"] for p in bad], [p[key] for p in bad], pen=None,
                              symbol="x", symbolPen="#f87171", symbolSize=10)
                self.polar_plots.append(plot)
            polar_layout.addWidget(graphics, 1)
        else:
            note = QtWidgets.QLabel("Серия углов не рассчитана. Выберите стационарный режим, включите «Поляра по углу атаки» в параметрах профиля и запустите расчёт.")
            note.setWordWrap(True)
            polar_layout.addWidget(note)
        table = QtWidgets.QTableWidget(len(polar), 5)
        table.setHorizontalHeaderLabels(["α, °", "Cl", "Cd", "Cl/Cd", "Сходимость"])
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        table.setMaximumHeight(160)
        for row, point in enumerate(polar):
            values = [f"{point['angle']:+.1f}", f"{point['cl']:.5f}", f"{point['cd']:.5f}",
                "—" if point["cl_cd"] is None else f"{point['cl_cd']:.4f}", "Да" if point["converged"] else "Нет"]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, col, item)
        polar_layout.addWidget(table)
        tabs.addTab(polar_widget, "Поляра по углу атаки")
        note = QtWidgets.QLabel("Силы: давление + стеночное трение, только профиль, единичный размах. "
            "Крестики — несошедшиеся углы. При Cd ≤ 1e−8 отношение не определяется. Точность не проверена по аэродинамической трубе.")
        note.setWordWrap(True)
        layout.addWidget(note)
        controls = QtWidgets.QHBoxLayout()
        for text, polar_flag in (("История CSV…", False), ("Поляра CSV…", True)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, polar_flag=polar_flag: self.save_csv(polar_flag))
            controls.addWidget(button)
        close = QtWidgets.QPushButton("Закрыть")
        close.clicked.connect(self.close)
        controls.addWidget(close)
        layout.addLayout(controls)

    def save_csv(self, polar):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Коэффициенты профиля", "polar.csv" if polar else "coefficients.csv", "CSV (*.csv)")
        if path:
            try:
                export_csv(self.result, path, polar)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Ошибка экспорта", str(exc))
