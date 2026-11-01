"""Exercise rendering, playback, and the real spawned calculation process headlessly."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import time
import numpy as np
from PySide6 import QtCore, QtWidgets
from flowdesk.gui import Window

_app = None  # Keep QApplication alive until all QGraphicsItem wrappers are gone.


def test_desktop_calculate_play_save_load(tmp_path):
    global _app
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _app = app
    window = Window()
    window.show()
    window.mode.setCurrentIndex(1)
    assert window.scenario.count() == 6
    window.grid.setCurrentIndex(6)
    assert window.nx.value() == window.ny.value() == 1024
    assert "12 кадров" in window.memory.text()
    window.grid.setCurrentIndex(0)
    window.nx.setValue(80)
    window.ny.setValue(20)
    window.top.setCurrentIndex(1)
    window.duration.setValue(.2)
    window.scenario.setCurrentIndex(1)
    window.start()
    deadline = time.monotonic()+30
    while window.process is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    try:
        assert window.result is not None, window.info.text()
        assert window.process is None
        assert window.result.config.scenario == "step"
        assert (window.result.config.nx, window.result.config.ny) == (80, 20)
        assert window.result.config.top == "open"
        rect = window.image.boundingRect()
        mapped = window.image.mapRectToParent(rect)
        assert np.isclose(mapped.width(), .6)
        assert np.isclose(mapped.height(), .04)
        window.seek(4000)
        window.field.setCurrentIndex(1)
        window.toggle()
        before = window.position
        time.sleep(.05)
        window.tick()
        assert window.position != before
        window.toggle()
        path = tmp_path/"desktop.npz"
        window.result.save(path)
        from flowdesk.solver import Result
        window.accept_result(Result.load(path))
        assert window.top.currentIndex() == 1
        assert window.nx.value() == 80
        window.show_lines.setChecked(False)
        app.processEvents()
        assert not window.line.isVisible()
        assert window.run_button.isEnabled()
        assert window.grab().width() > 0
    finally:
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def test_steady_gui_reports_iteration_limit_and_animation():
    global _app
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _app = app
    window = Window()
    window.show()
    window.set_mesh(32, 12)
    window.iterations.setValue(10)
    assert window.mode.currentIndex() == 0
    assert not window.frames.isEnabled()
    window.start()
    deadline = time.monotonic()+60
    while window.process is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    try:
        assert window.result is not None
        assert window.result.config.mode == "steady"
        assert "НЕ СОШЛОСЬ" in window.info.text()
        assert "Анимация" in window.time_label.text()
        assert not window.result.diagnostics["converged"]
        np.testing.assert_array_equal(window.result.u[0], window.result.u[1])
    finally:
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
