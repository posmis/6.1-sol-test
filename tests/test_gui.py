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
    assert window.scenario.count() == 7
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


def test_mesh_editor_to_desktop_calculation(tmp_path, monkeypatch):
    global _app
    from flowdesk.mesh_gui import MeshEditor
    from flowdesk.cad import example_geometry
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _app = app
    window = Window()
    window.open_mesh_editor()
    editor = window.mesh_editors[-1]
    g = example_geometry()
    g.loops[0][1][0] = g.loops[0][2][0] = .12
    editor.geometry = g
    editor.update_geometry()
    editor.add_zone(values=(.04, .01, .08, .03, .004))
    editor.zones[0].setPos((.045, .01))
    editor.roi_changed(editor.zones[0])
    assert editor.zone_values()[0][0] == .045
    editor.zone_table.selectRow(0)
    editor.remove_zone()
    editor.kind.setCurrentIndex(1)
    editor.method.setCurrentIndex(1)
    editor.target.setValue(100)
    editor.show()
    editor.build()
    assert editor.cancel_button.isEnabled()
    deadline = time.monotonic()+60
    while editor.process is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    try:
        assert editor.project is not None, editor.info.text()
        path = tmp_path/"desktop.flowmesh"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **kw: (str(path), ""))
        editor.use()
        assert path.exists()
        assert window.scenario.currentIndex() == 6
        assert window.mode.currentIndex() == 0
        assert not window.nx.isEnabled()
        window.iterations.setValue(250)
        window.start()
        deadline = time.monotonic()+90
        while window.process is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert window.result is not None, window.info.text()
        assert window.result.diagnostics["converged"]
        assert window.result.native is not None
        assert len(window.result.native["u"]) == len(editor.project.cells)
        assert window.native_export.isEnabled()
        assert window.custom_lines["wall"].isVisible()
        editor.build()
        assert editor.process is not None
        window.close()
        assert editor.process is None
    finally:
        for widget in (editor, window):
            widget.close()
            widget.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
