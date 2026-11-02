import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace
from PySide6 import QtCore, QtWidgets
from flowdesk.aero_gui import AeroDialog

_app = None


def test_both_graphs_and_unconverged_polar_markers():
    global _app
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _app = app
    history = [dict(iteration=1, cl=0., cd=0., cl_cd=None),
               dict(iteration=2, cl=.4, cd=.1, cl_cd=4.)]
    polar = [dict(angle=-4., cl=-.2, cd=.1, cl_cd=-2., converged=True),
             dict(angle=0., cl=.1, cd=.2, cl_cd=.5, converged=False),
             dict(angle=4., cl=.4, cd=.1, cl_cd=4., converged=True)]
    result = SimpleNamespace(config=SimpleNamespace(mode="steady"), diagnostics=dict(converged=True, aerodynamics=dict(
        angle_deg=4., reynolds_chord=100000., final=history[-1], history=history, polar=polar)))
    dialog = AeroDialog(result)
    dialog.show()
    try:
        app.processEvents()
        assert len(dialog.history_plots) == 3
        assert len(dialog.polar_plots) == 3
        assert dialog.findChild(QtWidgets.QTabWidget).count() == 2
        assert dialog.findChild(QtWidgets.QTableWidget).item(1, 4).text() == "Нет"
        assert len(dialog.polar_plots[0].listDataItems()) == 2
        assert dialog.grab().width() > 0
    finally:
        dialog.close()
        dialog.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def test_ready_profile_desktop_path_and_graph_button():
    global _app
    from flowdesk.gui import Window, PROFILE_INDEX
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _app = app
    window = Window()
    window.scenario.setCurrentIndex(PROFILE_INDEX)
    assert window.speed.value() == 20.
    assert window.ny.value() >= 128
    assert window.mode.currentIndex() == 1
    assert window.mode.isEnabled()
    assert not window.profile_polar.isEnabled()
    window.mode.setCurrentIndex(0)
    window.profile_cells.setValue(400)
    window.profile_polar.setChecked(False)
    window.iterations.setValue(10)
    window.start()
    deadline = time.monotonic()+90
    while window.process is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    try:
        assert window.result is not None, window.info.text()
        assert not window.result.diagnostics["converged"]
        assert window.aero_button.isEnabled()
        assert window.result.config.rho == 1.225
        window.show_aerodynamics()
        app.processEvents()
        assert len(window.aero_dialogs) == 1
        assert len(window.aero_dialogs[0].history_plots) == 3
    finally:
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
