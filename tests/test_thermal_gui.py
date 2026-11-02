import os
import time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtWidgets, QtCore
from flowdesk.gui import Window, PROFILE_INDEX
from flowdesk.solver import Config, Result, Solver

_app = None


def test_desktop_energy_checkbox_calculation_animation_and_reload(tmp_path):
    global _app
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _app = app
    w = Window()
    w.show()
    try:
        assert w.turbulence.isChecked() and not w.energy.isChecked()
        assert not w.field.model().item(2).isEnabled()
        w.energy.setChecked(True)
        w.turbulence.setChecked(False)
        assert w.temperature_inlet.isEnabled() and not w.turbulent_prandtl.isEnabled()
        assert not w.temperature_wall.isEnabled()
        w.thermal_wall.setCurrentIndex(1)
        assert w.temperature_wall.isEnabled()
        w.set_mesh(32, 12)
        w.duration.setValue(.03)
        w.frames.setValue(4)
        w.heat_source.setValue(1e8)
        w.start()
        deadline = time.monotonic()+45
        while w.process is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert w.result is not None, w.info.text()
        assert not w.result.config.turbulence and w.result.config.energy
        assert w.field.model().item(2).isEnabled()
        w.field.setCurrentIndex(2)
        w.seek(0)
        first = w.image.image.copy()
        w.seek(10000)
        assert np.linalg.norm(np.nan_to_num(w.image.image-first)) > .1
        assert 'Температура' in w.info.text()
        w.result.save(tmp_path/'desktop-energy.npz')
        w.accept_result(Result.load(tmp_path/'desktop-energy.npz'))
        assert w.energy.isChecked() and not w.turbulence.isChecked()
        assert w.heat_source.value() == 1e8
        cold = Solver(Config(nx=16, ny=8, duration=.001, frames=2)).run()
        w.accept_result(cold)
        assert w.field.currentIndex() == 0
        assert not w.field.model().item(2).isEnabled()
        w.scenario.setCurrentIndex(PROFILE_INDEX)
        assert w.heat_capacity.value() == 1005.
        assert w.conductivity.value() == .0257
    finally:
        w.close()
        w.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
