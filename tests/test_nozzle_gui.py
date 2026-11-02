import os
import time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtWidgets, QtCore
from flowdesk.gui import Window, NOZZLE_INDEX
from flowdesk.solver import Config, Solver, Result

_app = None


def test_nozzle_settings_real_worker_mach_density_and_loading(tmp_path):
    global _app
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _app = app
    w = Window()
    w.show()
    try:
        w.scenario.setCurrentIndex(NOZZLE_INDEX)
        assert w.nozzle_group.isVisible()
        assert w.energy.isChecked() and not w.energy.isEnabled()
        assert not w.turbulence.isChecked() and not w.turbulence.isEnabled()
        assert not w.speed.isEnabled() and not w.top.isEnabled()
        assert w.nx.isEnabled()
        w.nozzle_converging_length.setValue(.1)
        w.nozzle_throat_height.setValue(.014)
        assert w.length.value() == .22
        w.set_mesh(32, 8)
        w.frames.setValue(4)
        w.duration.setValue(.0005)
        w.iterations.setValue(45000)
        w.show_lines.setChecked(False)
        w.start()
        deadline = time.monotonic()+60
        while w.process is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert w.result is not None, w.info.text()
        assert w.result.config.is_compressible
        assert w.result.config.nozzle_throat_height == .014
        assert 'M выхода' in w.mesh_info.text()
        assert w.field.model().item(3).isEnabled() and w.field.model().item(4).isEnabled()
        w.field.setCurrentIndex(3)
        np.testing.assert_allclose(w.image.image[w.result.fluid], w.result.mach[-1, w.result.fluid])
        w.field.setCurrentIndex(4)
        assert (w.image.image[w.result.fluid] > 0.).all()
        w.result.save(tmp_path/'nozzle.npz')
        w.scenario.setCurrentIndex(0)
        w.accept_result(Result.load(tmp_path/'nozzle.npz'))
        assert w.iterations.value() == 45000
        assert w.nozzle_converging_length.value() == .1
        assert w.nozzle_throat_height.value() == .014
        assert not w.energy.isEnabled()
        # Leaving the forced gas model must restore a loaded result's own switches.
        liquid = Solver(Config(nx=16, ny=8, duration=.001, frames=2,
                              turbulence=False, energy=True)).run()
        w.accept_result(liquid)
        assert w.energy.isChecked() and not w.turbulence.isChecked()
        assert w.energy.isEnabled() and not w.field.model().item(3).isEnabled()
        assert w.field.currentIndex() == 0
        w.scenario.setCurrentIndex(NOZZLE_INDEX)
        w.nozzle_model.setCurrentIndex(1)
        assert w.energy.isEnabled() and w.turbulence.isEnabled() and w.speed.isEnabled()
    finally:
        w.close()
        w.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
