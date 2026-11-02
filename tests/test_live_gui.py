import os
import time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PySide6 import QtWidgets, QtCore
from flowdesk.gui import Window
from flowdesk.solver import Config, Solver

_app = None


def test_fields_update_while_worker_runs_and_cancel_restores_previous_result():
    global _app
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _app = app
    w = Window()
    w.show()
    baseline = Solver(Config(nx=32, ny=12, duration=.001, frames=2)).run()
    w.accept_result(baseline)
    w.show_lines.setChecked(False)
    w.mode.setCurrentIndex(0)
    w.coarse.setChecked(False)
    w.set_mesh(256, 64)
    w.iterations.setValue(1000)
    w.energy.setChecked(True)
    w.start()
    try:
        assert not w.play.isEnabled() and not w.slider.isEnabled()
        assert not w.save_button.isEnabled() and not w.load_button.isEnabled()
        observations = []
        last_iteration = -1
        deadline = time.monotonic()+45
        while w.process is not None and time.monotonic() < deadline:
            app.processEvents()
            if w.preview is not None and w.preview.iteration != last_iteration:
                assert w.result is baseline
                last_iteration = w.preview.iteration
                w.field.setCurrentIndex(1)
                np.testing.assert_allclose(w.image.image[w.preview.fluid], w.preview.pressure[w.preview.fluid])
                observations.append(w.preview.pressure.copy())
                assert 'ПРЕДВАРИТЕЛЬНОЕ' in w.caption.text()
                assert 'итерация' in w.time_label.text()
                if len(observations) >= 2:
                    w.field.setCurrentIndex(2)
                    assert w.preview.temperature is not None
                    assert np.isfinite(w.image.image[w.preview.fluid]).all()
                    w.cancel()
                    break
            time.sleep(.01)
        assert len(observations) >= 2, w.info.text()
        assert not np.array_equal(observations[0], observations[-1])
        deadline = time.monotonic()+20
        while w.process is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert w.process is None
        assert w.preview is None and w.preview_mailbox is None
        assert w.result is baseline
        assert 'отменён' in w.info.text()
        assert w.play.isEnabled() and w.slider.isEnabled()
        assert w.save_button.isEnabled() and w.load_button.isEnabled()
        assert not w.field.model().item(2).isEnabled()
    finally:
        w.close()
        w.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
