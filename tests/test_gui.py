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
    window.grid.setCurrentIndex(0)
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
