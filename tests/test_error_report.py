import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import queue
from threading import Event
from PySide6 import QtCore, QtWidgets
from flowdesk.error_report import ErrorDialog
from flowdesk.gui import calculate
from flowdesk.solver import Config

_app = None


def test_failed_calculation_writes_complete_local_diagnostics():
    messages = queue.Queue()
    calculate(Config(scenario="invalid"), messages, Event())
    kind, payload = messages.get_nowait()
    assert kind == "error"
    assert payload["summary"].startswith("ValueError:")
    assert "Traceback" in payload["text"]
    assert '"python"' in payload["text"]
    assert '"packages"' in payload["text"]
    assert Path(payload["file"]).read_text(encoding="utf-8") == payload["text"]


def test_long_error_is_copyable_scrollable_and_saved(tmp_path, monkeypatch):
    global _app
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _app = app
    text = "\n".join(f"frame {i}" for i in range(2000))+"\nValueError: terminal failure"
    dialog = ErrorDialog("Calculation error", dict(text=text, summary="ValueError: terminal failure"))
    dialog.show()
    try:
        app.processEvents()
        assert dialog.editor.verticalScrollBar().maximum() > 0
        assert dialog.editor.toPlainText().endswith("ValueError: terminal failure")
        dialog.copy_button.click()
        assert QtWidgets.QApplication.clipboard().text() == text
        path = tmp_path/"report.txt"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **kw: (str(path), ""))
        dialog.save_button.click()
        assert path.read_text(encoding="utf-8") == text
    finally:
        dialog.close()
        dialog.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()
