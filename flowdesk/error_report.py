"""Local, copyable diagnostics without reading environment variables or secrets."""
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from PySide6 import QtCore, QtWidgets, QtGui


def error_payload(traceback_text, config=None, summary=None):
    versions = {}
    for package in ("flowdesk", "numpy", "scipy", "FiPy", "gmsh", "PySide6", "numba", "pyamg"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    details = dict(python=sys.version, platform=platform.platform(), packages=versions)
    from . import __version__
    details["source_version"] = __version__
    if config is not None:
        details["configuration"] = asdict(config)
        if config.mesh_path:
            try:
                import numpy as np
                with np.load(config.mesh_path, allow_pickle=False) as mesh:
                    details["mesh_statistics"] = json.loads(str(mesh["statistics"]))
                    details["mesh_settings"] = json.loads(str(mesh["settings"]))
            except Exception as exc:
                details["mesh_diagnostics"] = f"{type(exc).__name__}: {exc}"
    report = "FlowDesk error report\n"+json.dumps(details, ensure_ascii=False, indent=2, default=str)+"\n\n"+traceback_text
    log_path = None
    try:
        path = Path(tempfile.gettempdir())/"flowdesk"/f"last-error-{os.getpid()}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        log_path = str(path)
    except OSError:
        pass  # Diagnostics must never conceal the original failure.
    return dict(text=report, file=log_path,
                summary=summary or next((line for line in reversed(traceback_text.splitlines()) if line.strip()), "Ошибка расчёта"))


class ErrorDialog(QtWidgets.QDialog):
    def __init__(self, title, payload, parent=None):
        super().__init__(parent)
        if isinstance(payload, str):
            payload = dict(text=payload, summary=next((s for s in reversed(payload.splitlines()) if s.strip()), title))
        self.payload = payload
        self.setWindowTitle(title)
        self.resize(1000, 650)
        layout = QtWidgets.QVBoxLayout(self)
        summary_text = str(payload.get("summary", title))
        summary = QtWidgets.QLabel(summary_text[:1200]+("…" if len(summary_text) > 1200 else ""))
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)
        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.editor.setPlainText(payload["text"])
        layout.addWidget(self.editor, 1)
        self.location = QtWidgets.QLabel("Автоматическая копия: "+str(payload.get("file") or "не удалось сохранить; используйте кнопку ниже"))
        self.location.setWordWrap(True)
        self.location.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.location)
        buttons = QtWidgets.QHBoxLayout()
        self.copy_button = QtWidgets.QPushButton("Копировать всё")
        self.copy_button.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.editor.toPlainText()))
        self.save_button = QtWidgets.QPushButton("Сохранить отчёт…")
        self.save_button.clicked.connect(self.save)
        close = QtWidgets.QPushButton("Закрыть")
        close.clicked.connect(self.accept)
        for button in (self.copy_button, self.save_button, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)

    def save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить полную ошибку", "flowdesk-error.txt", "Text (*.txt)")
        if path:
            try:
                Path(path).write_text(self.editor.toPlainText(), encoding="utf-8")
                self.location.setText("Отчёт сохранён: "+path)
            except OSError as exc:
                self.location.setText("Не удалось сохранить: "+str(exc))
