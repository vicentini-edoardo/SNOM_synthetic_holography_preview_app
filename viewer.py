from __future__ import annotations

import os
import sys

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.widgets import RangeSlider
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from .processing import (
        AMPLITUDE_ONLY_STAGES,
        PASSAGE_TO_KEYS,
        PROCESSING_MODES,
        ProcessingError,
        STAGE_LABELS,
        export_all_views,
        get_view_image,
        load_passage,
    )
else:
    from processing import (
        AMPLITUDE_ONLY_STAGES,
        PASSAGE_TO_KEYS,
        PROCESSING_MODES,
        ProcessingError,
        STAGE_LABELS,
        export_all_views,
        get_view_image,
        load_passage,
    )


VIEW_CMAPS = {
    ("raw", "amplitude"): "hot",
    ("raw", "phase"): "bwr",
    ("processed", "amplitude"): "hot",
    ("processed", "phase"): "bwr",
    ("mag_signal_ft", "amplitude"): "hot",
    ("filtered_shift", "amplitude"): "hot",
}

COLORS = {
    "bg": "#16181a",
    "panel": "#1a1e22",
    "panel_alt": "#22282d",
    "line": "#394147",
    "text": "#f1eadf",
    "muted": "#b6aca0",
    "accent": "#f1c981",
    "accent_strong": "#d8a85a",
    "danger": "#d17c67",
}

APP_QSS = f"""
QMainWindow {{
    background: {COLORS["bg"]};
}}
QWidget {{
    color: {COLORS["text"]};
    font-family: "Helvetica Neue", "Helvetica", "Arial", sans-serif;
    font-size: 13px;
}}
QFrame#panel, QGroupBox#panel {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {COLORS["panel"]}, stop:1 {COLORS["panel_alt"]});
    border: 1px solid {COLORS["line"]};
    border-radius: 14px;
}}
QGroupBox#panel {{
    margin-top: 14px;
    padding-top: 14px;
}}
QGroupBox#panel::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {COLORS["accent"]};
    font-weight: 600;
}}
QLineEdit, QComboBox, QPlainTextEdit {{
    background: rgba(18, 20, 22, 0.94);
    border: 1px solid {COLORS["line"]};
    border-radius: 8px;
    padding: 8px 10px;
    selection-background-color: rgba(241, 201, 129, 0.35);
    selection-color: {COLORS["text"]};
}}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border-color: {COLORS["accent"]};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 34px;
    border-left: 1px solid {COLORS["line"]};
    background: rgba(34, 40, 45, 0.96);
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QComboBox::down-arrow {{
    image: none;
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid {COLORS["accent"]};
}}
QComboBox QAbstractItemView {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["line"]};
    selection-background-color: rgba(241, 201, 129, 0.22);
}}
QPushButton {{
    background: rgba(32, 37, 41, 0.95);
    border: 1px solid {COLORS["line"]};
    border-radius: 10px;
    padding: 8px 16px;
    color: {COLORS["text"]};
}}
QPushButton:hover {{
    border-color: {COLORS["accent_strong"]};
    background: rgba(40, 46, 51, 0.98);
}}
QPushButton:disabled {{
    color: rgba(241, 234, 223, 0.45);
    background: rgba(24, 27, 30, 0.85);
    border-color: rgba(57, 65, 71, 0.8);
}}
QPushButton#secondary {{
    background: rgba(26, 30, 34, 0.96);
    border-color: rgba(57, 65, 71, 0.95);
}}
QPushButton#secondary:hover {{
    border-color: {COLORS["accent"]};
}}
QPushButton#accent {{
    border-color: rgba(241, 201, 129, 0.48);
    color: {COLORS["accent"]};
    background: rgba(28, 30, 32, 0.98);
}}
QPushButton#accent:hover {{
    background: rgba(37, 32, 25, 0.98);
    border-color: {COLORS["accent"]};
}}
QPushButton#primary {{
    background: rgba(241, 201, 129, 0.16);
    border-color: {COLORS["accent"]};
    color: #fff4df;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: rgba(241, 201, 129, 0.22);
    border-color: #f6d797;
}}
QRadioButton, QCheckBox {{
    spacing: 8px;
}}
QRadioButton::indicator, QCheckBox::indicator {{
    width: 16px;
    height: 16px;
}}
QRadioButton::indicator::unchecked, QCheckBox::indicator::unchecked {{
    border: 1px solid {COLORS["line"]};
    background: rgba(18, 20, 22, 0.95);
}}
QRadioButton::indicator::checked, QCheckBox::indicator::checked {{
    border: 1px solid {COLORS["accent"]};
    background: {COLORS["accent"]};
}}
QLabel#sectionTitle {{
    color: {COLORS["accent"]};
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.6px;
}}
QLabel#metricLabel {{
    color: {COLORS["muted"]};
    font-size: 11px;
}}
QLabel#metricValue {{
    color: #fff4df;
    background: rgba(241, 201, 129, 0.08);
    border: 1px solid rgba(241, 201, 129, 0.24);
    border-radius: 6px;
    padding: 3px 8px;
    font-family: "Menlo", "Monaco", monospace;
    font-size: 12px;
    min-height: 18px;
}}
QLineEdit#tuningInput {{
    background: rgba(18, 20, 22, 0.98);
    border: 1px solid rgba(241, 201, 129, 0.28);
    color: #fff4df;
    font-family: "Menlo", "Monaco", monospace;
    font-size: 12px;
    padding: 7px 10px;
}}
QPlainTextEdit {{
    color: {COLORS["text"]};
}}
"""


class HologramViewerWindow(QMainWindow):
    def __init__(self, initial_folder: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Hologram Viewer")
        self.resize(1500, 980)
        self.loaded = None
        self.loaded_by_passage: dict[str, object] = {}
        self.override_by_passage: dict[str, dict[str, dict[str, int]]] = {}
        self.loaded_folder_path: str | None = None
        self.processing_mode = "two_sideband"
        self._sliders: list[tuple[RangeSlider, int]] = []
        self._apply_theme()
        self._build_ui()
        self._sync_tuning_settings()

        if initial_folder:
            self.folder_edit.setText(initial_folder)
            self.load_current_folder()

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(APP_QSS)
            font = QFont("Helvetica", 11)
            app.setFont(font)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        attribution_label = QLabel(
            "Developed by Edoardo Vicentini. Citation: Vicentini, E. (2026). "
            "SNOM synthetic holography preview app [Computer software]. GitHub. "
            "https://github.com/vicentini-edoardo/SNOM_synthetic_holography_preview_app"
        )
        attribution_label.setObjectName("metricLabel")
        attribution_label.setWordWrap(True)
        root_layout.addWidget(attribution_label, 0)

        shell_row = QWidget()
        shell_layout = QHBoxLayout(shell_row)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(12)

        self.command_rail = QWidget()
        command_rail_layout = QVBoxLayout(self.command_rail)
        command_rail_layout.setContentsMargins(0, 0, 0, 0)
        command_rail_layout.setSpacing(12)
        self.dataset_panel = self._build_dataset_panel()
        self.acquisition_panel = self._build_acquisition_panel()
        self.tuning_panel = self._build_tuning_panel()
        command_rail_layout.addWidget(self.dataset_panel)
        command_rail_layout.addWidget(self.acquisition_panel)
        command_rail_layout.addWidget(self.tuning_panel)
        command_rail_layout.addStretch(1)
        self.command_rail.setMinimumWidth(320)
        self.command_rail.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        shell_layout.addWidget(self.command_rail, 0)

        self.viewer_workspace = QWidget()
        content_layout = QVBoxLayout(self.viewer_workspace)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._build_plot_panel(), 1)
        shell_layout.addWidget(self.viewer_workspace, 1)
        root_layout.addWidget(shell_row, 1)

        self.telemetry_panel = self._build_log_panel()
        root_layout.addWidget(self.telemetry_panel, 0)

    def _panel(self, title: str | None = None) -> QFrame | QGroupBox:
        if title:
            panel = QGroupBox(title)
        else:
            panel = QFrame()
        panel.setObjectName("panel")
        return panel

    def _build_dataset_panel(self) -> QWidget:
        panel = self._panel("Dataset")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(10)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select a hologram folder")
        layout.addWidget(self.folder_edit)

        button_row = QHBoxLayout()
        self.browse_button = QPushButton("Browse")
        self.browse_button.setObjectName("secondary")
        self.browse_button.clicked.connect(self.browse_folder)
        self.load_button = QPushButton("Load")
        self.load_button.setObjectName("primary")
        self.load_button.clicked.connect(self.load_current_folder)
        self.export_button = QPushButton("Export All")
        self.export_button.setObjectName("accent")
        self.export_button.clicked.connect(self.export_current_folder)

        button_row.addWidget(self.browse_button)
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.export_button)
        layout.addLayout(button_row)
        return panel

    def _build_acquisition_panel(self) -> QWidget:
        panel = self._panel("Acquisition")
        layout = QGridLayout(panel)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

        passage_title = QLabel("Passage")
        passage_title.setObjectName("sectionTitle")
        layout.addWidget(passage_title, 0, 0)
        passage_row = QHBoxLayout()
        self.forward_radio = QRadioButton(PASSAGE_TO_KEYS["forward"]["label"])
        self.reverse_radio = QRadioButton(PASSAGE_TO_KEYS["reverse"]["label"])
        self.forward_radio.setChecked(True)
        self.passage_group = QButtonGroup(self)
        self.passage_group.addButton(self.forward_radio)
        self.passage_group.addButton(self.reverse_radio)
        self.forward_radio.toggled.connect(self._on_passage_change)
        self.reverse_radio.toggled.connect(self._on_passage_change)
        passage_row.addWidget(self.forward_radio)
        passage_row.addWidget(self.reverse_radio)
        passage_row.addStretch(1)
        layout.addLayout(passage_row, 0, 1)

        harmonic_title = QLabel("Harmonic")
        harmonic_title.setObjectName("sectionTitle")
        layout.addWidget(harmonic_title, 0, 2)
        self.harmonic_combo = QComboBox()
        self.harmonic_combo.addItems([str(i) for i in range(6)])
        self.harmonic_combo.setCurrentText("2")
        self.harmonic_combo.setMinimumWidth(150)
        self.harmonic_combo.currentIndexChanged.connect(self._on_harmonic_change)
        layout.addWidget(self.harmonic_combo, 0, 3)

        view_title = QLabel("Stage")
        view_title.setObjectName("sectionTitle")
        layout.addWidget(view_title, 1, 0)
        self.view_combo = QComboBox()
        for key in STAGE_LABELS:
            self.view_combo.addItem(key)
        self.view_combo.setCurrentText("processed")
        self.view_combo.setMinimumWidth(180)
        self.view_combo.currentIndexChanged.connect(self.refresh_plot)
        layout.addWidget(self.view_combo, 1, 1)
        layout.addWidget(QLabel("Mode"), 1, 2)
        self.processing_mode_combo = QComboBox()
        for mode_key, mode_label in PROCESSING_MODES.items():
            self.processing_mode_combo.addItem(mode_label, mode_key)
        self.processing_mode_combo.setCurrentIndex(self.processing_mode_combo.findData(self.processing_mode))
        self.processing_mode_combo.currentIndexChanged.connect(self._on_processing_mode_change)
        layout.addWidget(self.processing_mode_combo, 1, 3)
        return panel

    def _build_tuning_panel(self) -> QWidget:
        panel = self._panel("Tuning")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(12)

        header_row = QHBoxLayout()
        self.tuning_toggle_button = QPushButton("Show Manual Tuning")
        self.tuning_toggle_button.setObjectName("secondary")
        self.tuning_toggle_button.setCheckable(True)
        self.tuning_toggle_button.toggled.connect(self._update_tuning_panel)
        header_row.addWidget(self.tuning_toggle_button)
        header_row.addStretch(1)
        layout.addLayout(header_row)

        self.tuning_content = QWidget()
        tuning_layout = QGridLayout(self.tuning_content)
        tuning_layout.setContentsMargins(0, 0, 0, 0)
        tuning_layout.setHorizontalSpacing(10)
        tuning_layout.setVerticalSpacing(10)

        self.auto_shift_value = self._metric_row(tuning_layout, 0, "Detected Shift (px)")
        self.auto_width_value = self._metric_row(tuning_layout, 1, "Detected Width (px)")
        self.zero_order_width_value = self._metric_row(tuning_layout, 2, "0th Order Width (px)")
        self.first_order_width_value = self._metric_row(tuning_layout, 3, "1st Order Width (px)")
        self.rotation_angle_value = self._metric_row(tuning_layout, 4, "Rotation Angle (deg)")

        tuning_layout.addWidget(QLabel("Current Shift (px)"), 5, 0)
        self.shift_edit = QLineEdit()
        self.shift_edit.setObjectName("tuningInput")
        self.shift_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.shift_edit.textChanged.connect(self._sync_width_slider_bounds)
        tuning_layout.addWidget(self.shift_edit, 5, 1)
        tuning_layout.addWidget(QLabel("Current Width (px)"), 6, 0)
        width_row = QHBoxLayout()
        width_row.setSpacing(10)
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider.setRange(0, 0)
        self.width_slider.valueChanged.connect(self._on_width_slider_change)
        width_row.addWidget(self.width_slider, 1)
        self.width_value_label = QLabel("0")
        self.width_value_label.setObjectName("metricValue")
        self.width_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.width_value_label.setMinimumWidth(64)
        width_row.addWidget(self.width_value_label, 0)
        tuning_layout.addLayout(width_row, 6, 1)
        tuning_layout.addWidget(QLabel("Filter Alpha"), 7, 0)
        self.alpha_edit = QLineEdit()
        self.alpha_edit.setObjectName("tuningInput")
        self.alpha_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.alpha_edit.setText("0.3")
        tuning_layout.addWidget(self.alpha_edit, 7, 1)
        tuning_layout.addWidget(QLabel("Padding Factor"), 8, 0)
        self.pad_fact_edit = QLineEdit()
        self.pad_fact_edit.setObjectName("tuningInput")
        self.pad_fact_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.pad_fact_edit.setText("4")
        self.pad_fact_edit.textChanged.connect(self._sync_width_slider_bounds)
        tuning_layout.addWidget(self.pad_fact_edit, 8, 1)

        button_row = QHBoxLayout()
        self.apply_tuning_button = QPushButton("Apply")
        self.apply_tuning_button.setObjectName("primary")
        self.apply_tuning_button.clicked.connect(self.apply_tuning_settings)
        self.reset_tuning_button = QPushButton("Reset to Auto")
        self.reset_tuning_button.setObjectName("secondary")
        self.reset_tuning_button.clicked.connect(self.reset_tuning_settings)
        button_row.addWidget(self.apply_tuning_button)
        button_row.addWidget(self.reset_tuning_button)
        tuning_layout.addLayout(button_row, 9, 0, 1, 2)

        layout.addWidget(self.tuning_content)
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self._update_tuning_panel(False)
        return panel

    def _metric_row(self, layout: QGridLayout, row: int, label_text: str) -> QLabel:
        label = QLabel(label_text)
        label.setObjectName("metricLabel")
        layout.addWidget(label, row, 0)
        value = QLabel("-")
        value.setObjectName("metricValue")
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(value, row, 1)
        return value

    def _build_plot_panel(self) -> QWidget:
        panel = self._panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.figure = Figure(figsize=(11, 7), dpi=100)
        self.figure.patch.set_facecolor(COLORS["bg"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, panel)
        self.toolbar.setIconSize(self.toolbar.iconSize())
        self.toolbar.setStyleSheet(
            f"QToolBar {{ background: {COLORS['panel']}; border: 1px solid {COLORS['line']}; border-radius: 10px; spacing: 6px; }}"
            f"QToolButton {{ background: rgba(32,37,41,0.94); border: 1px solid {COLORS['line']}; border-radius: 8px; padding: 6px; color: {COLORS['text']}; }}"
            f"QToolButton:hover {{ border-color: {COLORS['accent_strong']}; }}"
            f"QToolButton:checked, QToolButton:pressed {{ background: rgba(241,201,129,0.14); border-color: {COLORS['accent']}; }}"
        )
        self._recolor_toolbar_icons()
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.toolbar, 0)
        return panel

    def _tint_icon(self, icon: QIcon, color: str) -> QIcon:
        size = self.toolbar.iconSize()
        tinted_icon = QIcon()
        color_value = QColor(color)
        state_pairs = (
            (QIcon.Mode.Normal, QIcon.State.Off),
            (QIcon.Mode.Active, QIcon.State.Off),
            (QIcon.Mode.Selected, QIcon.State.Off),
            (QIcon.Mode.Disabled, QIcon.State.Off),
        )
        for mode, state in state_pairs:
            source = icon.pixmap(size, mode, state)
            if source.isNull():
                continue
            tinted = QPixmap(source.size())
            tinted.fill(Qt.GlobalColor.transparent)
            painter = QPainter(tinted)
            painter.drawPixmap(0, 0, source)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(tinted.rect(), color_value)
            painter.end()
            tinted_icon.addPixmap(tinted, mode, state)
        return tinted_icon if not tinted_icon.isNull() else icon

    def _recolor_toolbar_icons(self) -> None:
        accent_map = {
            "Home": COLORS["accent"],
            "Back": COLORS["text"],
            "Forward": COLORS["text"],
            "Pan": COLORS["accent_strong"],
            "Zoom": COLORS["accent_strong"],
            "Subplots": COLORS["muted"],
            "Customize": COLORS["muted"],
            "Save": COLORS["accent"],
        }
        for action in self.toolbar.actions():
            icon = action.icon()
            if icon.isNull():
                continue
            accent = accent_map.get(action.text(), COLORS["text"])
            action.setIcon(self._tint_icon(icon, accent))

    def _build_log_panel(self) -> QWidget:
        panel = self._panel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        title = QLabel("Log Output")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.status_label = QLabel("Select a folder and click Load.")
        layout.addWidget(self.status_label)
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(150)
        layout.addWidget(self.log_widget)
        return panel

    def log(self, message: str) -> None:
        print(message)
        self.status_label.setText(message)
        self.log_widget.appendPlainText(message)

    def browse_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select SNOM image folder")
        if selected:
            self.folder_edit.setText(selected)
            self.load_current_folder()

    def _current_passage(self) -> str:
        return "forward" if self.forward_radio.isChecked() else "reverse"

    def _current_harmonic(self) -> int:
        return int(self.harmonic_combo.currentText())

    def _current_stage(self) -> str:
        return self.view_combo.currentText()

    def _current_processing_mode(self) -> str:
        return self.processing_mode

    def _current_override_bucket(self) -> dict[str, int]:
        passage = self._current_passage()
        mode = self._current_processing_mode()
        return self.override_by_passage.setdefault(passage, {}).setdefault(mode, {})

    def _clear_loaded_state(self) -> None:
        self.loaded = None
        self.loaded_by_passage = {}
        self.override_by_passage = {}
        self.loaded_folder_path = None
        self._sync_tuning_settings()

    def _set_tuning_controls_enabled(self, enabled: bool) -> None:
        self.shift_edit.setEnabled(enabled)
        self.width_slider.setEnabled(enabled)
        self.alpha_edit.setEnabled(enabled)
        self.pad_fact_edit.setEnabled(enabled)
        self.apply_tuning_button.setEnabled(enabled)
        self.reset_tuning_button.setEnabled(enabled)

    def _update_tuning_panel(self, checked: bool | None = None) -> None:
        if checked is None:
            checked = self.tuning_toggle_button.isChecked()
        self.tuning_toggle_button.setText("Hide Manual Tuning" if checked else "Show Manual Tuning")
        self.tuning_content.setVisible(bool(checked))

    def _sync_tuning_settings(self) -> None:
        if self.loaded is None:
            for label in (
                self.auto_shift_value,
                self.auto_width_value,
                self.zero_order_width_value,
                self.first_order_width_value,
                self.rotation_angle_value,
            ):
                label.setText("-")
            self.shift_edit.setText("")
            self.width_slider.setRange(0, 0)
            self.width_slider.setValue(0)
            self.width_value_label.setText("0")
            self.alpha_edit.setText("0.3")
            self.pad_fact_edit.setText("4")
            self._set_tuning_controls_enabled(False)
            self.tuning_toggle_button.setChecked(False)
            return

        settings = self.loaded.processing_settings
        self.auto_shift_value.setText(str(settings["auto_shift_y"]))
        self.auto_width_value.setText(str(settings["auto_filter_width_y"]))
        self.zero_order_width_value.setText(f"{settings['zero_order_width_y']:.2f}")
        self.first_order_width_value.setText(f"{settings['first_order_width_y']:.2f}")
        self.shift_edit.setText(str(settings["current_shift_y"]))
        self.alpha_edit.setText(str(settings.get("alpha", 0.3)))
        self.pad_fact_edit.setText(str(settings.get("pad_fact", 1)))
        self._set_manual_width_value(int(settings["current_filter_width_y"]))
        harmonic_index = self._current_harmonic()
        diagnostics = settings.get("diagnostics", {})
        rotation_by_harmonic = diagnostics.get(
            "rotation_angle_deg_by_harmonic",
            settings.get("rotation_angle_deg_by_harmonic", []),
        )
        if settings.get("processing_mode") == "two_sideband" and harmonic_index < len(rotation_by_harmonic):
            rotation_angle = rotation_by_harmonic[harmonic_index]
            self.rotation_angle_value.setText("-" if not np.isfinite(rotation_angle) else f"{rotation_angle:.2f}")
        else:
            self.rotation_angle_value.setText("-")
        self._set_tuning_controls_enabled(True)

    def _on_width_slider_change(self, value: int) -> None:
        self.width_value_label.setText(str(int(value)))

    def _current_width_limit(self) -> int:
        if self.loaded is None:
            return 0

        settings = self.loaded.processing_settings
        fft_center_row = int(settings.get("fft_center_row", 0))
        loaded_pad_fact = int(settings.get("pad_fact", 4))
        if fft_center_row <= 0 or loaded_pad_fact <= 0:
            return 0

        try:
            current_pad_fact = int(self.pad_fact_edit.text().strip())
        except ValueError:
            current_pad_fact = loaded_pad_fact
        if current_pad_fact <= 0:
            current_pad_fact = loaded_pad_fact

        scale_ratio = current_pad_fact / loaded_pad_fact
        adjusted_fft_center_row = int(round(fft_center_row * scale_ratio))
        adjusted_fft_height = max(0, 2 * adjusted_fft_center_row)
        return adjusted_fft_height

    def _set_manual_width_value(self, width: int) -> None:
        slider_max = self._current_width_limit()
        clamped_width = max(0, min(int(round(width)), slider_max))
        self.width_slider.blockSignals(True)
        self.width_slider.setRange(0, slider_max)
        self.width_slider.setValue(clamped_width)
        self.width_slider.blockSignals(False)
        self.width_value_label.setText(str(clamped_width))

    def _sync_width_slider_bounds(self, *_args) -> None:
        if self.loaded is None:
            self.width_slider.setRange(0, 0)
            self.width_slider.setValue(0)
            self.width_value_label.setText("0")
            return
        self._set_manual_width_value(self.width_slider.value())

    def _on_passage_change(self) -> None:
        folder = self.folder_edit.text().strip()
        if folder and os.path.isdir(folder):
            self.load_current_folder()
        else:
            self._sync_tuning_settings()

    def _on_harmonic_change(self) -> None:
        self.refresh_plot()
        self._sync_tuning_settings()

    def _on_processing_mode_change(self) -> None:
        mode = self.processing_mode_combo.currentData()
        if not isinstance(mode, str) or mode == self.processing_mode:
            return
        self.processing_mode = mode
        self.loaded = None
        self.loaded_by_passage = {}
        folder = self.folder_edit.text().strip()
        if folder and os.path.isdir(folder):
            self.load_current_folder(force_reload=True)
        else:
            self._sync_tuning_settings()

    def load_current_folder(self, force_reload: bool = False) -> None:
        folder = self.folder_edit.text().strip()
        if not folder:
            QMessageBox.critical(self, "Missing folder", "Select a folder before loading.")
            return
        if not os.path.isdir(folder):
            QMessageBox.critical(self, "Invalid folder", f"Folder does not exist:\n{folder}")
            return

        folder = os.path.abspath(folder)
        if self.loaded_folder_path != folder:
            self._clear_loaded_state()
            self.loaded_folder_path = folder

        passage = self._current_passage()
        if not force_reload and passage in self.loaded_by_passage:
            self.loaded = self.loaded_by_passage[passage]
            self.log(
                f"Loaded cached {PASSAGE_TO_KEYS[self.loaded.passage]['label']} from {self.loaded.folder_path} "
                f"with cache {self.loaded.cache_path} in {PROCESSING_MODES[self.processing_mode]} mode."
            )
            self._sync_tuning_settings()
            self.refresh_plot()
            return

        overrides = self.override_by_passage.get(passage, {}).get(self.processing_mode, {})
        try:
            self.loaded = load_passage(
                folder,
                passage,
                carrier_row_override=overrides.get("center_row"),
                filter_width_override=overrides.get("filter_width_y"),
                pad_fact=overrides.get("pad_fact", 4),
                alpha=overrides.get("alpha", 0.3),
                processing_mode=self.processing_mode,
            )
        except ProcessingError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self.log(f"Load failed: {exc}")
            return

        self.loaded_by_passage[passage] = self.loaded
        self.log(
            f"Loaded {PASSAGE_TO_KEYS[self.loaded.passage]['label']} from {self.loaded.folder_path} "
            f"with cache {self.loaded.cache_path} in {PROCESSING_MODES[self.processing_mode]} mode."
        )
        self._sync_tuning_settings()
        self.refresh_plot()

    def apply_tuning_settings(self) -> None:
        if self.loaded is None:
            return
        try:
            shift_y = int(self.shift_edit.text().strip())
            filter_width_y = int(self.width_value_label.text().strip())
            alpha = float(self.alpha_edit.text().strip())
            pad_fact = int(self.pad_fact_edit.text().strip())
        except ValueError:
            QMessageBox.critical(
                self,
                "Tuning settings",
                "Shift, width, and padding factor must be integers, and alpha must be numeric.",
            )
            self.log("Tuning settings error: shift, width, and padding factor must be integers, and alpha must be numeric.")
            return
        if not 0.0 <= alpha <= 1.0:
            QMessageBox.critical(self, "Tuning settings", "Filter alpha must be between 0 and 1.")
            self.log("Tuning settings error: filter alpha must be between 0 and 1.")
            return
        if not 1 <= pad_fact <= 32:
            QMessageBox.critical(self, "Tuning settings", "Padding factor must be an integer between 1 and 32.")
            self.log("Tuning settings error: padding factor must be an integer between 1 and 32.")
            return

        fft_center_row = self.loaded.processing_settings["fft_center_row"]
        loaded_pad_fact = int(self.loaded.processing_settings.get("pad_fact", 4))
        if loaded_pad_fact <= 0:
            loaded_pad_fact = 4
        scale_ratio = pad_fact / loaded_pad_fact
        adjusted_fft_center_row = int(round(fft_center_row * scale_ratio))
        adjusted_shift_y = int(round(shift_y * scale_ratio))
        adjusted_filter_width_y = int(round(filter_width_y * scale_ratio))
        overrides = self._current_override_bucket()
        overrides.clear()
        overrides.update({
            "center_row": adjusted_fft_center_row - adjusted_shift_y,
            "filter_width_y": adjusted_filter_width_y,
            "alpha": alpha,
            "pad_fact": pad_fact,
        })
        passage = self._current_passage()
        self.loaded_by_passage.pop(passage, None)
        self.load_current_folder(force_reload=True)

    def reset_tuning_settings(self) -> None:
        if self.loaded is None:
            return
        passage = self._current_passage()
        passage_overrides = self.override_by_passage.get(passage, {})
        passage_overrides.pop(self.processing_mode, None)
        if not passage_overrides:
            self.override_by_passage.pop(passage, None)
        self.loaded_by_passage.pop(passage, None)
        self.load_current_folder(force_reload=True)

    def _get_data_range(self, image: np.ndarray) -> tuple[float, float]:
        finite_values = image[np.isfinite(image)]
        if finite_values.size == 0:
            return 0.0, 1.0
        data_min = float(np.min(finite_values))
        data_max = float(np.max(finite_values))
        if data_min == data_max:
            delta = 1.0 if data_min == 0 else abs(data_min) * 0.05
            return data_min - delta, data_max + delta
        return data_min, data_max

    def _style_matplotlib(self) -> None:
        self.figure.patch.set_facecolor(COLORS["bg"])

    def _style_axis(self, axis, title: str) -> None:
        axis.set_facecolor("#101315")
        axis.set_title(title, color=COLORS["accent"], fontsize=18, pad=12)
        axis.set_xlabel("x", color=COLORS["text"])
        axis.set_ylabel("y", color=COLORS["text"])
        axis.tick_params(colors=COLORS["text"], labelsize=11)
        for spine in axis.spines.values():
            spine.set_edgecolor(COLORS["accent_strong"])
            spine.set_linewidth(1.15)

    def _style_colorbar(self, colorbar) -> None:
        colorbar.outline.set_edgecolor(COLORS["line"])
        colorbar.outline.set_linewidth(1.0)
        colorbar.ax.tick_params(colors=COLORS["text"], labelsize=11)
        colorbar.ax.set_facecolor("#101315")

    def _add_color_slider(
        self,
        slider_axis,
        image_artist,
        image: np.ndarray,
        label: str,
        cmap_name: str,
        accent_color: str = COLORS["accent"],
    ) -> None:
        data_min, data_max = self._get_data_range(image)
        slider_min, slider_max = data_min, data_max

        slider_axis.set_facecolor(COLORS["panel_alt"])
        for spine in slider_axis.spines.values():
            spine.set_edgecolor(COLORS["line"])
        gradient = np.linspace(0, 1, 512)[np.newaxis, :]
        slider_axis.imshow(
            gradient,
            aspect="auto",
            cmap=colormaps[cmap_name],
            extent=(data_min, data_max, 0, 1),
            interpolation="bicubic",
            alpha=0.9,
            zorder=0,
        )
        slider = RangeSlider(slider_axis, label, data_min, data_max, valinit=(slider_min, slider_max))
        slider.label.set_color(accent_color)
        slider.valtext.set_color(COLORS["accent"])
        slider.track.set_color((1.0, 1.0, 1.0, 0.12))
        slider.poly.set_color(QColor(accent_color).lighter(150).name())
        slider.poly.set_alpha(0.25)
        for handle in slider._handles:
            handle.set_color(QColor(accent_color).name())
            handle.set_markeredgecolor(COLORS["accent"])
            handle.set_markeredgewidth(1.4)

        def _update(_value) -> None:
            try:
                image_artist.set_clim(*slider.val)
                self.canvas.draw_idle()
            except (NotImplementedError, ValueError):
                # A stale slider can outlive the previous figure during a redraw.
                return

        callback_id = slider.on_changed(_update)
        self._sliders.append((slider, callback_id))

    def _disconnect_sliders(self) -> None:
        for slider, callback_id in self._sliders:
            try:
                slider.disconnect(callback_id)
            except Exception:
                pass
        self._sliders = []

    def _overlay_fft_marker(self, axis, stage_name: str, image_shape: tuple[int, int]) -> None:
        if self.loaded is None or stage_name not in {"mag_signal_ft", "filtered_shift"}:
            return

        settings = self.loaded.processing_settings
        row = None
        filter_width_y = None
        if stage_name == "mag_signal_ft":
            row = settings.get("current_center_row")
            filter_width_y = settings.get("current_filter_width_y")
        elif stage_name == "filtered_shift":
            row = settings.get("fft_center_row")

        if row is None:
            return

        if stage_name == "mag_signal_ft" and filter_width_y is not None:
            half_width = float(filter_width_y) / 2.0
            for boundary_row in (row - half_width, row + half_width):
                axis.axhline(
                    boundary_row,
                    color=COLORS["accent_strong"],
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=4,
                )

        col = image_shape[1] // 2
        axis.plot(
            [col],
            [row],
            marker="*",
            markersize=14,
            markerfacecolor=COLORS["accent"],
            markeredgecolor="#fff4df",
            markeredgewidth=1.2,
            linestyle="None",
            zorder=5,
        )

    def refresh_plot(self) -> None:
        self._disconnect_sliders()
        self.figure.clear()
        self._style_matplotlib()

        if self.loaded is None:
            placeholder = np.zeros((32, 32))
            grid = self.figure.add_gridspec(2, 2, height_ratios=[0.10, 1.0], wspace=0.18, hspace=0.28)
            amplitude_slider_axis = self.figure.add_subplot(grid[0, 0])
            phase_slider_axis = self.figure.add_subplot(grid[0, 1])
            amplitude_axis = self.figure.add_subplot(grid[1, 0])
            phase_axis = self.figure.add_subplot(grid[1, 1])

            amplitude_artist = amplitude_axis.imshow(placeholder, aspect="auto", cmap=VIEW_CMAPS[("processed", "amplitude")])
            phase_artist = phase_axis.imshow(placeholder, aspect="auto", cmap=VIEW_CMAPS[("processed", "phase")])

            self._style_axis(amplitude_axis, "Processed Amplitude")
            self._style_axis(phase_axis, "Processed Phase")
            amplitude_axis.text(
                0.5,
                0.5,
                "Load a folder to inspect hologram stages",
                ha="center",
                va="center",
                color=COLORS["muted"],
                transform=amplitude_axis.transAxes,
            )
            phase_axis.text(
                0.5,
                0.5,
                "Load a folder to inspect hologram stages",
                ha="center",
                va="center",
                color=COLORS["muted"],
                transform=phase_axis.transAxes,
            )

            amplitude_colorbar = self.figure.colorbar(amplitude_artist, ax=amplitude_axis, fraction=0.046, pad=0.04)
            phase_colorbar = self.figure.colorbar(phase_artist, ax=phase_axis, fraction=0.046, pad=0.04)
            self._style_colorbar(amplitude_colorbar)
            self._style_colorbar(phase_colorbar)

            self._add_color_slider(
                amplitude_slider_axis,
                amplitude_artist,
                placeholder,
                "Amplitude",
                VIEW_CMAPS[("processed", "amplitude")],
                accent_color=COLORS["accent_strong"],
            )
            self._add_color_slider(
                phase_slider_axis,
                phase_artist,
                placeholder,
                "Phase",
                VIEW_CMAPS[("processed", "phase")],
                accent_color=COLORS["accent"],
            )
            self.canvas.draw_idle()
            return

        try:
            harmonic_index = self._current_harmonic()
            stage_name = self._current_stage()
            amplitude_image = get_view_image(
                self.loaded.stage_stacks,
                harmonic_index,
                stage_name,
                "amplitude",
                processing_settings=self.loaded.processing_settings,
            )
            phase_image = None
            if stage_name not in AMPLITUDE_ONLY_STAGES:
                phase_image = get_view_image(
                    self.loaded.stage_stacks,
                    harmonic_index,
                    stage_name,
                    "phase",
                    processing_settings=self.loaded.processing_settings,
                )
        except ProcessingError as exc:
            QMessageBox.critical(self, "Display error", str(exc))
            self.log(f"Display error: {exc}")
            return

        grid = self.figure.add_gridspec(2, 2, height_ratios=[0.10, 1.0], wspace=0.18, hspace=0.28)
        amplitude_slider_axis = self.figure.add_subplot(grid[0, 0])
        phase_slider_axis = self.figure.add_subplot(grid[0, 1])
        amplitude_axis = self.figure.add_subplot(grid[1, 0])
        phase_axis = self.figure.add_subplot(grid[1, 1])

        amplitude_artist = amplitude_axis.imshow(
            amplitude_image,
            aspect="auto",
            cmap=VIEW_CMAPS[(stage_name, "amplitude")],
        )
        self._style_axis(amplitude_axis, f"H{harmonic_index} {STAGE_LABELS[stage_name]} Amplitude")
        self._overlay_fft_marker(amplitude_axis, stage_name, amplitude_image.shape)
        amplitude_colorbar = self.figure.colorbar(amplitude_artist, ax=amplitude_axis, fraction=0.046, pad=0.04)
        self._style_colorbar(amplitude_colorbar)
        self._add_color_slider(
            amplitude_slider_axis,
            amplitude_artist,
            amplitude_image,
            "Amplitude",
            VIEW_CMAPS[(stage_name, "amplitude")],
            accent_color=COLORS["accent_strong"],
        )

        if phase_image is None:
            phase_slider_axis.set_visible(False)
            self._style_axis(phase_axis, f"H{harmonic_index} {STAGE_LABELS[stage_name]} Phase")
            phase_axis.text(
                0.5,
                0.5,
                "Phase unavailable",
                ha="center",
                va="center",
                color=COLORS["muted"],
                transform=phase_axis.transAxes,
            )
            phase_axis.set_xticks([])
            phase_axis.set_yticks([])
        else:
            phase_artist = phase_axis.imshow(phase_image, aspect="auto", cmap=VIEW_CMAPS[(stage_name, "phase")])
            self._style_axis(phase_axis, f"H{harmonic_index} {STAGE_LABELS[stage_name]} Phase")
            phase_colorbar = self.figure.colorbar(phase_artist, ax=phase_axis, fraction=0.046, pad=0.04)
            self._style_colorbar(phase_colorbar)
            self._add_color_slider(
                phase_slider_axis,
                phase_artist,
                phase_image,
                "Phase",
                VIEW_CMAPS[(stage_name, "phase")],
                accent_color=COLORS["accent"],
            )

        self.canvas.draw_idle()

    def export_current_folder(self) -> None:
        if self.loaded is None:
            self.load_current_folder()
            if self.loaded is None:
                return
        try:
            exported_files = export_all_views(self.loaded)
        except ProcessingError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            self.log(f"Export failed: {exc}")
            return
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            self.log(f"Export failed: {exc}")
            return

        for output_path in exported_files:
            self.log(f"Wrote {output_path}")
        QMessageBox.information(self, "Export complete", f"Exported {len(exported_files)} files.")


def run_app(initial_folder: str | None = None) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = HologramViewerWindow(initial_folder=initial_folder)
    window.show()
    app.exec()


def main() -> None:
    initial_folder = sys.argv[1] if len(sys.argv) > 1 else None
    run_app(initial_folder=initial_folder)


if __name__ == "__main__":
    main()
