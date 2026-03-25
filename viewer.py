from __future__ import annotations

import os
import sys

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.widgets import RangeSlider
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
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
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from .processing import (
        AMPLITUDE_ONLY_STAGES,
        PASSAGE_TO_KEYS,
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
    "bg": "#090d1b",
    "panel": "#0d1327",
    "panel_alt": "#111a33",
    "line": "#284d80",
    "text": "#c8e8ff",
    "muted": "#85a8cf",
    "cyan": "#6de7ff",
    "cyan_strong": "#17c5ff",
    "magenta": "#f36dff",
    "magenta_strong": "#cc43ff",
    "danger": "#ff4fc7",
}

APP_QSS = f"""
QMainWindow {{
    background: {COLORS["bg"]};
}}
QWidget {{
    color: {COLORS["text"]};
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 14px;
}}
QFrame#panel, QGroupBox#panel {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {COLORS["panel"]}, stop:1 {COLORS["panel_alt"]});
    border: 1px solid {COLORS["line"]};
    border-radius: 12px;
}}
QGroupBox#panel {{
    margin-top: 14px;
    padding-top: 14px;
}}
QGroupBox#panel::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {COLORS["cyan"]};
}}
QLineEdit, QComboBox, QPlainTextEdit {{
    background: rgba(6, 15, 32, 0.88);
    border: 1px solid {COLORS["line"]};
    border-radius: 8px;
    padding: 8px 10px;
    selection-background-color: {COLORS["cyan_strong"]};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 34px;
    border-left: 1px solid {COLORS["line"]};
    background: rgba(18, 39, 69, 0.9);
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QComboBox::down-arrow {{
    image: none;
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid {COLORS["cyan"]};
}}
QComboBox QAbstractItemView {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["line"]};
    selection-background-color: rgba(23, 197, 255, 0.3);
}}
QPushButton {{
    background: rgba(10, 27, 48, 0.95);
    border: 1px solid {COLORS["cyan_strong"]};
    border-radius: 10px;
    padding: 8px 16px;
    color: {COLORS["text"]};
}}
QPushButton:hover {{
    background: rgba(16, 54, 84, 0.95);
}}
QPushButton#accent {{
    border-color: {COLORS["magenta"]};
    color: #ffd6ff;
}}
QPushButton#accent:hover {{
    background: rgba(67, 14, 73, 0.95);
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
    background: rgba(5, 13, 26, 0.95);
}}
QRadioButton::indicator::checked, QCheckBox::indicator::checked {{
    border: 1px solid {COLORS["cyan"]};
    background: {COLORS["cyan_strong"]};
}}
QLabel#sectionTitle {{
    color: {COLORS["cyan"]};
    font-size: 15px;
    font-weight: 600;
}}
QLabel#metricValue {{
    color: #f3fbff;
}}
"""


class HologramViewerWindow(QMainWindow):
    def __init__(self, initial_folder: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Hologram Viewer")
        self.resize(1500, 980)
        self.loaded = None
        self.loaded_by_passage: dict[str, object] = {}
        self.override_by_passage: dict[str, dict[str, int]] = {}
        self.loaded_folder_path: str | None = None
        self._sliders: list[RangeSlider] = []
        self._apply_theme()
        self._build_ui()
        self._sync_advanced_settings()

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

        root_layout.addWidget(self._build_command_bar())
        root_layout.addWidget(self._build_control_bar())

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        content_layout.addWidget(self._build_sidebar(), 0)
        content_layout.addWidget(self._build_plot_panel(), 1)
        root_layout.addLayout(content_layout, 1)

        root_layout.addWidget(self._build_log_panel(), 0)

    def _panel(self, title: str | None = None) -> QFrame | QGroupBox:
        if title:
            panel = QGroupBox(title)
        else:
            panel = QFrame()
        panel.setObjectName("panel")
        return panel

    def _build_command_bar(self) -> QWidget:
        panel = self._panel()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select a hologram folder")
        layout.addWidget(self.folder_edit, 1)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_current_folder)
        export_button = QPushButton("Export All")
        export_button.clicked.connect(self.export_current_folder)
        export_button.setObjectName("accent")

        layout.addWidget(browse_button)
        layout.addWidget(load_button)
        layout.addWidget(export_button)
        return panel

    def _build_control_bar(self) -> QWidget:
        panel = self._panel()
        layout = QGridLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(10)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(5, 1)

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
        layout.addLayout(passage_row, 0, 1, 1, 3)

        harmonic_title = QLabel("Harmonic")
        harmonic_title.setObjectName("sectionTitle")
        layout.addWidget(harmonic_title, 0, 4)
        self.harmonic_combo = QComboBox()
        self.harmonic_combo.addItems([str(i) for i in range(1, 6)])
        self.harmonic_combo.setCurrentText("1")
        self.harmonic_combo.setMinimumWidth(150)
        self.harmonic_combo.currentIndexChanged.connect(self.refresh_plot)
        layout.addWidget(self.harmonic_combo, 0, 5)

        view_title = QLabel("View")
        view_title.setObjectName("sectionTitle")
        layout.addWidget(view_title, 1, 0)
        self.view_combo = QComboBox()
        for key in STAGE_LABELS:
            self.view_combo.addItem(key)
        self.view_combo.setCurrentText("processed")
        self.view_combo.setMinimumWidth(180)
        self.view_combo.currentIndexChanged.connect(self.refresh_plot)
        layout.addWidget(self.view_combo, 1, 1)
        return panel

    def _build_sidebar(self) -> QWidget:
        panel = self._panel("Advanced Settings")
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setSpacing(12)

        self.show_advanced_checkbox = QCheckBox("Show")
        self.show_advanced_checkbox.toggled.connect(self._update_advanced_panel)
        layout.addWidget(self.show_advanced_checkbox)

        self.advanced_content = QWidget()
        advanced_layout = QGridLayout(self.advanced_content)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setHorizontalSpacing(10)
        advanced_layout.setVerticalSpacing(10)

        self.auto_shift_value = self._metric_row(advanced_layout, 0, "Detected Shift (px)")
        self.auto_width_value = self._metric_row(advanced_layout, 1, "Detected Width (px)")
        self.zero_order_width_value = self._metric_row(advanced_layout, 2, "0th Order Width (px)")
        self.first_order_width_value = self._metric_row(advanced_layout, 3, "1st Order Width (px)")

        advanced_layout.addWidget(QLabel("Current Shift (px)"), 4, 0)
        self.shift_edit = QLineEdit()
        advanced_layout.addWidget(self.shift_edit, 4, 1)
        advanced_layout.addWidget(QLabel("Current Width (px)"), 5, 0)
        self.width_edit = QLineEdit()
        advanced_layout.addWidget(self.width_edit, 5, 1)

        button_row = QHBoxLayout()
        self.apply_advanced_button = QPushButton("Apply")
        self.apply_advanced_button.clicked.connect(self.apply_advanced_settings)
        self.reset_advanced_button = QPushButton("Reset to Auto")
        self.reset_advanced_button.clicked.connect(self.reset_advanced_settings)
        self.reset_advanced_button.setObjectName("accent")
        button_row.addWidget(self.apply_advanced_button)
        button_row.addWidget(self.reset_advanced_button)
        advanced_layout.addLayout(button_row, 6, 0, 1, 2)

        layout.addWidget(self.advanced_content)
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self._update_advanced_panel(False)
        return panel

    def _metric_row(self, layout: QGridLayout, row: int, label_text: str) -> QLabel:
        layout.addWidget(QLabel(label_text), row, 0)
        value = QLabel("-")
        value.setObjectName("metricValue")
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
            f"QToolBar {{ background: {COLORS['panel']}; border: 1px solid {COLORS['line']}; spacing: 6px; }}"
            f"QToolButton {{ background: rgba(10,27,48,0.92); border: 1px solid {COLORS['line']}; border-radius: 8px; padding: 6px; color: {COLORS['text']}; }}"
            f"QToolButton:hover {{ border-color: {COLORS['cyan_strong']}; }}"
        )
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.toolbar, 0)
        return panel

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
            self._clear_loaded_state()
            self.refresh_plot()

    def _current_passage(self) -> str:
        return "forward" if self.forward_radio.isChecked() else "reverse"

    def _current_harmonic(self) -> int:
        return int(self.harmonic_combo.currentText())

    def _current_stage(self) -> str:
        return self.view_combo.currentText()

    def _clear_loaded_state(self) -> None:
        self.loaded = None
        self.loaded_by_passage = {}
        self.override_by_passage = {}
        self.loaded_folder_path = None
        self._sync_advanced_settings()

    def _set_advanced_controls_enabled(self, enabled: bool) -> None:
        self.shift_edit.setEnabled(enabled)
        self.width_edit.setEnabled(enabled)
        self.apply_advanced_button.setEnabled(enabled)
        self.reset_advanced_button.setEnabled(enabled)

    def _update_advanced_panel(self, checked: bool | None = None) -> None:
        if checked is None:
            checked = self.show_advanced_checkbox.isChecked()
        self.advanced_content.setVisible(bool(checked))

    def _sync_advanced_settings(self) -> None:
        if self.loaded is None:
            for label in (
                self.auto_shift_value,
                self.auto_width_value,
                self.zero_order_width_value,
                self.first_order_width_value,
            ):
                label.setText("-")
            self.shift_edit.setText("")
            self.width_edit.setText("")
            self._set_advanced_controls_enabled(False)
            return

        settings = self.loaded.processing_settings
        self.auto_shift_value.setText(str(settings["auto_shift_y"]))
        self.auto_width_value.setText(str(settings["auto_filter_width_y"]))
        self.zero_order_width_value.setText(f"{settings['zero_order_width_y']:.2f}")
        self.first_order_width_value.setText(f"{settings['first_order_width_y']:.2f}")
        self.shift_edit.setText(str(settings["current_shift_y"]))
        self.width_edit.setText(str(settings["current_filter_width_y"]))
        self._set_advanced_controls_enabled(True)

    def _on_passage_change(self) -> None:
        folder = self.folder_edit.text().strip()
        if folder and os.path.isdir(folder):
            self.load_current_folder()
        else:
            self._sync_advanced_settings()

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
                f"with cache {self.loaded.cache_path}."
            )
            self._sync_advanced_settings()
            self.refresh_plot()
            return

        overrides = self.override_by_passage.get(passage, {})
        try:
            self.loaded = load_passage(
                folder,
                passage,
                carrier_row_override=overrides.get("center_row"),
                filter_width_override=overrides.get("filter_width_y"),
            )
        except ProcessingError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self.log(f"Load failed: {exc}")
            return

        self.loaded_by_passage[passage] = self.loaded
        self.log(
            f"Loaded {PASSAGE_TO_KEYS[self.loaded.passage]['label']} from {self.loaded.folder_path} "
            f"with cache {self.loaded.cache_path}."
        )
        self._sync_advanced_settings()
        self.refresh_plot()

    def apply_advanced_settings(self) -> None:
        if self.loaded is None:
            return
        try:
            shift_y = int(self.shift_edit.text().strip())
            filter_width_y = int(self.width_edit.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Advanced settings", "Shift and width must be integer pixel values.")
            self.log("Advanced settings error: shift and width must be integer pixel values.")
            return
        if filter_width_y <= 0:
            QMessageBox.critical(self, "Advanced settings", "Filter width must be greater than zero.")
            self.log("Advanced settings error: filter width must be greater than zero.")
            return

        fft_center_row = self.loaded.processing_settings["fft_center_row"]
        passage = self._current_passage()
        self.override_by_passage[passage] = {
            "center_row": fft_center_row - shift_y,
            "filter_width_y": filter_width_y,
        }
        self.loaded_by_passage.pop(passage, None)
        self.load_current_folder(force_reload=True)

    def reset_advanced_settings(self) -> None:
        if self.loaded is None:
            return
        passage = self._current_passage()
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
        axis.set_facecolor("#0a1020")
        axis.set_title(title, color=COLORS["cyan"], fontsize=18, pad=12)
        axis.set_xlabel("x", color=COLORS["text"])
        axis.set_ylabel("y", color=COLORS["text"])
        axis.tick_params(colors=COLORS["text"], labelsize=11)
        for spine in axis.spines.values():
            spine.set_edgecolor(COLORS["cyan_strong"])
            spine.set_linewidth(1.15)

    def _style_colorbar(self, colorbar) -> None:
        colorbar.outline.set_edgecolor(COLORS["line"])
        colorbar.outline.set_linewidth(1.0)
        colorbar.ax.tick_params(colors=COLORS["text"], labelsize=11)
        colorbar.ax.set_facecolor("#0a1020")

    def _add_color_slider(
        self,
        slider_axis,
        image_artist,
        image: np.ndarray,
        label: str,
        cmap_name: str,
        accent_color: str = COLORS["cyan_strong"],
    ) -> None:
        data_min, data_max = self._get_data_range(image)
        slider_min, slider_max = data_min, data_max

        slider_axis.set_facecolor("#0a1020")
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
        slider.label.set_color(COLORS["text"])
        slider.valtext.set_color(COLORS["text"])
        slider.track.set_color((1.0, 1.0, 1.0, 0.08))
        slider.poly.set_color((1.0, 1.0, 1.0, 0.0))
        for handle in slider._handles:
            handle.set_color(QColor(accent_color).name())

        def _update(_value) -> None:
            image_artist.set_clim(*slider.val)
            self.canvas.draw_idle()

        slider.on_changed(_update)
        self._sliders.append(slider)

    def refresh_plot(self) -> None:
        self.figure.clear()
        self._sliders = []
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
                accent_color=COLORS["cyan_strong"],
            )
            self._add_color_slider(
                phase_slider_axis,
                phase_artist,
                placeholder,
                "Phase",
                VIEW_CMAPS[("processed", "phase")],
                accent_color=COLORS["magenta_strong"],
            )
            self.canvas.draw_idle()
            return

        try:
            harmonic_index = self._current_harmonic()
            stage_name = self._current_stage()
            amplitude_image = get_view_image(self.loaded.stage_stacks, harmonic_index, stage_name, "amplitude")
            phase_image = None
            if stage_name not in AMPLITUDE_ONLY_STAGES:
                phase_image = get_view_image(self.loaded.stage_stacks, harmonic_index, stage_name, "phase")
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
        amplitude_colorbar = self.figure.colorbar(amplitude_artist, ax=amplitude_axis, fraction=0.046, pad=0.04)
        self._style_colorbar(amplitude_colorbar)
        self._add_color_slider(
            amplitude_slider_axis,
            amplitude_artist,
            amplitude_image,
            "Amplitude",
            VIEW_CMAPS[(stage_name, "amplitude")],
            accent_color=COLORS["cyan_strong"],
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
                accent_color=COLORS["magenta_strong"],
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
