from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from hologram_opening import (
    PROCESSING_MODES,
    ProcessingError,
    build_vertical_profile,
    measure_profile_width,
    processed_phase,
    reconstruct_hologram,
)


PASSAGE_TO_KEYS = {
    "forward": {"z": "Z", "o": "O", "m": "M", "label": "Forward (O)"},
    "reverse": {"z": "R-Z", "o": "R-O", "m": "R-M", "label": "Reverse (R-O)"},
}

EXPORT_HARMONICS = (0, 1, 2, 3, 4, 5)
STAGE_LABELS = {
    "raw": "Raw",
    "processed": "Processed",
    "mag_signal_ft": "Mag Signal FT",
    "filtered_shift": "Filtered Shift",
}
AMPLITUDE_ONLY_STAGES = {"mag_signal_ft", "filtered_shift"}


@dataclass
class LoadedData:
    folder_path: str
    image_name: str
    passage: str
    z: np.ndarray
    o: np.ndarray
    m: np.ndarray | None
    stage_stacks: dict[str, np.ndarray]
    processing_settings: dict[str, Any]
    processing_diagnostics: dict[str, Any]
    cache_path: str
    metadata: dict[str, Any]


def gsf2mat(file_path: str) -> np.ndarray | None:
    try:
        with open(file_path, "rb") as file:
            content = file.read().decode("latin1")
            pos_scaling = content.find("erScaling=1") + 10
            pos_scaling = -(-pos_scaling // 4) * 4
            pos_res_x = content.find("XRes=")
            pos_res_y = content.find("YRes=")
            pos_res_y_incomplete = content.find("YResIncomplete=")
            res_x = int(content[pos_res_x + 5 : pos_res_y].strip())
            res_y = int(content[pos_res_y + 5 : pos_res_y_incomplete].strip())
            file.seek(pos_scaling)
            return np.fromfile(file, dtype=np.float32, count=res_x * res_y).reshape(res_y, res_x)
    except FileNotFoundError:
        return None
    except Exception as exc:
        raise ProcessingError(f"Failed to read GSF file '{file_path}': {exc}") from exc


def _mode_file_paths(image_folder: str, image_name: str, mode: str, harmonic: int) -> tuple[str, str]:
    amp_file = os.path.join(image_folder, f"{image_name} {mode}{harmonic}A raw.gsf")
    phase_file = os.path.join(image_folder, f"{image_name} {mode}{harmonic}P raw.gsf")
    return amp_file, phase_file


def _harmonic_presence(image_folder: str, image_name: str, mode: str) -> list[int]:
    available = []
    for harmonic in range(6):
        amp_file, phase_file = _mode_file_paths(image_folder, image_name, mode, harmonic)
        if os.path.exists(amp_file) and os.path.exists(phase_file):
            available.append(harmonic)
    return available


def _harmonic_is_present(stack: np.ndarray, harmonic: int) -> bool:
    slice_ = stack[:, :, harmonic]
    return not np.all(np.isnan(np.real(slice_)) & np.isnan(np.imag(slice_)))


def build_or_load_cache(folder_path: str, force_rebuild: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        raise ProcessingError(f"Folder does not exist: {folder_path}")

    image_name = os.path.basename(os.path.normpath(folder_path))
    cache_path = os.path.join(folder_path, f"{image_name}_all_data.npz")

    if force_rebuild or not os.path.exists(cache_path):
        all_data: dict[str, np.ndarray] = {}

        for base in ("Z", "R-Z"):
            base_path = os.path.join(folder_path, f"{image_name} {base} raw.gsf")
            data = gsf2mat(base_path)
            if data is not None:
                all_data[base] = data

        for mode in ("M", "R-M", "O", "R-O"):
            harmonic_data: dict[int, np.ndarray] = {}
            for harmonic in range(6):
                amp = gsf2mat(_mode_file_paths(folder_path, image_name, mode, harmonic)[0])
                phase = gsf2mat(_mode_file_paths(folder_path, image_name, mode, harmonic)[1])
                if amp is not None and phase is not None:
                    harmonic_data[harmonic] = amp * np.exp(1j * phase)
            if harmonic_data:
                first = next(iter(harmonic_data.values()))
                combined_data = np.full((first.shape[0], first.shape[1], 6), np.nan + 1j * np.nan, dtype=complex)
                for harmonic, complex_data in harmonic_data.items():
                    combined_data[:, :, harmonic] = complex_data
                all_data[mode] = combined_data

        if not all_data:
            raise ProcessingError(f"No valid SNOM data found in folder: {folder_path}")

        np.savez_compressed(cache_path, **all_data)

    cache = np.load(cache_path, allow_pickle=True)
    data = {key: cache[key] for key in cache.files}
    metadata = {
        "folder_path": folder_path,
        "image_name": image_name,
        "cache_path": cache_path,
        "cache_keys": sorted(cache.files),
        "available_gsf_harmonics": {
            "forward": _harmonic_presence(folder_path, image_name, "O"),
            "reverse": _harmonic_presence(folder_path, image_name, "R-O"),
        },
    }
    return data, metadata


def validate_passage_data(data: dict[str, np.ndarray], passage: str) -> dict[str, str]:
    passage = passage.lower()
    if passage not in PASSAGE_TO_KEYS:
        raise ProcessingError(f"Unknown passage: {passage}")

    keys = PASSAGE_TO_KEYS[passage]
    missing = [value for value in (keys["z"], keys["o"]) if value not in data]
    if missing:
        raise ProcessingError(
            f"Selected passage '{keys['label']}' is missing required cached arrays: {', '.join(missing)}."
        )

    o = data[keys["o"]]
    if o.ndim != 3:
        raise ProcessingError(f"Expected 3D harmonic stack for '{keys['o']}', got shape {o.shape}.")

    required_count = max(EXPORT_HARMONICS) + 1
    if o.shape[2] < required_count:
        raise ProcessingError(
            f"Selected passage '{keys['label']}' has {o.shape[2]} harmonics in cache, "
            f"but export/view requires indices 0..5."
        )

    missing_harmonics = [harmonic for harmonic in EXPORT_HARMONICS if not _harmonic_is_present(o, harmonic)]
    if missing_harmonics:
        raise ProcessingError(
            f"Selected passage '{keys['label']}' is missing required harmonics: {missing_harmonics}."
        )

    return keys


def _empty_stage_stack(reference_image: np.ndarray, harmonic_count: int) -> np.ndarray:
    return np.full(
        (reference_image.shape[0], reference_image.shape[1], harmonic_count),
        np.nan + 1j * np.nan,
        dtype=complex,
    )


def _result_stage_map(result: Any) -> dict[str, np.ndarray]:
    return {
        "processed": result.view_stages.processed,
        "mag_signal_ft": result.view_stages.mag_signal_ft,
        "filtered_shift": result.view_stages.filtered_shift,
    }


def _result_diagnostics(result: Any) -> dict[str, float | int]:
    diagnostics = result.diagnostics
    diagnostics["carrier_row"] = int(result.geometry.carrier_row)
    diagnostics["filter_width_y"] = int(result.geometry.filter_width_y)
    return diagnostics


def process_stack(
    raw_stack: np.ndarray,
    pad_fact: int = 4,
    alpha: float = 0.3,
    carrier_row_override: int | None = None,
    filter_width_override: int | None = None,
    processing_mode: str = "two_sideband",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if raw_stack.ndim != 3:
        raise ProcessingError(f"Expected raw stack with shape [row, col, harmonic], got {raw_stack.shape}.")
    if processing_mode not in PROCESSING_MODES:
        raise ProcessingError(f"Unknown processing mode: {processing_mode}")

    reference_harmonic = 2
    if raw_stack.shape[2] <= reference_harmonic:
        raise ProcessingError("The raw stack does not contain harmonic index 2 required for hologram opening.")
    if not _harmonic_is_present(raw_stack, reference_harmonic):
        raise ProcessingError("The raw stack is missing harmonic index 2 required for hologram opening.")

    auto_result = reconstruct_hologram(
        raw_stack[:, :, reference_harmonic],
        pad_fact=pad_fact,
        alpha=alpha,
        processing_mode=processing_mode,
    )
    auto_geometry = auto_result.geometry
    auto_vertical_profile = auto_result.analysis.vertical_profile
    zero_order_width_y = measure_profile_width(auto_vertical_profile, auto_geometry.fft_center_row)
    first_order_width_y = measure_profile_width(auto_vertical_profile, auto_geometry.carrier_row)

    current_center_row = auto_geometry.carrier_row if carrier_row_override is None else int(round(carrier_row_override))
    current_filter_width_y = (
        auto_geometry.filter_width_y if filter_width_override is None else int(round(filter_width_override))
    )
    is_manual = carrier_row_override is not None or filter_width_override is not None

    reference_result = auto_result
    if is_manual:
        reference_result = reconstruct_hologram(
            raw_stack[:, :, reference_harmonic],
            pad_fact=pad_fact,
            alpha=alpha,
            carrier_row=current_center_row,
            filter_width_y=current_filter_width_y,
            processing_mode=processing_mode,
        )
        current_center_row = reference_result.geometry.carrier_row
        current_filter_width_y = reference_result.geometry.filter_width_y

    harmonic_count = raw_stack.shape[2]
    stage_stacks = {"raw": raw_stack.copy()}
    for stage_name, reference_image in _result_stage_map(reference_result).items():
        stage_stacks[stage_name] = _empty_stage_stack(reference_image, harmonic_count)

    per_harmonic_diagnostics: list[dict[str, float | int] | None] = [None] * harmonic_count
    for stage_name, image in _result_stage_map(reference_result).items():
        stage_stacks[stage_name][:, :, reference_harmonic] = image
    per_harmonic_diagnostics[reference_harmonic] = _result_diagnostics(reference_result)

    for idx in range(harmonic_count):
        if idx == reference_harmonic or not _harmonic_is_present(raw_stack, idx):
            continue
        result = reconstruct_hologram(
            raw_stack[:, :, idx],
            pad_fact=pad_fact,
            alpha=alpha,
            carrier_row=current_center_row,
            filter_width_y=current_filter_width_y,
            processing_mode=processing_mode,
        )
        for stage_name, image in _result_stage_map(result).items():
            stage_stacks[stage_name][:, :, idx] = image
        per_harmonic_diagnostics[idx] = _result_diagnostics(result)

    rotation_angle_rad_by_harmonic = [
        float("nan") if diagnostic is None else float(diagnostic["rotation_angle_rad"]) for diagnostic in per_harmonic_diagnostics
    ]
    rotation_angle_deg_by_harmonic = [
        float("nan") if diagnostic is None else float(diagnostic["rotation_angle_deg"]) for diagnostic in per_harmonic_diagnostics
    ]
    mirror_row_by_harmonic = [
        -1 if diagnostic is None else int(diagnostic["mirror_row"]) for diagnostic in per_harmonic_diagnostics
    ]

    processing_diagnostics = {
        "reference_harmonic": reference_harmonic,
        "auto_geometry": asdict(auto_geometry),
        "current_geometry": asdict(reference_result.geometry),
        "is_manual": is_manual,
        "per_harmonic": per_harmonic_diagnostics,
        "rotation_angle_rad_by_harmonic": rotation_angle_rad_by_harmonic,
        "rotation_angle_deg_by_harmonic": rotation_angle_deg_by_harmonic,
        "mirror_row_by_harmonic": mirror_row_by_harmonic,
    }
    processing_settings = {
        "processing_mode": processing_mode,
        "pad_fact": pad_fact,
        "alpha": alpha,
        "fft_center_row": auto_geometry.fft_center_row,
        "auto_center_row": auto_geometry.carrier_row,
        "auto_filter_width_y": auto_geometry.filter_width_y,
        "current_center_row": current_center_row,
        "current_filter_width_y": current_filter_width_y,
        "auto_shift_y": auto_geometry.fft_center_row - auto_geometry.carrier_row,
        "current_shift_y": auto_geometry.fft_center_row - current_center_row,
        "zero_order_width_y": zero_order_width_y,
        "first_order_width_y": first_order_width_y,
        "diagnostics": processing_diagnostics,
    }
    return stage_stacks, processing_settings


def load_passage(
    folder_path: str,
    passage: str,
    force_rebuild: bool = False,
    carrier_row_override: int | None = None,
    filter_width_override: int | None = None,
    pad_fact: int = 4,
    alpha: float = 0.3,
    processing_mode: str = "two_sideband",
) -> LoadedData:
    data, metadata = build_or_load_cache(folder_path, force_rebuild=force_rebuild)
    keys = validate_passage_data(data, passage)
    stage_stacks, processing_settings = process_stack(
        data[keys["o"]],
        pad_fact=pad_fact,
        alpha=alpha,
        carrier_row_override=carrier_row_override,
        filter_width_override=filter_width_override,
        processing_mode=processing_mode,
    )
    processing_diagnostics = processing_settings.get("diagnostics", {})

    loaded = LoadedData(
        folder_path=metadata["folder_path"],
        image_name=metadata["image_name"],
        passage=passage.lower(),
        z=data[keys["z"]],
        o=data[keys["o"]],
        m=data.get(keys["m"]),
        stage_stacks=stage_stacks,
        processing_settings=processing_settings,
        processing_diagnostics=processing_diagnostics,
        cache_path=metadata["cache_path"],
        metadata=metadata,
    )
    return loaded


def get_view_image(
    stage_stacks: dict[str, np.ndarray],
    harmonic_index: int,
    stage_name: str,
    representation: str,
    processing_settings: dict[str, Any] | None = None,
) -> np.ndarray:
    if stage_name not in stage_stacks:
        raise ProcessingError(f"Unknown view stage: {stage_name}")

    stack = stage_stacks[stage_name]
    if harmonic_index < 0 or harmonic_index >= stack.shape[2]:
        raise ProcessingError(f"Invalid harmonic index {harmonic_index} for stack shape {stack.shape}.")
    if not _harmonic_is_present(stack, harmonic_index):
        raise ProcessingError(f"Harmonic {harmonic_index} is not available in the selected passage.")
    image = stack[:, :, harmonic_index]

    if stage_name in AMPLITUDE_ONLY_STAGES and representation != "amplitude":
        raise ProcessingError(f"{STAGE_LABELS[stage_name]} only supports amplitude visualization.")
    if representation == "amplitude":
        return np.abs(image)
    if representation == "phase":
        if stage_name == "processed":
            processing_mode = "two_sideband"
            if processing_settings is not None:
                processing_mode = processing_settings.get("processing_mode", "two_sideband")
            return processed_phase(image, processing_mode=processing_mode)
        phase = np.unwrap(np.angle(image))
        return phase
    raise ProcessingError(f"Unknown representation: {representation}")


def _export_figure(image: np.ndarray, title: str, output_path: str, cmap: str) -> None:
    figure = Figure(figsize=(6, 5), dpi=150)
    axis = figure.add_subplot(111)
    im = axis.imshow(image, aspect="auto", cmap=cmap)
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    figure.colorbar(im, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path)


def export_all_views(loaded: LoadedData) -> list[str]:
    validate_passage_data(
        {PASSAGE_TO_KEYS[loaded.passage]["o"]: loaded.o, PASSAGE_TO_KEYS[loaded.passage]["z"]: loaded.z},
        loaded.passage,
    )
    exported_files: list[str] = []

    for harmonic in EXPORT_HARMONICS:
        views = {
            "raw_amplitude": ("raw", "amplitude", "hot"),
            "raw_phase": ("raw", "phase", "bwr"),
            "processed_amplitude": ("processed", "amplitude", "hot"),
            "processed_phase": ("processed", "phase", "bwr"),
        }
        for name, (stage_name, representation, cmap) in views.items():
            image = get_view_image(
                loaded.stage_stacks,
                harmonic,
                stage_name,
                representation,
                processing_settings=loaded.processing_settings,
            )
            output_path = os.path.join(loaded.folder_path, f"h{harmonic}_{name}.png")
            title = f"H{harmonic} {STAGE_LABELS[stage_name]} {representation.capitalize()} - {loaded.image_name}"
            _export_figure(image, title, output_path, cmap)
            exported_files.append(output_path)
    return exported_files
