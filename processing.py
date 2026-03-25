from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from scipy import signal
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.linalg import lstsq


PASSAGE_TO_KEYS = {
    "forward": {"z": "Z", "o": "O", "m": "M", "label": "Forward (O)"},
    "reverse": {"z": "R-Z", "o": "R-O", "m": "R-M", "label": "Reverse (R-O)"},
}

EXPORT_HARMONICS = (1, 2, 3, 4, 5)
STAGE_LABELS = {
    "raw": "Raw",
    "processed": "Processed",
    "mag_signal_ft": "Mag Signal FT",
    "filtered_shift": "Filtered Shift",
}
AMPLITUDE_ONLY_STAGES = {"mag_signal_ft", "filtered_shift"}


class ProcessingError(RuntimeError):
    pass


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
            f"but export/view requires indices 1..5."
        )

    missing_harmonics = [harmonic for harmonic in EXPORT_HARMONICS if not _harmonic_is_present(o, harmonic)]
    if missing_harmonics:
        raise ProcessingError(
            f"Selected passage '{keys['label']}' is missing required harmonics: {missing_harmonics}."
        )

    return keys


def tukey_filter_func(width: int, length: int, alpha: float) -> np.ndarray:
    width = max(2, min(int(width), length))
    if width % 2 == 1:
        width += 1
    width = min(width, length)

    filter_window = np.zeros(length)
    start = (length - width) // 2
    stop = start + width
    filter_window[start:stop] = signal.windows.tukey(width, alpha)
    return filter_window


def _build_vertical_profile(magnitude_ft: np.ndarray) -> np.ndarray:
    profile = np.log1p(np.mean(magnitude_ft, axis=1))
    kernel_size = max(5, min(21, profile.size // 32))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    return np.convolve(profile, kernel, mode="same")


def _refine_peak_subpixel(profile: np.ndarray, peak_idx: int) -> float:
    if peak_idx <= 0 or peak_idx >= profile.size - 1:
        return float(peak_idx)

    left = profile[peak_idx - 1]
    center = profile[peak_idx]
    right = profile[peak_idx + 1]
    denominator = left - 2 * center + right
    if abs(denominator) < 1e-12:
        return float(peak_idx)

    offset = 0.5 * (left - right) / denominator
    return float(np.clip(peak_idx + offset, peak_idx - 0.5, peak_idx + 0.5))


def _find_vertical_carrier(profile: np.ndarray) -> float:
    center_idx = profile.size // 2
    exclusion_half_width = max(8, profile.size // 32)
    edge_margin = 2
    search_stop = center_idx - exclusion_half_width

    peaks, peak_properties = signal.find_peaks(
        profile[:search_stop],
        distance=max(4, profile.size // 64),
        prominence=max(np.std(profile) * 0.25, 1e-6),
    )
    valid_peaks = [peak for peak in peaks if edge_margin <= peak < search_stop - edge_margin]
    if valid_peaks:
        prominences = peak_properties["prominences"]
        peak_order = {peak: idx for idx, peak in enumerate(peaks)}
        peak_idx = int(max(valid_peaks, key=lambda peak: prominences[peak_order[peak]] * profile[peak]))
        return _refine_peak_subpixel(profile, peak_idx)

    search_profile = profile.copy()
    search_profile[:edge_margin] = -np.inf
    search_profile[search_stop:] = -np.inf
    peak_idx = int(np.argmax(search_profile))
    if not np.isfinite(search_profile[peak_idx]):
        raise ProcessingError("Unable to find a valid vertical carrier away from the zero order.")
    return _refine_peak_subpixel(profile, peak_idx)


def _estimate_filter_width(profile: np.ndarray, carrier_row: float) -> int:
    center_idx = profile.size // 2
    distance_to_zero_order = abs(carrier_row - center_idx)
    carrier_row_int = int(round(carrier_row))
    distance_to_edge = min(carrier_row_int, profile.size - carrier_row_int - 1)
    width = int(min(distance_to_zero_order, distance_to_edge))
    if width < 2:
        raise ProcessingError("Detected carrier is too close to the zero order to build a stable filter.")
    return width


def _measure_profile_width(profile: np.ndarray, row_idx: int) -> float:
    if row_idx < 0 or row_idx >= profile.size:
        raise ProcessingError(f"Cannot measure profile width at invalid row {row_idx}.")

    search_radius = max(3, profile.size // 64)
    start = max(0, row_idx - search_radius)
    stop = min(profile.size, row_idx + search_radius + 1)
    local_idx = int(np.argmax(profile[start:stop])) + start

    peaks, _ = signal.find_peaks(profile[start:stop])
    if peaks.size > 0:
        peak_candidates = peaks + start
        local_idx = int(min(peak_candidates, key=lambda idx: abs(idx - row_idx)))

    width = float(signal.peak_widths(profile, [local_idx], rel_height=0.5)[0][0])
    if width <= 0:
        threshold = 0.5 * profile[local_idx]
        above = np.flatnonzero(profile >= threshold)
        if above.size > 0:
            contiguous = above[(above >= start) & (above < stop)]
            if contiguous.size > 0:
                width = float(contiguous[-1] - contiguous[0] + 1)
    return width


def _normalize_filter_width(width: int, length: int) -> int:
    width = max(2, min(int(round(width)), length))
    if width % 2 == 1:
        width += 1
    return min(width, length)


def _band_bounds(center_row: int, width: int, length: int) -> tuple[int, int]:
    width = _normalize_filter_width(width, length)
    start = center_row - width // 2
    stop = start + width
    if start < 0:
        stop -= start
        start = 0
    if stop > length:
        start -= stop - length
        stop = length
    start = max(0, start)
    stop = min(length, stop)
    if stop - start < 2:
        raise ProcessingError("Vertical filter band is too narrow after clamping.")
    return start, stop


def _validate_vertical_filter(length: int, center_row: int, filter_width_y: int) -> tuple[int, int]:
    center_idx = length // 2
    exclusion_half_width = max(8, length // 32)
    if center_row >= center_idx - exclusion_half_width:
        raise ProcessingError("Carrier center must stay above the zero-order exclusion band.")
    if center_row < 1 or center_row >= length - 1:
        raise ProcessingError("Carrier center must stay inside the Fourier-space image bounds.")

    width = _normalize_filter_width(filter_width_y, length)
    start, stop = _band_bounds(center_row, width, length)
    if stop >= center_idx - exclusion_half_width + 1:
        raise ProcessingError("Filter width reaches the zero order; reduce the width or move the center upward.")
    return center_row, width


def _build_sideband_filter(length: int, center_row: int, filter_width_y: int, alpha: float) -> np.ndarray:
    start, stop = _band_bounds(center_row, filter_width_y, length)
    filter_window = np.zeros(length)
    filter_window[start:stop] = signal.windows.tukey(stop - start, alpha)
    return filter_window


def _shift_band_to_center(filtered_ft: np.ndarray, source_center_row: int) -> np.ndarray:
    length = filtered_ft.shape[0]
    target_center_row = length // 2
    shift = target_center_row - source_center_row
    shifted = np.zeros_like(filtered_ft)

    src_start = max(0, -shift)
    src_stop = min(length, length - shift)
    dst_start = max(0, shift)
    dst_stop = min(length, length + shift)
    shifted[dst_start:dst_stop, :] = filtered_ft[src_start:src_stop, :]
    return shifted


def open_hologram_2d(
    image_complex_in_2d: np.ndarray,
    pad_fact: int = 1,
    alpha: float = 0.3,
    carrier_row: int | None = None,
    filter_width_y: int = 0,
) -> tuple[dict[str, np.ndarray], int, int]:
    signal_in = image_complex_in_2d
    n_x, n_y = signal_in.shape[1], signal_in.shape[0]

    n_x2, n_y2 = n_x, n_y * pad_fact
    signal_pad = np.zeros((n_y2, n_x2), dtype=complex)
    start_y, start_x = (n_y2 - n_y) // 2, (n_x2 - n_x) // 2
    signal_pad[start_y : start_y + n_y, start_x : start_x + n_x] = signal_in

    signal_ft = fftshift(fft2(signal_pad))
    mag_signal_ft = np.abs(signal_ft)
    vertical_profile = _build_vertical_profile(mag_signal_ft)

    if carrier_row is None:
        carrier_row = int(round(_find_vertical_carrier(vertical_profile)))
    if filter_width_y <= 0:
        filter_width_y = _estimate_filter_width(vertical_profile, carrier_row)
    carrier_row, filter_width_y = _validate_vertical_filter(n_y2, int(round(carrier_row)), filter_width_y)

    # Keep only the detected vertical sideband in shifted Fourier space, then
    # translate that band directly to the Fourier center without wraparound.
    sideband_filter = _build_sideband_filter(n_y2, carrier_row, filter_width_y, alpha)[:, np.newaxis]
    filtered_sideband = signal_ft * sideband_filter
    filtered_shift = _shift_band_to_center(filtered_sideband, carrier_row)

    filtered = ifft2(ifftshift(filtered_shift))
    image_complex_out = filtered[start_y : start_y + n_y, start_x : start_x + n_x]
    stage_images = {
        "processed": image_complex_out,
        "mag_signal_ft": mag_signal_ft,
        "filtered_shift": filtered_shift,
    }
    return stage_images, carrier_row, filter_width_y


def process_stack(
    raw_stack: np.ndarray,
    pad_fact: int = 1,
    alpha: float = 0.3,
    carrier_row_override: int | None = None,
    filter_width_override: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if raw_stack.ndim != 3:
        raise ProcessingError(f"Expected raw stack with shape [row, col, harmonic], got {raw_stack.shape}.")

    reference_harmonic = 2
    fft_center_row = (raw_stack.shape[0] * pad_fact) // 2
    if raw_stack.shape[2] <= reference_harmonic:
        raise ProcessingError("The raw stack does not contain harmonic index 2 required for hologram opening.")
    if not _harmonic_is_present(raw_stack, reference_harmonic):
        raise ProcessingError("The raw stack is missing harmonic index 2 required for hologram opening.")

    auto_reference_stages, auto_center_row, auto_filter_width_y = open_hologram_2d(
        raw_stack[:, :, reference_harmonic], pad_fact, alpha
    )
    auto_mag_signal_ft = auto_reference_stages["mag_signal_ft"]
    auto_vertical_profile = _build_vertical_profile(auto_mag_signal_ft)
    zero_order_width_y = _measure_profile_width(auto_vertical_profile, fft_center_row)
    first_order_width_y = _measure_profile_width(auto_vertical_profile, auto_center_row)
    current_center_row = auto_center_row if carrier_row_override is None else int(round(carrier_row_override))
    current_filter_width_y = (
        auto_filter_width_y if filter_width_override is None else int(round(filter_width_override))
    )
    is_manual = carrier_row_override is not None or filter_width_override is not None

    if is_manual:
        reference_stages, current_center_row, current_filter_width_y = open_hologram_2d(
            raw_stack[:, :, reference_harmonic],
            pad_fact,
            alpha,
            carrier_row=current_center_row,
            filter_width_y=current_filter_width_y,
        )
    else:
        reference_stages = auto_reference_stages

    stage_stacks = {"raw": raw_stack.copy()}
    for stage_name, reference_image in reference_stages.items():
        stage_stacks[stage_name] = np.full(
            (reference_image.shape[0], reference_image.shape[1], raw_stack.shape[2]),
            np.nan + 1j * np.nan,
            dtype=complex,
        )

    for stage_name, reference_image in reference_stages.items():
        stage_stacks[stage_name][:, :, reference_harmonic] = reference_image

    for idx in range(raw_stack.shape[2]):
        if idx == reference_harmonic:
            continue
        if not _harmonic_is_present(raw_stack, idx):
            continue
        processed_stages, _, _ = open_hologram_2d(
            raw_stack[:, :, idx],
            pad_fact=pad_fact,
            alpha=alpha,
            carrier_row=current_center_row,
            filter_width_y=current_filter_width_y,
        )
        for stage_name, image in processed_stages.items():
            stage_stacks[stage_name][:, :, idx] = image

    processing_settings = {
        "fft_center_row": fft_center_row,
        "auto_center_row": auto_center_row,
        "auto_filter_width_y": auto_filter_width_y,
        "current_center_row": current_center_row,
        "current_filter_width_y": current_filter_width_y,
        "auto_shift_y": fft_center_row - auto_center_row,
        "current_shift_y": fft_center_row - current_center_row,
        "zero_order_width_y": zero_order_width_y,
        "first_order_width_y": first_order_width_y,
        "is_manual": is_manual,
        "pad_fact": pad_fact,
        "alpha": alpha,
    }
    return stage_stacks, processing_settings


def correct_baseline_slope(z_values: np.ndarray) -> np.ndarray:
    rows, cols = z_values.shape
    x_vals, y_vals = np.meshgrid(np.arange(cols), np.arange(rows))
    matrix_a = np.c_[x_vals.flatten(), y_vals.flatten(), np.ones(rows * cols)]
    matrix_b = z_values.flatten()
    coeffs, _, _, _ = lstsq(matrix_a, matrix_b)
    z_fit = coeffs[0] * x_vals + coeffs[1] * y_vals + coeffs[2]
    return z_values - z_fit


def load_passage(
    folder_path: str,
    passage: str,
    force_rebuild: bool = False,
    carrier_row_override: int | None = None,
    filter_width_override: int | None = None,
) -> LoadedData:
    data, metadata = build_or_load_cache(folder_path, force_rebuild=force_rebuild)
    keys = validate_passage_data(data, passage)
    stage_stacks, processing_settings = process_stack(
        data[keys["o"]],
        carrier_row_override=carrier_row_override,
        filter_width_override=filter_width_override,
    )

    loaded = LoadedData(
        folder_path=metadata["folder_path"],
        image_name=metadata["image_name"],
        passage=passage.lower(),
        z=data[keys["z"]],
        o=data[keys["o"]],
        m=data.get(keys["m"]),
        stage_stacks=stage_stacks,
        processing_settings=processing_settings,
        cache_path=metadata["cache_path"],
        metadata=metadata,
    )
    return loaded


def get_view_image(
    stage_stacks: dict[str, np.ndarray],
    harmonic_index: int,
    stage_name: str,
    representation: str,
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
        phase = np.unwrap(np.angle(image))
        if stage_name == "processed":
            return correct_baseline_slope(phase)
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
    validate_passage_data({PASSAGE_TO_KEYS[loaded.passage]["o"]: loaded.o, PASSAGE_TO_KEYS[loaded.passage]["z"]: loaded.z}, loaded.passage)
    exported_files: list[str] = []

    for harmonic in EXPORT_HARMONICS:
        harmonic_index = harmonic
        views = {
            "raw_amplitude": ("raw", "amplitude", "hot"),
            "raw_phase": ("raw", "phase", "bwr"),
            "processed_amplitude": ("processed", "amplitude", "hot"),
            "processed_phase": ("processed", "phase", "bwr"),
        }
        for name, (stage_name, representation, cmap) in views.items():
            image = get_view_image(loaded.stage_stacks, harmonic_index, stage_name, representation)
            output_path = os.path.join(loaded.folder_path, f"h{harmonic}_{name}.png")
            title = f"H{harmonic} {STAGE_LABELS[stage_name]} {representation.capitalize()} - {loaded.image_name}"
            _export_figure(image, title, output_path, cmap)
            exported_files.append(output_path)

    return exported_files
