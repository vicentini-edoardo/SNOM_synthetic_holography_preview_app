"""Reusable hologram opening utilities for SNOM/synthetic holography workflows.

This module is independent from the GUI and data-loading code so it can be
imported directly in scripts and notebooks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal, stats
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.linalg import lstsq


PROCESSING_MODES = {
    "one_sideband": "One Sideband",
    "two_sideband": "Two Sidebands",
}


class ProcessingError(RuntimeError):
    pass


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


def build_vertical_profile(magnitude_ft: np.ndarray) -> np.ndarray:
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


def measure_profile_width(profile: np.ndarray, row_idx: int) -> float:
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


def _data_angle(x: np.ndarray, y: np.ndarray | None = None) -> float:
    if y is None:
        xx = np.real(x).ravel()
        yy = np.imag(x).ravel()
    else:
        xx = np.asarray(x).ravel()
        yy = np.asarray(y).ravel()

    valid = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[valid]
    yy = yy[valid]
    if xx.size < 2:
        raise ProcessingError("Need at least two finite samples to estimate the complex-plane angle.")
    if np.min(xx) == np.max(xx):
        return float(np.pi / 2)
    if np.min(yy) == np.max(yy):
        return 0.0

    regress_hor = stats.linregress(xx, yy)
    regress_ver = stats.linregress(yy, xx)
    if regress_hor.stderr < regress_ver.stderr:
        return float(np.arctan(regress_hor.slope))
    return float(np.arctan(1.0 / regress_ver.slope))


def _rotate_to_real_axis(image_complex_in_2d: np.ndarray) -> tuple[np.ndarray, float]:
    angle = _data_angle(image_complex_in_2d)
    rotated = image_complex_in_2d * np.exp(-1j * angle)
    return rotated, angle


def open_hologram_2d(
    image_complex_in_2d: np.ndarray,
    pad_fact: int = 1,
    alpha: float = 0.3,
    carrier_row: int | None = None,
    filter_width_y: int | None = 0,
    processing_mode: str = "two_sideband",
) -> tuple[dict[str, np.ndarray], int, int, dict[str, float | int]]:
    if processing_mode not in PROCESSING_MODES:
        raise ProcessingError(f"Unknown processing mode: {processing_mode}")

    rotation_angle = float("nan")
    if processing_mode == "two_sideband":
        rotated_complex, rotation_angle = _rotate_to_real_axis(image_complex_in_2d)
        signal_in = np.real(rotated_complex)
        signal_dtype = float
    else:
        signal_in = image_complex_in_2d
        signal_dtype = complex

    n_x, n_y = signal_in.shape[1], signal_in.shape[0]

    n_x2, n_y2 = n_x, n_y * pad_fact
    signal_pad = np.zeros((n_y2, n_x2), dtype=signal_dtype)
    start_y, start_x = (n_y2 - n_y) // 2, (n_x2 - n_x) // 2
    signal_pad[start_y : start_y + n_y, start_x : start_x + n_x] = signal_in

    signal_ft = fftshift(fft2(signal_pad))
    mag_signal_ft = np.abs(signal_ft)
    vertical_profile = build_vertical_profile(mag_signal_ft)

    if carrier_row is None:
        carrier_row = int(round(_find_vertical_carrier(vertical_profile)))
    if filter_width_y is None or filter_width_y <= 0:
        filter_width_y = _estimate_filter_width(vertical_profile, carrier_row)
    carrier_row, filter_width_y = _validate_vertical_filter(n_y2, int(round(carrier_row)), filter_width_y)

    if processing_mode == "two_sideband":
        mirror_row = n_y2 - carrier_row
        if mirror_row >= n_y2:
            mirror_row = n_y2 - 1

        sideband_filter_pos = _build_sideband_filter(n_y2, carrier_row, filter_width_y, alpha)[:, np.newaxis]
        sideband_filter_neg = _build_sideband_filter(n_y2, mirror_row, filter_width_y, alpha)[:, np.newaxis]
        filtered_pos = signal_ft * sideband_filter_pos
        filtered_neg = signal_ft * sideband_filter_neg
        shifted_pos = _shift_band_to_center(filtered_pos, carrier_row)
        shifted_neg = _shift_band_to_center(filtered_neg, mirror_row)

        field_pos_full = ifft2(ifftshift(shifted_pos))
        field_neg_full = ifft2(ifftshift(shifted_neg))
        filtered_full_rotated = 0.5 * (np.conj(field_pos_full) + field_neg_full)
        filtered_full = filtered_full_rotated * np.exp(1j * rotation_angle)
        filtered_shift = fftshift(fft2(filtered_full))
        filtered = filtered_full
    else:
        sideband_filter = _build_sideband_filter(n_y2, carrier_row, filter_width_y, alpha)[:, np.newaxis]
        filtered_sideband = signal_ft * sideband_filter
        filtered_shift = _shift_band_to_center(filtered_sideband, carrier_row)
        filtered = ifft2(ifftshift(filtered_shift))
        mirror_row = -1

    image_complex_out = filtered[start_y : start_y + n_y, start_x : start_x + n_x]
    stage_images = {
        "processed": image_complex_out,
        "mag_signal_ft": mag_signal_ft,
        "filtered_shift": filtered_shift,
    }
    diagnostics = {
        "rotation_angle_rad": rotation_angle,
        "rotation_angle_deg": float(np.degrees(rotation_angle)) if np.isfinite(rotation_angle) else float("nan"),
        "mirror_row": int(mirror_row),
    }
    return stage_images, carrier_row, filter_width_y, diagnostics


def correct_baseline_slope(z_values: np.ndarray) -> np.ndarray:
    rows, cols = z_values.shape
    x_vals, y_vals = np.meshgrid(np.arange(cols), np.arange(rows))
    edge_y = max(1, rows // 20)
    edge_x = max(1, cols // 20)
    fit_mask = np.ones((rows, cols), dtype=bool)
    if rows > 2 * edge_y and cols > 2 * edge_x:
        fit_mask[:edge_y, :] = False
        fit_mask[-edge_y:, :] = False
        fit_mask[:, :edge_x] = False
        fit_mask[:, -edge_x:] = False
    fit_mask &= np.isfinite(z_values)
    if np.count_nonzero(fit_mask) < 3:
        fit_mask = np.isfinite(z_values)

    matrix_a = np.c_[x_vals[fit_mask], y_vals[fit_mask], np.ones(np.count_nonzero(fit_mask))]
    matrix_b = z_values[fit_mask]
    coeffs, _, _, _ = lstsq(matrix_a, matrix_b)
    z_fit = coeffs[0] * x_vals + coeffs[1] * y_vals + coeffs[2]
    return z_values - z_fit


def processed_phase(image: np.ndarray, processing_mode: str = "two_sideband") -> np.ndarray:
    if processing_mode not in PROCESSING_MODES:
        raise ProcessingError(f"Unknown processing mode: {processing_mode}")
    phase_sign = -1.0 if processing_mode == "one_sideband" else 1.0
    phase = np.unwrap(phase_sign * np.angle(image), axis=0)
    return correct_baseline_slope(phase)
