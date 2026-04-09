"""Reusable hologram opening utilities for SNOM/synthetic holography workflows.

This module is independent from the GUI and data-loading code so it can be
imported directly in scripts and notebooks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
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


@dataclass(frozen=True)
class VerticalCropBounds:
    start_y: int
    stop_y: int


@dataclass(frozen=True)
class HologramGeometry:
    pad_fact: int
    fft_height: int
    image_width: int
    crop_y: VerticalCropBounds
    carrier_row: int
    mirror_row: int | None
    filter_width_y: int
    fft_center_row: int

    @property
    def fft_shape(self) -> tuple[int, int]:
        return (self.fft_height, self.image_width)


@dataclass(frozen=True)
class FourierAnalysis:
    signal_padded: np.ndarray
    signal_ft: np.ndarray
    magnitude_ft: np.ndarray
    vertical_profile: np.ndarray


@dataclass(frozen=True)
class HologramViewStages:
    raw: np.ndarray
    processed: np.ndarray
    mag_signal_ft: np.ndarray
    filtered_shift: np.ndarray


@dataclass(frozen=True)
class HologramReconstruction:
    processing_mode: str
    rotation_angle_rad: float
    geometry: HologramGeometry
    analysis: FourierAnalysis
    processed_full: np.ndarray
    processed_cropped: np.ndarray
    filtered_shift: np.ndarray
    view_stages: HologramViewStages

    @property
    def diagnostics(self) -> dict[str, float | int]:
        mirror_row = -1 if self.geometry.mirror_row is None else int(self.geometry.mirror_row)
        return {
            "rotation_angle_rad": self.rotation_angle_rad,
            "rotation_angle_deg": float(np.degrees(self.rotation_angle_rad))
            if np.isfinite(self.rotation_angle_rad)
            else float("nan"),
            "mirror_row": mirror_row,
        }


def tukey_filter_func(width: int, length: int, alpha: float) -> np.ndarray:
    width = max(2, min(int(round(width)), length))
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
    carrier_row_int = int(round(carrier_row))
    distance_to_zero_order = abs(carrier_row_int - center_idx)
    distance_to_edge = min(carrier_row_int, profile.size - carrier_row_int - 1)
    second_order_cap = carrier_row_int - abs(center_idx - 2 * carrier_row_int)
    width = int(min(distance_to_zero_order, distance_to_edge, second_order_cap))
    if width < 2:
        raise ProcessingError(
            "Detected carrier is too close to the edge or zero order to build a stable filter without second-order aliasing."
        )
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


def prepare_signal_for_mode(image_complex_in_2d: np.ndarray, processing_mode: str) -> tuple[np.ndarray, float]:
    if processing_mode not in PROCESSING_MODES:
        raise ProcessingError(f"Unknown processing mode: {processing_mode}")

    if processing_mode == "two_sideband":
        rotated_complex, rotation_angle = _rotate_to_real_axis(image_complex_in_2d)
        return np.real(rotated_complex), rotation_angle

    return image_complex_in_2d, float("nan")


def pad_vertical_and_fft(signal_in: np.ndarray, pad_fact: int) -> tuple[FourierAnalysis, VerticalCropBounds]:
    if pad_fact <= 0:
        raise ProcessingError("Padding factor must be a positive integer.")

    n_y, image_width = signal_in.shape
    fft_height = n_y * pad_fact
    signal_pad = np.zeros((fft_height, image_width), dtype=signal_in.dtype)
    start_y = (fft_height - n_y) // 2
    stop_y = start_y + n_y
    signal_pad[start_y:stop_y, :] = signal_in

    signal_ft = fftshift(fft2(signal_pad))
    magnitude_ft = np.abs(signal_ft)
    vertical_profile = build_vertical_profile(magnitude_ft)
    analysis = FourierAnalysis(
        signal_padded=signal_pad,
        signal_ft=signal_ft,
        magnitude_ft=magnitude_ft,
        vertical_profile=vertical_profile,
    )
    return analysis, VerticalCropBounds(start_y=start_y, stop_y=stop_y)


def resolve_filter_geometry(
    vertical_profile: np.ndarray,
    pad_fact: int,
    fft_height: int,
    image_width: int,
    crop_y: VerticalCropBounds,
    carrier_row_override: int | None = None,
    filter_width_override: int | None = None,
    processing_mode: str = "two_sideband",
) -> HologramGeometry:
    fft_center_row = fft_height // 2

    carrier_row = int(round(_find_vertical_carrier(vertical_profile))) if carrier_row_override is None else int(
        round(carrier_row_override)
    )
    filter_width_y = (
        _estimate_filter_width(vertical_profile, carrier_row)
        if filter_width_override is None or filter_width_override <= 0
        else int(round(filter_width_override))
    )
    filter_width_y = _normalize_filter_width(filter_width_y, fft_height)

    exclusion_half_width = max(8, fft_height // 32)
    if carrier_row >= fft_center_row - exclusion_half_width:
        raise ProcessingError("Carrier center must stay above the zero-order exclusion band.")
    if carrier_row < 1 or carrier_row >= fft_height - 1:
        raise ProcessingError("Carrier center must stay inside the Fourier-space image bounds.")

    start, stop = _band_bounds(carrier_row, filter_width_y, fft_height)

    mirror_row = None
    if processing_mode == "two_sideband":
        mirror_row = 2 * fft_center_row - carrier_row
        if mirror_row < 1 or mirror_row >= fft_height - 1:
            raise ProcessingError("Mirrored carrier falls outside the Fourier-space image bounds.")

    return HologramGeometry(
        pad_fact=pad_fact,
        fft_height=fft_height,
        image_width=image_width,
        crop_y=crop_y,
        carrier_row=carrier_row,
        mirror_row=mirror_row,
        filter_width_y=filter_width_y,
        fft_center_row=fft_center_row,
    )


def analyze_vertical_spectrum(
    analysis: FourierAnalysis,
    pad_fact: int,
    crop_y: VerticalCropBounds,
    carrier_row_override: int | None = None,
    filter_width_override: int | None = None,
    processing_mode: str = "two_sideband",
) -> HologramGeometry:
    return resolve_filter_geometry(
        analysis.vertical_profile,
        pad_fact=pad_fact,
        fft_height=analysis.signal_ft.shape[0],
        image_width=analysis.signal_ft.shape[1],
        crop_y=crop_y,
        carrier_row_override=carrier_row_override,
        filter_width_override=filter_width_override,
        processing_mode=processing_mode,
    )


def _build_band_mask(length: int, center_row: int, filter_width_y: int, alpha: float) -> np.ndarray:
    start, stop = _band_bounds(center_row, filter_width_y, length)
    filter_window = np.zeros(length)
    filter_window[start:stop] = signal.windows.tukey(stop - start, alpha)
    return filter_window[:, np.newaxis]


def _shift_rows_to_center(filtered_ft: np.ndarray, source_center_row: int, target_center_row: int) -> np.ndarray:
    length = filtered_ft.shape[0]
    shift = target_center_row - source_center_row
    shifted = np.zeros_like(filtered_ft)

    src_start = max(0, -shift)
    src_stop = min(length, length - shift)
    dst_start = max(0, shift)
    dst_stop = min(length, length + shift)
    shifted[dst_start:dst_stop, :] = filtered_ft[src_start:src_stop, :]
    return shifted


def reconstruct_from_sidebands(
    analysis: FourierAnalysis,
    geometry: HologramGeometry,
    processing_mode: str,
    alpha: float,
    rotation_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fft_height = geometry.fft_height
    fft_center_row = geometry.fft_center_row
    crop_y = geometry.crop_y

    positive_mask = _build_band_mask(fft_height, geometry.carrier_row, geometry.filter_width_y, alpha)
    filtered_positive = analysis.signal_ft * positive_mask
    centered_positive = _shift_rows_to_center(filtered_positive, geometry.carrier_row, fft_center_row)

    if processing_mode == "two_sideband":
        if geometry.mirror_row is None:
            raise ProcessingError("Two-sideband processing requires a mirrored carrier row.")
        negative_mask = _build_band_mask(fft_height, geometry.mirror_row, geometry.filter_width_y, alpha)
        filtered_negative = analysis.signal_ft * negative_mask
        centered_negative = _shift_rows_to_center(filtered_negative, geometry.mirror_row, fft_center_row)

        field_positive = ifft2(ifftshift(centered_positive))
        field_negative = ifft2(ifftshift(centered_negative))
        processed_full_rotated = 0.5 * (np.conj(field_positive) + field_negative)
        processed_full = processed_full_rotated * np.exp(1j * rotation_angle_rad)
        filtered_shift = fftshift(fft2(processed_full))
    else:
        processed_full = ifft2(ifftshift(centered_positive))
        filtered_shift = centered_positive

    processed_cropped = processed_full[crop_y.start_y : crop_y.stop_y, :]
    return processed_full, processed_cropped, filtered_shift


def build_view_stages(raw_image: np.ndarray, reconstruction: HologramReconstruction) -> HologramViewStages:
    return HologramViewStages(
        raw=raw_image,
        processed=reconstruction.processed_cropped,
        mag_signal_ft=reconstruction.analysis.magnitude_ft,
        filtered_shift=reconstruction.filtered_shift,
    )


def reconstruct_hologram(
    image_complex_in_2d: np.ndarray,
    pad_fact: int = 4,
    alpha: float = 0.3,
    carrier_row: int | None = None,
    filter_width_y: int | None = None,
    processing_mode: str = "two_sideband",
) -> HologramReconstruction:
    signal_in, rotation_angle_rad = prepare_signal_for_mode(image_complex_in_2d, processing_mode)
    analysis, crop_y = pad_vertical_and_fft(signal_in, pad_fact)
    geometry = analyze_vertical_spectrum(
        analysis,
        pad_fact=pad_fact,
        crop_y=crop_y,
        carrier_row_override=carrier_row,
        filter_width_override=filter_width_y,
        processing_mode=processing_mode,
    )
    processed_full, processed_cropped, filtered_shift = reconstruct_from_sidebands(
        analysis,
        geometry,
        processing_mode=processing_mode,
        alpha=alpha,
        rotation_angle_rad=rotation_angle_rad,
    )
    reconstruction = HologramReconstruction(
        processing_mode=processing_mode,
        rotation_angle_rad=rotation_angle_rad,
        geometry=geometry,
        analysis=analysis,
        processed_full=processed_full,
        processed_cropped=processed_cropped,
        filtered_shift=filtered_shift,
        view_stages=HologramViewStages(
            raw=image_complex_in_2d,
            processed=processed_cropped,
            mag_signal_ft=analysis.magnitude_ft,
            filtered_shift=filtered_shift,
        ),
    )
    return reconstruction


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
