# MATLAB Two-Sideband Export

This folder contains a pure MATLAB implementation of the repository's two-sideband hologram opening workflow.

The public entrypoint is:

```matlab
summary = export_two_sideband_holograms(folderPath)
```

Optional processing inputs are also supported:

```matlab
summary = export_two_sideband_holograms(folderPath, 'padFact', 4, 'alphaValue', 0.3)
summary = export_two_sideband_holograms(folderPath, ...
    'padFact', 4, ...
    'alphaValue', 0.3, ...
    'carrierRow', 120, ...
    'filterWidthY', 250)
```

Defaults:

- `padFact = 4`
- `alphaValue = 0.3`
- `carrierRow = []` for automatic carrier detection
- `filterWidthY = []` for automatic width detection

Legacy positional inputs are still accepted for compatibility:

```matlab
summary = export_two_sideband_holograms(folderPath, 4, 0.3)
```

The function scans one SNOM dataset folder, detects the available harmonic pairs for the forward `O` and reverse `R-O` passages, processes every detected harmonic `0..5`, and writes one `.mat` file per harmonic into:

```text
<folderPath>/matlab-two-sideband-export/
```

## Expected Folder Layout

The input folder name must match the dataset base name used by the existing `.gsf` files. For a folder named:

```text
/path/to/2026-03-24 171434 AFM 2.52THz_B
```

the MATLAB exporter expects files like:

```text
2026-03-24 171434 AFM 2.52THz_B O0A raw.gsf
2026-03-24 171434 AFM 2.52THz_B O0P raw.gsf
2026-03-24 171434 AFM 2.52THz_B O2A raw.gsf
2026-03-24 171434 AFM 2.52THz_B O2P raw.gsf
2026-03-24 171434 AFM 2.52THz_B R-O3A raw.gsf
2026-03-24 171434 AFM 2.52THz_B R-O3P raw.gsf
```

Forward files use the `O` prefix and reverse files use the `R-O` prefix.

## Usage

From MATLAB, either change into this `matlab/` folder or add it to the MATLAB path, then call:

```matlab
folderPath = '/path/to/2026-03-24 171434 AFM 2.52THz_B';
summary = export_two_sideband_holograms(folderPath);
summary_custom = export_two_sideband_holograms(folderPath, ...
    'padFact', 4, ...
    'alphaValue', 0.3, ...
    'carrierRow', 120, ...
    'filterWidthY', 250);
```

## MATLAB Requirements

- MATLAB R2017a+
- Signal Processing Toolbox, because the vertical FFT carrier detection uses `findpeaks`

`summary` is a struct array with one element per exported passage. Each element reports:

- the passage name
- the detected harmonics
- the skipped harmonics
- the shared carrier row and filter width reused across that passage
- the list of written `.mat` files

## Output Files

The exporter writes one file per passage and harmonic. Example names:

```text
forward_h0_two_sideband.mat
forward_h2_two_sideband.mat
reverse_h3_two_sideband.mat
```

Each `.mat` file contains MATLAB-native scalars and arrays only:

- `image_name`
- `folder_path`
- `passage`
- `harmonic_index`
- `processing_mode`
- `raw_hologram`
- `processed_hologram`
- `raw_amplitude`
- `raw_phase`
- `processed_amplitude`
- `processed_phase`
- `carrier_row`
- `filter_width_y`
- `fft_center_row`
- `mirror_row`
- `rotation_angle_rad`
- `rotation_angle_deg`
- `pad_fact`
- `alpha`

## Processing Notes

This MATLAB implementation mirrors the Python logic in [`hologram_opening.py`](../hologram_opening.py) and [`processing.py`](../processing.py) for two-sideband processing only.

Behavior that intentionally matches the Python app:

- only two-sideband processing is implemented
- harmonic 2 is the reference harmonic used for automatic carrier detection and filter-width estimation
- when `carrierRow` and `filterWidthY` are not provided, the detected `carrier_row` and `filter_width_y` from harmonic 2 are reused for all other harmonics in the same passage
- when `carrierRow` and/or `filterWidthY` are provided, those manual values are applied to the reference harmonic and then reused for all other harmonics in the same passage
- each harmonic still stores its own `mirror_row` and rotation-angle diagnostics
- manual `filterWidthY` is only limited by the FFT/image height normalization; it may overlap the zero order or other bands by user choice
- manual `carrierRow` is only constrained by Fourier-space image bounds needed for valid indexing

## Failure Cases

The function raises an error when:

- the input folder does not exist
- a passage contains an incomplete amplitude/phase pair for any harmonic
- a detected passage is missing harmonic 2
- `.gsf` headers do not contain the expected dimension metadata
- automatic detection cannot find a usable sideband away from the zero order
- a manual or automatic `carrierRow` falls outside the Fourier-space image bounds

If a passage is completely absent, it is skipped without error.
