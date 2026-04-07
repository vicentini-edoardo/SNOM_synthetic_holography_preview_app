import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MATLAB_EXPORT_FILE = REPO_ROOT / "matlab" / "export_two_sideband_holograms.m"
MATLAB_README_FILE = REPO_ROOT / "matlab" / "README.md"


class MatlabExportContractTests(unittest.TestCase):
    def test_matlab_entrypoint_exists_with_expected_signature(self) -> None:
        self.assertTrue(MATLAB_EXPORT_FILE.exists(), "Expected MATLAB export entrypoint to exist.")
        content = MATLAB_EXPORT_FILE.read_text(encoding="utf-8")
        self.assertIn("function summary = export_two_sideband_holograms(folderPath, padFact, alphaValue)", content)
        self.assertIn("matlab-two-sideband-export", content)
        self.assertIn("two_sideband", content)

    def test_matlab_entrypoint_mentions_expected_export_variables(self) -> None:
        self.assertTrue(MATLAB_EXPORT_FILE.exists(), "Expected MATLAB export entrypoint to exist.")
        content = MATLAB_EXPORT_FILE.read_text(encoding="utf-8")
        for required_name in (
            "raw_hologram",
            "processed_hologram",
            "raw_amplitude",
            "raw_phase",
            "processed_amplitude",
            "processed_phase",
            "carrier_row",
            "filter_width_y",
            "fft_center_row",
            "mirror_row",
            "rotation_angle_rad",
            "rotation_angle_deg",
            "pad_fact",
            "alpha",
        ):
            with self.subTest(required_name=required_name):
                self.assertIn(required_name, content)

    def test_matlab_entrypoint_uses_findpeaks_for_vertical_carrier_detection(self) -> None:
        self.assertTrue(MATLAB_EXPORT_FILE.exists(), "Expected MATLAB export entrypoint to exist.")
        content = MATLAB_EXPORT_FILE.read_text(encoding="utf-8")
        self.assertIn("findpeaks(", content)

    def test_matlab_entrypoint_uses_one_based_fft_row_coordinates(self) -> None:
        self.assertTrue(MATLAB_EXPORT_FILE.exists(), "Expected MATLAB export entrypoint to exist.")
        content = MATLAB_EXPORT_FILE.read_text(encoding="utf-8")
        self.assertIn("'fft_center_row', floor((size(rawStack, 1) * padFact) / 2) + 1", content)
        self.assertIn("mirrorRow = nY2 - carrierRow + 1;", content)
        self.assertNotIn("refine_peak_subpixel(profile, bestPeak - 1)", content)
        self.assertNotIn("refine_peak_subpixel(profile, idx - 1)", content)

    def test_matlab_readme_documents_usage_and_outputs(self) -> None:
        self.assertTrue(MATLAB_README_FILE.exists(), "Expected dedicated MATLAB README to exist.")
        content = MATLAB_README_FILE.read_text(encoding="utf-8")
        self.assertIn("export_two_sideband_holograms(folderPath)", content)
        self.assertIn("matlab-two-sideband-export", content)
        self.assertIn("forward_h0_two_sideband.mat", content)
        self.assertIn("reverse_h3_two_sideband.mat", content)
        self.assertIn("harmonic 2", content)
        self.assertIn("Signal Processing Toolbox", content)


if __name__ == "__main__":
    unittest.main()
