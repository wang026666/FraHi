"""CASMI label compatibility without importing the model or chemistry stack."""

import unittest

from Prepare_scripts.input_utils import normalize_casmi_ion_mode


class CasmiIonModeTests(unittest.TestCase):
    def test_standard_labels(self):
        self.assertEqual(normalize_casmi_ion_mode("POSITIVE"), "P")
        self.assertEqual(normalize_casmi_ion_mode("NEGATIVE"), "N")

    def test_historical_leading_space_labels(self):
        self.assertEqual(normalize_casmi_ion_mode(" POSITIVE"), "P")
        self.assertEqual(normalize_casmi_ion_mode(" NEGATIVE"), "N")

    def test_case_and_surrounding_whitespace(self):
        for value, expected in ((" positive ", "P"), ("\tNegative\n", "N")):
            with self.subTest(value=value):
                self.assertEqual(normalize_casmi_ion_mode(value), expected)

    def test_missing_or_unknown_labels_fail_explicitly(self):
        for value in (None, float("nan"), 1, "", " ", "UNKNOWN", "P"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "CASMI ION_MODE"):
                    normalize_casmi_ion_mode(value)


if __name__ == "__main__":
    unittest.main()
