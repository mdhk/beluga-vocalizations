from beluga_vocalizations.paths import ROOT

from tests.base import TestCaseBase


class TestPaths(TestCaseBase):
    def test_ROOT(self):
        self.assertEqual(str(ROOT.stem), "beluga-vocalizations")
        self.assertPathExists(ROOT)
