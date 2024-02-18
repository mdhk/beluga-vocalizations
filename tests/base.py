import pathlib
import unittest


class TestCaseBase(unittest.TestCase):
    def assertPathExists(self, path):
        if not pathlib.Path(path).exists():
            raise AssertionError(f"File or directory does not exist: {str(path)}")
