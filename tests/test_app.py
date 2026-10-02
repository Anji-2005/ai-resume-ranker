import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitSmokeTest(unittest.TestCase):
    def test_app_renders_without_loading_model(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=20).run()
        self.assertEqual(app.exception, [])
        self.assertGreaterEqual(len(app.title), 1)
        self.assertIn("Resume Review Assistant", app.title[0].value)


if __name__ == "__main__":
    unittest.main()
