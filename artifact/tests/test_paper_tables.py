"""Exercise real offline exports, their evidence, and text fit at output DPI."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artifact import paper_tables


class PaperTableRendererTests(unittest.TestCase):
    def test_real_exports_preserve_csv_and_fit_notes_inside_final_png(self):
        from matplotlib.figure import Figure
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "table.csv"
            source.write_text("gpu,status,hardware_peak_tflops,ideal_mma_status,ideal_mma_tflops,unit_tflops\n"
                              "A100,complete,156,measured,155.59,112\n")
            original = source.read_bytes()
            savefig = Figure.savefig
            checked = []

            def capture(figure, path, *args, **kwargs):
                if Path(path).suffix == ".png":
                    self.assertEqual(figure.dpi, kwargs["dpi"])
                    figure.canvas.draw()
                    for axis in figure.axes:
                        for text in axis.texts:
                            box = text.get_window_extent(figure.canvas.get_renderer())
                            self.assertGreaterEqual(box.x0, 0, text.get_text())
                            self.assertLessEqual(box.x1, figure.bbox.x1, text.get_text())
                            self.assertGreaterEqual(box.y0, 0, text.get_text())
                            self.assertLessEqual(box.y1, figure.bbox.y1, text.get_text())
                            checked.append(text.get_text())
                return savefig(figure, path, *args, **kwargs)

            with patch.object(Figure, "savefig", capture):
                state = paper_tables.render("table4", directory)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(state["new_gpu_measurements"])
            self.assertEqual(state["source_csv_sha256"], hashlib.sha256(original).hexdigest())
            self.assertTrue((directory / "table.pdf").read_bytes().startswith(b"%PDF-"))
            self.assertTrue((directory / "table.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            for name, digest in state["outputs"].items():
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(json.loads((directory / "table-render.json").read_text()), state)
            html = (directory / "table.html").read_text()
            self.assertIn("Hardware Peak FLOPS", html)
            self.assertIn("155.59", html)
            self.assertIn("—", html)
            self.assertTrue(any("Hardware peaks are published" in text for text in checked))

    def test_failed_export_does_not_replace_prior_presentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "table.csv").write_text("gpu\n")
            for name in paper_tables.OUTPUTS:
                (directory / name).write_bytes(b"previous output")
            with patch.object(paper_tables, "_write_html", side_effect=OSError("disk full")), \
                 self.assertRaisesRegex(OSError, "disk full"):
                paper_tables.render("table4", directory)
            for name in paper_tables.OUTPUTS:
                self.assertEqual((directory / name).read_bytes(), b"previous output")

    def test_html_math_is_local_and_escaped(self):
        self.assertEqual(paper_tables._html_text(r"$1.23\times 10^{-4}$"), "1.23× 10<sup>-4</sup>")
        self.assertEqual(paper_tables._html_text(r"$N\log_2 N$"), "Nlog<sub>2</sub> N")
        self.assertEqual(paper_tables._html_text("<script>&"), "&lt;script&gt;&amp;")


if __name__ == "__main__":
    unittest.main()
