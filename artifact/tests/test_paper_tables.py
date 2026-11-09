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
            (directory / "table.md").write_text("original hashed measurement notes")
            (directory / "comparison.csv").write_text("original scientific comparison")
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
            self.assertEqual((directory / "table.md").read_text(), "original hashed measurement notes")
            self.assertEqual((directory / "comparison.csv").read_text(), "original scientific comparison")
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
            self.assertIn('data-view="delta"', html)
            self.assertIn("paper-comparison.csv", html)
            with (directory / "paper-comparison.csv").open() as stream:
                comparisons = list(csv.DictReader(stream))
            ideal = next(row for row in comparisons if "Ideal" in row["row"] and row["column"] == "A100")
            self.assertAlmostEqual(float(ideal["relative_difference_pct"]), (155.59 - 147.22) / 147.22 * 100)
            self.assertTrue((directory / "comparison.md").is_file())
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

    def test_comparison_preserves_missing_zero_reference_and_raw_precision(self):
        panel = {"rows": [["Case", "Metric"], ["raw", "1.00"], ["missing", "—"], ["zero", "2.00"]],
                 "widths": [1, 1], "header_rows": 1, "comparisons": [
                     {"row": r, "col": 1, "paper": old, "current": new, "paper_text": str(old),
                      "current_text": text, "status": "measured" if new is not None else "missing"}
                     for r, old, new, text in ((1, 1.0, 1.004, "1.00"), (2, 2.0, None, "—"), (3, 0.0, 2.0, "2.00"))]}
        document = {"number": 3, "panels": [panel]}
        view = paper_tables.comparison_view(document)
        records = view["comparison_records"]
        self.assertAlmostEqual(records[0]["relative_difference_pct"], .4)
        self.assertIsNone(records[1]["relative_difference_pct"])
        self.assertIsNone(records[2]["relative_difference_pct"])
        self.assertEqual(panel["rows"][1][1], "1.00")
        self.assertEqual(view["panels"][0]["rows"][1][1], "1.0\n1.00\n+0.4%")
        panel["comparisons"][0]["current"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            paper_tables.comparison_view(document)


if __name__ == "__main__":
    unittest.main()
