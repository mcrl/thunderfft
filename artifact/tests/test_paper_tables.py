"""Exercise real offline exports, their evidence, and text fit at output DPI."""
import csv
import hashlib
from html.parser import HTMLParser
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
        self.assertEqual(view["panels"][0]["rows"][2][1:], ["1.0", "1.00", "+0.4%"])
        self.assertEqual(view["panels"][0]["rows"][1][1:], ["Paper", "New", "Δ%"])
        self.assertEqual(records[0], {"table": 3, "panel": "", "row": "raw", "column": "Metric",
                         "paper_value": 1.0, "new_value": 1.004,
                         "paper_display": "1.0", "new_display": "1.00",
                         "relative_difference_pct": (1.004 - 1.0) * 100,
                         "relative_error_pct": (1.004 - 1.0) * 100, "status": "measured"})
        panel["comparisons"][0]["current"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            paper_tables.comparison_view(document)

    def empty_view(self, evaluation):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        (directory / "table.csv").write_text("unused\n")
        return paper_tables.comparison_view(paper_tables.build(evaluation, directory))

    def test_table2_repeats_numeric_rowspans_without_merging_triplet_values(self):
        view = self.empty_view("table2")
        panel = view["panels"][0]
        self.assertEqual(len(panel["widths"]), 18)
        self.assertEqual(panel["spacer_columns"], [4, 10, 14])
        self.assertEqual(panel["rows"][0][5:7], ["Design", "MMA Shape"])
        self.assertEqual(panel["align"], {5: "left"})
        cells = {(r, c): (nr, nc, text) for r, c, nr, nc, text in paper_tables._cells(panel)}
        for row, rowspan in ((2, 2), (4, 3)):
            for col in (1, 2, 3):
                self.assertEqual(cells[row, col][:2], (rowspan, 1))
                for covered_row in range(row + 1, row + rowspan):
                    self.assertNotIn((covered_row, col), cells)
        self.assertEqual(cells[0, 0][:2], (2, 1))
        self.assertEqual(cells[0, 1][:2], (1, 3))
        self.assertTrue(all(not row[col] for row in panel["rows"] for col in panel["spacer_columns"]))

    def test_table3_preserves_header_hierarchy_empty_average_cells_and_rule_coordinates(self):
        view = self.empty_view("table3")
        panel = view["panels"][0]
        self.assertEqual(len(panel["widths"]), 72)
        self.assertEqual(len(panel["spacer_columns"]), 17)
        self.assertEqual(panel["header_rows"], 4)
        self.assertGreater(view["width"], 30)
        self.assertEqual(view["font_size"], 8.5)
        cells = {(r, c): (nr, nc, text) for r, c, nr, nc, text in paper_tables._cells(panel)}
        self.assertEqual(cells[0, 0], (4, 1, "N"))
        self.assertEqual(cells[0, 1], (1, 35, "FP16"))
        self.assertEqual(cells[0, 37], (1, 35, "TF32 / FP32"))
        self.assertEqual(cells[1, 1], (1, 11, "cuFFTDx"))
        for col, gpu in ((1, "A100"), (5, "H100"), (9, "B200")):
            self.assertEqual(cells[2, col], (1, 3, gpu))
            self.assertEqual(panel["rows"][3][col:col + 3], ["Paper", "New", "Δ%"])
        for old_col in (1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15):
            for col in panel["original_columns"][old_col]:
                self.assertEqual(panel["rows"][-1][col], "")
        for col in panel["spacer_columns"]:
            self.assertTrue(all(not row[col] for row in panel["rows"]))
        self.assertIn({"col": 5, "row": 2.1, "end": 10.9, "weight": .35}, panel["vrules"])
        self.assertTrue(any(rule["row"] == 4 and rule["end"] == 72 for rule in panel["rules"]))
        self.assertTrue(any(rule["row"] == 11 and rule["end"] == 72 for rule in panel["rules"]))
        self.assertEqual(len(view["comparison_records"]), 264)
        self.assertEqual(view["comparison_records"][0]["column"], "FP16 / cuFFTDx / A100")

    def test_table5_grouped_headers_and_scientific_values_keep_separate_columns(self):
        view = self.empty_view("table5")
        self.assertEqual(len(view["panels"]), 2)
        for panel in view["panels"]:
            self.assertEqual(len(panel["widths"]), 16)
            cells = {(r, c): (nr, nc, text) for r, c, nr, nc, text in paper_tables._cells(panel)}
            self.assertEqual(cells[0, 1], (1, 7, "FP16"))
            self.assertEqual(cells[1, 1], (1, 3, "xFFT"))
            self.assertEqual(cells[1, 0], (2, 1, "$N$"))
        self.assertEqual(view["panels"][0]["rows"][3][1:4],
                         [r"$4.34\times 10^{-4}$", "—", "—"])

    def test_html_view_switches_have_complete_grids_and_individual_value_cells(self):
        class Tables(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tables, self.sections = [], []
                self.current = None
                self.cell = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "section":
                    self.sections.append(attrs)
                elif tag == "table":
                    self.current = {"columns": 0, "rows": []}
                    self.tables.append(self.current)
                elif tag == "col" and self.current is not None:
                    self.current["columns"] += 1
                elif tag == "tr":
                    self.current["rows"].append([])
                elif tag in ("td", "th"):
                    self.cell = [int(attrs["rowspan"]), int(attrs["colspan"]), attrs.get("class"), ""]
                    self.current["rows"][-1].append(self.cell)

            def handle_data(self, data):
                if self.cell is not None:
                    self.cell[3] += data

            def handle_endtag(self, tag):
                if tag in ("td", "th"):
                    self.cell = None
                elif tag == "table":
                    self.current = None

        view = self.empty_view("table3")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.html"
            paper_tables._write_html(view, path)
            text = path.read_text()
        parser = Tables()
        parser.feed(text)
        self.assertEqual([section["data-table-view"] for section in parser.sections],
                         ["compare", "paper", "new", "delta"])
        self.assertNotIn("hidden", parser.sections[0])
        self.assertTrue(all("hidden" in section for section in parser.sections[1:]))
        self.assertEqual([table["columns"] for table in parser.tables], [72, 72, 19, 19, 19, 19, 19, 19])
        for table in parser.tables:
            occupied = set()
            for row, cells in enumerate(table["rows"]):
                col = 0
                for nr, nc, role, value in cells:
                    while (row, col) in occupied:
                        col += 1
                    area = {(r, c) for r in range(row, row + nr) for c in range(col, col + nc)}
                    self.assertFalse(area & occupied)
                    occupied.update(area)
                    if role in ("paper-value", "new-value", "delta-value", "spacer"):
                        self.assertEqual(nc, 1)
                        self.assertNotIn("\n", value)
                    if role == "spacer":
                        self.assertEqual(value, "")
                    col += nc
                self.assertEqual({c for r, c in occupied if r == row}, set(range(table["columns"])))
            self.assertEqual(len(occupied), len(table["rows"]) * table["columns"])
        self.assertIn("overflow-x:auto", text)
        self.assertIn("section.hidden=section.dataset.tableView!==button.dataset.view", text)


if __name__ == "__main__":
    unittest.main()
