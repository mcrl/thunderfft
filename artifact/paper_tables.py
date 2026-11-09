#!/usr/bin/env python3
"""Render Tables 2–7 with paper/new values in the paper's layout, without GPU work.

Usage: python3 -m artifact.paper_tables ASSEMBLED_DIRECTORY [MORE_DIRECTORIES ...]
An individual table result directory is also accepted. Source CSVs are never edited.
"""

import argparse
import copy
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import tempfile


TABLES = tuple(f"table{n}" for n in range(2, 8))
OUTPUTS = ("table.pdf", "table.png", "table.html", "comparison.md", "paper-comparison.csv")
PAPER_SHA256 = "02ab4855e16b721789c35b80a4b2488bc1e7849fa6da3f12f0bae861bb109fd9"
COMPARISON_FIELDS = ("table", "panel", "row", "column", "paper_value", "new_value",
                     "paper_display", "new_display", "relative_difference_pct",
                     "relative_error_pct", "status")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(evaluation, directory):
    if evaluation not in TABLES:
        raise ValueError(f"no paper table layout for {evaluation!r}")
    if evaluation in TABLES[:3]:
        from artifact.paper_table_performance import build as layout
    else:
        from artifact.paper_table_accuracy import build as layout
    return layout(evaluation, Path(directory))


def _cells(panel):
    """Return visible cells, validating merges before either output is drawn."""
    rows, widths = panel["rows"], panel["widths"]
    if not rows or not widths or any(w <= 0 for w in widths):
        raise ValueError("paper table must have rows and positive column widths")
    if any(len(row) != len(widths) for row in rows):
        raise ValueError("paper table row width does not match its columns")
    merged, covered = {}, set()
    for span in panel.get("spans", []):
        row, col = span["row"], span["col"]
        nr, nc = span.get("rowspan", 1), span.get("colspan", 1)
        if min(row, col) < 0 or min(nr, nc) < 1 or row + nr > len(rows) or col + nc > len(widths):
            raise ValueError("paper table span is outside the grid")
        area = {(r, c) for r in range(row, row + nr) for c in range(col, col + nc)}
        if area & covered:
            raise ValueError("paper table spans overlap")
        covered.update(area)
        merged[row, col] = nr, nc
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            if (r, c) not in covered or (r, c) in merged:
                nr, nc = merged.get((r, c), (1, 1))
                yield r, c, nr, nc, str(text)


def _rules(panel):
    rules = list(panel.get("rules", [
        {"row": panel.get("header_rows", 1), "start": 0, "end": len(panel["widths"]), "weight": .55}]))
    for edge in (0, len(panel["rows"])):
        if not any(rule["row"] == edge for rule in rules):
            rules.append({"row": edge, "start": 0, "end": len(panel["widths"]), "weight": 1})
    return rules


def _html_text(value):
    """Offline HTML for the small math vocabulary used by these paper tables."""
    text = html.escape(str(value))
    for source, target in ((r"\alpha", "α"), (r"\times", "×"), (r"\dagger", "†"),
                           (r"\mu", "μ"), (r"\geq", "≥"), (r"\ge", "≥"),
                           (r"\leq", "≤"), (r"\le", "≤"), (r"\cdot", "·"), (r"\log", "log"),
                           (r"\%", "%"), (r"\,", " "), (r"\;", " ")):
        text = text.replace(source, target)
    text = re.sub(r"\\(?:mathrm|mathit|mathbf|text)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", text)
    text = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", text)
    text = re.sub(r"\^(-?\d+|[a-zA-Z])", r"<sup>\1</sup>", text)
    text = re.sub(r"_([0-9a-zA-Z])", r"<sub>\1</sub>", text)
    return text.replace("$", "").replace("\n", "<br>")


def comparison_view(document):
    """Preserve the paper's grid and compare unrounded, qualified numbers."""
    document = copy.deepcopy(document)
    records = []
    for panel in document["panels"]:
        cells = list(_cells(panel))
        visible = {(r, c) for r, c, *_ in cells}
        seen = set()
        for item in panel.get("comparisons", []):
            r, c = item["row"], item["col"]
            if (r, c) not in visible or (r, c) in seen or r < panel.get("header_rows", 1):
                raise ValueError("invalid or duplicate paper comparison cell")
            seen.add((r, c))
            paper, current = item["paper"], item["current"]
            if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value)) for value in (paper, current)):
                raise ValueError("paper comparison requires finite numeric values or missing values")
            delta = (current - paper) / abs(paper) * 100 if paper not in (None, 0) and current is not None else None
            item["relative_difference_pct"] = delta
            item["delta_text"] = f"{delta:+.1f}%" if delta is not None else "—"
            row_labels = [str(panel["rows"][r][column]) for column in range(len(panel["widths"]))
                          if (r, column) not in {(x["row"], x["col"]) for x in panel.get("comparisons", [])}
                          and panel["rows"][r][column]]
            column_labels = [text for rr, cc, nr, nc, text in cells
                             if rr < panel.get("header_rows", 1) and cc <= c < cc + nc and text]
            records.append({"table": document["number"], "panel": panel.get("title", ""),
                            "row": " / ".join(row_labels), "column": " / ".join(column_labels),
                            "paper_value": paper, "new_value": current,
                            "paper_display": item["paper_text"], "new_display": item["current_text"],
                            "relative_difference_pct": delta,
                            "relative_error_pct": abs(delta) if delta is not None else None,
                            "status": item["status"]})
        # Labels must be captured before replacing numeric strings with multi-line cells.
        for item in panel.get("comparisons", []):
            panel["rows"][item["row"]][item["col"]] = "\n".join(
                (item["paper_text"], item["current_text"], item["delta_text"]))
    document["notes"] = [
        "Each numeric cell: Paper / New / Δ%. Δ% = (New − Paper) / |Paper| × 100; its absolute value is the relative error.",
        "Paper values are the rounded numbers printed in the supplied PDF. — means missing or not exactly comparable; model values and estimates retain their stated scope.",
        *document.get("notes", [])]
    document["comparison_records"] = records
    return document


def _write_html(document, path):
    parts = ["<!doctype html>", '<html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f"<title>Table {document['number']}</title>", "<style>",
             "body{font-family:'Times New Roman',Times,serif;color:#111;background:#fff;margin:24px}",
             f"main{{max-width:{document.get('width', 9)}in;margin:auto}}",
             "h1{font-size:1.1rem;font-weight:normal}h2{font-size:1rem;text-align:center;font-weight:normal}",
             ".scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;table-layout:fixed;margin:12px 0 20px}",
             "td,th{padding:5px 4px;text-align:center;font-weight:normal;white-space:nowrap}",
             ".cell-value{display:block;line-height:1.45}.new-value{font-weight:bold}.paper-value{color:#555}",
             "body[data-view='new'] .paper-value,body[data-view='new'] .delta-value,"
             "body[data-view='paper'] .new-value,body[data-view='paper'] .delta-value,"
             "body[data-view='delta'] .paper-value,body[data-view='delta'] .new-value{display:none}",
             "nav{font-family:Arial,sans-serif;font-size:.85rem;margin:12px 0}button{margin-right:6px;padding:5px 9px;cursor:pointer}",
             "p{font-size:.85rem;line-height:1.3}sup,sub{font-size:.72em}", "</style><main>",
             f"<h1><strong>Table {document['number']}.</strong> {_html_text(document['caption'])}</h1>",
             '<nav aria-label="Table view"><button data-view="compare">Paper / New / Δ%</button>'
             '<button data-view="paper">Paper</button><button data-view="new">New</button>'
             '<button data-view="delta">Δ%</button> '
             '<a href="table.pdf">PDF</a> · <a href="paper-comparison.csv">Comparison CSV</a></nav>',
             '<p>Each numeric cell: Paper / New / Δ%. Δ% = (New − Paper) / |Paper| × 100.</p>']
    for panel in document["panels"]:
        if panel.get("title"):
            parts.append(f"<h2>{_html_text(panel['title'])}</h2>")
        parts += [f'<div class="scroll"><table style="min-width:{document.get("width", 9) - .36}in">', "<colgroup>"]
        total = sum(panel["widths"])
        parts.extend(f'<col style="width:{width / total * 100:.4f}%">' for width in panel["widths"])
        parts.append("</colgroup>")
        cells = {(r, c): (nr, nc, text) for r, c, nr, nc, text in _cells(panel)}
        comparisons = {(item["row"], item["col"]): item for item in panel.get("comparisons", [])}
        rules = _rules(panel)
        for r in range(len(panel["rows"])):
            parts.append("<tr>")
            for c in range(len(panel["widths"])):
                if (r, c) not in cells:
                    continue
                nr, nc, text = cells[r, c]
                tag = "th" if r < panel.get("header_rows", 1) else "td"
                align = panel.get("align", {}).get(c, "center")
                style = [f"text-align:{align}"]
                for rule in rules:
                    if rule.get("start", 0) <= c and c + nc <= rule.get("end", len(panel["widths"])):
                        side = "top" if rule["row"] == r else "bottom" if rule["row"] == r + nr else None
                        if side:
                            style.append(f"border-{side}:{rule.get('weight', .6)}pt solid black")
                for rule in panel.get("vrules", []):
                    if rule.get("row", 0) <= r and r + nr <= rule.get("end", len(panel["rows"])):
                        side = "left" if rule["col"] == c else "right" if rule["col"] == c + nc else None
                        if side:
                            style.append(f"border-{side}:{rule.get('weight', .4)}pt solid black")
                if tag == "th" and panel.get("bold_header", False):
                    style.append("font-weight:bold")
                item = comparisons.get((r, c))
                if item:
                    content = ''.join(f'<span class="cell-value {kind}-value" title="{label}">{_html_text(value)}</span>'
                                      for kind, label, value in (("paper", "Paper", item["paper_text"]),
                                          ("new", html.escape("New: " + item["status"], quote=True), item["current_text"]),
                                          ("delta", "Relative difference (%)", item["delta_text"])))
                else:
                    content = _html_text(text)
                parts.append(f'<{tag} rowspan="{nr}" colspan="{nc}" style="{";".join(style)}">'
                             f'{content}</{tag}>')
            parts.append("</tr>")
        parts.append("</table></div>")
    parts += [f"<p>{_html_text(note)}</p>" for note in document.get("notes", [])]
    parts.append("</main><script>document.querySelectorAll('button[data-view]').forEach(function(button){"
                 "button.addEventListener('click',function(){document.body.dataset.view=button.dataset.view;});});"
                 "</script></html>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_images(document, directory):
    try:
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as error:
        raise RuntimeError("Paper table rendering requires matplotlib; run bash artifact/setup.sh or "
                           "python3 -m pip install -r artifact/requirements.txt") from error
    with matplotlib.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                                "pdf.fonttype": 42, "ps.fonttype": 42}):
        width, margin = document.get("width", 9), .18
        body_width = width - 2 * margin
        font = document.get("font_size", 10)
        # Measure at the PNG's final DPI: font hinting changes physical text
        # widths enough that measuring at a lower DPI can clip long notes.
        figure = Figure(figsize=(width, 1), dpi=180)
        FigureCanvasAgg(figure)
        renderer = figure.canvas.get_renderer()
        measured = {}

        def text_width(text, size):
            key = text, size
            if key not in measured:
                artist = figure.text(0, 0, text, fontsize=size)
                measured[key] = artist.get_window_extent(renderer).width / figure.dpi
                artist.remove()
            return measured[key]

        def wrap(text, size):
            lines, current = [], ""
            for token in re.findall(r"\$[^$]*\$|\S+", text):
                candidate = current + (" " if current else "") + token
                if current and text_width(candidate, size) > body_width - .06:
                    lines.append(current)
                    current = token
                else:
                    current = candidate
            return lines + ([current] if current else [])

        caption_size = max(10, font + 1)
        caption = wrap(f"Table {document['number']}. {document['caption']}", caption_size)
        note_size = max(8, font - 1)
        notes = [wrap(note, note_size) for note in document.get("notes", [])]
        y = margin + len(caption) * caption_size / 72 * 1.3 + .14
        positions = []
        for panel in document["panels"]:
            if panel.get("title"):
                y += .27
            heights = [panel.get("row_height", .27) * (1 + .65 * max(str(t).count("\n") for t in row))
                       for row in panel["rows"]]
            positions.append((y, heights))
            y += sum(heights) + .25
        note_y = y - .04
        height = note_y + sum(len(lines) * note_size / 72 * 1.25 + .055 for lines in notes) + margin
        figure.set_size_inches(width, height)
        ax = figure.add_axes((0, 0, 1, 1), xlim=(0, width), ylim=(height, 0))
        ax.set_axis_off()
        for i, line in enumerate(caption):
            ax.text(margin, margin + i * caption_size / 72 * 1.3, line,
                    fontsize=caption_size, va="top", ha="left")
        for panel, (top, heights) in zip(document["panels"], positions):
            if panel.get("title"):
                ax.text(width / 2, top - .14, panel["title"], fontsize=font + 1,
                        ha="center", va="center")
            x = [margin]
            for size in panel["widths"]:
                x.append(x[-1] + body_width * size / sum(panel["widths"]))
            ys = [top]
            for size in heights:
                ys.append(ys[-1] + size)
            for r, c, nr, nc, text in _cells(panel):
                align = panel.get("align", {}).get(c, "center")
                x_pos = x[c] + .04 if align == "left" else x[c + nc] - .04 if align == "right" else (x[c] + x[c + nc]) / 2
                available = x[c + nc] - x[c] - .07
                actual = text_width(text, font) if text else 0
                size = min(font, font * available / actual) if actual else font
                ax.text(x_pos, (ys[r] + ys[r + nr]) / 2, text, ha=align, va="center",
                        fontsize=size, linespacing=1.15,
                        fontweight="bold" if panel.get("bold_header") and r < panel.get("header_rows", 1) else "normal")
            for rule in _rules(panel):
                start, end = rule.get("start", 0), rule.get("end", len(panel["widths"]))
                pad = .04 if start or end != len(panel["widths"]) else 0
                ax.plot([x[start] + pad, x[end] - pad], [ys[rule["row"]]] * 2,
                        color="black", linewidth=rule.get("weight", .6), solid_capstyle="butt")
            for rule in panel.get("vrules", []):
                def row_y(row):
                    base = int(row)
                    return ys[base] + (row - base) * heights[base] if base < len(heights) else ys[-1]
                ax.plot([x[rule["col"]]] * 2,
                        [row_y(rule.get("row", 0)), row_y(rule.get("end", len(heights)))],
                        color="black", linewidth=rule.get("weight", .4), solid_capstyle="butt")
        for lines in notes:
            for line in lines:
                ax.text(margin, note_y, line, fontsize=note_size, va="top", ha="left")
                note_y += note_size / 72 * 1.25
            note_y += .055
        figure.savefig(directory / "table.pdf", metadata={"Title": f"Table {document['number']}",
                       "Creator": "ThunderFFT artifact paper table renderer", "CreationDate": None})
        figure.savefig(directory / "table.png", dpi=180)
        figure.clear()


def render(evaluation, directory):
    directory = Path(directory)
    source = directory / "table.csv"
    if not source.is_file():
        raise ValueError(f"assembled table CSV is missing: {source}")
    document = comparison_view(build(evaluation, directory))
    for panel in document["panels"]:
        list(_cells(panel))
    with tempfile.TemporaryDirectory(prefix=".table-render-", dir=directory) as temporary:
        staging = Path(temporary)
        _write_images(document, staging)
        _write_html(document, staging / "table.html")
        with (staging / "paper-comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerows(document["comparison_records"])
        (staging / "comparison.md").write_text(
            f"# Table {document['number']}\n\n{document['caption']}\n\n"
            "[HTML: Paper / New / Δ%](table.html) · [PDF](table.pdf) · [Comparison CSV](paper-comparison.csv)\n\n"
            f"[![Table {document['number']}](table.png)](table.html)\n\n"
            + "\n\n".join(document["notes"]) + "\n", encoding="utf-8")
        state = {"schema_version": 1, "evaluation": evaluation, "operation": "offline_paper_table_render",
                 "new_gpu_measurements": False, "paper_layout_sha256": PAPER_SHA256,
                 "source_csv_sha256": digest(source),
                 "inputs": {name: digest(directory / name) for name in ("table.csv", "speedup.csv", "metadata.json")
                            if (directory / name).is_file()},
                 "paper_reference_inputs": {name: digest(Path(__file__).parent / name) for name in
                                             ("paper_table_references.json", "table3/paper-table3-all-targets.csv",
                                              "paper_table_accuracy.py")},
                 "outputs": {name: digest(staging / name) for name in OUTPUTS},
                 "comparison_formula": "100 * (new - paper) / abs(paper); absolute value is relative error",
                 "notes": document.get("notes", [])}
        (staging / "table-render.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        for name in (*OUTPUTS, "table-render.json"):
            (staging / name).replace(directory / name)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        if directory.name in TABLES:
            targets = [(directory.name, directory)]
        elif (directory / "table.csv").is_file() and (directory / "metadata.json").is_file():
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            evaluation = metadata.get("evaluation") if isinstance(metadata, dict) else None
            targets = [(evaluation, directory)] if evaluation in TABLES else []
        else:
            targets = [(evaluation, directory / evaluation) for evaluation in TABLES
                       if (directory / evaluation / "table.csv").is_file()]
        if not targets:
            parser.error(f"no assembled table CSVs in {directory}")
        for evaluation, target in targets:
            render(evaluation, target)
            print(f"{evaluation}: {target / 'table.html'} (PDF: {target / 'table.pdf'})")


if __name__ == "__main__":
    main()
