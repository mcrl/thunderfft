"""CPU checks for bounded telemetry, terminal redraws, and quiet redirected logs."""
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact import progress


def task(done=3, total=10, **patch):
    return {"schema_version": 1, "done": done, "total": total, "unit": "cases",
            "detail": "N=1024", "status": "measuring", "updated_at": 123.0, **patch}


def aggregate(done=3, **patch):
    return {"schema_version": 1, "completed": 1, "total": 8, "evaluation": "table3",
            "status": "running", "detail": task(done), "updated_at": 123.0, **patch}


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "nested" / "progress.json"

    def test_report_is_opt_in_and_replaces_one_file(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(progress, "write_snapshot") as write:
            progress.report(1, 2)
            write.assert_not_called()
        with mock.patch.dict(os.environ, {progress.PROGRESS_FILE_ENV: str(self.path)}):
            progress.report(1, 2, detail="first")
            progress.report(2, 2, unit="configs", detail="last")
        self.assertEqual(progress.read_snapshot(self.path)["done"], 2)
        self.assertEqual(progress.read_snapshot(self.path)["unit"], "configs")
        self.assertEqual([entry.name for entry in self.path.parent.iterdir()], ["progress.json"])

    def test_atomic_replacement_leaves_previous_snapshot_readable(self):
        progress.write_snapshot(self.path, task(1))
        replace = os.replace
        def checked_replace(source, target):
            self.assertEqual(progress.read_snapshot(target)["done"], 1)
            self.assertEqual(progress.read_snapshot(source)["done"], 2)
            replace(source, target)
        with mock.patch.object(progress.os, "replace", side_effect=checked_replace):
            self.assertTrue(progress.write_snapshot(self.path, task(2)))
        self.assertEqual(progress.read_snapshot(self.path)["done"], 2)

    def test_io_failure_does_not_break_measurements_or_leave_temporary_files(self):
        with mock.patch.dict(os.environ, {progress.PROGRESS_FILE_ENV: str(self.path)}), \
             mock.patch.object(progress.os, "replace", side_effect=OSError("full disk")):
            progress.report(1, 2)
            self.assertFalse(progress.write_snapshot(self.path, aggregate()))
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_aggregate_roundtrip_is_sanitized(self):
        value = aggregate(detail=task(detail="\x1b[31mN=1024\x1b[0m\n\x1b]0;secret title\x07next\r\x07"),
                          evaluation="table3\n\x1b[2J", unexpected="not transported")
        self.assertTrue(progress.write_snapshot(self.path, value))
        actual = progress.read_snapshot(self.path)
        self.assertEqual(actual["evaluation"], "table3")
        self.assertEqual(actual["detail"]["detail"], "N=1024 next")
        self.assertNotIn("unexpected", actual)

    def test_rejects_bad_counts_and_nested_aggregate(self):
        for value in (task(-1), task(11), task(True), task(1, 1.5), task(schema_version=True),
                      task(updated_at=float("inf")), task(updated_at=10 ** 10000),
                      task(updated_at="2026-11-09"), aggregate(detail=aggregate()),
                      aggregate(detail="not a task snapshot"), {}):
            with self.subTest(value=str(value)[:100] if value.get("updated_at") != 10 ** 10000 else "huge timestamp"):
                self.assertIsNone(progress.validate_snapshot(value))
        self.assertIsNotNone(progress.validate_snapshot(aggregate(completed=0, total=0, detail=None)))

    def test_read_ignores_missing_truncated_oversized_and_deep_json(self):
        self.assertIsNone(progress.read_snapshot(self.path))
        self.path.parent.mkdir()
        for value in (b'{"done":', b"x" * (progress.MAX_SNAPSHOT_BYTES + 1),
                      b"[" * 2000 + b"]" * 2000, b"\xff"):
            self.path.write_bytes(value)
            self.assertIsNone(progress.read_snapshot(self.path))

    def test_detail_is_bounded(self):
        actual = progress.validate_snapshot(task(detail="x" * 100000))
        self.assertEqual(len(actual["detail"]), 512)


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class DashboardTests(unittest.TestCase):
    def test_tty_reuses_three_rows_and_restores_cursor(self):
        stream = FakeTTY()
        display = progress.Dashboard(stream=stream, mode="tty")
        display.refresh(force=True)
        display.update("A100", {"status": "running", "progress": aggregate()})
        display.refresh(force=True)
        display.close()
        text = stream.getvalue()
        self.assertEqual(text.count("\x1b[?25l"), 1)
        self.assertEqual(text.count("\x1b[?25h"), 1)
        self.assertEqual(text.count("\x1b[2A"), 2)
        self.assertIn("Table3 3/10 cases", text)
        self.assertTrue(text.endswith("\n\x1b[?25h"))
        display.close()
        self.assertEqual(stream.getvalue(), text)

    def test_redirected_ticks_only_print_stage_and_evaluation_transitions(self):
        stream = io.StringIO()
        display = progress.Dashboard(("A100",), stream=stream)
        display.update("A100", {"status": "running", "progress": aggregate(0)})
        for count in range(11):
            display.update("A100", {"progress": aggregate(count)})
            display.refresh(force=True)
        display.update("A100", {"progress": aggregate(evaluation="figure5", detail=None)})
        display.refresh(force=True)
        display.update("A100", {"status": "cleaning"})
        display.refresh(force=True)
        display.update("A100", {"status": "completed"})
        display.close()
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(all("\x1b" not in line for line in lines))
        self.assertIn("completed", lines[-1])

    def test_auto_uses_plain_for_dumb_terminal(self):
        with mock.patch.dict(os.environ, {"TERM": "dumb"}):
            display = progress.Dashboard(stream=FakeTTY())
        self.assertEqual(display.mode, "plain")

    def test_refresh_is_rate_limited(self):
        stream = FakeTTY()
        with mock.patch.object(progress.time, "monotonic", return_value=10.0):
            display = progress.Dashboard(("A100",), stream=stream, mode="tty")
            display.refresh()
            first = stream.getvalue()
            for _ in range(20):
                display.refresh()
            self.assertEqual(stream.getvalue(), first)
            display.close()

    def test_narrow_lines_never_wrap_and_scrub_controls(self):
        display = progress.Dashboard(("A100",), stream=FakeTTY(), mode="tty")
        display.update("A100", {"status": "running\x1b[2J", "progress": aggregate(
            detail=task(detail="긴 설명 " * 100)), "detail": "\n\r\x1b[31munsafe"})
        for width in (1, 10, 20, 40, 60, 90):
            line = display._line("A100", width)
            self.assertLessEqual(sum(progress._cell_width(char) for char in line), width)
            self.assertNotIn("\x1b", line)
            self.assertNotIn("\n", line)
        self.assertIn("3/10", display._line("A100", 35))

    def test_cleanup_is_not_displayed_as_completed(self):
        display = progress.Dashboard(("B200",), stream=FakeTTY(), mode="tty")
        display.update("B200", {"status": "cleaning", "progress": aggregate(
            completed=8, status="completed", detail=task(10))})
        line = display._line("B200", 150)
        self.assertIn("cleaning", line)
        self.assertIn("8/8 steps", line)
        self.assertNotIn("[##########]", line)
        display.update("B200", {"status": "completed"})
        self.assertIn("[##########]", display._line("B200", 150))

    def test_task_completion_does_not_claim_gpu_is_finished(self):
        display = progress.Dashboard(("A100",), stream=FakeTTY(), mode="tty")
        display.update("A100", {"status": "running", "progress": aggregate(status="completed")})
        self.assertIn("finishing", display._line("A100", 150))
        self.assertNotIn("[##########]", display._line("A100", 150))

    def test_failed_counts_remain_incomplete(self):
        display = progress.Dashboard(("H100",), stream=FakeTTY(), mode="tty")
        display.update("H100", {"status": "failed", "progress": aggregate(3)})
        line = display._line("H100", 150)
        self.assertIn("failed", line)
        self.assertIn("1/8 steps", line)
        self.assertIn("3/10 cases", line)
        self.assertNotIn("[##########]", line)

    def test_final_clock_does_not_jump_to_shorter_measurement_duration(self):
        with mock.patch.object(progress.time, "monotonic", return_value=10.0) as clock:
            display = progress.Dashboard(("A100",), stream=FakeTTY(), mode="tty")
            clock.return_value = 1010.0
            display.update("A100", {"status": "completed", "elapsed_seconds": 30.0})
            clock.return_value = 2010.0
            self.assertIn("16:40", display._line("A100", 150))

    def test_exception_context_restores_cursor(self):
        stream = FakeTTY()
        with self.assertRaisesRegex(RuntimeError, "measurement"):
            with progress.Dashboard(stream=stream, mode="tty") as display:
                display.refresh(force=True)
                raise RuntimeError("measurement failed")
        self.assertTrue(stream.getvalue().endswith("\x1b[?25h"))

    def test_broken_pipe_does_not_interrupt_measurements(self):
        class BrokenStream(FakeTTY):
            def write(self, _value):
                raise BrokenPipeError("output consumer exited")
        display = progress.Dashboard(stream=BrokenStream(), mode="tty")
        display.refresh(force=True)
        display.close()
        self.assertFalse(display._cursor_hidden)

    def test_log_file_keeps_verbose_messages_out_of_panel(self):
        stream = FakeTTY()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pipeline.log"
            with progress.Dashboard(stream=stream, mode="tty", log_path=path) as display:
                display.log("provider polling detail")
            self.assertEqual(path.read_text(), "provider polling detail\n")
        self.assertNotIn("provider polling", stream.getvalue())

    def test_off_produces_no_terminal_output(self):
        stream = io.StringIO()
        with progress.Dashboard(stream=stream, mode="off") as display:
            display.update("A100", {"status": "completed"})
            display.refresh(force=True)
            display.log("quiet")
        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
