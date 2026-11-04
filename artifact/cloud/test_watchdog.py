"""Exercise real detached acknowledgement/disarming without any API requests."""
import json
from pathlib import Path
import os
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact.cloud.coordinator import arm_watchdog
from artifact.cloud.watchdog import write_json


class DetachedWatchdogTests(unittest.TestCase):
    def test_real_process_acknowledges_before_creation_and_disarms_without_network(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            key = folder / "vast_api_key"
            key.write_text("offline-test-credential-never-sent")
            key.chmod(0o600)
            state = {"label": "thunderfft-ae-offline-fixture", "api_key_file": str(key),
                     "deadline": time.time() + 120, "created_at": time.time(),
                     "create_attempted": False, "finished": False, "cleanup_confirmed": False}
            write_json(folder / "state.json", state)
            with (folder / "lock").open("a") as lock:
                process = arm_watchdog(folder, lock)
                try:
                    ready = json.loads((folder / "watchdog-ready.json").read_text())
                    self.assertEqual(ready["pid"], process.pid)
                    self.assertEqual(os.getpgid(process.pid), process.pid)
                    self.assertIsNone(process.poll())
                    state["finished"] = True
                    write_json(folder / "state.json", state)
                    self.assertEqual(process.wait(timeout=5), 0)
                    self.assertFalse((folder / "watchdog-cleanup.json").exists())
                    self.assertFalse((folder / "watchdog-billing-error.json").exists())
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
