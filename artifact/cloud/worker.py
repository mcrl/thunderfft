#!/usr/bin/env python3
"""Runs inside one rented container; receives no coordinator credentials."""
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tarfile
import time


TERMINATE_SECONDS = 20
PACK_SECONDS = 5


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def safe_environment(environment):
    """Provider credentials belong only to its onstart cleanup process."""
    return {key: value for key, value in environment.items()
            if not any(word in key.upper() for word in ("TOKEN", "KEY", "SECRET", "PASSWORD"))}


def stop(process):
    if process.poll() is not None:
        return
    # The runner records terminal checkpoints and joins its separately started
    # adapter process groups before forced termination is necessary.
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=TERMINATE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    except ProcessLookupError:
        pass


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


def time_left(deadline, maximum):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("worker execution deadline reached")
    return min(remaining, maximum)


def validate_gpu_inventory(text, gpu):
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("rental must expose exactly one GPU with complete inventory")
    identity, name, memory, driver = [value.strip() for value in rows[0]]
    normalized = re.sub(r"[ _-]+", " ", name.upper()).removeprefix("NVIDIA ")
    match = re.fullmatch(r"([0-9]+)(?:\s+MiB)?", memory)
    if not identity.startswith("GPU-") or not driver or match is None:
        raise ValueError("GPU inventory has invalid UUID, memory, or driver fields")
    memory_mib = int(match.group(1))
    tokens = normalized.split()
    if gpu in ("A100", "H100"):
        correct = (gpu in tokens and "PCIE" in tokens and
                   not any(word.startswith("SXM") or word == "NVL" for word in tokens)
                   and 79000 <= memory_mib <= 84000)
    elif gpu == "B200":
        correct = normalized == "B200" and memory_mib >= 175000
    else:
        correct = False
    if not correct:
        raise ValueError("observed GPU does not match the requested paper GPU SKU and memory")
    return {"uuid": identity, "name": name, "memory_mib": memory_mib, "driver": driver}


def verify_source(root, deadline):
    source = json.loads((root / ".thunderfft-cloud-source.json").read_text())
    for relative, expected in source["files"].items():
        time_left(deadline, 1)
        path = root / relative
        pure = PurePosixPath(relative)
        if (pure.is_absolute() or ".." in pure.parts or path.resolve() != path
                or not path.is_file()):
            raise ValueError("transferred source contains an unsafe path")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                time_left(deadline, 1)
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("transferred source hash mismatch")
    return source


def run_jobs(root, request, output, deadline):
    source = verify_source(root, deadline)
    atomic_json(output / "source.json", source)
    environment = safe_environment(os.environ)
    observations = {}
    probes = {
        "gpu": ["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader"],
        "cuda": ["nvcc", "--version"], "cmake": ["cmake", "--version"],
        "python": [sys.executable, "--version"],
        "python_packages": [sys.executable, "-m", "pip", "freeze"],
        "os": ["cat", "/etc/os-release"],
    }
    for name, command in probes.items():
        response = subprocess.run(command, text=True, capture_output=True,
                                  timeout=time_left(deadline, 20), env=environment)
        observations[name] = {"exit_code": response.returncode, "stdout": response.stdout,
                              "stderr": response.stderr}
        atomic_json(output / "environment.json", observations)
    if observations["gpu"]["exit_code"]:
        raise ValueError("GPU inventory query failed")
    observations["validated_gpu"] = validate_gpu_inventory(observations["gpu"]["stdout"], request["gpu"])
    atomic_json(output / "environment.json", observations)
    environment.update({"THUNDERFFT_ARCHS": request["arch"], "PYTHON": sys.executable,
                        "THUNDERFFT_PYTHON": sys.executable, "MPLBACKEND": "Agg",
                        "PYTHONUNBUFFERED": "1"})
    if request.get("mathdx"):
        mathdx_root = root.parent / "mathdx/nvidia/mathdx/25.06"
        if not (mathdx_root / "include/cufftdx.hpp").is_file():
            raise ValueError("requested MathDx was not installed by bootstrap")
        environment["MATHDX_ROOT"] = str(mathdx_root)
    if request.get("cudnn"):
        cudnn_root = root.parent / "cudnn"
        if not (cudnn_root / "include/cudnn.h").is_file() or not (cudnn_root / "lib/libcudnn.so").is_file():
            raise ValueError("requested cuDNN was not installed by bootstrap")
        environment["CUDNN_HOME"] = str(cudnn_root)
        environment["LD_LIBRARY_PATH"] = str(cudnn_root / "lib") + (
            os.pathsep + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    if request.get("torch_headers"):
        environment["FLASHFFTCONV_PYTHON"] = sys.executable
    states = []
    atomic_json(output / "jobs.json", states)
    with (output / "jobs.log").open("w") as log:
        for index, job in enumerate(request["jobs"]):
            state = {"part": job["part"], "evaluations": job["evaluations"],
                     "status": "running", "exit_code": None}
            states.append(state)
            process = None
            try:
                remaining = time_left(deadline, request["execution_seconds"])
                command = ["bash", str(root / "run.sh"), *job["evaluations"], "--backend", "local", "--profile", request["profile"],
                           "--device", "0", "--jobs", str(request["build_jobs"]),
                           "--result-dir", str(root / "result"), "--build-dir", str(root / "build_artifact"),
                           "--timeout", str(request["task_timeout"])]
                if request["collection"]:
                    command += ["--collection", request["collection"], "--part", job["part"]]
                job_environment = dict(environment, **job["environment"])
                # Defense in depth if a request file was edited after controller validation.
                if safe_environment(job_environment) != job_environment:
                    raise ValueError("job overrides must not contain credential variables")
                log.write(json.dumps({"command": command, "part": job["part"]}) + "\n")
                log.flush()
                atomic_json(output / "jobs.json", states)
                process = subprocess.Popen(command, cwd=root, env=job_environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["exit_code"] = process.wait(timeout=remaining)
                state["status"] = "completed" if state["exit_code"] == 0 else "partial" if state["exit_code"] == 2 else "failed"
            except (TimeoutError, subprocess.TimeoutExpired):
                if process is not None:
                    stop(process)
                state.update(exit_code=124, status="timed_out" if process is not None else "not_started_deadline")
            except KeyboardInterrupt:
                if process is not None:
                    stop(process)
                state.update(exit_code=130, status="interrupted")
                states.extend({"part": future["part"], "evaluations": future["evaluations"],
                               "exit_code": 130, "status": "not_started_interrupted"}
                              for future in request["jobs"][index + 1:])
                raise
            except BaseException:
                if process is not None:
                    stop(process)
                state.update(exit_code=1, status="failed")
                raise
            finally:
                atomic_json(output / "jobs.json", states)
    codes = [state["exit_code"] for state in states]
    return 124 if 124 in codes else 1 if any(code not in (0, 2) for code in codes) else 2 if 2 in codes else 0


class DeadlineReader:
    def __init__(self, stream, deadline):
        self.stream, self.deadline = stream, deadline

    def read(self, size=-1):
        time_left(self.deadline, 1)
        return self.stream.read(size)


def pack_results(result, archive_path, max_bytes, deadline):
    files, total = [], 0
    for path in itertools.chain((result,), result.rglob("*")):
        time_left(deadline, 1)
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
            raise ValueError("result bundle contains a link or special file")
        if stat.S_ISREG(mode):
            total += path.stat().st_size
        files.append(path)
        if total > max_bytes or len(files) > 100000:
            raise ValueError("result bundle exceeds the transfer bound")
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    try:
        # Avoid hard-link optimization: each evidence file is an ordinary member.
        with tarfile.open(temporary, "w:gz", dereference=True) as archive:
            for path in sorted(files):
                time_left(deadline, 1)
                info = archive.gettarinfo(path, arcname=str(path.relative_to(result.parent)))
                if info.isfile():
                    with path.open("rb") as stream:
                        archive.addfile(info, DeadlineReader(stream, deadline))
                else:
                    archive.addfile(info)
        temporary.replace(archive_path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    started = time.monotonic()
    root, request_file = [Path(value).resolve() for value in (sys.argv[1:] if argv is None else argv)]
    parent = root.parent
    (parent / "worker.pid").write_text(str(os.getpid()) + "\n")
    output = root / "result/cloud-worker"
    output.mkdir(parents=True, exist_ok=True)
    original_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    code, archive_ready, request = 1, False, None
    deadline = started + 60
    try:
        request = json.loads(request_file.read_text())
        if (not isinstance(request["execution_seconds"], int) or isinstance(request["execution_seconds"], bool)
                or request["execution_seconds"] < 60):
            raise ValueError("execution_seconds must be at least 60")
        if not isinstance(request["max_result_bytes"], int) or request["max_result_bytes"] < 1:
            raise ValueError("max_result_bytes must be a positive integer")
        deadline = started + request["execution_seconds"]
        code = run_jobs(root, request, output, deadline - TERMINATE_SECONDS - PACK_SECONDS)
    except KeyboardInterrupt:
        code = 130
        atomic_json(output / "error.json", {"error": "worker interrupted", "exit_code": code})
    except (TimeoutError, subprocess.TimeoutExpired):
        code = 124
        atomic_json(output / "error.json", {"error": "worker execution deadline reached", "exit_code": code})
    except Exception as error:
        code = 1
        # The command arguments contain only source paths and validated settings;
        # provider credentials and environment dumps never enter this record.
        atomic_json(output / "error.json", {"error_type": type(error).__name__, "error": str(error)[:1000]})
    finally:
        try:
            if request is not None and isinstance(request.get("max_result_bytes"), int) and request["max_result_bytes"] > 0:
                pack_results(root / "result", parent / "results.tar.gz", request["max_result_bytes"], deadline)
                archive_ready = True
        except BaseException as error:
            atomic_json(output / "pack-error.json", {"error_type": type(error).__name__,
                                                      "error": "result packaging failed or exceeded its deadline"})
            if code == 0:
                code = 1
        finally:
            atomic_json(parent / "worker.exit.json", {"exit_code": code, "result_archive_ready": archive_ready,
                                                      "elapsed_seconds": time.monotonic() - started})
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)
    return code


if __name__ == "__main__":
    sys.exit(main())
