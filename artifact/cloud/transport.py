"""Small, bounded SSH transport and source/result archive helpers.

Private credentials stay on the controller. Source archives contain the current
tracked files and an explicit provenance manifest, never Git history. Returned
archives are untrusted input and are validated before publishing any files.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time


SOURCE_MANIFEST = ".thunderfft-cloud-source.json"
_SECRET_NAMES = {"cloud_key", "cloud_keys", ".ssh", ".aws", "credentials.json",
                 "api_key", "api-key", "vast_api_key", "id_rsa", "id_dsa",
                 "id_ecdsa", "id_ed25519", "authorized_keys", "known_hosts"}
_SECRET_CONTENT = re.compile(
    rb"(?m)^-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----|"
    rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bgh[pousr]_[A-Za-z0-9]{36,}\b|"
    rb"\bgithub_pat_[A-Za-z0-9_]{50,}\b|"
    rb"(?i:\b(?:vast(?:ai)?[_-]?)?(?:api[_-]?key|api[_-]?token|access[_-]?token|secret[_-]?key))"
    rb"[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_/-]{24,}[\"']")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=30).stdout


def _safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or name.startswith("/") or "\\" in name
            or any(ord(c) < 32 or ord(c) == 127 for c in name)
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("unsafe archive path")
    return path


def _excluded_source(path: PurePosixPath) -> bool:
    parts = [p.lower() for p in path.parts]
    return (parts[0] in ("result", "build") or ".git" in parts
            or parts[0].startswith("build_")
            or any(p in _SECRET_NAMES or p == ".env" or p.startswith(".env.")
                   or p.startswith("api_key.") or p.endswith((".pem", ".p12", ".pfx", ".key", ".pub"))
                   for p in parts))


_GENERATED_DIRS = {"__pycache__", "CMakeFiles", "_local_artifacts"}
_GENERATED_FILES = {"CMakeCache.txt"}


def _generated(path: PurePosixPath) -> bool:
    return (any(part in _GENERATED_DIRS for part in path.parts)
            or path.name in _GENERATED_FILES or path.suffix in (".pyc", ".pyo")
            or (path.parent == PurePosixPath("sample/conv1d") and path.suffix == ".log"))


def is_distribution(root: Path) -> bool:
    root = Path(root)
    return not (root / ".git").exists() and (root / SOURCE_MANIFEST).is_file()


def distribution_status(root: Path) -> dict:
    root = Path(root).resolve()
    manifest_path = root / SOURCE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modified, missing = [], []
    for name, evidence in sorted(manifest["files"].items()):
        path = root / _safe_relative(name)
        if path.is_symlink() or not path.is_file():
            missing.append(name)
        elif _sha256_file(path) != evidence["sha256"]:
            modified.append(name)
    untracked = []
    for directory, dirs, files in os.walk(root):
        relative = PurePosixPath(Path(directory).relative_to(root).as_posix())
        dirs[:] = sorted(d for d in dirs
                         if not _excluded_source(relative / d) and not _generated(relative / d))
        for name in sorted(files):
            path = (relative / name) if str(relative) != "." else PurePosixPath(name)
            if (path.as_posix() == SOURCE_MANIFEST or path.as_posix() in manifest["files"]
                    or _excluded_source(path) or _generated(path)):
                continue
            untracked.append(path.as_posix())
    return {"git": manifest["git"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "modified": modified, "missing": missing, "untracked": untracked,
            "unmodified": not (modified or missing or untracked)}


def _git_sources(repo: Path, allow_dirty: bool) -> tuple[dict, list[str], list[str]]:
    if Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve() != repo:
        raise ValueError("source directory must be the repository root")
    if _git(repo, "ls-files", "--others", "--exclude-standard", "-z"):
        raise ValueError("untracked files are present; commit or explicitly ignore them before cloud execution")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=no").decode("utf-8")
    if status and not allow_dirty:
        raise ValueError("source tree is dirty; commit it or use --cloud-allow-dirty")
    manifest = {
        "schema_version": 1,
        "git": {"head": _git(repo, "rev-parse", "HEAD").decode().strip(),
                "branch": _git(repo, "branch", "--show-current").decode().strip(),
                "status_porcelain": status, "dirty": bool(status)},
        "files": {}, "excluded_files": [],
    }
    names, excluded = [], []
    for record in _git(repo, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        index, raw_name = record.split(b"\t", 1)
        git_mode, _oid, stage = index.decode("ascii").split()
        name = raw_name.decode("utf-8")
        path = _safe_relative(name)
        if name == SOURCE_MANIFEST:
            raise ValueError("reserved cloud source manifest is already tracked")
        if _excluded_source(path):
            excluded.append(name)
            continue
        if stage != "0" or git_mode not in ("100644", "100755"):
            raise ValueError(f"unsupported or conflicted tracked source: {name}")
        names.append(name)
    return manifest, names, excluded


def _distribution_sources(repo: Path, allow_dirty: bool) -> tuple[dict, list[str], list[str]]:
    status = distribution_status(repo)
    if status["missing"]:
        raise ValueError("distributed files are missing: " + ", ".join(status["missing"][:10]))
    if status["untracked"]:
        raise ValueError("files outside the distribution manifest are present; move them out of the "
                         "source directory before cloud execution: " + ", ".join(status["untracked"][:10]))
    if status["modified"] and not allow_dirty:
        raise ValueError("distributed files were modified; restore them or use --cloud-allow-dirty: "
                         + ", ".join(status["modified"][:10]))
    original = json.loads((repo / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    git = dict(original["git"])
    git["dirty"] = bool(git.get("dirty")) or bool(status["modified"])
    manifest = {
        "schema_version": 1, "git": git, "files": {},
        "excluded_files": list(original.get("excluded_files", [])),
        "distribution": {"manifest_sha256": status["manifest_sha256"],
                         "modified_files": status["modified"]},
    }
    return manifest, sorted(original["files"]), []


def build_source_bundle(repo: Path, dest: Path, *, allow_dirty: bool = False,
                        max_bytes: int = 512 * 1024 * 1024,
                        max_files: int = 50000) -> dict:
    """Archive a safe tracked working tree, rejecting omitted untracked work.

    ``allow_dirty`` permits changed/staged tracked files, not untracked files,
    missing tracked files, submodules, symlinks or credential content. The
    embedded manifest records the original revision and each actual file hash.
    """
    repo, dest = Path(repo).resolve(), Path(dest).absolute()
    if max_bytes < 1 or max_files < 1:
        raise ValueError("archive limits must be positive")
    distribution = is_distribution(repo)
    if distribution:
        manifest, names, excluded = _distribution_sources(repo, allow_dirty)
    else:
        manifest, names, excluded = _git_sources(repo, allow_dirty)
    manifest["excluded_files"].extend(excluded)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".source-", suffix=".tar.gz", delete=False) as out:
            temporary = Path(out.name)
            with tarfile.open(fileobj=out, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
                for name in names:
                    source = repo / _safe_relative(name)
                    if source.resolve() != source or not stat.S_ISREG(source.lstat().st_mode):
                        raise ValueError(f"tracked source must be a regular file without symlink parents: {name}")
                    if len(manifest["files"]) >= max_files:
                        raise ValueError("source archive exceeds file count limit")
                    with source.open("rb") as stream:
                        data = stream.read(max_bytes - total_bytes + 1)
                    total_bytes += len(data)
                    if total_bytes > max_bytes:
                        raise ValueError("source archive exceeds byte limit")
                    if _SECRET_CONTENT.search(data):
                        raise ValueError(f"credential-like content found in tracked source: {name}")
                    mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
                    manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(),
                                               "size": len(data), "mode": mode}
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = len(data), mode
                    archive.addfile(info, io.BytesIO(data))
                encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
                info = tarfile.TarInfo(SOURCE_MANIFEST)
                info.size, info.mode = len(encoded), 0o644
                archive.addfile(info, io.BytesIO(encoded))
        # Revisions and status may change while files are read. Do not label such
        # a snapshot with an earlier checkout's identity.
        if not distribution and (
                _git(repo, "rev-parse", "HEAD").decode().strip() != manifest["git"]["head"]
                or _git(repo, "status", "--porcelain=v1", "--untracked-files=no").decode("utf-8")
                != manifest["git"]["status_porcelain"]):
            raise ValueError("repository changed while preparing the source archive")
        # An already-dirty file can change without altering porcelain status.
        # Check the captured bytes again before declaring the snapshot ready.
        for name, evidence in manifest["files"].items():
            source = repo / name
            if (source.resolve() != source or not stat.S_ISREG(source.lstat().st_mode)
                    or _sha256_file(source) != evidence["sha256"]):
                raise ValueError("source files changed while preparing the source archive")
        temporary.replace(dest)
        temporary = None
        manifest["archive_sha256"] = _sha256_file(dest)
        manifest["archive_bytes"] = dest.stat().st_size
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_private_key(path: Path, repo_root: Path | None = None) -> Path:
    path = Path(path).expanduser().absolute()
    if path.resolve() != path:
        raise ValueError("SSH private key and its parents must not be symlinks")
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("SSH private key must be an owned regular file with mode 0600")
    if repo_root is not None and path.is_relative_to(Path(repo_root).resolve()):
        raise ValueError("SSH private key must be outside the repository")
    return path


class SSH:
    """Identity-only SSH with one known-host file per cloud session.

    The first connection uses OpenSSH's accept-new policy; later connections
    require the pinned key. Changed host keys always fail. No agent, user SSH
    configuration, local forwarding or global known-host file is consulted.
    """

    def __init__(self, host: str, port: int, user: str, key_path: Path,
                 known_hosts_path: Path, repo_root: Path | None = None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", host):
            raise ValueError("invalid SSH host")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", user):
            raise ValueError("invalid SSH user")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("invalid SSH port")
        self.host, self.port, self.user = host, port, user
        self.cancel_event = None
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.key_path = validate_private_key(key_path, self.repo_root)
        self.known_hosts_path = Path(known_hosts_path).absolute()
        self.known_hosts_path.parent.mkdir(parents=True, exist_ok=True)
        if self.known_hosts_path.resolve() != self.known_hosts_path:
            raise ValueError("known-host file and its parents must not be symlinks")
        if not self.known_hosts_path.exists():
            descriptor = os.open(self.known_hosts_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            os.close(descriptor)
        self._check_known_hosts()

    def _check_known_hosts(self):
        info = self.known_hosts_path.lstat()
        if (self.known_hosts_path.resolve() != self.known_hosts_path
                or not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("known-host file must be an owned regular file with mode 0600")

    def _command(self, argv: list[str]) -> list[str]:
        validate_private_key(self.key_path, self.repo_root)
        self._check_known_hosts()
        if not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
            raise ValueError("remote command requires nonempty string arguments without NUL")
        policy = "yes" if self.known_hosts_path.stat().st_size else "accept-new"
        return ["ssh", "-F", "/dev/null", "-T", "-p", str(self.port), "-i", str(self.key_path),
                "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none",
                "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
                "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=3", "-o", "GlobalKnownHostsFile=/dev/null",
                "-o", f"UserKnownHostsFile={self.known_hosts_path}",
                "-o", f"StrictHostKeyChecking={policy}", f"{self.user}@{self.host}", shlex.join(argv)]

    def run(self, argv: list[str], *, timeout: float = 60, input: str | None = None,
            check: bool = True) -> subprocess.CompletedProcess:
        if timeout <= 0:
            raise ValueError("SSH timeout must be positive")
        return self._run(self._command(argv), input=input, timeout=timeout, check=check)

    def _check_cancelled(self):
        event = getattr(self, "cancel_event", None)
        if event is not None and event.is_set():
            raise InterruptedError("SSH operation cancelled")

    def _run(self, command, *, timeout, input=None, stdin=None, check=True):
        """Keep ordinary calls unchanged; group calls can cancel a blocked SSH."""
        if getattr(self, "cancel_event", None) is None:
            return subprocess.run(command, input=input, stdin=stdin, text=True, capture_output=True,
                                  timeout=timeout, check=check)
        self._check_cancelled()
        deadline = time.monotonic() + timeout
        with subprocess.Popen(command, stdin=subprocess.PIPE if input is not None else stdin,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            pending_input = input
            last_timeout = None
            try:
                while True:
                    self._check_cancelled()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout,
                            output=last_timeout.output if last_timeout else None,
                            stderr=last_timeout.stderr if last_timeout else None)
                    try:
                        stdout, stderr = process.communicate(input=pending_input, timeout=min(remaining, .1))
                        break
                    except subprocess.TimeoutExpired as error:
                        last_timeout = error
                        pending_input = None
                result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                if check:
                    result.check_returncode()
                return result
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.communicate()
                raise

    def upload(self, local: Path, remote: str, *, timeout: float = 120) -> subprocess.CompletedProcess:
        local = Path(local)
        if local.resolve() == self.key_path or not local.is_file() or local.is_symlink():
            raise ValueError("upload requires a regular noncredential file")
        _remote_path(remote)
        if timeout <= 0:
            raise ValueError("SSH timeout must be positive")
        script = ("import os,shutil,sys,tempfile; "
                  "p=sys.argv[1]; fd,t=tempfile.mkstemp(prefix='.upload-',dir=os.path.dirname(p)); "
                  "f=os.fdopen(fd,'wb'); shutil.copyfileobj(sys.stdin.buffer,f); f.close(); os.replace(t,p)")
        with local.open("rb") as stream:
            return self._run(self._command(["python3", "-c", script, remote]), stdin=stream,
                             timeout=timeout, check=True)

    def download(self, remote: str, local: Path, *, timeout: float = 120,
                 max_bytes: int = 1024 * 1024 * 1024) -> subprocess.CompletedProcess:
        """Stream a file through SSH, enforcing byte and wall-clock limits."""
        _remote_path(remote)
        if timeout <= 0 or max_bytes < 1:
            raise ValueError("download limits must be positive")
        self._check_cancelled()
        local = Path(local).absolute()
        local.parent.mkdir(parents=True, exist_ok=True)
        command = self._command(["cat", "--", remote])
        process = None
        temporary = None
        stderr = bytearray()
        received = 0
        deadline = time.monotonic() + timeout
        try:
            with tempfile.NamedTemporaryFile(dir=local.parent, prefix=".download-", delete=False) as output:
                temporary = Path(output.name)
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                    while selector.get_map():
                        self._check_cancelled()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, timeout)
                        for key, _events in selector.select(min(remaining, 0.25)):
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            if key.data == "stdout":
                                received += len(chunk)
                                if received > max_bytes:
                                    raise ValueError("download exceeds byte limit")
                                output.write(chunk)
                            else:
                                stderr.extend(chunk)
                                if len(stderr) > 1024 * 1024:
                                    raise ValueError("download stderr exceeds byte limit")
                while True:
                    self._check_cancelled()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        code = process.wait(timeout=min(.1, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                error_text = stderr.decode("utf-8", errors="replace")
                if code:
                    raise subprocess.CalledProcessError(code, command, stderr=error_text)
            temporary.replace(local)
            temporary = None
            return subprocess.CompletedProcess(command, 0, stdout="", stderr=error_text)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                process.stdout.close()
                process.stderr.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _remote_path(path: str):
    if not isinstance(path, str) or not path.startswith("/") or any(c in path for c in ("\0", "\n", "\r")):
        raise ValueError("remote file path must be absolute and contain no control characters")


def safe_extract_result(archive: Path, dest: Path, *, max_bytes: int = 2 * 1024 * 1024 * 1024,
                        max_files: int = 100000) -> Path:
    """Validate a result/ archive and atomically publish a fresh destination.

    A destination is never merged or overwritten. Links, special files,
    duplicate names, excessive output and paths outside result/ are rejected.
    """
    archive, dest = Path(archive), Path(dest).absolute()
    if max_bytes < 1 or max_files < 1:
        raise ValueError("extraction limits must be positive")
    if dest.exists() or dest.is_symlink():
        raise ValueError("result extraction destination already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=dest.parent))
    published = False
    try:
        with tarfile.open(archive, mode="r:*") as bundle:
            members = []
            names = set()
            size = 0
            for member in bundle:
                # GNU tar commonly spells directory members with a trailing '/'.
                name = member.name.rstrip("/") if member.isdir() else member.name
                path = _safe_relative(name)
                if path.parts[0] != "result" or (len(path.parts) == 1 and not member.isdir()):
                    raise ValueError("archive members must be under result/")
                if not member.isfile() and not member.isdir():
                    raise ValueError("archive links and special files are not accepted")
                if getattr(member, "sparse", None):
                    raise ValueError("sparse result files are not accepted")
                if name in names:
                    raise ValueError("duplicate result archive path")
                names.add(name)
                size += member.size
                if member.size < 0 or size > max_bytes or len(names) > max_files:
                    raise ValueError("result archive exceeds extraction limits")
                members.append((member, path))
            if not members:
                raise ValueError("empty result archive")
            regular_names = {path.as_posix() for member, path in members if member.isfile()}
            for _member, path in members:
                if any(parent.as_posix() in regular_names for parent in path.parents):
                    raise ValueError("result archive file conflicts with a parent directory")
            for member, path in members:
                target = staging.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    target.chmod(0o700)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if target.stat().st_size != member.size:
                    raise ValueError("result archive member size mismatch")
                target.chmod(0o600)
        if dest.exists() or dest.is_symlink():
            raise ValueError("result extraction destination appeared during extraction")
        staging.rename(dest)
        published = True
        return dest
    finally:
        if not published:
            shutil.rmtree(staging)
