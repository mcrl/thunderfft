"""Small Vast.ai API adapter and conservative, explicit USD cost planning.

API contracts: https://docs.vast.ai/api-reference/search/search-offers,
instances/{create-instance,attach-ssh-key,destroy-instance}, accounts/show-user,
and billing/show-charges (checked 2026-11-04). No credentials are sent to workers.
The caller owns the independent watchdog, polling, and finally-cleanup. Requests,
especially creation, are never automatically retried by this module.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request


API_ORIGIN = "https://console.vast.ai"
DEFAULT_KEY_FILE = Path("~/cloud_key/thunderfft/vast_api_key")
MAX_BUDGET_USD = 10.0
GPU_NAMES = {"A100": "A100 PCIE", "H100": "H100 PCIE", "B200": "B200"}


class ProviderError(RuntimeError):
    """Sanitized operational error suitable for an audit log."""


class CreateOutcomeUnknown(ProviderError):
    """A create request may have succeeded: discover by its unique label, never retry."""


class CreateRejected(ProviderError):
    """A definitive HTTP rejection prevented creation; no reconciliation is needed."""


class HTTPStatusError(ProviderError):
    def __init__(self, code: int):
        self.code = code
        super().__init__(f"Vast.ai returned HTTP {code}; response body suppressed")


def read_secret_file(path: Path | str, *, repo_root: Path | str | None = None) -> str:
    """Read an owner-only regular file; reject links, repo files, and unsafe modes."""
    path = Path(path).expanduser().absolute()
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    try:
        if path.resolve().is_relative_to(root):
            raise ProviderError("Credentials must be stored outside the repository")
        directory = path.parent.stat()
        if directory.st_uid != os.getuid() or stat.S_IMODE(directory.st_mode) != 0o700:
            raise ProviderError("Credential directory must be owned by the current user with permissions 0700")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ProviderError("Credential file must be a regular file owned by the current user")
            if stat.S_IMODE(info.st_mode) != 0o600:
                raise ProviderError("Credential file permissions must be 0600")
            value = stream.read(16385).strip()
    except ProviderError:
        raise
    except (OSError, UnicodeError):
        raise ProviderError("Cannot read credential file safely (check path, owner, and permissions)") from None
    if not value or len(value) > 16384 or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in value):
        raise ProviderError("API key file must contain one nonempty ASCII token")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _transport(method: str, url: str, headers: dict, payload: dict | None,
               timeout: float) -> dict:
    """No redirect or request retry; bearer credentials stay on the fixed HTTPS origin."""
    request = urllib.request.Request(
        url, method=method, headers=headers,
        data=None if payload is None else json.dumps(payload).encode("utf-8"))
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                raise HTTPStatusError(response.status)
            data = response.read(8 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as error:
        raise HTTPStatusError(error.code) from None
    if len(data) > 8 * 1024 * 1024:
        raise ProviderError("Vast.ai response exceeds the size limit")
    return json.loads(data)


def _number(value, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        raise ProviderError(f"Missing or invalid {name}")
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ProviderError(f"Missing or invalid {name}") from None
    if not math.isfinite(result) or result < minimum:
        raise ProviderError(f"Missing or invalid {name}")
    return result


def _id(value) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ProviderError("Invalid Vast.ai resource ID")
    return int(value)


def gpu_matches(gpu: str, offer: dict) -> bool:
    """Paper SKU matching: PCIe 80 GB A100/H100; B200, never SXM/H100 NVL/GB200."""
    if gpu not in GPU_NAMES:
        raise ProviderError("GPU must be A100, H100, or B200")
    name = re.sub(r"[ _-]+", " ", str(offer.get("gpu_name", "")).upper()).strip()
    name = re.sub(r"^NVIDIA ", "", name)
    if name not in {GPU_NAMES[gpu], GPU_NAMES[gpu] + " 80GB"}:
        return False
    try:
        memory = _number(offer.get("gpu_ram"), "GPU memory")
        count = _number(offer.get("num_gpus"), "GPU count")
    except ProviderError:
        return False
    if count != 1:
        return False
    return 79000 <= memory <= 84000 if gpu in ("A100", "H100") else memory >= 175000


def plan_cost(offer: dict, *, disk_gb: float, startup_seconds: float,
              execution_seconds: float, download_seconds: float,
              cleanup_seconds: float = 120, billing_delay_seconds: float = 360,
              upload_gb: float, download_gb: float, budget_usd: float = 1.5,
              reserve_usd: float = .25) -> dict:
    """Fail closed on unknown prices or plans exceeding the all-in ceiling.

    ``upload_gb`` means all ingress to the rented machine (source, image, packages),
    ``download_gb`` means all egress. Include retries in these bounds. Storage uses
    a 30-day month and is deliberately added to dph_total even if an offer already
    includes it, so ambiguous quoted totals cannot underestimate storage.
    """
    budget = _number(budget_usd, "budget", minimum=.25)
    reserve = _number(reserve_usd, "cleanup reserve", minimum=.25)
    if budget > MAX_BUDGET_USD:
        raise ProviderError(f"Per-execution budget must not exceed USD {MAX_BUDGET_USD:g}")
    disk = _number(disk_gb, "disk size", minimum=1)
    phases = {
        "startup_seconds": _number(startup_seconds, "startup timeout", minimum=1),
        "execution_seconds": _number(execution_seconds, "execution timeout", minimum=1),
        "download_seconds": _number(download_seconds, "download timeout", minimum=1),
        "cleanup_seconds": _number(cleanup_seconds, "cleanup timeout", minimum=1),
        "billing_delay_seconds": _number(billing_delay_seconds, "billing delay", minimum=360),
    }
    compute_rate = _number(offer.get("dph_total"), "dph_total")
    storage_rate = _number(offer.get("storage_cost"), "storage_cost ($/GB/month)")
    ingress_rate = _number(offer.get("inet_down_cost"), "inet_down_cost ($/GB)")
    egress_rate = _number(offer.get("inet_up_cost"), "inet_up_cost ($/GB)")
    ingress = _number(upload_gb, "upload bound (GB)")
    egress = _number(download_gb, "download bound (GB)")
    billed_seconds = sum(phases.values())
    storage_hourly = storage_rate * disk / (30 * 24)
    compute_cost = compute_rate * billed_seconds / 3600
    storage_cost = storage_hourly * billed_seconds / 3600
    network_cost = ingress_rate * ingress + egress_rate * egress
    total = compute_cost + storage_cost + network_cost + reserve
    if not math.isfinite(total) or total > budget + 1e-12:
        raise ProviderError(f"Conservative all-in plan USD {total:.4f} exceeds budget USD {budget:.2f}")
    return {
        "currency": "USD", "budget_usd": budget, "hard_ceiling_usd": MAX_BUDGET_USD,
        "reserve_usd": reserve, **phases,
        "billed_seconds": billed_seconds,
        "active_seconds": billed_seconds - phases["billing_delay_seconds"],
        "compute_usd_per_hour": compute_rate, "storage_usd_per_hour": storage_hourly,
        "ingress_usd_per_gb": ingress_rate, "egress_usd_per_gb": egress_rate,
        "upload_gb": ingress, "download_gb": egress, "disk_gb": disk,
        "compute_usd": compute_cost, "storage_usd": storage_cost,
        "network_usd": network_cost, "planned_total_usd": total,
        "storage_accounting": "dph_total plus explicit storage (conservative if already included)",
        "provider_enforced_cap": False,
    }


class VastClient:
    def __init__(self, api_key_file: Path | str = DEFAULT_KEY_FILE, *,
                 transport: Callable | None = None, timeout: float = 20,
                 repo_root: Path | str | None = None):
        self._key = read_secret_file(api_key_file, repo_root=repo_root)
        self._transport = transport or _transport
        self.timeout = _number(timeout, "API timeout", minimum=1)
        self._last_instances_request = None

    def _request(self, method: str, path: str, payload: dict | None = None,
                 query: dict | None = None) -> dict:
        if not path.startswith("/api/") or "?" in path or "#" in path:
            raise ProviderError("Invalid API endpoint")
        url = API_ORIGIN + path
        if query:
            url += "?" + urllib.parse.urlencode({
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, bool)) else value
                for key, value in query.items()})
        try:
            data = self._transport(method, url, {"Authorization": "Bearer " + self._key,
                "Content-Type": "application/json", "Accept": "application/json"}, payload, self.timeout)
        except HTTPStatusError:
            raise
        except Exception:
            # Transport exceptions and provider error bodies can contain URLs or
            # headers. Never echo them or retain their chained traceback.
            raise ProviderError("Vast.ai request failed; transport details suppressed") from None
        if not isinstance(data, dict):
            raise ProviderError("Invalid Vast.ai response structure")
        if data.get("success") is False or "error" in data:
            raise ProviderError("Vast.ai rejected the request; response body suppressed")
        return data

    def search_offers(self, gpu: str, disk_gb: float, limit: int = 100) -> list[dict]:
        if gpu not in GPU_NAMES:
            raise ProviderError("GPU must be A100, H100, or B200")
        query = {
            "verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False},
            "num_gpus": {"eq": 1}, "gpu_name": {"eq": GPU_NAMES[gpu]},
            "gpu_ram": {"gte": 79000 if gpu != "B200" else 175000},
            "disk_space": {"gte": _number(disk_gb, "disk size", minimum=1)},
            "reliability": {"gte": .98}, "direct_port_count": {"gte": 1},
            "cuda_max_good": {"gte": 12.9 if gpu == "A100" else 13.0}, "cpu_arch": {"eq": "amd64"},
            "type": "ondemand", "allocated_storage": disk_gb,
            "limit": min(_id(limit), 1000), "order": [["dph_total", "asc"]],
        }
        data = self._request("POST", "/api/v0/bundles/", query)
        offers = data.get("offers")
        if not isinstance(offers, list) or any(not isinstance(row, dict) for row in offers):
            raise ProviderError("Vast.ai offer response is not a list")
        return [row for row in offers if gpu_matches(gpu, row)]

    def account_summary(self) -> dict:
        data = self._request("GET", "/api/v0/users/current/")
        # Never persist names, email, SSH keys, session IDs, or payment details.
        return {"balance_usd": _number(data.get("balance"), "account balance", minimum=-1e12),
                "observed_at": time.time()}

    def instances(self) -> list[dict]:
        # The v0 list route returns HTTP 410. v1 is paginated (25 rows/page),
        # and absence is meaningful only after every page has been read.
        columns = ["id", "label", "actual_status", "ssh_host", "ssh_port",
                   "gpu_name", "gpu_ram", "num_gpus", "start_date"]
        query = {"select_filters": {}, "select_cols": columns,
                 "order_by": [{"col": "id", "dir": "asc"}], "limit": 25}
        rows, seen_tokens, seen_ids = [], set(), set()
        for _ in range(100):
            # Also space separate calls: cleanup lists immediately before and
            # after DELETE. Without this, the second list may repeatedly hit
            # v1's one-request/second limit even when deletion succeeded.
            if self._last_instances_request is not None:
                wait = 1.05 - (time.monotonic() - self._last_instances_request)
                if wait > 0:
                    time.sleep(wait)
            self._last_instances_request = time.monotonic()
            data = self._request("GET", "/api/v1/instances/", query=query)
            batch = data.get("instances")
            if ("next_token" not in data or not isinstance(batch, list)
                    or any(not isinstance(row, dict) or "id" not in row for row in batch)):
                raise ProviderError("Invalid instance listing; absence cannot be established")
            for row in batch:
                identity = _id(row["id"])
                if identity in seen_ids:
                    raise ProviderError("Duplicate instance across pages; absence cannot be established")
                seen_ids.add(identity)
                # Defensive redaction even if select_cols is ignored upstream.
                rows.append({key: row.get(key) for key in columns})
            token = data["next_token"]
            if token is None:
                return rows
            if not isinstance(token, str) or not token or token in seen_tokens:
                raise ProviderError("Invalid instance pagination; absence cannot be established")
            seen_tokens.add(token)
            query["after_token"] = token
        raise ProviderError("Instance pagination limit reached; absence cannot be established")

    def billing_preflight(self) -> dict:
        """Verify billing_read access before incurring charges; retain no rows."""
        observed = time.time()
        day = int(observed // 86400) * 86400
        data = self._request("GET", "/api/v0/charges/", query={
            "select_filters": {"day": {"gte": day, "lte": day + 86400},
                               "type": {"in": ["instance"]}},
            "format": "table", "limit": 1})
        if not isinstance(data.get("results"), list):
            raise ProviderError("Invalid billing response; billing permission was not verified")
        return {"billing_read_verified": True, "observed_at": observed}

    def create_instance(self, offer_id: int, *, image: str, disk_gb: float,
                        label: str, onstart: str = "") -> int:
        offer_id = _id(offer_id)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", label):
            raise ProviderError("Instance label must be a unique simple identifier")
        if not image or any(c.isspace() for c in image):
            raise ProviderError("Invalid container image")
        payload = {"client_id": "me", "image": image,
                   "disk": _number(disk_gb, "disk size", minimum=1),
                   "label": label, "runtype": "ssh_direc ssh_proxy",
                   "cancel_unavail": True, "env": {}, "onstart": onstart}
        try:
            data = self._request("PUT", f"/api/v0/asks/{offer_id}/", payload)
            if data.get("success") is not True:
                raise ProviderError("Missing creation acknowledgement")
            return _id(data.get("new_contract"))
        except ProviderError as error:
            if isinstance(error, HTTPStatusError) and error.code in {400, 401, 403, 404, 405, 422}:
                raise CreateRejected(f"Vast.ai rejected creation with HTTP {error.code}; response body suppressed") from None
            # Even a malformed/failed acknowledgement must first be reconciled
            # against the unique label; the caller must never duplicate rental.
            raise CreateOutcomeUnknown(
                "Create outcome requires reconciliation by unique label; do not retry creation") from None

    def attach_ssh_key(self, instance_id: int, public_key: str) -> None:
        # Comments need not be disclosed to the provider.
        fields = public_key.strip().split()
        if len(fields) < 2 or fields[0] not in {"ssh-ed25519", "ssh-rsa", "ecdsa-sha2-nistp256"} or not re.fullmatch(r"[A-Za-z0-9+/=]+", fields[1]):
            raise ProviderError("Invalid SSH public key")
        data = self._request("POST", f"/api/v0/instances/{_id(instance_id)}/ssh/",
                             {"ssh_key": " ".join(fields[:2])})
        if data.get("success") is not True:
            raise ProviderError("SSH key attachment was not acknowledged")

    def destroy_instance(self, instance_id: int) -> None:
        try:
            data = self._request("DELETE", f"/api/v0/instances/{_id(instance_id)}/")
        except HTTPStatusError as error:
            if error.code == 404:
                return  # Caller still has to verify absence using the listing.
            raise
        if data.get("success") is not True:
            raise ProviderError("Instance destruction was not acknowledged; verify and retry cleanup")

    def verify_absent(self, instance_id: int) -> bool:
        target = _id(instance_id)
        return all(_id(row["id"]) != target for row in self.instances())

    def billing_evidence(self, start_epoch: float, end_epoch: float,
                         instance_id: int | None = None) -> dict:
        """Read per-instance charge rows; disappearance is never final settlement.

        Charges are grouped by UTC day by Vast, so the filter spans complete days.
        Payment invoices contain unrelated financial details and are not fetched.
        At most 20 pages are read; a truncation is explicit in the returned record.
        """
        start = _number(start_epoch, "billing start")
        end = _number(end_epoch, "billing end", minimum=start)
        target = _id(instance_id) if instance_id is not None else None
        query = {"select_filters": {"day": {"gte": int(start // 86400) * 86400,
                                             "lte": (int(end // 86400) + 1) * 86400},
                                    "type": {"in": ["instance"]}},
                 "limit": 100, "format": "table", "latest_first": True}
        rows, token, seen = [], None, set()
        for _ in range(20):
            data = self._request("GET", "/api/v0/charges/", query=query)
            batch = data.get("results")
            if not isinstance(batch, list) or any(not isinstance(row, dict) for row in batch):
                raise ProviderError("Invalid billing response")
            for row in batch:
                match = re.fullmatch(r"instance-([1-9][0-9]*)", str(row.get("source", "")))
                if not match or (target is not None and int(match[1]) != target):
                    continue
                rows.append(self._charge_row(row))
            token = data.get("next_token")
            if not token:
                break
            if not isinstance(token, str) or token in seen:
                raise ProviderError("Invalid billing pagination token")
            seen.add(token)
            query["after_token"] = token
        return {"observed_at": time.time(), "instance_id": target,
                "requested_start_epoch": start, "requested_end_epoch": end,
                "charge_rows": rows, "observed_charge_usd": sum(row["amount_usd"] for row in rows),
                "pagination_complete": not bool(token), "settlement_verified": False,
                "settlement_note": "Charges may be delayed; repeated stable reads do not prove final settlement.",
                "account": self.account_summary()}

    @staticmethod
    def _charge_row(row: dict, depth: int = 0) -> dict:
        if depth > 8:
            raise ProviderError("Invalid billing item nesting")
        result = {"type": str(row.get("type", ""))[:64],
                  "source": str(row.get("source", ""))[:64],
                  "start": row.get("start"), "end": row.get("end"),
                  "amount_usd": _number(row.get("amount"), "charge amount", minimum=-1e12)}
        # Keep numerical usage breakdown, not free-form descriptions or metadata.
        children = row.get("items", [])
        if not isinstance(children, list) or any(not isinstance(item, dict) for item in children):
            raise ProviderError("Invalid billing breakdown")
        result["items"] = [VastClient._charge_row(item, depth + 1) for item in children]
        return result
