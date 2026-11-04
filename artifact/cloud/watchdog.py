"""Detached coordinator watchdog, armed and acknowledged before rental creation."""
import json
import os
from pathlib import Path
import sys
import time

from .provider import VastClient, ProviderError


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def destroy_owned(client, state):
    """Delete only this execution's known ID or unguessable unique label."""
    instances = client.instances()
    found = [row for row in instances if row.get("label") == state["label"]]
    known = state.get("instance_id")
    if known:
        found.extend(row for row in instances if int(row["id"]) == known and row not in found)
    if found and not known:
        # Preserve a discovered ambiguous creation even if the subsequent
        # deletion or confirmation request fails.
        known = state["instance_id"] = int(found[0]["id"])
    for row in found:
        client.destroy_instance(int(row["id"]))
    after = client.instances()
    absent = not any(row.get("label") == state["label"] or (known and int(row["id"]) == known) for row in after)
    # An ambiguous creation with no discovered ID may still materialize later.
    confirmed = absent and (bool(known) or bool(found) or not state.get("create_attempted"))
    return {"absent": absent, "confirmed": confirmed,
            "destroyed_ids": sorted({int(row["id"]) for row in found}), "observed_at": time.time()}


def main():
    folder = Path(sys.argv[1]).resolve()
    state = json.loads((folder / "state.json").read_text())
    client = VastClient(state["api_key_file"])
    write_json(folder / "watchdog-ready.json", {"pid": os.getpid(), "armed_at": time.time(), "deadline": state["deadline"]})
    while True:
        state = json.loads((folder / "state.json").read_text())
        if state.get("cleanup_confirmed") or not state.get("create_attempted") and state.get("finished"):
            return 0
        if time.time() < state["deadline"]:
            time.sleep(max(0, min(2, state["deadline"] - time.time())))
            continue
        try:
            cleanup = destroy_owned(client, state)
            write_json(folder / "watchdog-cleanup.json", cleanup)
            if cleanup["confirmed"]:
                # Read charges after deletion as well. A later --cloud-recover
                # refreshes evidence; no claim of settled billing is made.
                for instance_id in set(cleanup["destroyed_ids"] + ([state["instance_id"]] if state.get("instance_id") else [])):
                    try:
                        evidence = client.billing_evidence(state["created_at"], time.time(), instance_id)
                        write_json(folder / f"watchdog-billing-{instance_id}.json", evidence)
                    except ProviderError:
                        write_json(folder / "watchdog-billing-error.json", {"status": "unavailable", "settlement_verified": False})
                return 0
        except ProviderError as error:
            write_json(folder / "watchdog-error.json", {"error": str(error), "observed_at": time.time()})
        if state.get("instance_id"):
            latest = json.loads((folder / "state.json").read_text())
            latest["instance_id"] = state["instance_id"]
            write_json(folder / "state.json", latest)
        # Stay alive on API errors and ambiguous create responses. No second
        # resource is ever created, and the inherited coordinator lock stays held.
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
