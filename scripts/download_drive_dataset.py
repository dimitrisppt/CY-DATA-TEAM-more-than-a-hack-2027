"""Download a Drive dataset through the user's existing Codex connector login.

The app-server is used only for connector RPC; no model turns are started.
Files are streamed to disk, checked against provider sizes, and hashed locally.
Signed download references stay in memory and are never written to the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from possession.storage import sha256_file, write_json

FOLDER_URL = "https://drive.google.com/drive/folders/1eg7eByB8GlHkQm79bhiyUzCxQl-hwcd-"
# The connector reports HTTP 413 above this per-file download limit.
CONNECTOR_FILE_LIMIT = 268_435_456
TACTICAL_INPUTS = {
    "BAR-ATH_tactical_atd.txt", "BAR-ATH_tactical_atd.xml",
    "BAR-ATH_tactical_homography.csv", "BAR-ATH_tactical_smart_tagging.xml",
}


class DriveClient:
    """A small stdio RPC client using the installed, authenticated Codex runtime."""

    def __init__(self, cwd, log_path):
        self.resources = ExitStack()
        # ExitStack closes the log when the RPC process is closed.
        self.log = self.resources.enter_context(Path(log_path).open("a"))  # noqa: SIM115
        self.process = subprocess.Popen(
            ["codex", "app-server", "--stdio"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
        )
        self.messages = queue.Queue()
        self.request_id = 0

        def reader():
            for line in self.process.stdout:
                try:
                    self.messages.put(json.loads(line))
                except ValueError:
                    pass
            self.messages.put({"local_error": "Codex connector process exited"})

        threading.Thread(target=reader, daemon=True).start()
        self.call("initialize", {
            "clientInfo": {"name": "dataset_connector_client", "version": "1.0"},
            "capabilities": {"experimentalApi": True},
        })
        self.send({"method": "initialized"})
        self.call("app/installed", {"forceRefresh": True})
        response = self.call("thread/start", {"cwd": str(cwd), "ephemeral": True})
        self.thread_id = response["thread"]["id"]

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def call(self, method, params, *, timeout=1200, label=None):
        self.request_id += 1
        request_id = self.request_id
        self.send({"id": request_id, "method": method, "params": params})
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            try:
                value = self.messages.get(timeout=min(20, timeout - (time.monotonic() - started)))
            except queue.Empty:
                if label:
                    print(json.dumps({"phase": "staging_from_drive", "file": label,
                                      "elapsed_seconds": int(time.monotonic() - started)}), flush=True)
                continue
            if "local_error" in value:
                raise RuntimeError(value["local_error"])
            if value.get("id") == request_id and "method" not in value:
                if "error" in value:
                    raise RuntimeError(json.dumps(value["error"]))
                return value.get("result")
            if "id" in value and "method" in value:
                # Never fabricate answers to an interactive authentication/approval request.
                self.send({"id": value["id"], "error": {"code": -32601,
                           "message": "Interactive requests require the user-facing client."}})
                raise RuntimeError(f"Connector requested an interactive action: {value['method']}")
        raise TimeoutError(f"Connector request timed out: {method}")

    def fetch(self, url, *, raw=False, label=None):
        arguments = {"url": url}
        if raw:
            arguments.update(download_raw_file=True, include_base64=False)
        response = self.call("mcpServer/tool/call", {
            "threadId": self.thread_id, "server": "codex_apps",
            "tool": "google_drive.fetch", "arguments": arguments,
        }, label=label)
        if response.get("isError"):
            messages = " ".join(c.get("text", "") for c in response.get("content", [])
                                if c.get("type") == "text")
            messages = re.sub(r"https?://\S+", "<URL>", messages)
            raise RuntimeError(f"Google Drive fetch failed for {label or 'dataset folder'}: {messages[:1000]}")
        return response["structuredContent"]

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.resources.close()


def safe_name(name):
    if (not name or name in (".", "..") or Path(name).name != name
            or "/" in name or "\\" in name):
        raise ValueError("Unsafe dataset filename")
    return name


def inventory(client, folder_url, prefix="", seen=None):
    seen = set() if seen is None else seen
    response = client.fetch(folder_url)
    folder_id = response["id"]
    if folder_id in seen:
        raise ValueError("Repeated or cyclic dataset folder")
    seen.add(folder_id)
    children = json.loads(response["content"])["files"]
    if len(children) >= 100:
        raise ValueError("Folder reached the provider listing limit; refuse a partial download")
    result = []
    for child in children:
        relative = str(Path(prefix) / safe_name(child["title"]))
        if child["file_or_folder"] == "folder":
            result.extend(inventory(client, child["url"], relative, seen))
        else:
            result.append({"path": relative, "id": child["id"], "url": child["url"],
                           "bytes": int(child["size"]), "mime_type": child["mime_type"],
                           "modified_time": child.get("modified_time")})
    if len({row["path"] for row in result}) != len(result):
        raise ValueError("Duplicate destination paths in dataset")
    return result


def download(client, entry, output):
    final = output / entry["path"]
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.with_name(final.name + ".part")
    for attempt in range(3):
        response = client.fetch(entry["url"], raw=True, label=entry["path"])
        if (response.get("id") != entry["id"]
                or response.get("file_size_bytes") != entry["bytes"]
                or response.get("modified_time") != entry["modified_time"]):
            raise ValueError(f"Source changed during download: {entry['path']}")
        reference = response.get("file_uri")
        if not isinstance(reference, dict) or not reference.get("download_url"):
            raise ValueError("Connector did not provide a downloadable raw-file reference")
        digest, count = hashlib.sha256(), 0
        started = last_report = time.monotonic()
        try:
            with urllib.request.urlopen(reference["download_url"], timeout=90) as source:
                length = source.headers.get("Content-Length")
                if length is not None and int(length) != entry["bytes"]:
                    raise ValueError(f"HTTP length differs for {entry['path']}")
                with partial.open("wb") as target:
                    while block := source.read(4 * 1024 * 1024):
                        target.write(block)
                        digest.update(block)
                        count += len(block)
                        if time.monotonic() - last_report >= 15:
                            print(json.dumps({"phase": "downloading", "file": entry["path"],
                                              "bytes": count, "total_bytes": entry["bytes"]}), flush=True)
                            last_report = time.monotonic()
            if count != entry["bytes"]:
                raise ValueError(f"Downloaded size differs for {entry['path']}")
            partial.replace(final)
            print(json.dumps({"phase": "verified", "file": entry["path"], "bytes": count,
                              "seconds": round(time.monotonic() - started, 1)}), flush=True)
            return {**entry, "sha256": digest.hexdigest()}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(f"Transfer failed for {entry['path']}: {type(exc).__name__}") from None
            print(json.dumps({"phase": "retry", "file": entry["path"], "attempt": attempt + 1}), flush=True)
        finally:
            partial.unlink(missing_ok=True)


def run(folder_url, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "download-manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous.get("source") != folder_url:
        raise ValueError("Existing download manifest belongs to a different source")
    previous_files = {r["path"]: r for r in previous.get("files", [])}
    client = DriveClient(Path.cwd(), output / "connector.log")
    try:
        all_files = inventory(client, folder_url)
        files = [r for r in all_files if r["bytes"] <= CONNECTOR_FILE_LIMIT]
        oversized = [{**r, "reason": "connector_per_file_size_limit",
                      "limit_bytes": CONNECTOR_FILE_LIMIT}
                     for r in all_files if r["bytes"] > CONNECTOR_FILE_LIMIT]
        needed = sum(r["bytes"] for r in files if r["path"] not in previous_files)
        if shutil.disk_usage(output).free < needed + 1024**3:
            raise ValueError("Dataset download requires a 1 GiB working reserve")
        manifest = {"source": folder_url, "retrieved_via": "authenticated Google Drive Codex connector",
                    "started_utc": previous.get("started_utc", datetime.now(UTC).isoformat()),
                    "status": "downloading", "expected_files": len(all_files),
                    "expected_bytes": sum(r["bytes"] for r in all_files),
                    "skipped_files": oversized, "files": []}
        print(json.dumps({"phase": "inventory", "files": len(files),
                          "bytes": sum(r["bytes"] for r in files),
                          "oversized_files": len(oversized)}), flush=True)
        # Get model inputs first; videos are downloaded last.
        files.sort(key=lambda r: (r["mime_type"].startswith("video/"),
                                 r["path"] != "tactical/BAR-ATH_tactical_atd.txt", r["bytes"]))
        for entry in files:
            final = output / entry["path"]
            old = previous_files.get(entry["path"])
            if final.exists():
                if (old is None or any(old.get(k) != v for k, v in entry.items())
                        or final.stat().st_size != entry["bytes"]
                        or sha256_file(final) != old["sha256"]):
                    raise ValueError(f"Existing file differs or lacks provenance: {entry['path']}")
                record = old
            else:
                print(json.dumps({"phase": "fetching", "file": entry["path"],
                                  "bytes": entry["bytes"]}), flush=True)
                record = download(client, entry, output)
            # Preserve all previous receipts if a resumed transfer stops before
            # revisiting files that already exist on disk.
            previous_files[entry["path"]] = record
            expected_paths = {row["path"] for row in files}
            manifest["files"] = [row for path, row in previous_files.items()
                                 if path in expected_paths]
            write_json(manifest_path, manifest)
            tactical = [r for r in manifest["files"]
                        if r["path"].startswith("tactical/") and Path(r["path"]).name in TACTICAL_INPUTS]
            if len(tactical) == len(TACTICAL_INPUTS):
                write_json(output / "input-manifest.json", {
                    "source": folder_url, "scope": "tactical automatic inputs only, no video or labels",
                    "files": [{k: r[k] for k in ("path", "bytes", "sha256", "id", "url")}
                              for r in sorted(tactical, key=lambda r: r["path"])],
                })
        manifest.update(status="partial_provider_size_limit" if oversized else "complete",
                        downloaded_bytes=sum(r["bytes"] for r in manifest["files"]),
                        completed_utc=datetime.now(UTC).isoformat())
        write_json(manifest_path, manifest)
        return manifest
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-url", default=FOLDER_URL)
    parser.add_argument("--output", type=Path, default=Path("data/target/2027"))
    args = parser.parse_args()
    manifest = run(args.folder_url, args.output)
    print(json.dumps({"status": manifest["status"], "files": len(manifest["files"]),
                      "bytes": manifest["downloaded_bytes"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
