"""Airscope helper daemon.

Serves index.html on http://127.0.0.1:8765/ and provides a small JSON API
the page uses to discover and manage scopes:

    GET /discover        -> merged registry + last-known live state
    GET /rescan          -> re-issue /whoami against every registered scope
    GET /scan            -> probe ARP neighbors for /whoami (local LAN)
    GET /add-ip?ip=...   -> probe one IP, register on success
    GET /remove?mac=...  -> drop a scope from the registry
    GET /save?title=...  -> open a folder picker, return the chosen path

Also answers UDP "TIME" requests on 0.0.0.0:12345 with the timestamp
string the scope firmware uses as its SD folder name.

Stdlib only. Python 3.9+.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import ipaddress
import json
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8765
UDP_BIND = "0.0.0.0"
UDP_PORT = 12345
WHOAMI_TIMEOUT_S = 2.0
REGISTRY_PATH = HERE / "scopes_registry.json"
INDEX = "index.html"


# -- UDP time responder -------------------------------------------------------

def _udp_timestamp() -> str:
    return _dt.datetime.now().strftime("/%d-%b-%y/%H-%M-%S-%f")[:-3]


def udp_loop(stop: threading.Event) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_BIND, UDP_PORT))
    sock.settimeout(0.5)
    print(f"[udp] listening on {UDP_BIND}:{UDP_PORT}")
    while not stop.is_set():
        try:
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            continue
        except OSError as e:
            print(f"[udp] socket error: {e}")
            break
        if data.strip() == b"TIME":
            reply = _udp_timestamp().encode()
            sock.sendto(reply, addr)
            print(f"[udp] TIME from {addr[0]} -> {reply.decode()}")
    sock.close()


# -- network helpers ----------------------------------------------------------

def host_lan_ip() -> str | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def probe_whoami(ip: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://{ip}/whoami",
                                    timeout=WHOAMI_TIMEOUT_S) as resp:
            if resp.status != 200:
                return None
            info = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(info, dict) or "mac" not in info:
        return None
    info.setdefault("ip", ip)
    return info


async def probe_whoami_async(ip: str) -> dict | None:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, 80), timeout=0.5)
    except (asyncio.TimeoutError, OSError):
        return None
    try:
        writer.write(f"GET /whoami HTTP/1.0\r\nHost: {ip}\r\n"
                     f"Connection: close\r\n\r\n".encode())
        await writer.drain()
        data = await asyncio.wait_for(reader.read(4096), timeout=1.5)
    except (asyncio.TimeoutError, OSError):
        data = b""
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
    if b"\r\n\r\n" not in data:
        return None
    try:
        info = json.loads(data.split(b"\r\n\r\n", 1)[1]
                          .decode("utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(info, dict) or "mac" not in info:
        return None
    info["ip"] = ip
    return info


_IFACE_RE = re.compile(r"Interface:\s+(\d+\.\d+\.\d+\.\d+)")
_ARP_RE = re.compile(
    r"^\s*(\d+\.\d+\.\d+\.\d+)\s+"
    r"([0-9a-fA-F]{2}(?:[-:][0-9a-fA-F]{2}){5})\s+(\w+)"
)


def arp_neighbors(host_ip: str) -> list[tuple[str, str]]:
    """Parse `arp -a` for dynamic neighbors on the interface matching host_ip."""
    try:
        out = subprocess.check_output(["arp", "-a"], text=True, errors="ignore")
    except Exception as e:
        print(f"[arp] failed: {e}")
        return []
    neighbors: list[tuple[str, str]] = []
    in_iface = False
    for line in out.splitlines():
        m = _IFACE_RE.search(line)
        if m:
            in_iface = m.group(1) == host_ip
            continue
        if not in_iface:
            continue
        m = _ARP_RE.match(line)
        if not m:
            continue
        ip, mac, kind = m.group(1), m.group(2), m.group(3).lower()
        if kind != "dynamic" or ip == host_ip:
            continue
        first, last = int(ip.split(".")[0]), int(ip.split(".")[-1])
        if first >= 224 or last == 255:
            continue
        neighbors.append((ip, mac))
    return neighbors


# -- registry ----------------------------------------------------------------

class Registry:
    """Persistent list of known scopes, keyed by MAC."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._scopes: dict[str, dict] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("scopes", []):
                mac = entry.get("mac")
                if mac:
                    self._scopes[mac.upper()] = entry
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[registry] load failed ({e}); starting empty")

    def _save_locked(self) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"scopes": list(self._scopes.values())}, indent=2),
            encoding="utf-8")
        tmp.replace(self._path)

    def upsert(self, mac: str, device_name: str | None,
               ip: str | None, source: str) -> None:
        if not mac:
            return
        mac = mac.upper()
        with self._lock:
            entry = self._scopes.setdefault(mac, {"mac": mac})
            if device_name:
                entry["device_name"] = device_name
            if ip:
                entry["ip"] = ip
            entry["last_seen"] = time.time()
            entry["last_source"] = source
            self._save_locked()

    def remove(self, mac: str) -> bool:
        mac = (mac or "").upper()
        with self._lock:
            if mac not in self._scopes:
                return False
            del self._scopes[mac]
            self._save_locked()
            return True

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(s) for s in self._scopes.values()]

    def known_ips(self) -> list[str]:
        with self._lock:
            return [s["ip"] for s in self._scopes.values() if s.get("ip")]


# -- discovery ---------------------------------------------------------------

class Discovery:
    """Combines the registry with live /whoami results + on-demand scans."""

    def __init__(self, registry: Registry) -> None:
        self._lock = threading.Lock()
        self._live: dict[str, dict] = {}
        self._host_ip: str | None = None
        self._last_poll_ts = 0.0
        self._last_scan_ts = 0.0
        self._last_scan_summary: dict = {}
        self._registry = registry

    def snapshot(self) -> dict:
        with self._lock:
            live = dict(self._live)
            host_ip = self._host_ip
            last_poll = self._last_poll_ts
            last_scan = self._last_scan_ts
            last_scan_summary = dict(self._last_scan_summary)
        scopes = []
        for entry in self._registry.all():
            mac = entry.get("mac", "").upper()
            merged = dict(entry)
            if mac in live:
                merged["online"] = True
                merged.update(live[mac])
            else:
                merged["online"] = False
            scopes.append(merged)
        return {
            "host_ip": host_ip,
            "last_poll_ts": last_poll,
            "last_scan_ts": last_scan,
            "last_scan": last_scan_summary,
            "scopes": scopes,
        }

    def poll_registry(self) -> None:
        """Hit /whoami at every registered IP in parallel."""
        ips = self._registry.known_ips()
        results: dict[str, dict | None] = {}

        def worker(ip):
            results[ip] = probe_whoami(ip)

        threads = [threading.Thread(target=worker, args=(ip,), daemon=True)
                   for ip in ips]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=WHOAMI_TIMEOUT_S + 0.4)

        live: dict[str, dict] = {}
        for ip, info in results.items():
            if not info:
                continue
            mac = (info.get("mac") or "").upper()
            if mac:
                live[mac] = info
                self._registry.upsert(mac, info.get("device_name"),
                                      info.get("ip"), source="whoami")
        host_ip = host_lan_ip()
        with self._lock:
            self._live = live
            self._last_poll_ts = time.time()
            self._host_ip = host_ip
        print(f"[discover] live={len(live)} "
              f"registry={len(self._registry.all())} host_ip={host_ip}")

    def scan_local(self) -> dict:
        host_ip = host_lan_ip()
        if not host_ip:
            return {"ok": False, "error": "no LAN IP on this host"}
        t0 = time.time()
        try:
            found, neighbors, total = asyncio.run(
                self._scan_local_async(host_ip))
        except Exception as e:
            return {"ok": False, "error": str(e)}
        for info in found:
            mac = (info.get("mac") or "").upper()
            if mac:
                self._registry.upsert(mac, info.get("device_name"),
                                      info.get("ip"), source="scan")
        elapsed = time.time() - t0
        summary = {
            "ok": True, "host_ip": host_ip, "scanned": total,
            "found": len(found), "elapsed_s": round(elapsed, 2),
            "neighbors": neighbors, "scopes": found,
        }
        with self._lock:
            self._last_scan_ts = time.time()
            self._last_scan_summary = {k: v for k, v in summary.items()
                                       if k not in ("scopes", "neighbors")}
        print(f"[scan] host_ip={host_ip} neighbors={total} "
              f"found={len(found)} elapsed={elapsed:.2f}s")
        return summary

    @staticmethod
    async def _scan_local_async(host_ip: str
                                ) -> tuple[list[dict], list[dict], int]:
        neighbors = arp_neighbors(host_ip)
        if not neighbors:
            return [], [], 0
        results = await asyncio.gather(
            *(probe_whoami_async(ip) for ip, _ in neighbors))
        found, summary = [], []
        for (ip, mac), info in zip(neighbors, results):
            is_scope = info is not None
            summary.append({"ip": ip, "mac": mac, "is_airscope": is_scope})
            if is_scope:
                found.append(info)
        return found, summary, len(neighbors)

    def add_by_ip(self, ip: str) -> dict:
        info = probe_whoami(ip)
        if not info:
            return {"ok": False, "ip": ip,
                    "error": "no /whoami response"}
        mac = (info.get("mac") or "").upper()
        if not mac:
            return {"ok": False, "ip": ip, "error": "response missing mac"}
        self._registry.upsert(mac, info.get("device_name"),
                              info.get("ip"), source="manual")
        return {"ok": True, "scope": info}

    def remove(self, mac: str) -> dict:
        return {"ok": self._registry.remove(mac), "mac": mac.upper()}


# -- folder picker ------------------------------------------------------------

def ask_folder(title: str) -> str | None:
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:
        return None
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        return filedialog.askdirectory(title=title) or None
    finally:
        root.destroy()


# -- HTTP --------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    discovery: Discovery  # set in main()

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[http] {self.address_string()} - {fmt % args}\n")

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND, "file not found")
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js":   "application/javascript",
            ".css":  "text/css",
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".svg":  "image/svg+xml",
            ".json": "application/json",
        }.get(path.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        d = self.discovery

        if url.path == "/discover":
            self._json(d.snapshot()); return
        if url.path == "/rescan":
            threading.Thread(target=d.poll_registry, daemon=True).start()
            self._json({"ok": True}); return
        if url.path == "/scan":
            self._json(d.scan_local()); return
        if url.path == "/add-ip":
            ip = (qs.get("ip") or [""])[0].strip()
            if not ip:
                self._json({"ok": False, "error": "missing ip"}, 400); return
            try:
                ipaddress.IPv4Address(ip)
            except ValueError:
                self._json({"ok": False, "error": "bad ip"}, 400); return
            self._json(d.add_by_ip(ip)); return
        if url.path == "/remove":
            mac = (qs.get("mac") or [""])[0].strip()
            if not mac:
                self._json({"ok": False, "error": "missing mac"}, 400); return
            self._json(d.remove(mac)); return
        if url.path == "/save":
            title = (qs.get("title") or ["Choose folder"])[0]
            self._json({"path": ask_folder(title) or ""}); return

        # static files (default → INDEX)
        rel = url.path.lstrip("/") or INDEX
        target = (HERE / rel).resolve()
        if HERE not in target.parents and target != HERE / rel:
            self.send_error(HTTPStatus.FORBIDDEN); return
        self._file(target)


# -- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-browser", action="store_true",
                    help="don't auto-open the browser")
    ap.add_argument("--port", type=int, default=HTTP_PORT)
    args = ap.parse_args()

    registry = Registry(REGISTRY_PATH)
    discovery = Discovery(registry)
    Handler.discovery = discovery

    stop = threading.Event()
    threading.Thread(target=udp_loop, args=(stop,), daemon=True).start()
    threading.Thread(target=discovery.poll_registry, daemon=True).start()

    server = ThreadingHTTPServer((HTTP_HOST, args.port), Handler)
    url = f"http://{HTTP_HOST}:{args.port}/"
    print(f"[http] serving {HERE} on {url}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[main] shutting down")
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
