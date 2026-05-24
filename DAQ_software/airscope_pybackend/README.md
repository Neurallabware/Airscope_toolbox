# Airscope host (clean Python edition)

A minimal local daq software that pairs with the Airscope firmware. Two files:

| File         | Role                                                              |
|--------------|-------------------------------------------------------------------|
| `server.py`  | Backend. HTTP server on `127.0.0.1:8765` + UDP responder on `:12345`. |
| `index.html` | Frontend. Single-page UI served by `server.py`.                   |

A third file, `scopes_registry.json`, is created on first run to remember
discovered scopes across restarts.

## Requirements

- Python 3.9+ (standard library only — no `pip install` needed)
- Windows / macOS / Linux. Local LAN scan uses `arp -a`, present on all three.
- A modern Chromium-based browser (Chrome / Edge / Opera) if you want the
  embedded Serial Monitor — it uses Web Serial.

## Running

```sh
python server.py
```

That's it. The script:

1. Starts the HTTP server on `http://127.0.0.1:8765/`.
2. Starts a UDP listener on `0.0.0.0:12345` that answers `TIME` packets
   with a timestamp string the firmware uses to name its SD folder.
3. Opens your default browser at the local URL.

Flags:

- `--no-browser` — don't auto-open the browser.
- `--port N` — serve HTTP on a different port (UDP port is fixed).

Stop with `Ctrl-C`.

## Frontend (index.html)

The page is what you actually interact with. Refer to wiki for usage instructions.

## Backend (server.py)

Endpoints the frontend calls (all `GET`):

| Path           | Purpose                                              |
|----------------|------------------------------------------------------|
| `/discover`    | Cached registry + last-known live state              |
| `/rescan`      | Re-issue `/whoami` against every registered scope    |
| `/scan`        | Probe ARP-table neighbors for `/whoami`              |
| `/add-ip?ip=`  | Probe one IP, register on success                    |
| `/remove?mac=` | Drop a scope from the registry                       |
| `/save?title=` | Open a folder picker (Tk), return the chosen path    |
| `/<filename>`  | Serve any file from this directory (default `index.html`) |

UDP: a single thread binds `0.0.0.0:12345` and replies to `TIME` packets
with `/DD-Mon-YY/HH-MM-SS-mmm`. Scopes use this to pick their SD folder
name when recording starts.
