#!/usr/bin/env python3
"""yuy: a semantic command router (EmbeddingGemma 2 + llama.cpp).

A plain sentence is matched against the commands in commands.json using embeddings,
and the best match is executed. Only commands from that list can ever run.

The embedding server (llama-server) is started in the background when it is not
running and is never started twice. The llama-server and model paths live in config.json.

Per-command options in commands.json:
  "args": "expression"  -> math expression extracted from the sentence
  "args": "query"       -> the whole sentence is passed as a single argument
  "keywords": [...]     -> words/phrases that add a score bonus when found in the sentence
  "sudo": true          -> run with sudo, always asks for confirmation
  "timeout": seconds    -> time limit (default 30, or 3600 for sudo)
"cwd": "." means the folder where yuy was started.
Placeholders in cmd and cwd: {ROOT} (project folder), {LLAMA_SERVER}.

Commands whose working directory is inside the home folder run without confirmation
(AUTO_ALLOW_HOME). sudo commands and folders outside home always ask first.

Usage:
  yuy "your sentence"
  yuy --list      list commands
  yuy --status    embedding server status
  yuy --stop      stop the embedding server
"""
import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = f"{HERE}/config.json"


def load_config():
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"Failed to read {CONFIG_FILE}: {e}")
        return {}


CONFIG = load_config()

# --- embedding server (llama-server) ---
HOST = "127.0.0.1"
PORT = int(CONFIG.get("port", 8081))
LLAMA_SERVER = os.path.expanduser(CONFIG.get("llama_server") or shutil.which("llama-server") or "")
EMBED_MODEL = os.path.expanduser(CONFIG.get("embed_model") or "")
SERVER_ARGS = ["--embeddings", "-ngl", "99", "-c", "8192", "-b", "2048", "-ub", "2048",
               "--host", HOST, "--port", str(PORT)]
LOG_FILE = f"{HERE}/llama-embed.log"
LOCK_FILE = f"{HERE}/.llama-embed.lock"
START_TIMEOUT = 120  # seconds to wait for the server to become ready

BASE_URL = f"http://{HOST}:{PORT}"
EMBED_URL = f"{BASE_URL}/v1/embeddings"
HEALTH_URL = f"{BASE_URL}/health"

# --- router ---
THRESHOLD = 0.75         # minimum score of the best command
MIN_MARGIN = 0.03        # minimum gap to the runner-up, so the choice is clear
BOOST = 0.10             # score bonus: sentence has an expression (for "expression") or a keyword
AUTO_ALLOW_HOME = True   # working directory inside home -> run without confirmation (except sudo)
COMMANDS_FILE = f"{HERE}/commands.json"

# word -> operator, applied before extracting the expression (Indonesian and English)
WORD_OPS = [
    (r"\b(?:ditambah|tambah|plus)\b", "+"),
    (r"\b(?:dikurangi|dikurang|kurang|minus)\b", "-"),
    (r"\b(?:dikalikan|dikali|kali|times|multiplied by)\b", "*"),
    (r"\b(?:dibagi|bagi|divided by)\b", "/"),
    (r"\b(?:pangkat|to the power of|power)\b", "^"),
    (r"\b(?:modulo|mod)\b", "%"),
    (r"(?<=\d)\s*[x×]\s*(?=\d)", "*"),
    (r"÷", "/"),
]
EXPR_CHARS = re.compile(r"[\d.()+\-*/%^\s]+")
HAS_EXPR = re.compile(r"\d\s*[+\-*/%^]\s*[\d(]|\)\s*[+\-*/%^]")


# ---------------------------------------------------------------- server

def find_servers(port):
    """PIDs of all llama-server processes using --port <port>."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                args = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if not args or os.path.basename(args[0]) != "llama-server":
            continue
        for i, a in enumerate(args):
            if a == f"--port={port}" or (a == "--port" and i + 1 < len(args) and args[i + 1] == str(port)):
                pids.append(int(entry))
                break
    return pids


def server_state():
    """ready | loading | down | busy (port used by another program)."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as r:
            return "ready" if r.status == 200 else "loading"
    except urllib.error.HTTPError as e:
        return "loading" if e.code == 503 else "busy"
    except (urllib.error.URLError, OSError):
        return "down"


def tail_log(n=15):
    try:
        with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(empty log)"


def ensure_server():
    """Make sure exactly one embedding llama-server is alive on PORT. True when ready."""
    if server_state() == "ready":
        return True

    with open(LOCK_FILE, "w") as lock:
        # other routers running at the same time wait here, so there is no double start
        fcntl.flock(lock, fcntl.LOCK_EX)

        state = server_state()
        if state == "ready":
            return True
        if state == "busy":
            print(f"Port {PORT} is used by another program that is not the embedding llama-server.")
            return False

        proc = None
        pids = find_servers(PORT)
        if state == "down" and not pids:
            for label, path in (("llama_server", LLAMA_SERVER), ("embed_model", EMBED_MODEL)):
                if not path or not os.path.isfile(path):
                    print(f"{label} is not set or the file was not found: {path or '(empty)'}")
                    print(f"Fill in {CONFIG_FILE} (see config.example.json).")
                    return False
            with open(LOG_FILE, "wb") as log:
                proc = subprocess.Popen(
                    [LLAMA_SERVER, "-m", EMBED_MODEL, *SERVER_ARGS],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True,  # keeps running after the router exits
                )
            print(f"Starting the embedding llama-server in the background (PID {proc.pid})...")
        else:
            shown = ", ".join(map(str, pids)) or "?"
            print(f"llama-server is already running (PID {shown}), waiting until it is ready...")

        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            if server_state() == "ready":
                print("Server ready.")
                return True
            if proc and proc.poll() is not None:
                print(f"llama-server exited with code {proc.returncode}. Last log lines:\n{tail_log()}")
                return False
            time.sleep(0.5)
        print(f"Server not ready after {START_TIMEOUT} seconds. Check the log: {LOG_FILE}")
        return False


def cmd_status():
    labels = {"ready": "ready", "loading": "loading the model",
              "down": "stopped", "busy": "port used by another program"}
    pids = find_servers(PORT)
    print(f"Embedding server on port {PORT}: {labels[server_state()]}")
    print(f"llama-server PID: {', '.join(map(str, pids)) if pids else '-'}")
    if len(pids) > 1:
        print("Warning: more than one llama-server is running on this port.")


def cmd_stop():
    pids = find_servers(PORT)
    if not pids:
        print(f"No llama-server on port {PORT}.")
        return
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    print(f"SIGTERM sent to PID {', '.join(map(str, pids))}.")


# ---------------------------------------------------------------- router

def normalize(query):
    text = query
    for pattern, op in WORD_OPS:
        text = re.sub(pattern, f" {op} ", text, flags=re.IGNORECASE)
    return text


def extract_expression(query):
    """Pick the part of the sentence that holds numbers and operators. None if there is none."""
    candidates = [m.strip() for m in EXPR_CHARS.findall(normalize(query)) if re.search(r"\d", m)]
    return max(candidates, key=len) if candidates else None


def keyword_hit(query, keywords):
    """True when any keyword appears as a whole word/phrase in the sentence."""
    return any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", query, re.IGNORECASE) for k in keywords)


def in_home(path):
    """True when path (after resolving symlinks) is inside the home folder."""
    home = os.path.realpath(HOME)
    real = os.path.realpath(path)
    return real == home or real.startswith(home + os.sep)


def expand(text):
    return os.path.expanduser(text.replace("{ROOT}", HERE).replace("{LLAMA_SERVER}", LLAMA_SERVER))


def load_commands():
    with open(COMMANDS_FILE, encoding="utf-8") as f:
        commands = json.load(f)
    for c in commands:
        c["_raw"] = " ".join(c["cmd"])
        c["cmd"] = [expand(a) for a in c["cmd"]]
        c["cwd"] = os.path.abspath(expand(c.get("cwd", "~")))
    return commands


def embed(texts):
    req = urllib.request.Request(
        EMBED_URL,
        data=json.dumps({"input": texts, "model": "embeddinggemma"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = sorted(json.load(r)["data"], key=lambda d: d["index"])
    out = []
    for d in data:
        v = d["embedding"]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def confirm(cmdline, cwd):
    return input(f"Run {cmdline} in {cwd}? [y/N] ").strip().lower() == "y"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    arg = sys.argv[1]
    if arg == "--status":
        return cmd_status()
    if arg == "--stop":
        return cmd_stop()

    try:
        commands = load_commands()
    except (OSError, json.JSONDecodeError) as e:
        print(f"Failed to read {COMMANDS_FILE}: {e}")
        return

    if arg == "--list":
        suffixes = {"expression": " <expression>", "query": " <sentence>"}
        for c in commands:
            shown = ("sudo " if c.get("sudo") else "") + c["_raw"] + suffixes.get(c.get("args"), "")
            print(f"{shown:55} {c['desc']}")
        return

    query = " ".join(sys.argv[1:])

    if not ensure_server():
        return

    try:
        vecs = embed(
            [f"task: search result | query: {query}"]
            + [f"title: none | text: {c['desc']}" for c in commands]
        )
    except urllib.error.URLError as e:
        print(f"Could not connect to the embedding server at {EMBED_URL} ({e.reason}).")
        return

    has_expr = bool(HAS_EXPR.search(normalize(query)))
    qv, dvs = vecs[0], vecs[1:]
    scored = []
    for dv, c in zip(dvs, commands):
        s = sum(a * b for a, b in zip(qv, dv))
        boosted = (has_expr and c.get("args") == "expression") or keyword_hit(query, c.get("keywords", []))
        scored.append((s + (BOOST if boosted else 0.0), boosted, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    print("== Matches ==")
    for score, boosted, c in scored[:3]:
        note = " (+bonus)" if boosted else ""
        print(f"[{score:.3f}] {c['_raw']}  ({c['desc']}){note}")

    best_score, _, best = scored[0]
    margin = best_score - scored[1][0] if len(scored) > 1 else best_score
    print(f"\nBest score {best_score:.3f}, gap to runner-up: {margin:.3f}")

    if best_score < THRESHOLD:
        print(f"Score below {THRESHOLD}, no matching command.")
        return
    if margin < MIN_MARGIN:
        print(f"Gap below {MIN_MARGIN}, the choice is unclear. Try a more specific sentence.")
        return

    cmd = list(best["cmd"])
    if best.get("args") == "expression":
        expr = extract_expression(query)
        if not expr:
            print("No math expression found in that sentence.")
            return
        cmd.append(expr)  # one whole argument, never goes through a shell
    elif best.get("args") == "query":
        cmd.append(query)  # one whole argument, never goes through a shell

    is_sudo = bool(best.get("sudo"))
    if is_sudo and os.geteuid() != 0:
        if not shutil.which("sudo"):
            print("This command needs sudo, but sudo was not found.")
            return
        cmd = ["sudo"] + cmd

    cmdline = " ".join(f'"{a}"' if " " in a else a for a in cmd)
    if is_sudo:
        print("WARNING: this command uses sudo (root privileges).")
        if not confirm(cmdline, best["cwd"]):
            print("Cancelled.")
            return
    elif AUTO_ALLOW_HOME and in_home(best["cwd"]):
        print(f"Running (auto-allowed, working directory inside home): {cmdline}")
    elif not confirm(cmdline, best["cwd"]):
        print("Cancelled.")
        return

    timeout = best.get("timeout", 3600 if is_sudo else 30)
    try:
        if is_sudo:
            # the terminal is passed through as-is: password prompt and output show up live
            res = subprocess.run(cmd, cwd=best["cwd"], timeout=timeout)
            print(f"Finished with exit code {res.returncode}.")
            return
        res = subprocess.run(cmd, cwd=best["cwd"], capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, PermissionError):
        print(f"Program or folder not found: {cmd[0]} in {best['cwd']}")
        return
    except subprocess.TimeoutExpired:
        print(f"Command exceeded the {timeout} second limit and was stopped.")
        return
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return
    print("== Output ==")
    print(res.stdout or res.stderr)


if __name__ == "__main__":
    main()
