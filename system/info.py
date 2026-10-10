#!/usr/bin/env python3
"""System info and controls for yuy (standard library only).

Usage: python3 info.py <os|cpu|gpu|battery|temp|time|volume|brightness|tree> ["sentence"]
"""
import datetime
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
import time

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# folders hidden from tree (add your own as needed)
TREE_IGNORE = {
    "node_modules", "dist", ".next", ".git", "__pycache__", ".venv", "venv",
    "build", "target", ".cache", ".turbo", ".vercel", ".nuxt", ".svelte-kit",
    "coverage", ".pytest_cache", ".mypy_cache", ".idea", ".vscode",
}
TREE_DEFAULT_DEPTH = 3
TREE_MAX_LINES = 400

VOLUME_STEP = 5
BRIGHTNESS_STEP = 10

# intent words (Indonesian and English)
MUTE_RE = r"\b(mute|unmute|bisu|bisukan|senyap|senyapkan)\b"
SET_RE = r"\b(?:ke|jadi|menjadi|to|set|atur|sampai)\s*(\d{1,3})\b"
UP_RE = (r"\b(naik|naikkan|naikin|tambah|tambahkan|up|increase|raise|keras|kerasin|"
         r"besarkan|louder|terang|cerah|brighter)\b")
DOWN_RE = (r"\b(turun|turunkan|turunin|kurang|kurangi|down|decrease|lower|pelan|"
           r"pelankan|kecilkan|quieter|gelap|redup|dimmer)\b")


# ---------------------------------------------------------------- helpers

def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def num(path):
    v = read(path)
    try:
        return float(v) if v is not None else None
    except ValueError:
        return None


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None


def row(label, value, width=12):
    print(f"{label:<{width}}: {value}")


def fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def parse_action(query, default_step, allow_mute=True):
    """Read the intent from a sentence: get | up | down | set | mute (plus the number)."""
    q = (query or "").lower()
    m = re.search(r"\b(\d{1,3})\b", q)
    n = int(m.group(1)) if m else None
    if allow_mute and re.search(MUTE_RE, q):
        return "mute", None
    s = re.search(SET_RE, q)
    if s:
        return "set", int(s.group(1))
    if re.search(UP_RE, q):
        return "up", n if n is not None else default_step
    if re.search(DOWN_RE, q):
        return "down", n if n is not None else default_step
    if n is not None:
        return "set", n
    return "get", None


# ---------------------------------------------------------------- os, cpu, gpu

def cmd_os(_q):
    osr = {}
    for line in (read("/etc/os-release") or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            osr[k] = v.strip('"')
    u = platform.uname()
    row("OS", osr.get("PRETTY_NAME", u.system))
    row("Kernel", u.release)
    row("Architecture", u.machine)
    row("Hostname", u.node)
    up = read("/proc/uptime")
    if up:
        secs = int(float(up.split()[0]))
        d, rem = divmod(secs, 86400)
        h, rem = divmod(rem, 3600)
        row("Uptime", f"{d} days {h} hours {rem // 60} minutes")
    desktop = os.environ.get("XDG_CURRENT_DESKTOP")
    if desktop:
        row("Desktop", f"{desktop} ({os.environ.get('XDG_SESSION_TYPE', '?')})")
    row("Shell", os.path.basename(os.environ.get("SHELL", "?")))


def cpu_times():
    fields = (read("/proc/stat") or "cpu 0 0 0 0 0 0 0 0").splitlines()[0].split()[1:]
    vals = [int(x) for x in fields[:8]]
    return sum(vals), vals[3] + vals[4]  # total, idle + iowait


def cmd_cpu(_q):
    model = "?"
    for line in (read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    t1, i1 = cpu_times()
    time.sleep(0.5)
    t2, i2 = cpu_times()
    usage = 100.0 * (1 - (i2 - i1) / max(t2 - t1, 1))
    l1, l5, l15 = os.getloadavg()
    row("Model", model)
    row("Threads", os.cpu_count())
    row("Usage", f"{usage:.0f}%")
    row("Load avg", f"{l1:.2f} / {l5:.2f} / {l15:.2f} (1/5/15 min)")
    freqs = []
    for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        v = num(p)
        if v:
            freqs.append(v / 1000)
    if freqs:
        row("Frequency", f"average {sum(freqs) / len(freqs):.0f} MHz, highest {max(freqs):.0f} MHz")


def cmd_gpu(_q):
    shown = False
    if shutil.which("lspci"):
        r = run(["lspci"])
        if r and r.returncode == 0:
            for line in r.stdout.splitlines():
                if any(k in line for k in ("VGA", "3D controller", "Display controller")):
                    print(line.split(" ", 1)[1] if " " in line else line)
                    shown = True
    for card in sorted(glob.glob("/sys/class/drm/card*")):
        name = os.path.basename(card)
        if not re.fullmatch(r"card\d+", name):
            continue
        dev = card + "/device"
        if not shown:
            print(f"{name}: vendor {read(dev + '/vendor')} device {read(dev + '/device')} "
                  "(install pciutils for the full name)")
            shown = True
        stats = []
        for label, used, total in (("VRAM", "mem_info_vram_used", "mem_info_vram_total"),
                                   ("GTT (shared RAM)", "mem_info_gtt_used", "mem_info_gtt_total")):
            u, t = read(f"{dev}/{used}"), read(f"{dev}/{total}")
            if u and t and u.isdigit() and t.isdigit():
                stats.append(f"{label}: {fmt_bytes(int(u))} / {fmt_bytes(int(t))}")
        busy = read(dev + "/gpu_busy_percent")
        if busy and busy.isdigit():
            stats.append(f"GPU load: {busy}%")
        if stats:
            print(f"[{name}]")
            for s in stats:
                print("  " + s)
    if not shown:
        print("No readable GPU info found.")


# ---------------------------------------------------------------- battery, temperature, time

STATUS_EN = {"Charging": "Charging", "Discharging": "Discharging (running on battery)",
             "Full": "Full", "Not charging": "Not charging", "Unknown": "Unknown"}


def cmd_battery(_q):
    bats = sorted(glob.glob("/sys/class/power_supply/BAT*"))
    if not bats:
        print("No battery found (probably a desktop PC).")
        return
    for b in bats:
        st = read(b + "/status") or "Unknown"
        print(os.path.basename(b))
        row("  Capacity", f"{read(b + '/capacity')}%")
        row("  Status", STATUS_EN.get(st, st))
        now, full, design, rate = (num(b + "/energy_now"), num(b + "/energy_full"),
                                   num(b + "/energy_full_design"), num(b + "/power_now"))
        if now is None:
            now, full, design, rate = (num(b + "/charge_now"), num(b + "/charge_full"),
                                       num(b + "/charge_full_design"), num(b + "/current_now"))
        if full and design:
            row("  Health", f"{100 * full / design:.0f}% of design capacity")
        if now is not None and rate and rate > 0:
            label = hrs = None
            if st == "Discharging":
                label, hrs = "Time left", now / rate
            elif st == "Charging" and full:
                label, hrs = "Until full", (full - now) / rate
            if hrs is not None:
                row("  " + label, f"{int(hrs)} h {int((hrs % 1) * 60)} min")


def cmd_temp(_q):
    found = False
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name = read(hw + "/name") or os.path.basename(hw)
        for t in sorted(glob.glob(hw + "/temp*_input")):
            raw = num(t)
            if raw is None:
                continue
            label = read(t.replace("_input", "_label")) or os.path.basename(t).replace("_input", "")
            print(f"{name:<12} {label:<16} {raw / 1000:5.1f} °C")
            found = True
    if not found:
        for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            raw = num(z + "/temp")
            if raw is not None:
                print(f"{read(z + '/type') or os.path.basename(z):<28} {raw / 1000:5.1f} °C")
                found = True
    if not found:
        print("No temperature sensors found.")


def cmd_time(_q):
    now = datetime.datetime.now().astimezone()
    day, month = DAYS[now.weekday()], MONTHS[now.month - 1]
    row("Time", f"{now:%H:%M:%S} {now.tzname()}")
    row("Day", day)
    row("Date", now.day)
    row("Month", f"{month} ({now.month})")
    row("Year", now.year)
    row("Full", f"{day}, {month} {now.day}, {now.year}")


# ---------------------------------------------------------------- volume

def vol_backend():
    for name in ("wpctl", "pactl", "amixer"):
        if shutil.which(name):
            return name
    return None


def vol_get(be):
    """(percent, muted) or None."""
    if be == "wpctl":
        r = run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
        if r and r.returncode == 0:
            m = re.search(r"([\d.]+)", r.stdout)
            if m:
                return round(float(m.group(1)) * 100), "[MUTED]" in r.stdout
    elif be == "pactl":
        v = run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
        m = run(["pactl", "get-sink-mute", "@DEFAULT_SINK@"])
        if v and v.returncode == 0:
            p = re.search(r"(\d+)%", v.stdout)
            if p:
                return int(p.group(1)), bool(m and "yes" in m.stdout)
    elif be == "amixer":
        r = run(["amixer", "get", "Master"])
        if r and r.returncode == 0:
            p = re.search(r"\[(\d+)%\]", r.stdout)
            if p:
                return int(p.group(1)), "[off]" in r.stdout
    return None


def vol_set(be, pct):
    cmds = {
        "wpctl": ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct}%"],
        "pactl": ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"],
        "amixer": ["amixer", "set", "Master", f"{pct}%"],
    }
    run(cmds[be])


def vol_toggle_mute(be):
    cmds = {
        "wpctl": ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"],
        "pactl": ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
        "amixer": ["amixer", "set", "Master", "toggle"],
    }
    run(cmds[be])


def cmd_volume(query):
    be = vol_backend()
    if not be:
        print("No wpctl, pactl, or amixer found. Install wireplumber (PipeWire) or alsa-utils.")
        return
    cur = vol_get(be)
    if cur is None:
        print(f"Failed to read the volume via {be}.")
        return
    pct, muted = cur
    action, n = parse_action(query, VOLUME_STEP)
    if action == "mute":
        vol_toggle_mute(be)
    elif action in ("up", "down", "set"):
        target = pct + n if action == "up" else pct - n if action == "down" else n
        vol_set(be, max(0, min(100, target)))
    new_pct, new_muted = vol_get(be) or (pct, muted)
    row("Volume", f"{new_pct}%{' (MUTED)' if new_muted else ''}")
    if action != "get":
        row("Previous", f"{pct}%{' (MUTED)' if muted else ''}")
    row("Backend", be)


# ---------------------------------------------------------------- brightness

def set_brightness(name, value):
    if shutil.which("brightnessctl"):
        r = run(["brightnessctl", "-d", name, "set", str(value)])
    else:  # via logind, no root needed for an active desktop session
        r = run(["busctl", "call", "org.freedesktop.login1",
                 "/org/freedesktop/login1/session/auto", "org.freedesktop.login1.Session",
                 "SetBrightness", "ssu", "backlight", name, str(value)])
    if r is None or r.returncode != 0:
        print("Failed to change brightness:", (r.stderr.strip() if r else "command did not run"))
        print("Try installing brightnessctl: sudo pacman -S brightnessctl")
        return False
    return True


def cmd_brightness(query):
    devs = sorted(glob.glob("/sys/class/backlight/*"))
    if not devs:
        print("No backlight found (external monitor or desktop PC).")
        return
    dev = devs[0]
    name = os.path.basename(dev)
    cur, mx = num(dev + "/brightness"), num(dev + "/max_brightness")
    if cur is None or not mx:
        print("Failed to read brightness.")
        return
    pct = round(100 * cur / mx)
    action, n = parse_action(query, BRIGHTNESS_STEP, allow_mute=False)
    if action in ("up", "down", "set"):
        target = pct + n if action == "up" else pct - n if action == "down" else n
        target = max(1, min(100, target))  # at least 1% so the screen never goes fully dark
        if not set_brightness(name, max(1, round(mx * target / 100))):
            return
        time.sleep(0.2)
        cur = num(dev + "/brightness") or cur
    row("Brightness", f"{round(100 * cur / mx)}%")
    if action != "get":
        row("Previous", f"{pct}%")
    row("Device", name)


# ---------------------------------------------------------------- tree

def cmd_tree(query):
    q = query or ""
    root = "."
    for tok in q.split():
        if tok.startswith(("~", "/", "./", "../")):
            p = os.path.expanduser(tok.rstrip(",;"))
            if os.path.isdir(p):
                root = p
                break
    m = re.search(r"\b(?:depth|level|kedalaman|dalam)\s*(\d{1,2})\b", q.lower())
    depth = int(m.group(1)) if m else TREE_DEFAULT_DEPTH
    root = os.path.abspath(root)

    lines, counts, truncated = [], {"dirs": 0, "files": 0}, False

    def walk(path, prefix, level):
        nonlocal truncated
        if level > depth:
            return
        try:
            entries = [e for e in os.scandir(path) if e.name not in TREE_IGNORE]
        except OSError:
            lines.append(prefix + "└── [unreadable]")
            return
        entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        for i, e in enumerate(entries):
            if len(lines) >= TREE_MAX_LINES:
                truncated = True
                return
            last = i == len(entries) - 1
            is_dir = e.is_dir(follow_symlinks=False)
            lines.append(f"{prefix}{'└── ' if last else '├── '}{e.name}{'/' if is_dir else ''}")
            if is_dir:
                counts["dirs"] += 1
                walk(e.path, prefix + ("    " if last else "│   "), level + 1)
            else:
                counts["files"] += 1

    walk(root, "", 1)
    print(root)
    print("\n".join(lines))
    print(f"\n{counts['dirs']} folders, {counts['files']} files (depth {depth}; "
          "hidden: node_modules, dist, .next, .git, etc.)")
    if truncated:
        print(f"Truncated at {TREE_MAX_LINES} lines. Use a more specific folder or a smaller depth.")


COMMANDS = {
    "os": cmd_os, "cpu": cmd_cpu, "gpu": cmd_gpu, "battery": cmd_battery,
    "temp": cmd_temp, "time": cmd_time, "volume": cmd_volume,
    "brightness": cmd_brightness, "tree": cmd_tree,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else "")


if __name__ == "__main__":
    main()
