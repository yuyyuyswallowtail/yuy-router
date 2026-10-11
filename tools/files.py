#!/usr/bin/env python3
"""yuy file and web tools, called by router.py with the whole sentence as one argument.

  files.py search  "search TODO in ~/Projects"   (a folder path or "search file ..." = files, otherwise web)
  files.py read    "read ~/notes.txt lines 10-40"
  files.py create  'create ~/notes.txt with "hello"'   |   "create folder ~/x/y"
  files.py edit    'edit ~/notes.txt replace "old" with "new" [all]'   |   'edit FILE append "text"'
  files.py delete  "delete ~/old.txt"        (moved to a trash folder, not removed)
  files.py fetch   "fetch https://example.com"
  files.py browser "open https://example.com in browser"
  files.py websearch "search the web for laravel 14"   (local SearXNG, then Bing, Brave, Mojeek, Yahoo)
  files.py enginetest "laravel 14"                     (which of those engines work from here)

Safety: every path must resolve inside $HOME (symlinks resolved). ~/.ssh ~/.gnupg ~/.aws
are never touched and .git is never written. create never overwrites. edit prints a diff and
keeps a backup. delete moves to ~/.local/share/yuy-trash. Standard library only.
"""
import base64
import difflib
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from html.parser import HTMLParser

HOME = os.path.realpath(os.path.expanduser("~"))
TRASH = os.path.join(HOME, ".local", "share", "yuy-trash")
DENY = {".ssh", ".gnupg", ".aws", ".password-store"}
WRITE_DENY = DENY | {".git"}
SKIP_DIRS = {"node_modules", "dist", "build", "target", "__pycache__", "venv"}
MAX_FILE = 1_000_000      # bytes: bigger files are not read, edited or searched
MAX_LINES = 200           # read: lines shown when no range is given
MAX_HITS = 30             # search: results per section
MAX_WEB = 8000            # fetch: characters shown
MAX_DOWNLOAD = 2_000_000  # fetch: bytes downloaded

QUOTED = re.compile(r'"([^"]*)"|\'([^\']*)\'|“([^”]*)”')
STOP = set("""search find cari carikan temukan for file files named called in di dalam inside
containing contain contains content contents with text teks berisi isi yang mengandung
folder directory direktori the a an all semua grep on at under about tentang pada from within""".split())
NOT_PATH = STOP | set("""read delete remove edit replace append create hapus baca buang ubah ganti
tambahkan buat buatkan bikin dengan lines line baris browser fetch open new""".split())
CONTENT = re.compile(r"\b(contain\w*|content\w*|inside|isi|berisi|mengandung|grep|text|teks)\b", re.I)
RANGE = re.compile(r"\b(?:lines?|baris)\s*(\d+)(?:\s*(?:-|to|sampai|hingga|until)\s*(\d+))?", re.I)
APPEND = re.compile(r"\b(append|tambahkan|tambah|add)\b", re.I)
ALL = re.compile(r"\b(all|semua|every|seluruh)\b", re.I)
DIRWORD = re.compile(r"\b(folder|dir|directory|direktori)\b", re.I)
# "search ..." means files when the sentence has a folder path or starts like "search file ...",
# and means the web otherwise
WEB_WORD = re.compile(r"(?<![\w./\-])(web|internet|online|bing|google)(?![\w./\-])", re.I)
FILE_HINT = re.compile(r"^\s*(?:search|find|cari|carikan|temukan)(?:\s+(?:for|the))?\s+"
                       r"(?:files?|folders?|directory|content|contents|isi|berisi)\b", re.I)
URL = re.compile(r"https?://[^\s\"'<>]+|(?<![\w@/.])(?:[\w\-]+\.)+[a-z]{2,}(?:/[^\s\"'<>]*)?", re.I)


def die(msg):
    print(msg)
    sys.exit(1)


def unescape(s):
    return s.replace("\\n", "\n").replace("\\t", "\t")


def stamp():
    return time.strftime("%Y%m%d-%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"


# ---------------------------------------------------------------- parsing and paths

def clean(tok):
    tok = tok.rstrip(",;:!?")
    return re.sub(r"(?<=\w)\.$", "", tok)


def looks_like_path(t):
    if not t or re.fullmatch(r"\d+(\.\d+)?", t):
        return False
    return (t in (".", "..") or t.startswith(("~", "/", "./", "../")) or "/" in t
            or bool(re.fullmatch(r"\.?[\w\-]+(?:\.[\w\-]+)+|\.[\w\-]+", t)))


def parse(sentence, quoted_path=True, bare=True):
    """-> (quoted strings, sentence without quotes and path, path or None)"""
    quoted = [a or b or c for a, b, c in QUOTED.findall(sentence)]
    words = QUOTED.sub(" ", sentence).split()
    path = None
    for i, w in enumerate(words):
        if looks_like_path(clean(w)):
            path = clean(w)
            del words[i]
            break
    if path is None and bare:  # bare name of something that exists here, e.g. "delete Makefile"
        for i, w in enumerate(words):
            c = clean(w)
            if c and c.lower() not in NOT_PATH and os.path.lexists(os.path.expanduser(c)):
                path = c
                del words[i]
                break
    if path is None and quoted_path and quoted:
        path = quoted.pop(0)
    return quoted, " ".join(words), path


def resolve(path, write=False, follow=True):
    p = os.path.expanduser(path)
    if follow:
        full = os.path.realpath(p)
    else:  # delete acts on the link itself, not on its target
        absp = os.path.abspath(p)
        full = os.path.join(os.path.realpath(os.path.dirname(absp)), os.path.basename(absp))
    if full != HOME and not full.startswith(HOME + os.sep):
        die(f"Blocked: path is outside your home folder: {full}")
    if set(full[len(HOME):].split(os.sep)) & (WRITE_DENY if write else DENY):
        die(f"Blocked: sensitive folder: {full}")
    if write and full == HOME:
        die("Blocked: refusing to modify the home folder itself.")
    return full


def load_text(full):
    if os.path.getsize(full) > MAX_FILE:
        die(f"File is larger than {MAX_FILE // 1000} KB, not handled: {full}")
    with open(full, "rb") as f:
        raw = f.read()
    if b"\0" in raw[:8192]:
        die(f"Binary file, not handled: {full}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        die(f"Not a UTF-8 text file: {full}")


def files_under(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in SKIP_DIRS]
        for f in files:
            yield os.path.join(d, f)


# ---------------------------------------------------------------- file actions

def do_search(s):
    quoted = [a or b or c for a, b, c in QUOTED.findall(s)]
    words = QUOTED.sub(" ", s).split()
    root_path = None
    for i in range(len(words) - 1, -1, -1):  # last existing folder wins: "main.go di /home/me"
        c = clean(words[i])
        if looks_like_path(c) and os.path.isdir(os.path.expanduser(c)):
            root_path = c
            del words[i]
            break
    explicit_path = any(clean(w).startswith(("~", "/", "./", "../")) for w in words)
    has_path = root_path is not None or explicit_path
    if (WEB_WORD.search(s) and not has_path) or not (has_path or FILE_HINT.search(s) or "*" in s):
        return do_websearch(s)
    if root_path is None:
        for w in words:
            c = clean(w)
            if c.startswith(("~", "/", "./", "../")):
                die(f"Not a folder: {os.path.abspath(os.path.expanduser(c))}")
    rest = " ".join(words)
    root = resolve(root_path or ".")
    term = quoted[0] if quoted else " ".join(w for w in rest.split() if w.lower().strip(",.") not in STOP)
    if not term:
        die('Nothing to search for. Example: search "TODO" in ~/Projects')
    needle = term.lower()
    glob = any(c in term for c in "*?[")
    content_only = bool(CONTENT.search(rest))
    print(f"Searching '{term}' in {root}")

    names = []
    if not content_only:
        for full in files_under(root):
            name = os.path.basename(full).lower()
            if (fnmatch.fnmatch(name, needle) if glob else needle in name):
                names.append(os.path.relpath(full, root))
                if len(names) >= MAX_HITS:
                    break
        if names:
            print(f"\n== Files by name ({len(names)}) ==")
            print("\n".join(names))

    if content_only or not names:
        hits = []
        for full in files_under(root):
            try:
                if os.path.getsize(full) > MAX_FILE:
                    continue
                with open(full, "rb") as f:
                    if b"\0" in f.read(8192):
                        continue
                    f.seek(0)
                    for n, line in enumerate(f, 1):
                        text = line.decode("utf-8", "replace")
                        if needle in text.lower():
                            hits.append(f"{os.path.relpath(full, root)}:{n}: {text.strip()[:120]}")
                            if len(hits) >= MAX_HITS:
                                break
            except OSError:
                continue
            if len(hits) >= MAX_HITS:
                break
        print(f"\n== Content matches ({len(hits)}{'+' if len(hits) >= MAX_HITS else ''}) ==")
        print("\n".join(hits) if hits else "(none)")


def do_read(s):
    quoted, rest, path = parse(s)
    if not path:
        die("Give a path. Example: read ~/notes.txt lines 10-40")
    full = resolve(path)
    if os.path.isdir(full):
        for name in sorted(os.listdir(full)):
            print(name + ("/" if os.path.isdir(os.path.join(full, name)) else ""))
        return
    if not os.path.isfile(full):
        die(f"File not found: {full}")
    lines = load_text(full).splitlines()
    m = RANGE.search(rest)
    if m:
        a = max(int(m.group(1)), 1)
        b = int(m.group(2) or a)
        a, b = min(a, b), max(a, b)
    else:
        a, b = 1, MAX_LINES
    print(f"== {full} ({len(lines)} lines) ==")
    chunk = lines[a - 1:b]
    for i, line in enumerate(chunk, a):
        print(f"{i:>5}  {line}")
    if not chunk:
        print("(no lines in that range)")
    elif not m and len(lines) > b:
        print(f"... {len(lines) - b} more lines. Use: read {path} lines {b + 1}-{b + MAX_LINES}")


def do_create(s):
    quoted, rest, path = parse(s)
    if not path:
        die('Give a path. Example: create ~/notes.txt with "hello"')
    full = resolve(path, write=True)
    if os.path.lexists(full):
        die(f"Already exists, not overwriting: {full} (use edit to change it)")
    if not quoted and not os.path.splitext(full)[1] and DIRWORD.search(rest):
        os.makedirs(full)
        print(f"Created folder: {full}")
        return
    content = unescape(quoted[0]) if quoted else ""
    if content and not content.endswith("\n"):
        content += "\n"
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created file: {full} ({len(content.encode())} bytes)")


def do_edit(s):
    quoted, rest, path = parse(s)
    if not path:
        die('Give a path. Example: edit ~/notes.txt replace "old" with "new"')
    full = resolve(path, write=True)
    if not os.path.isfile(full):
        die(f"File not found: {full}")
    text = load_text(full)
    if APPEND.search(rest) and quoted:
        add = unescape(quoted[0])
        sep = "" if not text or text.endswith("\n") else "\n"
        out = text + sep + add + ("" if add.endswith("\n") else "\n")
    elif len(quoted) >= 2:
        old, new = unescape(quoted[0]), unescape(quoted[1])
        if not old:
            die("The text to replace is empty.")
        n = text.count(old)
        if n == 0:
            die("Text not found, nothing changed.")
        if n > 1 and not ALL.search(rest):
            die(f"Found {n} times. Add the word 'all' to replace every one, or make the text more specific.")
        out = text.replace(old, new)
    else:
        die('Usage: edit FILE replace "old" with "new" [all]   or   edit FILE append "text"')
    if out == text:
        die("No change.")
    name = os.path.basename(full)
    sys.stdout.writelines(difflib.unified_diff(text.splitlines(True), out.splitlines(True), f"a/{name}", f"b/{name}"))
    os.makedirs(TRASH, exist_ok=True)
    bak = os.path.join(TRASH, f"{stamp()}-{name}.bak")
    shutil.copy2(full, bak)
    with open(full, "wb") as f:
        f.write(out.encode("utf-8"))
    print(f"\nEdited: {full}\nBackup: {bak}")


def do_delete(s):
    quoted, rest, path = parse(s)
    if not path:
        die("Give a path. Example: delete ~/old.txt")
    full = resolve(path, write=True, follow=False)
    if not os.path.lexists(full):
        die(f"Not found: {full}")
    os.makedirs(TRASH, exist_ok=True)
    dest = os.path.join(TRASH, f"{stamp()}-{os.path.basename(full)}")
    shutil.move(full, dest)
    print(f"Moved to trash: {dest}")
    print(f"Restore with:   mv {shlex.quote(dest)} {shlex.quote(full)}")


# ---------------------------------------------------------------- web actions

def term_width():
    try:
        cols = os.get_terminal_size(0).columns  # stdin is still the terminal when stdout is captured
    except OSError:
        cols = 100
    return max(60, min(cols, 110))


class TextExtractor(HTMLParser):
    """HTML -> readable text: headings as '#', list items as '-', page chrome (menus, footer) dropped."""
    SKIP_ALWAYS = {"script", "style", "noscript", "svg", "template", "iframe", "dialog", "select", "button"}
    SKIP_CHROME = {"nav", "footer", "aside", "form"}
    HEADING = {"h1": "# ", "h2": "## ", "h3": "## ", "h4": "### ", "h5": "### ", "h6": "### "}
    PARA = {"p", "ul", "ol", "table", "pre", "blockquote", "section", "article", "main", "figure",
            "details", "dl"}
    LINE = {"div", "br", "tr", "dt", "dd", "figcaption"}
    BREAK = "\n\x00\n"  # paragraph break marker; plain newlines are only line breaks

    def __init__(self, strict=True):
        super().__init__(convert_charrefs=True)
        self.strict = strict
        self.all, self.main, self.skipped = [], [], []
        self.title, self.in_title = "", False
        self.main_depth = self.article_depth = 0

    def out(self, text):
        if not self.skipped:
            self.all.append(text)
            if self.main_depth:
                self.main.append(text)

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
            return
        site_header = tag == "header" and not (self.main_depth or self.article_depth)
        if tag in self.SKIP_ALWAYS or (self.strict and (tag in self.SKIP_CHROME or site_header)):
            self.skipped.append(tag)
            return
        if tag == "main":
            self.main_depth += 1
        elif tag == "article":
            self.article_depth += 1
        if tag in self.HEADING:
            self.out(self.BREAK + self.HEADING[tag])
        elif tag == "li":
            self.out("\n- ")
        elif tag in self.PARA:
            self.out(self.BREAK)
        elif tag in self.LINE:
            self.out("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
            return
        if tag in self.skipped:  # also repairs a missing end tag inside a skipped block
            del self.skipped[len(self.skipped) - 1 - self.skipped[::-1].index(tag):]
            return
        if tag == "main":
            self.main_depth = max(self.main_depth - 1, 0)
        elif tag == "article":
            self.article_depth = max(self.article_depth - 1, 0)
        if tag in self.HEADING or tag in self.PARA:
            self.out(self.BREAK)
        elif tag in self.LINE:
            self.out("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        else:
            self.out(data)


def tidy(raw):
    """Collapse spaces, drop repeated short lines (menu leftovers), wrap to the terminal width."""
    width, seen, lines = term_width(), set(), []
    for line in raw.splitlines():
        if line == "\x00":  # paragraph break
            if lines and lines[-1] != "":
                lines.append("")
            continue
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line in ("-", "#", "##", "###") or len(line) < 3:
            continue
        if len(line) <= 40 and not line.startswith("#"):
            if line in seen:
                continue
            seen.add(line)
        lines.append(textwrap.fill(line, width=width, break_long_words=False, break_on_hyphens=False,
                                   subsequent_indent="  " if line.startswith("- ") else ""))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(html):
    """-> (title, text). Uses <main> when the page has one, falls back to a looser parse if too little is left."""
    for strict in (True, False):
        p = TextExtractor(strict)
        p.feed(html)
        main = tidy("".join(p.main))
        text = main if len(main) > 300 else tidy("".join(p.all))
        if len(text) > 200 or not strict:
            return " ".join(p.title.split()), text


def cut_text(text, limit):
    """-> (text cut at a paragraph or line end, whether something was cut)."""
    if len(text) <= limit:
        return text, False
    head = text[:limit]
    for sep in ("\n\n", "\n"):
        i = head.rfind(sep)
        if i > limit * 0.6:
            return head[:i].rstrip(), True
    return head.rstrip(), True


def get_url(s):
    m = URL.search(s)
    if not m:
        die("No URL found. Example: fetch https://example.com")
    url = m.group(0).rstrip(".,;:!?)")
    if not re.match(r"https?://", url, re.I):
        url = "https://" + url
    if not urllib.parse.urlparse(url).netloc:
        die(f"Invalid URL: {url}")
    return url


class FetchError(Exception):
    pass


def fetch_page(url, timeout=20):
    """-> (final url, status, content type, title, text, truncated). Raises FetchError."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; yuy-router)",
        "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.5",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(MAX_DOWNLOAD + 1)
            ctype = r.headers.get_content_type()
            charset = r.headers.get_content_charset() or "utf-8"
            status, final = r.status, r.geturl()
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} {e.reason}")
    except OSError as e:
        raise FetchError(str(getattr(e, "reason", e)))
    cut = len(raw) > MAX_DOWNLOAD
    try:
        text = raw[:MAX_DOWNLOAD].decode(charset, "replace")
    except LookupError:
        text = raw[:MAX_DOWNLOAD].decode("utf-8", "replace")
    title = ""
    if ctype in ("text/html", "application/xhtml+xml"):
        title, text = html_to_text(text)
    elif not (ctype.startswith("text/") or ctype.endswith(("json", "xml"))):
        raise FetchError(f"not a text page ({ctype})")
    return final, status, ctype, title, text, cut


def show_page(page, limit, label=""):
    final, status, ctype, title, text, cut = page
    print(f"== {label}{final} (HTTP {status}, {ctype}) ==")
    if title:
        print(f"Title: {title}")
    print()
    shown, more = cut_text(text, limit)
    print(shown)
    if more or cut:
        print(f"\n... cut here: showing {len(shown)} of {len(text)} characters.")


def do_fetch(s):
    if not URL.search(s):  # no URL in the sentence: treat it as a web search
        return do_websearch(s)
    url = get_url(s)
    try:
        show_page(fetch_page(url), MAX_WEB)
    except FetchError as e:
        die(f"Could not fetch {url}: {e}")


# ---------------------------------------------------------------- web search (Bing)

WEB_STOP = set("""websearch web search the on for cari carikan di internet online bing fetch from
browser tentang about lookup look up google""".split())
MAX_RESULTS = 8
AUTO_TRIES = 3     # websearch shows the first result that can be read, trying this many
AUTO_CHARS = 4000  # characters of that page to show


class BingParser(HTMLParser):
    """Collects title, url and snippet from <li class="b_algo"> result blocks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results, self.cur = [], None
        self.in_h2 = self.in_a = self.in_p = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "li" and "b_algo" in (a.get("class") or "").split():
            self.cur = {"title": "", "url": "", "snippet": ""}
            self.results.append(self.cur)
        elif self.cur is None:
            return
        elif tag == "h2":
            self.in_h2 = True
        elif tag == "a" and self.in_h2 and not self.cur["url"]:
            self.cur["url"] = a.get("href") or ""
            self.in_a = True
        elif tag == "p" and not self.in_h2 and not self.cur["snippet"].strip():
            self.in_p = True

    def handle_endtag(self, tag):
        if tag == "h2":
            self.in_h2 = self.in_a = False
        elif tag == "a":
            self.in_a = False
        elif tag == "p":
            self.in_p = False

    def handle_data(self, data):
        if self.cur is None:
            return
        if self.in_a:
            self.cur["title"] += data
        elif self.in_p:
            self.cur["snippet"] += data


def real_url(href):
    """Bing often wraps links as bing.com/ck/a?...&u=a1<base64 of the real url>."""
    q = urllib.parse.urlparse(href)
    if q.netloc.endswith("bing.com") and q.path.startswith("/ck/a"):
        u = urllib.parse.parse_qs(q.query).get("u", [""])[0]
        if u.startswith("a1"):
            try:
                return base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode("utf-8")
            except ValueError:
                pass
    return href


class SearchError(Exception):
    pass


LAST_HTML = ""  # raw page of the last http_get, saved by enginetest


def http_get(url, timeout=12):
    global LAST_HTML
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(MAX_DOWNLOAD)
            LAST_HTML = raw.decode(r.headers.get_content_charset() or "utf-8", "replace")
            return LAST_HTML
    except urllib.error.HTTPError as e:
        raise SearchError(f"HTTP {e.code}")
    except (OSError, LookupError) as e:
        raise SearchError(str(getattr(e, "reason", e)))


def bing_search(query):
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "setlang": "en-US"})
    parser = BingParser()
    parser.feed(http_get(url))
    return [{"title": x["title"], "url": real_url(x["url"]), "snippet": x["snippet"]} for x in parser.results]


# Engine-agnostic result finder: external links with a readable title, plus the text right after them.
DOMAINLIKE = re.compile(r"^(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)+(?:\s*[›»/].*)?$", re.I)


def unwrap(href):
    """Yahoo wraps links as r.search.yahoo.com/.../RU=<encoded real url>/RK=..."""
    m = re.search(r"/RU=([^/]+)/R[KS]=", href)
    return urllib.parse.unquote(m.group(1)) if m and "yahoo.com" in href else href


class SerpParser(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "nav", "footer", "form", "button", "select"}

    def __init__(self, own):
        super().__init__(convert_charrefs=True)
        self.own, self.results = own, []
        self.skip = 0
        self.cur = None   # external link being read: {"url": ..., "parts": [text pieces]}
        self.last = None  # last accepted result, collects the snippet text that follows it

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "a" and not self.skip:
            self.last = None
            url = unwrap(dict(attrs).get("href") or "")
            host = urllib.parse.urlparse(url).netloc.lower()
            if url.startswith(("http://", "https://")) and host and not host.endswith(self.own):
                self.cur = {"url": url, "parts": []}

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(self.skip - 1, 0)
        elif tag == "a" and self.cur is not None:
            kept = [p for p in self.cur["parts"] if "›" not in p and not DOMAINLIKE.match(p)]
            if len(kept) > 1 and len(kept[1]) > len(kept[0]) and kept[1].lower().startswith(kept[0].lower()):
                kept = kept[1:]  # a site name printed in front of the real title
            title = " ".join(kept)
            if len(title) >= 3:
                known = next((r for r in self.results if r["url"] == self.cur["url"]), None)
                if known:
                    if len(title) > len(known["title"]):
                        known["title"] = title
                    self.last = known
                else:
                    self.last = {"title": title, "url": self.cur["url"], "snippet": ""}
                    self.results.append(self.last)
            self.cur = None

    def handle_data(self, data):
        text = " ".join(data.split())
        if self.skip or not text:
            return
        if self.cur is not None:
            self.cur["parts"].append(text)
        elif self.last is not None and len(self.last["snippet"]) < 220:
            self.last["snippet"] = (self.last["snippet"] + " " + text).strip()


def serp_search(template, own):
    def search(query):
        parser = SerpParser(own)
        parser.feed(http_get(template.format(q=urllib.parse.quote_plus(query))))
        return parser.results
    return search


# A local SearXNG (https://docs.searxng.org) does the scraping with maintained parsers and answers in JSON.
SEARXNG = os.environ.get("YUY_SEARXNG_URL", "http://127.0.0.1:8888").rstrip("/")


def searxng_search(query):
    url = f"{SEARXNG}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    try:
        data = json.loads(http_get(url, timeout=20))
    except SearchError as e:
        if str(e) == "HTTP 403":
            raise SearchError("HTTP 403 (enable the json format: search.formats in settings.yml)")
        raise
    except ValueError:
        raise SearchError("did not answer with JSON")
    return [{"title": r.get("title") or "", "url": r.get("url") or "", "snippet": r.get("content") or ""}
            for r in data.get("results", [])]


# name, search function, whether to keep only results that mention the query (generic parsing also
# picks up stray links). DuckDuckGo is left out on purpose: it is blocked in Indonesia.
ENGINES = (
    ("SearXNG", searxng_search, False),
    ("Bing", bing_search, False),
    ("Brave", serp_search("https://search.brave.com/search?q={q}&source=web", ("brave.com", "bravesoftware.com")), True),
    ("Mojeek", serp_search("https://www.mojeek.com/search?q={q}", ("mojeek.com", "mojeek.net")), True),
    ("Yahoo", serp_search("https://search.yahoo.com/search?p={q}", ("yahoo.com", "yimg.com", "aol.com")), True),
)


def mentions(r, words):
    text = f'{r["title"]} {r["snippet"]} {r["url"]}'.lower()
    return any(w in text for w in words)


def usable(results, words, strict=False):
    """Keep entries with a title and an http(s) link. Unrelated results (an engine serving noise) are dropped."""
    good = [r for r in results if r["title"].strip() and r["url"].startswith(("http://", "https://"))]
    if words:
        if strict:
            good = [r for r in good if mentions(r, words)]
        elif not any(mentions(r, words) for r in good):
            good = []
    return good


def query_words(query):
    return [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]


def do_websearch(s):
    quoted = [a or b or c for a, b, c in QUOTED.findall(s)]
    query = quoted[0] if quoted else " ".join(
        w for w in QUOTED.sub(" ", s).split() if w.lower().strip(",.") not in WEB_STOP)
    if not query:
        die('Nothing to search for. Example: websearch "laravel 14 release date"')
    words = query_words(query)
    results, engine, notes = [], "", []
    for name, search, strict in ENGINES:
        try:
            results = usable(search(query), words, strict)
        except SearchError as e:
            notes.append(f"{name}: {e}")
            continue
        if results:
            engine = name
            break
        notes.append(f"{name}: no results that match the query")
    if not results:
        bing = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query})
        die("No usable web results.\n" + "\n".join(f"  {n}" for n in notes) +
            f'\nOpen it yourself: yuy "open {bing} in browser"')
    print(f"== Web search ({engine}): {query} ==")
    if notes:
        print("Note: " + "; ".join(notes))
    width = term_width()
    for i, r in enumerate(results[:MAX_RESULTS], 1):
        print(f"\n{i}. {' '.join(r['title'].split())}")
        print(f"   {r['url']}")
        snippet = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(r["snippet"].split()))
        if snippet:
            print(textwrap.fill(snippet, width=width, initial_indent="   ", subsequent_indent="   "))
    for i, r in enumerate(results[:AUTO_TRIES], 1):
        try:
            page = fetch_page(r["url"], timeout=10)
        except FetchError as e:
            print(f"\n(skipped result {i}: {e})")
            continue
        print()
        show_page(page, AUTO_CHARS, label=f"Result {i}: ")
        return
    print('\nCould not read any result. Read one yourself: yuy "fetch <url>"')


def do_enginetest(s):
    """Try every search engine and say which ones work from this machine. Raw pages go to /tmp."""
    global LAST_HTML
    query = " ".join(w for w in s.split() if w.lower() not in WEB_STOP) or "laravel"
    words = query_words(query)
    print(f"Query: {query}")
    for name, search, strict in ENGINES:
        LAST_HTML = ""  # so a failed engine does not save the previous engine's page
        started = time.time()
        try:
            results = usable(search(query), words, strict)
            status = f"{len(results)} usable results"
        except SearchError as e:
            results, status = [], f"failed: {e}"
        print(f"\n{name}: {status} ({time.time() - started:.1f}s)")
        for r in results[:2]:
            print(f"   {' '.join(r['title'].split())[:70]} -> {r['url']}")
        if LAST_HTML:
            path = f"/tmp/yuy-engine-{name.lower()}.html"
            with open(path, "w", encoding="utf-8") as f:
                f.write(LAST_HTML)
            print(f"   raw page: {path}")


def do_browser(s):
    url = get_url(s)
    opener = shutil.which("xdg-open")
    if not opener:
        die("xdg-open not found. Install xdg-utils.")
    subprocess.Popen([opener, url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    print(f"Opening in the browser: {url}")


ACTIONS = {"search": do_search, "read": do_read, "create": do_create, "edit": do_edit,
           "delete": do_delete, "fetch": do_fetch, "browser": do_browser,
           "websearch": do_websearch,
           "enginetest": do_enginetest}


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ACTIONS:
        print(__doc__)
        sys.exit(2)
    ACTIONS[sys.argv[1]](" ".join(sys.argv[2:]))


if __name__ == "__main__":
    main()
