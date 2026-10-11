#!/usr/bin/env python3
"""Relevance helpers for the yuy web search (used by files.py, standard library only).

rank_results   sorts search results by how well they match the query. It uses EmbeddingGemma
               through the local llama-server (the one router.py starts) and falls back to plain
               word matching when that server cannot be reached.
show_relevant  reads the most relevant pages in parallel and prints the paragraphs that match
               the query instead of the start of the page. A page that cannot be read is
               replaced by its search snippet.

Numbers in the query ("14" in "laravel 14") must appear in a result or paragraph to count.
"""
import functools
import json
import math
import os
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHRASE_BONUS = 0.10    # the query words appear next to each other, in order
NUM_BONUS = 0.15       # every number of the query appears
NUM_PENALTY = 0.20     # a number of the query is missing
MIN_WORD_SHARE = 0.6   # a result must hold this share of the query words (and all numbers) to be read
PAGES = 3              # pages shown at most
TRIES = 6              # candidate pages fetched (in parallel)
FETCH_TIMEOUT = 10     # seconds per page
EXCERPTS = 3           # paragraphs shown per page
EXCERPT_CHARS = 500    # characters per paragraph

STOP = set("""the a an of in on for to and or is are was were be what how why when where which who
about with web search websearch lookup look up cari carikan tentang di internet online dan yang
apa itu untuk dari""".split())
TOKEN = re.compile(r"\w+(?:\.\w+)*")


# ---------------------------------------------------------------- words and numbers

def query_terms(query):
    """-> (words, numbers). Keeps short tokens, so "14" in "laravel 14" counts."""
    words = []
    for w in TOKEN.findall(query.lower()):
        if w not in STOP and w not in words:
            words.append(w)
    return words, [w for w in words if any(c.isdigit() for c in w)]


@functools.lru_cache(maxsize=256)
def token_re(w):
    if any(c.isdigit() for c in w):  # "14" matches "14", "v14", "14.x" but not "2014" or "14.2"
        return re.compile(rf"(?<![\w.])v?{re.escape(w)}(?!\w|\.\d)", re.I)
    return re.compile(rf"(?<!\w){re.escape(w)}", re.I)


def has(text, w):
    return bool(token_re(w).search(text))


def phrase_re(words):
    """The query words next to each other in order ("laravel 14", "laravel-v14"). None for one word."""
    if len(words) < 2:
        return None
    sep = r"[\W_]{1,3}(?:v(?=\d))?"
    return re.compile(r"(?<!\w)" + sep.join(re.escape(w) for w in words), re.I)


# ---------------------------------------------------------------- embeddings

def embed_url():
    env = os.environ.get("YUY_EMBED_URL")
    if env:
        return env
    port = 8081
    try:
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
            port = int(json.load(f).get("port", 8081))
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return f"http://127.0.0.1:{port}/v1/embeddings"


def embed(texts, timeout=20):
    req = urllib.request.Request(
        embed_url(),
        data=json.dumps({"input": texts, "model": "embeddinggemma"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = sorted(json.load(r)["data"], key=lambda d: d["index"])
    out = []
    for d in data:
        v = d["embedding"]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def doc_text(r):
    return f"title: {' '.join(r['title'].split())} | text: {' '.join(r['snippet'].split())[:400]}"


# ---------------------------------------------------------------- ranking

def rank_results(query, results):
    """-> (results sorted by relevance, method). Sets "score", "relevant" and "rank" on every result."""
    words, nums = query_terms(query)
    phrase = phrase_re(words)
    cos, method = [None] * len(results), "word match"
    if results:
        try:
            vecs = embed([f"task: search result | query: {query}"] + [doc_text(r) for r in results])
            cos = [sum(a * b for a, b in zip(vecs[0], v)) for v in vecs[1:]]
            method = "EmbeddingGemma"
        except (urllib.error.URLError, OSError, KeyError, ValueError, IndexError):
            method = "word match, embedding server not reachable"

    for r, c in zip(results, cos):
        text = f"{r['title']} {r['snippet']} {r['url']}"
        share = sum(1 for w in words if has(text, w)) / len(words) if words else 1.0
        nums_ok = all(has(text, n) for n in nums)
        score = c if c is not None else share
        if phrase and phrase.search(text):
            score += PHRASE_BONUS
        if nums:
            score += NUM_BONUS if nums_ok else -NUM_PENALTY
        r["score"] = score
        r["relevant"] = nums_ok and share >= MIN_WORD_SHARE

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked, method


# ---------------------------------------------------------------- excerpts

def blocks(text):
    """Paragraphs and single list items of a page text, each as one line."""
    out = []
    for block in re.split(r"\n\s*\n", text):
        if not block.strip():
            continue
        if block.lstrip().startswith("- "):
            item = []
            for line in block.splitlines():
                if line.startswith("- ") and item:
                    out.append(" ".join(" ".join(item).split()))
                    item = []
                item.append(line.strip())
            if item:
                out.append(" ".join(" ".join(item).split()))
        else:
            out.append(" ".join(block.split()))
    return out


def window(block, focus, chars):
    """At most `chars` characters of a block, starting a little before the first match."""
    if len(block) <= chars:
        return block
    m = focus.search(block)
    start = max(0, (m.start() if m else 0) - chars // 4)
    if start:
        start = block.rfind(" ", 0, start) + 1
    piece = block[start:start + chars]
    if start + chars < len(block):
        piece = piece.rsplit(" ", 1)[0] + " ..."
    return ("... " if start else "") + piece


def excerpts(text, words, nums, phrase, strict):
    """Paragraphs that hold every query word and number (strict) or at least half the words."""
    need_nums = nums if strict else []
    min_share = 0.99 if strict else 0.5
    focus = phrase or token_re((nums or words or ["."])[0])
    scored = []
    for i, b in enumerate(blocks(text)):
        if len(b) < 25:
            continue
        hits = sum(1 for w in words if has(b, w))
        if words and hits / len(words) < min_share:
            continue
        if not all(has(b, n) for n in need_nums):
            continue
        score = hits / max(len(words), 1) + (0.5 if phrase and phrase.search(b) else 0.0)
        scored.append((score, i, b))
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:EXCERPTS]
    return [window(b, focus, EXCERPT_CHARS) for _, _, b in sorted(top, key=lambda x: x[1])]


# ---------------------------------------------------------------- output

def host_of(url):
    host = urllib.parse.urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def fetch_all(urls, fetch_page, fetch_error):
    """Fetch pages in parallel. -> [(page, None) or (None, error text)] in the same order."""
    if not urls:
        return []

    def one(url):
        try:
            return fetch_page(url, timeout=FETCH_TIMEOUT), None
        except fetch_error as e:
            return None, str(e)
        except Exception as e:  # one bad page must not stop the others
            return None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=min(len(urls), 4)) as pool:
        return list(pool.map(one, urls))


def bullet(text, width):
    return textwrap.fill(text, width=width, initial_indent="    - ", subsequent_indent="      ",
                         break_long_words=False, break_on_hyphens=False)


def show_relevant(query, results, fetch_page, fetch_error, width):
    """Read the most relevant results and print the paragraphs that match the query."""
    words, nums = query_terms(query)
    phrase = phrase_re(words)
    strict = True
    candidates = [r for r in results if r.get("relevant")]
    if not candidates:
        strict = False
        candidates = results[:3]
        print(f'\nNo result clearly mentions every part of "{query}". Showing the closest ones.')
    candidates = candidates[:TRIES]
    if not candidates:
        print("\nNo results to read.")
        return

    fetched = fetch_all([r["url"] for r in candidates], fetch_page, fetch_error)
    print("\n== Relevant pages ==")
    shown, skipped = 0, []
    for r, (page, err) in zip(candidates, fetched):
        if shown >= PAGES:
            break
        title = textwrap.shorten(" ".join(r["title"].split()), width=90, placeholder="...")
        head = f"[{r['rank']}] {title} ({host_of(r['url'])})"
        if page is None:  # blocked or not text: the search snippet still says something
            print(f"\n{head}\n    {r['url']}\n    (page not readable: {err}; showing the search snippet)")
            snippet = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(r["snippet"].split()))
            if snippet:
                print(bullet(snippet, width))
            shown += 1
            continue
        picks = excerpts(page[4], words, nums, phrase, strict)
        if not picks:
            skipped.append(host_of(r["url"]))
            continue
        shown += 1
        print(f"\n{head}\n    {r['url']}")
        for p in picks:
            print(bullet(p, width))

    if shown == 0:
        print('\nNo readable page matched. Read one yourself: yuy "fetch <url>"')
    elif skipped:
        print(f"\n(not shown, no matching paragraph: {', '.join(skipped)})")
