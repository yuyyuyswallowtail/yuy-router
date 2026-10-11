#!/usr/bin/env python3
"""Relevance helpers for the yuy web search (used by files.py, standard library only).

rank_results   sorts search results by how well they match the query. It uses EmbeddingGemma
               through the local llama-server (the one router.py starts) and falls back to plain
               word matching when that server cannot be reached.
show_relevant  reads the most relevant pages in parallel and prints a direct answer built from
               complete, self-contained sentences picked with EmbeddingGemma (extractive, nothing is
               generated). The answer has a Summary (one cohesive paragraph: a topic sentence that says
               what the subject is, then supporting facts in a logical order, joined with "It" and light
               transitions), Key facts (facts that did not fit in the paragraph), Highlights (feature
               names with a one-line explanation), and Sources numbered in order of appearance. Cut-off
               snippets, meta sentences, taglines, dangling references, opinions, advice, near
               duplicates, and forecasts whose date has already passed are left out or ranked lower.
               With -d/--debug (or YUY_DEBUG=1) the candidate table, how the headings of every page were
               read, and the matching paragraphs of every page are printed as well, and sources keep
               their result-list rank.

Numbers in the query ("14" in "laravel 14") must appear in a result, paragraph, or sentence to count.
"""
import datetime
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

PHRASE_BONUS = 0.10        # the query words appear next to each other, in order
NUM_BONUS = 0.15           # every number of the query appears
NUM_PENALTY = 0.20         # a number of the query is missing
MIN_WORD_SHARE = 0.6       # a result must hold this share of the query words (and all numbers) to be read
MIN_BLOCK_SHARE = 0.6      # a paragraph must hold this share of the query words to be shown
PROSE_BONUS = 0.25         # sentences are preferred over headings
CODE_DENSITY = 0.04        # share of code-like symbols above which a paragraph counts as a code sample
TRIES = 6                  # candidate pages fetched (in parallel)
PAGES = 3                  # pages shown at most in the debug view
FETCH_TIMEOUT = 10         # seconds per page
EXCERPTS = 3               # paragraphs shown per page in the debug view (at most one a heading)
EXCERPT_CHARS = 500        # characters per paragraph in the debug view
TRACE_LINES = 40           # heading decisions shown per page in the debug view

ANSWER_SENTENCES = 4       # sentences in the answer at most
ANSWER_WINDOW = 0.12       # only sentences within this score distance of the best one
PER_SOURCE_ANSWER = 2      # sentences per page at most
SENTENCES_PER_PAGE = 12    # candidate sentences per page that get embedded
MIN_SENT_CHARS = 35        # shortest sentence that can be part of the answer
MIN_SENT_WORDS = 6
AGREE_FLOOR = 0.85         # sentences from two pages agree only above this cosine ...
AGREE_PERCENTILE = 0.95    # ... and above this percentile of all cross-page sentence pairs
MAX_SUPPORT = 3            # pages named as support for one sentence at most
DUP = 0.88                 # cosine above which two sentences that state the same number are the same fact
DUP_SURE = 0.95            # cosine above which two sentences are near duplicates whatever they say
CONTAIN = 0.70             # share of the shorter sentence's content words found in a chosen one
CONSENSUS_BONUS = 0.08     # score bonus when other pages agree (two pages count at most)
SENT_PHRASE_BONUS = 0.05   # the sentence holds the query words next to each other
SUBJECT_BONUS = 0.04       # the sentence starts with the query phrase ("Laravel 13 requires ...")
LEAD_BONUS = 0.04          # one of the first sentences of its page (usually the summary)
LEAD_SENTENCES = 3
TOPIC_LEAD_BONUS = 0.05    # topic sentence: a page's opening sentence says what the subject is
TOPIC_SUBJECT_BONUS = 0.03 # topic sentence: starts with the subject
FACT_BONUS = 0.04          # dates, versions, and other numbers (three count at most)
RANK_BONUS = 0.04          # sentences from higher ranked pages score a little higher
YOU_PENALTY = 0.05         # "you/your" voice (tutorials, marketing)
STALE_PENALTY = 0.15       # a forecast ("expected", "scheduled", ...) whose dates are all in the past
OPINION_PENALTY = 0.08     # evaluative language instead of facts
ADVICE_PENALTY = 0.08      # advice and instructions instead of facts
VAGUE_PENALTY = 0.10       # "a number of new features": says nothing a name or a number would
MMR_WEIGHT = 0.30          # a sentence loses this much times its similarity to the ones already chosen
HIGHLIGHTS = 8             # key points listed at most
HL_MIN_PAGES = 2           # a key point must be listed by this many pages (when that many pages have a list)
HL_MATCH = 0.75            # share of content words two key point names must have in common to count as one
HL_OVERVIEW = 3            # a sentence naming this many key points is an overview: the list says it better
HL_RAW_CHARS = 3000        # text kept per feature section, to tell whether a short heading is part of it
FEATURE_SLOTS = 2          # answer sentences that name a key point, when key points exist
FEATURE_WINDOW = 0.30      # such a sentence may score this far below the best one
TAGS_SHOWN = 2             # source tags shown per line; "(+N)" says how many more pages agree
NOTE_CHARS = 170           # characters of the one-line explanation of a key point
PARA_SENTENCES = 4         # sentences in the Summary paragraph at most (topic sentence + supporting sentences)
PARA_WORDS = 110           # the Summary paragraph stops growing past this many words
PRONOUN_RUN = 2            # "It" is used this many times in a row at most, then the subject is named again

STOP = set("""the a an of in on for to and or is are was were be what how why when where which who
about with web search websearch lookup look up cari carikan tentang di internet online dan yang
apa itu untuk dari""".split())
FUNC = STOP | set("with as it its by that this from at up than so".split())
TOKEN = re.compile(r"\w+(?:\.\w+)*")
CODE_CHARS = set("$;{}<>[]=\\|`")
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(\[])')
SENT_END = re.compile(r"[.!?][\"”')\]]*$")
CUT_END = re.compile(r"(?:\.{2,}|…)$")
FLUFF_SPLIT = re.compile(r"\s[—–]\s")
BOILER = re.compile(
    r"\b(cookies?|subscribe|newsletter|sign up|log in|privacy policy|terms of (?:use|service)|"
    r"all rights reserved|follow us|share this|read more|advertisement|click here|table of contents|"
    r"this (?:post|article|page|guide)|as (?:they|it) (?:are|is) announced|stay tuned|"
    r"(?:that|this) (?:situation|issue|problem|approach)|as (?:mentioned|discussed|noted|described|shown) (?:above|earlier|before)|the (?:above|former|latter))\b",
    re.I)
# "simplifies this for smaller deployments": 'this' points at something in the sentence before
DANGLING = re.compile(r"\bthis\s+(?:for|by|with|to|in|on|at|so)\b", re.I)
# sentences that start like this only make sense with the sentence before them
ANAPHORA = re.compile(
    r"^(?:this|that|these|those|it|its|they|their|he|she|here|there|but|and|so|also|however|then|such|"
    r"none|nothing|whatever|anything|everything|otherwise|instead|thus|therefore|meanwhile)\b", re.I)
# sentences that describe the article instead of stating a fact, and page taglines
META = re.compile(
    r"^(?:what (?:is|are|we|you)\b|here(?:'s| is| are)\b|in this (?:article|post|blog|guide|video|tutorial|section|series)\b|"
    r"this (?:article|post|guide|video|tutorial)\b|everything you\b|learn\b|discover\b|find out\b|"
    r"read on\b|let's\b|let me\b|welcome\b|introduction\b|see (?:the|our|this|below|above|also)\b|"
    r"(?:in|on) (?:the )?(?:table|list) below\b|every (?:meaningful|major|new|single|important)\b|"
    r"(?:the )?(?:complete|ultimate|definitive) (?:guide|list)\b)", re.I)
OPINION = re.compile(
    r"\b(?:might feel|feels? like|one of the (?:most|best)|the (?:biggest|best|worst)|exactly why|"
    r"game[- ]?changer|amazing|awesome|incredible|honestly|most important|roundups?|listicles?|"
    r"in my opinion|i think|i believe|personally|"
    r"(?:is|was|are|were) an? (?:strong|solid|great|exciting|impressive|powerful|fantastic|excellent|remarkable)|"
    r"(?:significant|major|big|huge|giant) (?:step|leap|milestone)|step forward|represents another)\b", re.I)
ADVICE = re.compile(
    r"^(?:waiting|upgrading|migrating|updating|if you|when you|make sure|don't|do not|avoid|consider|"
    r"remember|keep|use|try|check|we (?:generally |highly |strongly )?(?:recommend|suggest|advise)|"
    r"it(?:'s| is) recommended|visit|click|download|refer to)\b", re.I)
VAGUE = re.compile(
    r"\b(?:a (?:number|variety|range|host|couple|lot) of|several|various|numerous|"
    r"(?:new )?features and (?:improvements|changes|enhancements)|"
    r"(?:aim(?:s|ing)|designed) to (?:simplify|improve|make|enhance)|"
    r"has (?:officially )?arrived|(?:are|is) (?:now )?considering|(?:across|around) the (?:globe|world)|"
    r"when or if|what you need to know|get ready)\b", re.I)
REF_MARK = re.compile(r"\s*\[(?:\d+|[a-z]|citation needed|edit)\]")  # footnote marks that would look like our [1][2] tags
HL_QUERY = re.compile(r"\b(?:new|features?|released?|changes?|changelog|updates?|upgrad\w*|version|highlights|"
                      r"improvements)\b|\d", re.I)
HL_HEAD = re.compile(r"\b(?:new|features?|highlights?|improvements?|changes?|added|introduc\w*|headline|releases?|"
                     r"announce\w*)\b", re.I)
HL_JUNK = re.compile(r"^(?:download|install\w*|contribut\w*|related|summary|conclusion|introduction|overview|faq|"
                     r"comments?|share|references?|see also|resources?|about|changelog|credits|sponsors?|"
                     r"final thoughts|upgrad\w+|migration|wrapping up|key takeaways|table of contents|what'?s next|"
                     r"getting started|support|requirements?|more|other|misc\w*|minor|bonus|extras?|how to|frequently asked|"
                     r"keep reading|need help|want to|further reading|read more)\b", re.I)
HL_REQ = re.compile(r"\b(?:minimum|requirements?|requires?)\b", re.I)  # a requirement is a fact, not a feature
NAME_BAD = re.compile(
    r"^(?:find|see|turn|get|test|run|make|read|start|stop|set|add|enable|disable|install|update|upgrade|migrate|"
    r"check|use|try|keep|avoid|consider|remember|learn|build|create|write|open|click|follow|take|go|let|wait|"
    r"review|audit|plan|prepare|decide|know|understand|choose|pick|ask|look|stay|watch|join|download)\s+"
    r"(?:on|out|up|off|your|the|a|an|to|for|what|where|how|why|if|when|all|every|any)\b|"
    r"\b(?:is|are|was|were|has|have|had|will|can|should|must)\b|n't\b|^[a-z_]\w*_\w+(?:\(\))?$", re.I)
HL_STRONG = re.compile(r"\b(?:new|features?|highlights?|improvements?|changes?|headline)\b", re.I)
HL_SPLIT = re.compile(r"\s+[—–-]\s+|:\s|\s\(|;\s")   # where a feature name ends and its explanation starts
NOTE_SPLIT = re.compile(r"\s+[—–-]\s+|:\s|;\s")
VERSIONED = re.compile(r"^[A-Za-z][\w .+-]*?\s+v?(\d+(?:\.\d+)*)\.?$")  # "Laravel 11": a product and a version only
CLOSING = re.compile(r"^(?:conclusions?|final thoughts|wrapping up|takeaways?|key takeaways|faq|comments?|"
                     r"related (?:posts|articles)|see also|references?|about the author|share this)\b", re.I)
YOU = re.compile(r"\b(?:you|your|yours|you'll|you've|you're|yourself)\b", re.I)

# a verb right after the subject: only then "Laravel 13 requires ..." can become "It requires ..."
SUBJECT_VERB = re.compile(
    r"^(?:is|was|are|were|will|would|can|could|should|must|has|have|had|now|also|"
    r"requires?|introduces?|brings?|drops?|removes?|adds?|includes?|continues?|supports?|ships?|comes?|offers?|"
    r"provides?|makes?|replaces?|gets?|improves?|raises?|changes?|delivers?|lets?|allows?|enables?|uses?|"
    r"targets?|focuses|extends?|deprecates?|ends?|starts?|launche[sd]|arrives?|lands?)\b", re.I)
CONTRAST = re.compile(r"\b(?:not|no|never|but|however|although|though|despite|unlike|instead|without|except|"
                      r"yet|still|cannot)\b|n't\b", re.I)
LOWER_FIRST = {"The", "A", "An", "It", "Its", "This", "These", "Those", "There"}
TRANSITIONS = {2: "In addition,", 3: "Moreover,"}   # by position in the paragraph (0 = topic sentence)
NOT_TOPIC = {"opinion", "advice", "vague", "you", "stale"}
# rhetoric instead of a definition: "isn't just about the present—it's laying groundwork for ..."
RHETORIC = re.compile(r"\b(?:isn[’']t just|not just|more than just|laying (?:the )?groundwork|groundwork for|"
                      r"what[’']s (?:coming|next|ahead))\b|\w—\w", re.I)
CODE_TOKEN = re.compile(r"\w\(\)|::|->|\$\w|#\[")   # an API name: too specific to say what the subject is
# headings where the list of features is over: how to upgrade, FAQ, other posts
END_HEAD = re.compile(r"^(?:how to\b|frequently asked|keep reading|need help|want to\b|further reading|read more|"
                      r"what[’']?s next|more from)", re.I)
TOPIC_DEFINE_BONUS = 0.04  # topic sentence: the subject followed by a verb ("Laravel 13 continues ...")

MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
DATE_Q = re.compile(r"\bQ([1-4])\s*(20\d\d)\b", re.I)
DATE_M = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
                    r"(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?(20\d\d)\b", re.I)
DATE_Y = re.compile(r"\b(20\d\d)\b")
FORECAST = re.compile(
    r"\b(?:expected|scheduled|will (?:be|launch|release|arrive|land|ship|come)|upcoming|"
    r"coming (?:in|soon)|set to|planned|slated|due|launch(?:es|ing) in|next major|should arrive)\b", re.I)


def debug_on():
    """True for yuy -d/--debug (read from the router's command line) or YUY_DEBUG=1."""
    if os.environ.get("YUY_DEBUG"):
        return True
    try:
        with open(f"/proc/{os.getppid()}/cmdline", "rb") as f:
            args = f.read().decode(errors="replace").split("\0")
    except OSError:
        return False
    return any(a in ("-d", "--debug") for a in args[1:])


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


def subject_re(words, phrase):
    """A sentence that starts with the query phrase (optionally after 'The') talks about the subject itself."""
    if phrase is not None:
        return re.compile(r"^(?:the\s+)?" + phrase.pattern, re.I)
    if words:
        return re.compile(r"^(?:the\s+)?" + token_re(words[0]).pattern, re.I)
    return None


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


def embed_many(texts, chunk=32):
    out = []
    for i in range(0, len(texts), chunk):
        out.extend(embed(texts[i:i + chunk], timeout=60))
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


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
            cos = [dot(vecs[0], v) for v in vecs[1:]]
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


# ---------------------------------------------------------------- paragraphs and sentences

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


def looks_like_noise(b):
    """Code samples, command lines, comment-thread headers, and tables of contents are not worth showing."""
    if b.startswith("#"):  # headings are fine
        return False
    if b.count("#") >= 3 or b.count("|") >= 3:
        return True
    if len(re.findall(r"(?:^|\s)--?[a-z][\w-]*", b)) >= 3:  # command-line flags
        return True
    symbols = sum(1 for ch in b if ch in CODE_CHARS)
    return symbols / len(b) > CODE_DENSITY


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
    """Prose that holds every query number and most query words (half the words when not strict)."""
    need_nums = nums if strict else []
    share = MIN_BLOCK_SHARE if strict else 0.5
    need_hits = math.ceil(len(words) * share - 1e-9) if words else 0
    focus = phrase or token_re((nums or words or ["."])[0])
    scored = []
    for i, b in enumerate(blocks(text)):
        if len(b) < 25 or looks_like_noise(b):
            continue
        hits = sum(1 for w in words if has(b, w))
        if hits < need_hits or not all(has(b, n) for n in need_nums):
            continue
        heading = b.startswith("#")
        score = (hits / max(len(words), 1) + (0.5 if phrase and phrase.search(b) else 0.0)
                 + (0.0 if heading else PROSE_BONUS))
        scored.append((score, i, heading, b))

    picked, headings = [], 0
    for score, i, heading, b in sorted(scored, key=lambda x: (-x[0], x[1])):
        if heading:
            if headings >= 1:
                continue
            headings += 1
        picked.append((i, b))
        if len(picked) == EXCERPTS:
            break
    return [window(b, focus, EXCERPT_CHARS) for _, b in sorted(picked)]


def trim_fluff(s):
    """Cut an opinion tail after a dash: 'Released on March 17 — and it is the best yet.' -> 'Released on March 17.'"""
    parts = FLUFF_SPLIT.split(s, maxsplit=1)
    if len(parts) == 2 and OPINION.search(parts[1]) and len(parts[0]) >= MIN_SENT_CHARS:
        return parts[0].rstrip(" ,;:") + "."
    return s


def good_sentence(s):
    """A complete sentence that still makes sense on its own (no titles, breadcrumbs, 'This is ...', cut-offs)."""
    return (MIN_SENT_CHARS <= len(s) <= 420 and len(s.split()) >= MIN_SENT_WORDS
            and (s[0].isupper() or s[0].isdigit())
            and not CUT_END.search(s) and bool(SENT_END.search(s)) and not s.endswith("?")
            and s.count("/") < 2
            and not ANAPHORA.match(s) and not META.match(s) and not DANGLING.search(s)
            and not BOILER.search(s) and not looks_like_noise(s))


def page_sentences(text):
    """Self-contained sentences of a page, without headings, code, menus, questions, and boilerplate."""
    out, closing = [], False
    for b in blocks(text):
        b = REF_MARK.sub("", b)
        if b.startswith("#"):
            closing = bool(CLOSING.match(b.lstrip("#").strip()))
            continue
        if closing or looks_like_noise(b):
            continue
        if b.startswith("- "):
            b = b[2:]
        for s in SENT_SPLIT.split(b):
            s = trim_fluff(s.strip())
            if good_sentence(s):
                out.append(s)
    return out


def snippet_sentences(snippet):
    """Complete sentences of a search snippet (for pages that could not be read).

    Search engines cut snippets with '...': the cut-off sentence is kept only up to its last complete clause.
    """
    s = " ".join(snippet.split())
    s = re.sub(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\s*(?:\.{2,}|…)?\s*", "", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s).strip()
    cut = bool(CUT_END.search(s))
    if cut:
        s = re.sub(r"\s*(?:\.{2,}|…)$", "", s)
    parts = [p.strip() for p in SENT_SPLIT.split(s) if p.strip()]
    if cut and parts and not SENT_END.search(parts[-1]):
        last = parts.pop()
        at = max(last.rfind(", "), last.rfind(" — "), last.rfind("; "))
        if at >= MIN_SENT_CHARS:
            parts.append(last[:at] + ".")
    return [p for p in (trim_fluff(p) for p in parts) if good_sentence(p)]


# ---------------------------------------------------------------- dates

def end_dates(text):
    """Dates a sentence mentions, as the last days of their quarter, month, or year."""
    out = []
    for q, y in DATE_Q.findall(text):
        out.append(datetime.date(int(y), int(q) * 3, 28))
    for m, y in DATE_M.findall(text):
        out.append(datetime.date(int(y), MONTHS[m.lower()[:3]], 28))
    if not out:  # a bare year only counts when nothing more precise is there
        out = [datetime.date(int(y), 12, 28) for y in DATE_Y.findall(text)]
    return out


def stale_forecast(text, today=None):
    """True for 'expected / scheduled / launches in ...' sentences whose dates are all already in the past."""
    dates = end_dates(text)
    return bool(dates) and max(dates) < (today or datetime.date.today()) and bool(FORECAST.search(text))


# ---------------------------------------------------------------- the answer

def query_variants(query):
    """Several ways to ask for the same thing, so facts beat sentences that merely repeat the keywords.
    Index 1 ("What is ...?") is also used to find the topic sentence."""
    return [query, f"What is {query}?", f"Key facts and details about {query}"]


def lexical(s, words, phrase):
    hits = sum(1 for w in words if has(s, w))
    return hits / max(len(words), 1) + (0.5 if phrase and phrase.search(s) else 0.0), hits


def build_candidates(pairs, words, nums, phrase, strict):
    """Sentences worth embedding: they hold the query numbers and at least half of the query words."""
    need_hits = math.ceil(len(words) * 0.5 - 1e-9) if words else 0
    plain = [w for w in words if w not in nums]
    cands, seen = [], set()
    for r, (page, _err) in pairs:
        readable = page is not None
        pool = page_sentences(page[4]) if readable else snippet_sentences(r["snippet"])
        scored = []
        for pos, s in enumerate(pool):
            score, hits = lexical(s, words, phrase)
            if hits < need_hits or (plain and not any(has(s, w) for w in plain)):
                continue
            if strict and not all(has(s, n) for n in nums):
                continue
            scored.append((score, pos, s))
        scored.sort(key=lambda x: -x[0])
        for score, pos, s in scored[:SENTENCES_PER_PAGE]:
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            cands.append({"text": s, "src": r["rank"], "lex": score, "support": [],
                          "pos": pos if readable else None})
    return cands


def assign_support(cands):
    """Name the pages that say the same as each sentence. -> the agreement threshold that was used."""
    sims = [dot(a["vec"], b["vec"]) for i, a in enumerate(cands) for b in cands[i + 1:]
            if a["src"] != b["src"]]
    if not sims:
        return None
    sims.sort()
    threshold = max(AGREE_FLOOR, sims[min(int(len(sims) * AGREE_PERCENTILE), len(sims) - 1)])
    for c in cands:
        best = {}
        for o in cands:
            if o["src"] != c["src"]:
                s = dot(c["vec"], o["vec"])
                if s >= threshold and s > best.get(o["src"], 0.0):
                    best[o["src"]] = s
        c["support"] = [src for src, _ in sorted(best.items(), key=lambda kv: -kv[1])[:MAX_SUPPORT]]
    return threshold


def content_words(text):
    return {w for w in TOKEN.findall(text.lower()) if w not in FUNC}


def duplicate(a, b):
    """Near duplicates: most content words of the shorter sentence are in the other, or the cosine is very high,
    or the cosine is high and both state the same number (a date, a version)."""
    wa, wb = content_words(a["text"]), content_words(b["text"])
    if wa and wb and len(wa & wb) / min(len(wa), len(wb)) >= CONTAIN:
        return True
    if not (a.get("vec") and b.get("vec")):
        return False
    cos = dot(a["vec"], b["vec"])
    return cos >= DUP_SURE or (cos >= DUP and bool(a.get("facts", set()) & b.get("facts", set())))


def similarity(a, b):
    """Cosine of two candidate sentences, or their content-word overlap when there are no embeddings."""
    if a.get("vec") and b.get("vec"):
        return dot(a["vec"], b["vec"])
    wa, wb = content_words(a["text"]), content_words(b["text"])
    return len(wa & wb) / max(min(len(wa), len(wb)), 1)


def select(cands, points=()):
    """Best sentence first, then the ones that add the most: score minus a share of the similarity to what
    is already chosen (maximal marginal relevance). No near duplicates, at most 2 per page."""
    pool = sorted(cands, key=lambda c: -c["score"])
    if not pool:
        return []
    top = pool[0]["score"]
    pool = [c for c in pool if c["score"] >= top - ANSWER_WINDOW]
    sims = sorted(similarity(x, y) for i, x in enumerate(pool) for y in pool[i + 1:])
    base = sims[len(sims) // 2] if sims else 0.0  # a typical pair: only similarity above it costs points

    def penalty(c):
        worst = max((similarity(c, x) for x in chosen), default=0.0)
        return MMR_WEIGHT * max(0.0, worst - base) / max(1.0 - base, 0.05)

    chosen, per_src = [], {}
    while pool and len(chosen) < ANSWER_SENTENCES:
        pool = [c for c in pool if per_src.get(c["src"], 0) < PER_SOURCE_ANSWER
                and not any(duplicate(c, x) for x in chosen)]
        if not pool:
            break
        best = max(pool, key=lambda c: c["score"] - penalty(c))
        chosen.append(best)
        per_src[best["src"]] = per_src.get(best["src"], 0) + 1
        pool.remove(best)
    if points and chosen:
        named = [c for c in chosen if mentions_point(c["text"], points)]
        top = max(c["score"] for c in cands)
        extra = sorted((c for c in cands if all(c is not x for x in chosen) and mentions_point(c["text"], points)
                        and c["score"] >= top - FEATURE_WINDOW
                        and not {"advice", "opinion", "vague", "you"} & set(c["flags"])),
                       key=lambda c: -c["score"])
        for c in extra:
            if len(named) >= FEATURE_SLOTS:
                break
            if any(duplicate(c, x) for x in chosen) or sum(x["src"] == c["src"] for x in chosen) >= PER_SOURCE_ANSWER:
                continue
            if len(chosen) >= ANSWER_SENTENCES:  # make room: drop the weakest sentence that names no key point
                spare = [x for x in chosen[1:] if all(x is not n for n in named)]
                if not spare:
                    break
                chosen.remove(min(spare, key=lambda x: x["score"]))
            chosen.append(c)
            named.append(c)
    return chosen


def score_candidate(c, qvecs, nums, phrase, subject, today):
    """Total score of one sentence. Also sets c["qcos"], c["qwhat"] (closeness to 'What is ...?') and c["flags"]."""
    text = c["text"]
    qcos = sum(dot(q, c["vec"]) for q in qvecs) / len(qvecs) if qvecs else c["lex"]
    c["qwhat"] = dot(qvecs[1], c["vec"]) if qvecs else c["lex"]
    facts = [t for t in re.findall(r"\d+(?:\.\d+)*", text) if t not in nums]
    c["facts"] = set(facts)
    flags = []
    score = qcos
    if phrase and phrase.search(text):
        score += SENT_PHRASE_BONUS
    score += CONSENSUS_BONUS * min(len(c["support"]), 2) / 2
    score += FACT_BONUS * min(len(facts), 3) / 3
    score += RANK_BONUS * (TRIES - min(c["src"], TRIES)) / TRIES
    if YOU.search(text):
        score -= YOU_PENALTY
        flags.append("you")
    if stale_forecast(text, today):
        score -= STALE_PENALTY
        flags.append("stale")
    if OPINION.search(text) or RHETORIC.search(text):
        score -= OPINION_PENALTY
        flags.append("opinion")
    if ADVICE.match(text):
        score -= ADVICE_PENALTY
        flags.append("advice")
    if VAGUE.search(text) and text.count(",") < 2:  # a list of named things is specific enough
        score -= VAGUE_PENALTY
        flags.append("vague")
    m = subject.match(text) if subject else None
    if m:
        score += SUBJECT_BONUS
        flags.append("subject")
        if SUBJECT_VERB.match(text[m.end():].lstrip()):
            flags.append("defines")
    if c.get("pos") is not None and c["pos"] < LEAD_SENTENCES:
        score += LEAD_BONUS
        flags.append("lead")
    c["qcos"], c["flags"], c["score"] = qcos, flags, score


# ---------------------------------------------------------------- key points (feature names)

def highlight_name(item):
    """'Property hooks: define get and set logic ...' -> 'Property hooks'. '' when it is a sentence, a
    requirement, or page chrome."""
    name = HL_SPLIT.split(item.strip(), maxsplit=1)[0]
    name = re.sub(r"^(?:\d+[.)]\s*|new:?\s+)", "", name, flags=re.I).strip(" .,;:*`\"'#")
    if not 4 <= len(name) <= 60 or len(name.split()) > 7 or name.endswith("?"):
        return ""
    if HL_JUNK.match(name) or HL_REQ.search(name) or NAME_BAD.search(name) or YOU.search(name) \
            or BOILER.search(name) or looks_like_noise(name):
        return ""
    return name


def note_sentence(block):
    """First complete sentence of a paragraph, as a one-line explanation of a feature. '' when none fits."""
    for s in SENT_SPLIT.split(REF_MARK.sub("", block))[:2]:
        s = trim_fluff(s.strip())
        if (25 <= len(s) <= 300 and len(s.split()) >= 5 and SENT_END.search(s) and not s.endswith("?")
                and not CUT_END.search(s) and not BOILER.search(s) and not META.match(s)
                and not DANGLING.search(s) and not OPINION.search(s) and not looks_like_noise(s)):
            return s
    return ""


def item_note(item):
    """The explanation after the name in a list item: 'Cache::touch(): extend a cache TTL' -> 'Extend a cache TTL.'"""
    parts = NOTE_SPLIT.split(item.strip(), maxsplit=1)
    if len(parts) < 2:
        return ""
    rest = SENT_SPLIT.split(parts[1].strip())[0].strip()
    if len(rest) < 20 or len(rest.split()) < 4 or OPINION.search(rest) or BOILER.search(rest) \
            or DANGLING.search(rest) or looks_like_noise(rest):
        return ""
    rest = rest[0].upper() + rest[1:]
    return rest if SENT_END.search(rest) else rest.rstrip(" ,;:") + "."


def part_of(name, raw):
    """True when a one-word heading ('Models') is just one of the things the section of the feature above it
    lists ('... your models, jobs, and commands ...'). Only plain single words qualify: 'PHP Attributes' or
    'Cache::touch()' are features of their own."""
    if len(name.split()) != 1 or not name.isalpha() or not raw:
        return False
    return bool(re.search(rf"\b{re.escape(name.lower().rstrip('s'))}s?\b", raw.lower()))


def page_features(text, nums=(), trace=None):
    """-> (features, strong). Features of a release page as {"name", "note", "raw", "parts"}: the headings one
    level below a 'New features' / 'Release' heading (deeper headings, and one-word headings that the section
    above them lists, become the parts of that feature; its first fitting paragraph gives the note), or else the
    short list items below one. strong: that heading itself says 'new' or 'features'. trace collects what was
    decided for every heading (shown with -d)."""
    def log(msg):
        if trace is not None:
            trace.append(msg)

    heads, items, parent, child, strong, cur = [], [], 0, 0, False, None
    for b in blocks(text):
        if b.startswith("#"):
            level, title = len(b) - len(b.lstrip("#")), b.lstrip("#").strip()
            tag = f"h{level} {textwrap.shorten(title, width=60, placeholder='...')!r}"
            other = VERSIONED.match(title)
            if other and nums and other.group(1) not in nums:  # a section about another version: stop collecting
                parent = child = 0
                cur = None
                log(f"{tag}: section about another version, stop")
            elif parent and (CLOSING.match(title) or END_HEAD.match(title)):
                parent = child = 0
                cur = None
                log(f"{tag}: closing section, stop")
            elif parent and level > parent:
                if not child:
                    child = level
                name = highlight_name(title)
                if level == child:
                    prev = heads[-1] if heads else None
                    if name and prev is not None and part_of(name, prev["raw"]):
                        prev["parts"].append(name)
                        cur = prev
                        log(f"{tag}: part of {prev['name']!r}")
                    elif name:
                        cur = {"name": name, "note": "", "raw": "", "parts": []}
                        heads.append(cur)
                        log(f"{tag}: feature {name!r}")
                    else:
                        cur = None
                        log(f"{tag}: not a feature name")
                elif cur is not None and name:
                    cur["parts"].append(name)
                    log(f"{tag}: sub-part of {cur['name']!r}")
                else:
                    log(f"{tag}: deeper heading, ignored")
            elif HL_HEAD.search(title):
                parent, child, cur = level, 0, None
                strong = strong or bool(HL_STRONG.search(title))
                log(f"{tag}: collecting below this heading")
            else:
                parent = child = 0
                cur = None
                log(f"{tag}: ends the section")
        elif parent and b.startswith("- "):
            name = highlight_name(b[2:])
            if name:
                items.append({"name": name, "note": item_note(b[2:]), "raw": b, "parts": []})
                log(f"item {name!r}")
        elif parent and cur is not None:
            if len(cur["raw"]) < HL_RAW_CHARS:
                cur["raw"] = f"{cur['raw']} {b}".strip()
            if not cur["note"]:
                cur["note"] = note_sentence(b)
    return (heads if len(heads) >= 3 else items), strong


def collect_highlights(pairs, words=(), nums=(), trace=None):
    """-> [{"name", "pages": [ranks], "note", "parts"}] for the key points several pages list, most agreed on
    first. A single page is only trusted when its heading itself says 'new' or 'features'. trace: {rank: lines}."""
    lists = []
    for r, (page, _err) in pairs:
        if page is not None:
            lines = [] if trace is not None else None
            feats, strong = page_features(page[4], nums, lines)
            if trace is not None and lines:
                trace[r["rank"]] = lines
            if feats:
                lists.append((r["rank"], feats, strong))
    groups, qwords = [], set(words)
    for rank, feats, _strong in lists:
        for f in feats:
            name = f["name"]
            w = content_words(name)
            rest = {x for x in w if not re.fullmatch(r"\d+(?:\.\d+)*", x)}
            if not rest or rest <= qwords:  # empty, or the query's own name again ("Laravel 13")
                continue
            for g in groups:
                if len(w & g["words"]) / min(len(w), len(g["words"])) >= HL_MATCH:
                    if rank not in g["pages"]:
                        g["pages"].append(rank)
                    if not g["note"]:
                        g["note"] = f["note"]
                    for p in f["parts"]:
                        if p not in g["parts"]:
                            g["parts"].append(p)
                    break
            else:
                groups.append({"name": name, "words": w, "pages": [rank], "order": len(groups),
                               "note": f["note"], "parts": list(f["parts"])})
    if len(lists) == 1 and not lists[0][2]:
        return []
    need = HL_MIN_PAGES if len(lists) >= HL_MIN_PAGES else 1
    keep = sorted((g for g in groups if len(g["pages"]) >= need), key=lambda g: (-len(g["pages"]), g["order"]))
    if len(keep) < 3:  # no agreement between pages: the best ranked page whose heading says 'new' or 'features'
        rank = next((rk for rk, nm, st in lists if st and len(nm) >= 3), None)
        keep = sorted((g for g in groups if rank in g["pages"]), key=lambda g: g["order"])
    return keep[:HIGHLIGHTS] if len(keep) >= 3 else []


def mentions_point(text, points):
    tw = content_words(text)
    return any(g["words"] and len(g["words"] & tw) / len(g["words"]) >= HL_MATCH for g in points)


# ---------------------------------------------------------------- the readable layout

class Tags:
    """Source numbers shown to the reader: the rank in the result list in debug mode, otherwise 1, 2, 3 ... in the
    order the sources first appear in the answer."""

    def __init__(self, keep_ranks):
        self.keep = keep_ranks
        self.numbers = {}  # result rank -> number shown

    def label(self, ranks, extra=True):
        ranks = list(dict.fromkeys(ranks))
        shown = []
        for r in ranks[:TAGS_SHOWN]:
            if r not in self.numbers:
                self.numbers[r] = r if self.keep else len(self.numbers) + 1
            shown.append(self.numbers[r])
        more = len(ranks) - TAGS_SHOWN
        return "".join(f"[{n}]" for n in sorted(shown)) + (f" (+{more})" if extra and more > 0 else "")


def fact_rank(text):
    """Order of facts: release first, then requirements and support, then the rest."""
    t = text.lower()
    if re.search(r"\b(?:releas\w*|launch\w*|available|ship\w*)\b", t):
        return 0
    if re.search(r"\b(?:requir\w*|minimum|bug fixes|security (?:fixes|updates)|support\w*)\b", t):
        return 1
    return 2


def src_ranks(c):
    return [c["src"]] + [s for s in c["support"] if s != c["src"]]


def pick_topic(cands, points):
    """The sentence that says what the subject is: the plain sentence (no numbers, no dates, no opinion or advice,
    not about one feature) closest to 'What is ...?' among ALL candidates, whether or not the selection kept it."""
    top = max(c["score"] for c in cands)
    pool = [c for c in cands
            if not c["facts"] and not end_dates(c["text"]) and not NOT_TOPIC & set(c["flags"])
            and not CODE_TOKEN.search(c["text"])
            and c["score"] >= top - FEATURE_WINDOW and not mentions_point(c["text"], points)]
    if not pool:
        return None
    return max(pool, key=lambda c: c["qwhat"] + (TOPIC_LEAD_BONUS if "lead" in c["flags"] else 0.0)
               + (TOPIC_SUBJECT_BONUS if "subject" in c["flags"] else 0.0)
               + (TOPIC_DEFINE_BONUS if "defines" in c["flags"] else 0.0))


def arrange(chosen, cands, points):
    """-> (topic sentence or None, supporting sentences in a logical order). A sentence about one or two key points
    becomes the explanation of that point when the page gave none; a sentence that lists three or more is an
    overview that the Highlights say better, so it is left out."""
    numeric, plain = [], []
    for c in chosen:
        tw = content_words(c["text"])
        hits = [g for g in points if g["words"] and len(g["words"] & tw) / len(g["words"]) >= HL_MATCH]
        if len(hits) >= HL_OVERVIEW:
            continue
        if hits:
            if not hits[0]["note"]:
                hits[0]["note"] = c["text"]
        elif c["facts"] or end_dates(c["text"]):
            numeric.append(c)
        else:
            plain.append(c)
    topic = pick_topic(cands, points)
    general = [c for c in plain if not CODE_TOKEN.search(c["text"])]
    if topic is None and general:
        topic = max(general, key=lambda c: ("lead" in c["flags"], c["score"]))
    rest = sorted(numeric, key=lambda c: (fact_rank(c["text"]), -c["score"]))
    rest += sorted((c for c in plain if c is not topic), key=lambda c: -c["score"])
    if topic is None and rest:
        topic, rest = rest[0], rest[1:]
    if topic is not None:
        rest = [c for c in rest if c is not topic and not duplicate(c, topic)]
    return topic, rest


def compose_paragraph(topic, supports, subject, nums):
    """One cohesive paragraph: the topic sentence first, then supporting facts in a logical order (release,
    requirements, the rest). Cohesion devices: "It" instead of repeating the subject (only where a verb follows
    the subject, never more than twice in a row), "In addition," / "Moreover," on the later sentences (not on
    sentences that contradict or contrast), and nothing in second person.
    -> ([(sentence, source ranks)], supporting sentences that did not fit)."""
    if topic is None:
        return [], list(supports)
    items = [(topic["text"], src_ranks(topic))]
    refers = bool(subject and nums and subject.match(topic["text"]))  # "It" needs a clear subject to point at
    words_used, leftover, run = len(topic["text"].split()), [], 0
    for c in supports:
        if (len(items) >= PARA_SENTENCES or words_used >= PARA_WORDS or YOU.search(c["text"])
                or CODE_TOKEN.search(c["text"])):
            leftover.append(c)
            continue
        text, pronoun = c["text"], False
        m = subject.match(text) if refers else None
        if m and run < PRONOUN_RUN and SUBJECT_VERB.match(text[m.end():].lstrip()):
            text, pronoun = "It " + text[m.end():].lstrip(), True
        lead = TRANSITIONS.get(len(items))
        if lead and not CONTRAST.search(text):
            text = f"{lead} {text[0].lower() + text[1:] if text.split()[0] in LOWER_FIRST else text}"
        run = run + 1 if pronoun else 0
        items.append((text, src_ranks(c)))
        words_used += len(text.split())
    return items, leftover


def fill(text, width, first, rest):
    return textwrap.fill(text, width=width, initial_indent=first, subsequent_indent=rest,
                         break_long_words=False, break_on_hyphens=False)


def short_note(s, limit=NOTE_CHARS):
    """A long explanation cut at the last clause before the limit, so it still reads as a sentence."""
    s = s.strip()
    if len(s) <= limit:
        return s
    cut = max(s.rfind(", ", 0, limit), s.rfind("; ", 0, limit), s.rfind(" — ", 0, limit), s.rfind(" - ", 0, limit))
    if cut >= 50:
        return s[:cut].rstrip(" ,;:—-") + "."
    return textwrap.shorten(s, width=limit, placeholder="...")


def highlight_line(g, label):
    name = g["name"]
    parts = [p for p in dict.fromkeys(g.get("parts", [])) if p.lower() != name.lower()][:4]
    extra = f" ({', '.join(parts)})" if parts else ""
    note = short_note(g["note"]) if g.get("note") else ""
    bare = re.sub(r"^(?:the|a|an)\s+", "", note, flags=re.I)
    if note and bare.lower().startswith(name.lower()):  # the sentence already starts with the name
        return f"{note}{extra} {label}"
    return f"{name}{extra}: {note} {label}" if note else f"{name}{extra} {label}"


def print_answer(query, pairs, words, nums, phrase, strict, width):
    """Print the direct answer. -> False when no sentence qualified."""
    cands = build_candidates(pairs, words, nums, phrase, strict)
    if not cands:
        return False

    variants = query_variants(query)
    qvecs, threshold = None, None
    try:
        vecs = embed_many([f"task: search result | query: {v}" for v in variants]
                          + [f"title: none | text: {c['text']}" for c in cands])
        qvecs = vecs[:len(variants)]
        for c, v in zip(cands, vecs[len(variants):]):
            c["vec"] = v
        threshold = assign_support(cands)
    except (urllib.error.URLError, OSError, KeyError, ValueError, IndexError):
        qvecs = None
        for c in cands:
            c.pop("vec", None)
            c["support"] = []

    debug = debug_on()
    today = datetime.date.today()
    subject = subject_re(words, phrase)
    for c in cands:
        score_candidate(c, qvecs, nums, phrase, subject, today)

    trace = {} if debug else None
    points = collect_highlights(pairs, words, nums, trace) if HL_QUERY.search(query) else []
    chosen = select(cands, points)
    if not chosen:
        return False

    topic, rest = arrange(chosen, cands, points)
    paragraph, leftover = compose_paragraph(topic, rest, subject, nums)
    tags = Tags(keep_ranks=debug)

    print(f"\n== Answer: {query} ==" if debug else "")
    if qvecs is None:
        print("(embedding server not reachable: sentences picked by word match only)\n")
    if paragraph:
        print("Summary")
        print(fill(" ".join(f"{text} {tags.label(ranks, extra=False)}" for text, ranks in paragraph),
                   width, "  ", "  "))
        print()
    if leftover:
        print("Key facts")
        for c in sorted(leftover, key=lambda c: (fact_rank(c["text"]), -c["score"])):
            print(fill(f"{c['text']} {tags.label(src_ranks(c))}", width, "  - ", "    "))
        print()
    if points:
        print("Highlights")
        for g in points:
            print(fill(highlight_line(g, tags.label(g["pages"])), width, "  - ", "    "))
        print()

    print("Sources")
    by_rank = {r["rank"]: r for r, _ in pairs}
    for rank, n in sorted(tags.numbers.items(), key=lambda kv: kv[1]):
        r = by_rank.get(rank)
        if r:
            title = textwrap.shorten(" ".join(r["title"].split()), width=80, placeholder="...")
            print(f"  [{n}] {host_of(r['url'])} - {title}\n      {r['url']}")

    if debug:
        print("\n== Answer candidates (debug): score, query cosine, supporting pages, flags, page, sentence ==")
        print("(page = rank in the result list; T = topic sentence of the Summary)")
        for c in sorted(cands, key=lambda c: -c["score"])[:10]:
            mark = "T" if c is topic else ("*" if c in chosen else " ")
            others = [x for x in chosen if x is not c]
            near = max((similarity(c, x) for x in others), default=0.0)
            tag = c["flags"] + (["dup"] if c not in chosen and any(duplicate(c, x) for x in others) else [])
            tag += ["feature"] if points and mentions_point(c["text"], points) else []
            flags = f" {','.join(tag)}" if tag else ""
            print(f"{mark} {c['score']:.3f} q={c['qcos']:.3f} what={c['qwhat']:.3f} sup={len(c['support'])} "
                  f"near={near:.2f}{flags} [{c['src']}] {textwrap.shorten(c['text'], width=90, placeholder='...')}")
        if topic is not None and all(topic is not c for c in sorted(cands, key=lambda c: -c["score"])[:10]):
            print(f"T {topic['score']:.3f} q={topic['qcos']:.3f} what={topic['qwhat']:.3f} [{topic['src']}] "
                  f"{textwrap.shorten(topic['text'], width=90, placeholder='...')}")
        if threshold is not None:
            print(f"(agreement threshold {threshold:.3f}, {len(cands)} candidate sentences)")
        if trace:
            print("\n== Highlights trace (debug): how the headings of every page were read ==")
            for rank, lines in sorted(trace.items()):
                print(f"[{rank}] {len(lines)} headings and items")
                for line in lines[:TRACE_LINES]:
                    print(f"    {line}")
    return True


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


def print_excerpts(pairs, words, nums, phrase, strict, width):
    """Debug view: the matching paragraphs of every page."""
    print("\n== Relevant pages ==")
    shown, skipped = 0, []
    for r, (page, err) in pairs:
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


def show_relevant(query, results, fetch_page, fetch_error, width):
    """Read the most relevant results and print a direct answer (plus the matching paragraphs with -d)."""
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

    pairs = list(zip(candidates, fetch_all([r["url"] for r in candidates], fetch_page, fetch_error)))
    answered = print_answer(query, pairs, words, nums, phrase, strict, width)
    if debug_on() or not answered:
        if not answered:
            print("\nNo direct answer could be put together; showing the matching paragraphs instead.")
        print_excerpts(pairs, words, nums, phrase, strict, width)
