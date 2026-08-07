#!/usr/bin/env python3
"""
scrape_wikipedia.py
===================

Reads a keyword/test CSV (columns: id, prompt, keyword) and fetches matching 
plain-text Wikipedia articles for each row using MediaWiki Search API.

Usage:
    python3 src/scrape_wikipedia.py [--csv PATH] [--out DIR] [--delay SECONDS] [--resume] [-v]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

import requests

# ----------------------------------------------------------------------
# Constants & Configuration
# ----------------------------------------------------------------------
API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_UA = "PhysicsMCQWikiScraper/1.0 (24f2001637@ds.study.iitm.ac.in)"

MAX_RETRIES = 6
RETRY_BASE_DELAY = 3.0
RETRY_STATUS_CODES = {403, 429, 500, 502, 503, 504}

# Resolve default paths relative to repository root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_CSV = (
    PROJECT_ROOT / "data" / "test_keywords.csv"
    if (PROJECT_ROOT / "data" / "test_keywords.csv").exists()
    else (
        PROJECT_ROOT / "data" / "test.csv"
        if (PROJECT_ROOT / "data" / "test.csv").exists()
        else Path("test_keywords.csv")
    )
)
DEFAULT_OUT = PROJECT_ROOT / "data" / "wikipedia_pages"

# Common stopwords & generic filler words to exclude during token matching
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "and", "or", "by", "with",
    "from", "into", "is", "are", "was", "were", "be", "as", "that", "this", "it", "its",
    "their", "his", "her", "they", "them", "we", "you", "what", "which", "who", "whom",
    "whose", "how", "why", "when", "where", "do", "does", "did", "can", "could", "would",
    "should", "may", "might", "will", "shall", "must", "have", "has", "had", "been",
    "being", "about", "between", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "then", "once", "here", "there",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "d", "ll",
    "m", "re", "ve", "y", "physics", "science", "theory", "law", "effect", "field", "force",
    "energy", "system", "structure", "mechanism", "framework", "method", "concept", "principle",
    "definition", "purpose", "role", "use", "used", "describe", "described", "following",
    "best", "answer", "correct", "option", "statement", "accurate", "based", "given", "context",
    "among", "listed", "options", "choose", "select", "identify", "determine", "pick", "carefully",
    "from", "choices", "model", "models", "equation", "equations", "function", "functions",
    "factor", "factors", "type", "types", "kind", "kinds", "class", "classes", "form", "forms",
    "formalism", "formulation",
}


# ----------------------------------------------------------------------
# API Helper Functions
# ----------------------------------------------------------------------
def _get_with_retry(session: requests.Session, params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    """GET MediaWiki API with exponential backoff on rate-limits & server errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(API_URL, params=params, timeout=timeout)
            if r.status_code in RETRY_STATUS_CODES:
                retry_after = r.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if (retry_after and retry_after.isdigit())
                    else (RETRY_BASE_DELAY * (2**attempt))
                )
                print(f"      [rate-limit {r.status_code}] sleeping {wait:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            wait = RETRY_BASE_DELAY * (2**attempt)
            print(f"      [error {e}] sleeping {wait:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
    if last_exc:
        raise last_exc
    raise RuntimeError("API request retries exhausted")


# ----------------------------------------------------------------------
# Text Processing & Matching Helpers
# ----------------------------------------------------------------------
def slugify(title: str) -> str:
    """Sanitize title for safe filesystem filenames."""
    title = title.replace("/", "_").replace("\\", "_")
    title = re.sub(r"[^\w\.\- ]", "_", title, flags=re.UNICODE)
    return re.sub(r"\s+", "_", title.strip())


def tokenize(text: str) -> set[str]:
    """Normalize text, remove accents, lowercases, and strip stopwords."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def title_overlap_count(query: str, title: str) -> int:
    """Number of meaningful query tokens present in article title."""
    return len(tokenize(query) & tokenize(title))


def snippet_relevance(query: str, snippet: str) -> int:
    """Number of meaningful query tokens present in search result snippet."""
    if not snippet:
        return 0
    plain = re.sub(r"<[^>]+>", "", snippet)
    return len(tokenize(query) & tokenize(plain))


# ----------------------------------------------------------------------
# Wikipedia Search & Extraction
# ----------------------------------------------------------------------
def search_pages(session: requests.Session, query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Query MediaWiki Search API and retrieve candidate articles + disambiguation status."""
    params = {
        "action": "query",
        "format": "json",
        "redirects": "1",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": str(limit),
        "gsrprop": "snippet",
        "prop": "pageprops",
        "ppprop": "disambiguation",
    }
    data = _get_with_retry(session, params, timeout=30)
    pages_dict = data.get("query", {}).get("pages", {})
    ordered = sorted(pages_dict.values(), key=lambda p: p.get("index", 999999))

    results: list[dict[str, Any]] = []
    for page in ordered:
        if page.get("ns", 0) != 0:  # Only main namespace articles
            continue
        title = page.get("title", "")
        pageprops = page.get("pageprops", {}) or {}
        is_disambig = "disambiguation" in pageprops or title.lower().endswith("(disambiguation)")
        results.append({
            "title": title,
            "snippet": page.get("snippet", ""),
            "pageid": page.get("pageid"),
            "disambiguation": is_disambig,
        })
    return results


def get_extract(session: requests.Session, title: str) -> Optional[str]:
    """Fetch clean plain-text extract for a Wikipedia page."""
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": "1",
        "format": "json",
        "redirects": "1",
    }
    data = _get_with_retry(session, params, timeout=60)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract")
        if extract:
            header = (
                f"Wikipedia article: {page.get('title', title)}\n"
                f"Source: https://en.wikipedia.org/wiki/{page.get('title', title).replace(' ', '_')}\n"
                f"PageID: {page.get('pageid', 'n/a')}\n"
                f"{'=' * 70}\n\n"
            )
            return header + extract
    return None


def find_best_page(session: requests.Session, query: str, verbose: bool = False) -> Optional[str]:
    """Select highest-relevance non-disambiguation Wikipedia page for the query."""
    candidates = search_pages(session, query, limit=8)
    if verbose:
        titles_repr = [c['title'] + (' [disambig]' if c['disambiguation'] else '') for c in candidates]
        print(f"      search returned {len(candidates)} candidates: {titles_repr}")

    for cand in candidates:
        title = cand["title"]
        if cand.get("disambiguation"):
            if verbose:
                print(f"      -> skip disambig: {title}")
            continue

        title_n = title_overlap_count(query, title)
        snip_n = snippet_relevance(query, cand.get("snippet", ""))
        q_token_count = len(tokenize(query))
        snip_threshold = 1 if q_token_count <= 1 else 2

        if title_n == 0 and snip_n < snip_threshold:
            if verbose:
                print(f"      -> skip unrelated: '{title}' (title_overlap={title_n}, snippet_overlap={snip_n})")
            continue

        if verbose:
            print(f"      -> ACCEPT: {title} (title_overlap={title_n}, snippet_overlap={snip_n})")
        return title

    return None


# ----------------------------------------------------------------------
# Processing Engine
# ----------------------------------------------------------------------
def process_csv(
    csv_path: Path,
    out_dir: Path,
    delay: float,
    limit: Optional[int],
    start_id: Optional[int],
    user_agent: str,
    verbose: bool,
    resume: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip"})

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if start_id is not None:
        rows = [r for r in rows if int(r["id"]) >= start_id]
    if limit is not None:
        rows = rows[:limit]

    print(f"Processing {len(rows)} rows -> {out_dir}")
    print(f"Delay: {delay}s | Resume: {resume} | User-Agent: {user_agent}")
    print("-" * 70)

    ok, not_found, skipped, resumed = 0, 0, 0, 0

    for i, r in enumerate(rows, 1):
        qid = r["id"]
        query = (r.get("keyword") or r.get("prompt") or "").strip()
        if not query:
            print(f"[{i:>4}/{len(rows)}] id={qid}  EMPTY QUERY — skipping")
            skipped += 1
            continue

        if resume:
            existing = list(out_dir.glob(f"{qid}_*.txt"))
            if existing:
                print(f"[{i:>4}/{len(rows)}] id={qid}  query={query!r}  -> resume skip ({existing[0].name})")
                resumed += 1
                continue

        print(f"[{i:>4}/{len(rows)}] id={qid}  query={query!r}")

        try:
            title = find_best_page(session, query, verbose=verbose)
        except requests.RequestException as e:
            print(f"      ERROR during search: {e}")
            title = None

        if title:
            try:
                extract = get_extract(session, title)
            except requests.RequestException as e:
                print(f"      ERROR fetching extract for {title}: {e}")
                extract = None

            if extract:
                out_path = out_dir / f"{qid}_{slugify(title)}.txt"
                out_path.write_text(extract, encoding="utf-8")
                print(f"      -> wrote {len(extract):>7} chars -> {out_path.name}")
                ok += 1
            else:
                out_path = out_dir / f"{qid}_NOT_FOUND.txt"
                out_path.write_text(f"NOT_FOUND\nid: {qid}\nkeyword: {query}\nExtract empty.\n", encoding="utf-8")
                print("      -> page found but extract empty; wrote NOT_FOUND")
                not_found += 1
        else:
            out_path = out_dir / f"{qid}_NOT_FOUND.txt"
            out_path.write_text(f"NOT_FOUND\nid: {qid}\nkeyword: {query}\nNo suitable article found.\n", encoding="utf-8")
            print(f"      -> wrote NOT_FOUND: {out_path.name}")
            not_found += 1

        time.sleep(delay)

    print("-" * 70)
    print(f"Done. OK={ok}  NOT_FOUND={not_found}  SKIPPED={skipped}  RESUMED={resumed}  TOTAL={len(rows)}")


# ----------------------------------------------------------------------
# Command Line Interface
# ----------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Path to input CSV (default: {DEFAULT_CSV})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Output directory (default: {DEFAULT_OUT})")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (default: 1.0)")
    p.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    p.add_argument("--start-id", type=int, default=None, help="Start from specific question ID")
    p.add_argument("--user-agent", default=DEFAULT_UA, help="Custom User-Agent header")
    p.add_argument("-v", "--verbose", action="store_true", help="Print detailed search candidates")
    p.add_argument("--resume", action="store_true", help="Skip rows whose output file already exists")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if not args.csv.is_file():
        sys.exit(f"CSV file not found: {args.csv}")
    process_csv(
        csv_path=args.csv,
        out_dir=args.out,
        delay=args.delay,
        limit=args.limit,
        start_id=args.start_id,
        user_agent=args.user_agent,
        verbose=args.verbose,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
