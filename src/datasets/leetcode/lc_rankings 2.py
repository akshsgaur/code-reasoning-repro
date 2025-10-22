#!/usr/bin/env python3
import argparse, json, re, sys, time
import requests
from typing import Dict, Any, List

RANKING_API = "https://leetcode.com/contest/api/ranking/{slug}/"

def extract_cookie_value(cookie_header: str, name: str) -> str | None:
    """Grab a single cookie value from a Cookie header string."""
    m = re.search(r'(^|;\s*){}=([^;]+)'.format(re.escape(name)), cookie_header)
    return m.group(2) if m else None

def build_session(cookie_header: str, referer: str) -> requests.Session:
    sess = requests.Session()
    # Send exactly what a browser would; the Cookie header is the key.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": referer,
        "Origin": "https://leetcode.com",
        "Connection": "keep-alive",
        "Cookie": cookie_header.strip(),
    }
    # Add CSRF header if present (not strictly required for GET, but helps).
    csrf = extract_cookie_value(cookie_header, "csrftoken")
    if csrf:
        headers["x-csrftoken"] = csrf
    sess.headers.update(headers)
    return sess

def fetch_page(sess: requests.Session, slug: str, page: int, region: str) -> Dict[str, Any]:
    url = RANKING_API.format(slug=slug.rstrip("/"))
    params = {"pagination": page, "region": region}
    r = sess.get(url, params=params, timeout=30)
    if r.status_code == 403:
        raise SystemExit(
            "403 Forbidden from LeetCode. Usually this means your cookie is "
            "expired or you’re not on the same IP as your browser session. "
            "Refresh the ranking page in your browser, recopy the Cookie header, "
            "and try again."
        )
    r.raise_for_status()
    return r.json()

def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep what’s most useful; you can customize if you want more/less.
    out = []
    for row in rows:
        out.append({
            "rank": row.get("rank"),
            "username": row.get("username"),
            "user_slug": row.get("user_slug"),
            "country_code": row.get("country_code"),
            "contest_id": row.get("contest_id"),
            "score": row.get("score"),
            "finish_time": row.get("finish_time"),
            "replays": row.get("replays", {}),
            "submissions": row.get("submissions", {}),
            "medal": row.get("medal"),
            "avatar_url": row.get("avatar_url"),
            "data_region": row.get("data_region"),
        })
    return out

def scrape(slug: str, cookie: str, region: str, max_pages: int, delay: float) -> Dict[str, Any]:
    referer = f"https://leetcode.com/contest/{slug}/ranking/"
    sess = build_session(cookie, referer)

    all_rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    page = 1

    while page <= max_pages:
        data = fetch_page(sess, slug, page, region)
        if page == 1:
            meta = {k: v for k, v in data.items() if k not in ("total_rank",)}
        rows = data.get("total_rank") or []
        if not rows:
            break
        all_rows.extend(normalize_rows(rows))
        page += 1
        if delay:
            time.sleep(delay)

    return {
        "contest_slug": slug,
        "region": region,
        "fetched_pages": page - 1,
        "user_num": meta.get("user_num"),
        "ak_info": meta.get("ak_info"),
        "total_rows": len(all_rows),
        "rows": all_rows,
    }

def main():
    p = argparse.ArgumentParser(description="Scrape LeetCode contest rankings (browser-like).")
    p.add_argument("slug", help="Contest slug, e.g. weekly-contest-470")
    p.add_argument("--cookie", required=True, help="Full Cookie header value from your browser.")
    p.add_argument("--region", default="global_v2", help="Region param (global_v2 or global).")
    p.add_argument("--max-pages", type=int, default=200, help="Safety cap on pagination.")
    p.add_argument("--delay", type=float, default=0.2, help="Delay between page requests (seconds).")
    p.add_argument("-o", "--out", default=None, help="Output JSON file (default: <slug>.rankings.json)")
    args = p.parse_args()

    result = scrape(args.slug, args.cookie, args.region, args.max_pages, args.delay)

    out_path = args.out or f"{args.slug}.rankings.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # Also print a tiny summary:
    print(f"Wrote {len(result['rows'])} rows from {result['fetched_pages']} pages → {out_path}")

if __name__ == "__main__":
    main()