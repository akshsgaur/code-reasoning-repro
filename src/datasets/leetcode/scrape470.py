#!/usr/bin/env python3
# scrape_lc_contest.py
# Examples:
#   python scrape_lc_contest.py weekly-contest-470 --pages 1
#   python scrape_lc_contest.py weekly-contest-470 --pages all -o out.json
#   python scrape_lc_contest.py https://leetcode.com/contest/weekly-contest-470/ranking/ --csv

import argparse, csv, json, math, re, time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def normalize_slug(s: str) -> str:
    s = s.strip()
    if s.startswith("http"):
        m = re.search(r"/contest/([^/]+)/", s) or re.search(r"/contest/([^/?#]+)", s)
        if not m:
            raise SystemExit("Could not parse contest slug from URL.")
        return m.group(1)
    return s.strip("/")

def ts_to_iso(ts: Optional[int]) -> Optional[str]:
    if not ts and ts != 0: return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None

def fetch_json(session: requests.Session, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://leetcode.com/",
    }
    r = session.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def get_problem_map(session: requests.Session, slug: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns {question_id_str: {title, title_slug}} by calling
    https://leetcode.com/contest/api/info/<slug>/
    """
    url = f"https://leetcode.com/contest/api/info/{slug}/"
    try:
        data = fetch_json(session, url, {})
        qlist = (data or {}).get("questions") or []
        out = {}
        for q in qlist:
            qid = str(q.get("question_id"))
            out[qid] = {
                "title": q.get("title"),
                "title_slug": q.get("title_slug") or q.get("titleSlug"),
            }
        return out
    except Exception:
        # If this endpoint ever breaks, we just return an empty map.
        return {}

def get_ranking_page(session: requests.Session, slug: str, page: int, region: str="global_v2") -> Dict[str, Any]:
    url = f"https://leetcode.com/contest/api/ranking/{slug}/"
    params = {"pagination": page, "region": region}
    try:
        return fetch_json(session, url, params)
    except requests.HTTPError as e:
        if region == "global_v2":
            # Fallback to 'global' if needed
            return fetch_json(session, url, {"pagination": page, "region": "global"})
        raise e

def flatten_user_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        out.append({
            "rank": r.get("rank"),
            "score": r.get("score"),
            "finish_time_sec": r.get("finish_time"),
            "finish_time_iso": ts_to_iso(r.get("finish_time")),
            "username": r.get("username"),
            "user_slug": r.get("user_slug"),
            "country_code": r.get("country_code"),
            "country_name": r.get("country_name"),
            "avatar": r.get("avatar_url") or r.get("avatar"),
        })
    return out

def long_submission_rows(rows: List[Dict[str, Any]], problem_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Produces a long table:
    one row per (user, problem) using the 'submissions' object you see in DevTools.
    """
    long_rows: List[Dict[str, Any]] = []
    for r in rows:
        base = {
            "rank": r.get("rank"),
            "username": r.get("username"),
            "user_slug": r.get("user_slug"),
        }
        subs = r.get("submissions") or {}
        # subs looks like: { "4019": {date: 123, lang: "python", submission_id: 179xxxx, fail_count: 0}, ... }
        for qid, info in subs.items():
            qinfo = problem_map.get(str(qid), {})
            long_rows.append({
                **base,
                "question_id": int(qid) if str(qid).isdigit() else qid,
                "question_title": qinfo.get("title"),
                "question_slug": qinfo.get("title_slug"),
                "lang": info.get("lang"),
                "submission_id": info.get("submission_id"),
                "fail_count": info.get("fail_count"),
                "submit_time_sec": info.get("date"),
                "submit_time_iso": ts_to_iso(info.get("date")),
            })
    return long_rows

def main():
    ap = argparse.ArgumentParser(description="Scrape LeetCode contest ranking API (no login).")
    ap.add_argument("contest", help="Contest slug or URL (e.g., weekly-contest-470)")
    ap.add_argument("--pages", default="all", help="'all' or a positive integer (default: all)")
    ap.add_argument("--region", default="global_v2", help="Ranking region: global_v2 (default) or global")
    ap.add_argument("--sleep", type=float, default=0.4, help="Delay between requests (s)")
    ap.add_argument("-o", "--out", default="", help="Write aggregated JSON to this file; else print to stdout")
    ap.add_argument("--csv", action="store_true", help="Also write rankings.csv and submissions.csv")
    args = ap.parse_args()

    slug = normalize_slug(args.contest)
    sess = requests.Session()

    # Optional: fetch mapping from problem id -> title/slug
    problem_map = get_problem_map(sess, slug)

    # Page 1
    first = get_ranking_page(sess, slug, page=1, region=args.region)
    rows = (first.get("total_rank") or first.get("total_ranks") or [])
    per_page = len(rows) if rows else 25
    user_num = first.get("user_num")  # total participants (if provided)

    # Decide page count
    if args.pages.lower() == "all":
        total_pages = math.ceil(user_num / per_page) if (user_num and per_page) else None
    else:
        try:
            total_pages = max(1, int(args.pages))
        except ValueError:
            raise SystemExit("--pages must be 'all' or an integer")

    all_rows = rows[:]
    page = 1
    while True:
        if total_pages is not None and page >= total_pages:
            break
        page += 1
        time.sleep(args.sleep)
        pg = get_ranking_page(sess, slug, page=page, region=args.region)
        rows = (pg.get("total_rank") or pg.get("total_ranks") or [])
        if not rows:
            break
        all_rows.extend(rows)

        if total_pages is None and not user_num:
            user_num = pg.get("user_num")
            if user_num and per_page:
                total_pages = math.ceil(user_num / per_page)

    # Build outputs
    users_compact = flatten_user_rows(all_rows)
    subs_long = long_submission_rows(all_rows, problem_map)

    payload = {
        "contest_slug": slug,
        "region": args.region,
        "participants_reported": user_num,
        "pages_fetched": page,
        "per_page_observed": per_page,
        "problems": problem_map,          # {question_id: {title, title_slug}}
        "users": users_compact,           # list[dict] per user
        "submissions": subs_long,         # long table per (user, problem)
    }

    # JSON output
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote JSON -> {args.out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    # CSV outputs (optional)
    if args.csv:
        with open("rankings.csv", "w", newline="", encoding="utf-8") as f:
            cols = ["rank","score","finish_time_sec","finish_time_iso","username","user_slug","country_code","country_name","avatar"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in users_compact:
                w.writerow({k: r.get(k) for k in cols})
        with open("submissions.csv", "w", newline="", encoding="utf-8") as f:
            cols = ["rank","username","user_slug","question_id","question_title","question_slug","lang","submission_id","fail_count","submit_time_sec","submit_time_iso"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in subs_long:
                w.writerow({k: r.get(k) for k in cols})
        print("Wrote CSV -> rankings.csv, submissions.csv")

if __name__ == "__main__":
    main()