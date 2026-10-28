"""
fetch_espn_matchups.py

Downloads this week's ESPN fantasy matchups and each team's ACTUAL points from
days that are already finished, for weekly_matchup_simulator_v3_1.py.

Uses the same league + .env login (ESPN_SWID, ESPN_S2) as fantasy_sim.py.

Run order each time you want a fresh projection:
    1. fantasy_sim.py                  (rosters, injury statuses, schedule flags)
    2. your current-season history rebuild (player_game_history_2027_current.csv)
    3. fetch_espn_matchups.py          (this file)
    4. weekly_matchup_simulator_v3_1.py

Output: fantasy_matchups.csv, one row per fantasy team:
    matchup_period, matchup_id, fantasy_team_id, opponent_team_id,
    completed_points, completed_days, first_sim_day_index, scoring_period_ids,
    current_scoring_period, fetched_at_utc

Why ESPN's points: only ESPN knows who each manager actually STARTED on the
days already played. Today's (possibly live) scoring period is never counted
as completed; the simulator simulates today in full.

Testing / debugging:
    python fetch_espn_matchups.py --save-raw              also saves espn_matchups_raw.json
    python fetch_espn_matchups.py --from-json FILE.json   parse a saved response (no network)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

LEAGUE_ID = 1421453942
SEASON = 2027
DAYS_PER_MATCHUP = 7          # Mon-Sun weekly matchups in this league

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "fantasy_matchups.csv")
RAW_FILE = os.path.join(BASE_DIR, "espn_matchups_raw.json")

LEAGUE_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
    f"seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"
)


# Lineup slots that do NOT score: 12 = bench, 13 = IR.
NON_SCORING_SLOTS = {12, 13}
# ESPN basketball stat ids: 40 = minutes, 42 = games played.
STAT_MINUTES, STAT_GP = "40", "42"


def _cookies() -> dict:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    swid, s2 = os.getenv("ESPN_SWID"), os.getenv("ESPN_S2")
    if not swid or not s2:
        raise SystemExit("ESPN_SWID or ESPN_S2 is missing from .env")
    return {"SWID": swid, "espn_s2": s2}


def download_roster_day(scoring_period: int) -> dict:
    """Every team's lineup and each player's stats for one finished ESPN day."""
    r = requests.get(LEAGUE_URL, params={"view": "mRoster", "scoringPeriodId": scoring_period},
                     cookies=_cookies(), timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"roster request for day {scoring_period} failed: HTTP {r.status_code}")
    return r.json()


def parse_day_starts(data: dict, scoring_period: int) -> dict:
    """
    {teamId: {"starts": n, "points": x}} for one day: players in a scoring
    lineup slot who actually played (minutes > 0 or a game played recorded).
    """
    out = {}
    for team in data.get("teams") or []:
        starts, points = 0, 0.0
        for entry in (team.get("roster") or {}).get("entries") or []:
            if entry.get("lineupSlotId") in NON_SCORING_SLOTS:
                continue
            player = (entry.get("playerPoolEntry") or {}).get("player") or {}
            for st in player.get("stats") or []:
                if st.get("scoringPeriodId") != scoring_period or st.get("statSourceId") != 0:
                    continue
                stats = st.get("stats") or {}
                mins = float(stats.get(STAT_MINUTES, 0) or 0)
                gp = float(stats.get(STAT_GP, 0) or 0)
                applied = float(st.get("appliedTotal") or 0.0)
                played = mins > 0 or gp > 0 or (STAT_MINUTES not in stats and STAT_GP not in stats and applied != 0)
                if played:
                    starts += 1
                    points += applied
                break
        out[team.get("id")] = {"starts": starts, "points": points}
    return out


def download_league_json() -> dict:
    params = [("view", v) for v in ("mMatchupScore", "mScoreboard", "mSettings", "mStatus", "mTeam")]
    r = requests.get(LEAGUE_URL, params=params, cookies=_cookies(), timeout=30)
    print("ESPN league API HTTP status:", r.status_code)
    if r.status_code != 200:
        print(r.text[:2000])
        raise SystemExit("ESPN request failed.")
    return r.json()


def _points_by_period(side: dict) -> dict[int, float]:
    raw = side.get("pointsByScoringPeriod") or {}
    out = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v or 0.0)
        except (TypeError, ValueError):
            pass
    return out


def parse_matchups(data: dict) -> pd.DataFrame:
    # ---- current scoring period (today's ESPN day) -------------------------
    status = data.get("status") or {}
    current_sp = data.get("scoringPeriodId") or status.get("latestScoringPeriod")
    if current_sp is None:
        raise SystemExit("Could not find the current scoring period in the ESPN response.")
    current_sp = int(current_sp)

    # ---- current matchup period --------------------------------------------
    schedule = data.get("schedule") or []
    current_mp = status.get("currentMatchupPeriod")
    if current_mp is None:
        # Fallback: the matchup period whose daily points include today.
        for m in schedule:
            for side in (m.get("home") or {}, m.get("away") or {}):
                if current_sp in _points_by_period(side):
                    current_mp = m.get("matchupPeriodId")
    if current_mp is None:
        raise SystemExit("Could not determine the current matchup period.")
    current_mp = int(current_mp)

    # ---- which ESPN scoring periods (days) make up this week ---------------
    # ESPN's settings list matchup periods, not days, so the week is found from
    # the daily points ESPN returns for this matchup period: the earliest day
    # with points is Monday (Monday always has NBA games). If no points exist
    # yet (Monday before tip-off), fall back to today's US Eastern weekday.
    keys = set()
    for m in schedule:
        if int(m.get("matchupPeriodId", -1)) != current_mp:
            continue
        for side in (m.get("home") or {}, m.get("away") or {}):
            keys |= set(_points_by_period(side))
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    first_sp_by_date = current_sp - today_et.weekday()
    if keys:
        first_sp = min(keys)
        if first_sp != first_sp_by_date:
            print(f"NOTE: week start from ESPN points (day {first_sp}) differs from the US calendar "
                  f"estimate (day {first_sp_by_date}); using ESPN points. This is normal around "
                  "ESPN's overnight day rollover.")
    else:
        first_sp = first_sp_by_date
        print("NOTE: no points recorded yet this week; week start taken from the US calendar.")
    period_sps = list(range(first_sp, first_sp + DAYS_PER_MATCHUP))
    stray = sorted(k for k in keys if k not in period_sps)
    if stray:
        raise SystemExit(
            f"ESPN points exist for days {stray} outside a 7-day week starting at day {first_sp}. "
            "This matchup period may not be a Mon-Sun week; run with --save-raw and send the file."
        )
    completed_sps = [sp for sp in period_sps if sp < current_sp]
    first_sim_day = len(completed_sps)        # 0 = Monday ... 7 = week finished
    week_start = today_et - timedelta(days=(current_sp - first_sp))

    team_names = {}
    for t in data.get("teams") or []:
        name = t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip()
        team_names[t.get("id")] = name

    rows = []
    fetched = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for m in data.get("schedule") or []:
        if int(m.get("matchupPeriodId", -1)) != current_mp:
            continue
        sides = [m.get("home") or {}, m.get("away") or {}]
        sides = [s for s in sides if s.get("teamId") is not None]
        for i, side in enumerate(sides):
            other = sides[1 - i] if len(sides) == 2 else {}
            pbp = _points_by_period(side)
            if pbp:
                completed = sum(pbp.get(sp, 0.0) for sp in completed_sps)
                source = "pointsByScoringPeriod"
            else:
                # Fallback: total points. Only exact if today's games haven't started.
                completed = float(side.get("totalPoints") or 0.0)
                source = "totalPoints (fallback)"
            rows.append({
                "matchup_period": current_mp,
                "matchup_id": m.get("id"),
                "fantasy_team_id": side.get("teamId"),
                "fantasy_team": team_names.get(side.get("teamId"), ""),
                "opponent_team_id": other.get("teamId"),
                "opponent_team": team_names.get(other.get("teamId"), "") if other else "BYE",
                "completed_points": round(completed, 2),
                "completed_days": first_sim_day,
                "first_sim_day_index": first_sim_day,
                "points_source": source,
                "completed_points_by_day": json.dumps({str(sp): pbp.get(sp, 0.0) for sp in completed_sps}),
                "scoring_period_ids": json.dumps(period_sps),
                "week_start": str(week_start),
                "current_scoring_period": current_sp,
                "fetched_at_utc": fetched,
            })
    if not rows:
        raise SystemExit(f"No matchups found for matchup period {current_mp}.")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", help="parse a saved ESPN response instead of downloading")
    ap.add_argument("--save-raw", action="store_true", help="also save the raw ESPN response")
    ap.add_argument("--roster-json-dir", help="with --from-json: folder of saved espn_roster_day_<N>.json files")
    ap.add_argument("--out", default=OUTPUT_FILE)
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = download_league_json()
        if args.save_raw:
            with open(RAW_FILE, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            print("Saved raw response:", os.path.basename(RAW_FILE))

    df = parse_matchups(data)

    # ---- actual starts on finished days (for PPP to date) -------------------
    sps = json.loads(df["scoring_period_ids"].iloc[0])
    first_idx = int(df["first_sim_day_index"].iloc[0])
    completed_sps = sps[:first_idx]
    totals = {}
    starts_ok = True
    for sp in completed_sps:
        try:
            if args.from_json:
                if not args.roster_json_dir:
                    raise RuntimeError("offline mode without --roster-json-dir")
                with open(os.path.join(args.roster_json_dir, f"espn_roster_day_{sp}.json"), encoding="utf-8") as fh:
                    day = json.load(fh)
            else:
                day = download_roster_day(sp)
                if args.save_raw:
                    with open(os.path.join(BASE_DIR, f"espn_roster_day_{sp}.json"), "w", encoding="utf-8") as fh:
                        json.dump(day, fh)
        except Exception as exc:
            print(f"WARNING: could not get lineups for ESPN day {sp} ({exc}); PPP to date unavailable.")
            starts_ok = False
            break
        for tid, v in parse_day_starts(day, sp).items():
            t = totals.setdefault(tid, {"starts": 0, "points": 0.0, "mismatch_days": 0})
            t["starts"] += v["starts"]
            t["points"] += v["points"]
            row = df[df["fantasy_team_id"] == tid]
            if len(row):
                espn_day = json.loads(row["completed_points_by_day"].iloc[0]).get(str(sp), 0.0)
                if abs(espn_day - v["points"]) > 0.5:
                    t["mismatch_days"] += 1
    if starts_ok:
        df["completed_starts"] = df["fantasy_team_id"].map(lambda t: totals.get(t, {}).get("starts", 0))
        df["lineup_points_check"] = df["fantasy_team_id"].map(lambda t: round(totals.get(t, {}).get("points", 0.0), 2))
        df["lineup_mismatch_days"] = df["fantasy_team_id"].map(lambda t: totals.get(t, {}).get("mismatch_days", 0))
    else:
        df["completed_starts"] = np.nan
        df["lineup_points_check"] = np.nan
        df["lineup_mismatch_days"] = np.nan

    df.to_csv(args.out, index=False)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    first = int(df["first_sim_day_index"].iloc[0])
    print()
    print(f"Matchup period {df['matchup_period'].iloc[0]} | ESPN scoring period now: "
          f"{df['current_scoring_period'].iloc[0]}")
    print("Completed days counted as actual points:", ", ".join(days[:first]) or "none")
    print("Days the simulator will simulate:", ", ".join(days[first:]) or "none (week finished)")
    print()
    pairs = df.drop_duplicates("matchup_id")
    for _, r in pairs.iterrows():
        opp = df[(df["matchup_id"] == r["matchup_id"]) & (df["fantasy_team_id"] != r["fantasy_team_id"])]
        opp_pts = float(opp["completed_points"].iloc[0]) if len(opp) else 0.0
        print(f"  {r['fantasy_team']:<28} {r['completed_points']:>8.1f}  vs  "
              f"{opp_pts:<8.1f} {r['opponent_team']}")
    if first and df["completed_starts"].notna().all():
        print()
        print("Starts so far (lineup players who played) and PPP to date:")
        for _, r in df.sort_values("fantasy_team").iterrows():
            st = int(r["completed_starts"])
            ppp = r["completed_points"] / st if st else float("nan")
            print(f"  {r['fantasy_team']:<28} {st:>3} starts   PPP {ppp:6.2f}")
        bad = df[df["lineup_mismatch_days"] > 0]
        if len(bad):
            print("\nWARNING: lineup points differ from ESPN's daily totals for:",
                  ", ".join(bad["fantasy_team"]), "- start counts may be off; run with --save-raw and send the files.")
    if (df["points_source"] != "pointsByScoringPeriod").any():
        print("\nWARNING: some teams used totalPoints fallback; run before today's games start for exact totals.")
    print("\nSaved:", os.path.basename(args.out))


if __name__ == "__main__":
    main()
