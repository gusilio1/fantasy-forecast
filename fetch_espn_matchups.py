"""
fetch_espn_matchups.py

Downloads this week's ESPN fantasy matchups and each team's ACTUAL points from
days that are already finished, for weekly_matchup_simulator_v3_2.py, plus the
whole season's daily lineups for the PPP table and league records.

Season history: every finished day's lineups are downloaded ONCE and kept in
league_cache/ (the hourly GitHub workflow saves that folder between runs). The
most recent REFRESH_RECENT_DAYS finished days are re-downloaded each run to pick
up ESPN stat corrections. Outputs:
    league_daily_starts.csv   one row per player in a scoring lineup slot per day
    league_team_days.csv      one row per team per day: ESPN points, starts

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
CACHE_DIR = os.path.join(BASE_DIR, "league_cache")
DAILY_STARTS_FILE = os.path.join(BASE_DIR, "league_daily_starts.csv")
TEAM_DAYS_FILE = os.path.join(BASE_DIR, "league_team_days.csv")
REFRESH_RECENT_DAYS = 2

# Weekly add limit if ESPN does not report your league's real one. ESPN stores
# "no limit" as 0 or -1; set e.g. 7 here if your league caps adds at 7 a week.
ACQUISITION_LIMIT_OVERRIDE = 3       # league rule: 3 adds per week

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


def parse_day_rows(data: dict, scoring_period: int) -> list[dict]:
    """
    One row per player in a SCORING lineup slot (not bench / IR) for one day,
    with his actual fantasy points and whether he played (minutes > 0 or a
    game played recorded).
    """
    rows = []
    for team in data.get("teams") or []:
        for entry in (team.get("roster") or {}).get("entries") or []:
            if entry.get("lineupSlotId") in NON_SCORING_SLOTS:
                continue
            player = (entry.get("playerPoolEntry") or {}).get("player") or {}
            played, applied = False, 0.0
            for st in player.get("stats") or []:
                if st.get("scoringPeriodId") != scoring_period or st.get("statSourceId") != 0:
                    continue
                stats = st.get("stats") or {}
                mins = float(stats.get(STAT_MINUTES, 0) or 0)
                gp = float(stats.get(STAT_GP, 0) or 0)
                applied = float(st.get("appliedTotal") or 0.0)
                played = mins > 0 or gp > 0 or (STAT_MINUTES not in stats and STAT_GP not in stats and applied != 0)
                break
            rows.append({
                "scoring_period": scoring_period,
                "fantasy_team_id": team.get("id"),
                "player_id": player.get("id", entry.get("playerId")),
                "player_name": player.get("fullName", ""),
                "lineup_slot_id": entry.get("lineupSlotId"),
                "played": bool(played),
                "points": applied if played else 0.0,
            })
    return rows


def parse_day_starts(data: dict, scoring_period: int) -> dict:
    """{teamId: {"starts": n, "points": x}} for one day (lineup players who played)."""
    out = {}
    for r in parse_day_rows(data, scoring_period):
        t = out.setdefault(r["fantasy_team_id"], {"starts": 0, "points": 0.0})
        if r["played"]:
            t["starts"] += 1
            t["points"] += r["points"]
    return out


def season_day_map(data: dict) -> dict[int, int]:
    """ESPN scoring period (day) -> matchup period, from every matchup's daily points."""
    out = {}
    for m in data.get("schedule") or []:
        mp = int(m.get("matchupPeriodId", -1))
        for side in (m.get("home") or {}, m.get("away") or {}):
            for sp in _points_by_period(side):
                out[sp] = mp
    return out


def manager_names(data: dict) -> dict:
    """teamId -> primary owner's first name (empty if ESPN does not return members)."""
    members = {m.get("id"): m for m in data.get("members") or []}
    out = {}
    for t in data.get("teams") or []:
        owner = t.get("primaryOwner") or ((t.get("owners") or [None])[0])
        m = members.get(owner) or {}
        first = (m.get("firstName") or "").strip()
        out[t.get("id")] = first.title() if first.isupper() or first.islower() else first
    return out


def load_season_history(data: dict, current_sp: int, roster_source) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily lineup rows for every finished day this season, using league_cache/.
    roster_source(sp) returns ESPN's mRoster JSON for that day (or raises).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    day_mp = season_day_map(data)
    finished = sorted(sp for sp in day_mp if sp < current_sp)
    frames, fetched, failed = [], 0, []
    for sp in finished:
        path = os.path.join(CACHE_DIR, f"day_{sp}.csv")
        stale = sp >= current_sp - REFRESH_RECENT_DAYS
        if os.path.exists(path) and not stale:
            frames.append(pd.read_csv(path, dtype={"player_id": str}))
            continue
        try:
            day = pd.DataFrame(parse_day_rows(roster_source(sp), sp))
            day.to_csv(path, index=False)
            frames.append(day)
            fetched += 1
        except Exception as exc:
            if os.path.exists(path):
                frames.append(pd.read_csv(path, dtype={"player_id": str}))
            failed.append(f"{sp} ({exc})")
    print(f"\nSeason lineups: {len(finished)} finished days, {fetched} downloaded this run, "
          f"{len(finished) - fetched - len(failed)} from cache" + (f", FAILED: {', '.join(failed)}" if failed else ""))
    starts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["scoring_period", "fantasy_team_id", "player_id", "player_name", "lineup_slot_id", "played", "points"])
    if len(starts):
        starts["matchup_period"] = starts["scoring_period"].map(day_mp)

    # ESPN's own daily team points (what actually counts), plus starts from lineups.
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    rows = []
    for m in data.get("schedule") or []:
        mp = int(m.get("matchupPeriodId", -1))
        for side in (m.get("home") or {}, m.get("away") or {}):
            if side.get("teamId") is None:
                continue
            for sp, pts in _points_by_period(side).items():
                if sp >= current_sp:
                    continue
                rows.append({"scoring_period": sp, "matchup_period": mp, "fantasy_team_id": side.get("teamId"),
                             "date": str(today_et - timedelta(days=current_sp - sp)), "espn_points": pts})
    team_days = pd.DataFrame(rows, columns=["scoring_period", "matchup_period", "fantasy_team_id", "date", "espn_points"])
    if len(team_days) and len(starts):
        agg = (starts[starts["played"].astype(bool)].groupby(["scoring_period", "fantasy_team_id"])
               .agg(starts=("player_id", "size"), lineup_points=("points", "sum")).reset_index())
        team_days = team_days.merge(agg, on=["scoring_period", "fantasy_team_id"], how="left")
        team_days[["starts", "lineup_points"]] = team_days[["starts", "lineup_points"]].fillna(0)
    else:
        team_days["starts"] = np.nan
        team_days["lineup_points"] = np.nan
    if len(starts):
        dates = {sp: str(today_et - timedelta(days=int(current_sp - sp))) for sp in starts["scoring_period"].unique()}
        starts["date"] = starts["scoring_period"].map(dates)
    return starts, team_days


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

    # ---- league acquisition limits (ESPN: -1 = unlimited) -------------------
    acq = (data.get("settings") or {}).get("acquisitionSettings") or {}

    def _limit(v):
        # ESPN uses 0 or -1 for "no limit" (leagues with a 0 limit still record adds).
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
        return None if v <= 0 else v

    matchup_limit = _limit(acq.get("matchupAcquisitionLimit"))
    limit_per_day = bool(acq.get("matchupLimitPerScoringPeriod", False))
    if ACQUISITION_LIMIT_OVERRIDE is not None:
        matchup_limit, limit_per_day = int(ACQUISITION_LIMIT_OVERRIDE), False
    season_limit = _limit(acq.get("acquisitionLimit"))
    remaining_days = DAYS_PER_MATCHUP - first_sim_day

    team_info = {}
    for t in data.get("teams") or []:
        tc = t.get("transactionCounter") or {}
        totals = {}
        for k, v in (tc.get("matchupAcquisitionTotals") or {}).items():
            try:
                totals[int(k)] = int(v or 0)
            except (TypeError, ValueError):
                pass
        used_now = totals.get(current_mp, 0)
        past = [totals.get(p, 0) for p in range(1, current_mp)]
        season_used = tc.get("acquisitions")
        if matchup_limit is None:
            left_week = None
        elif limit_per_day:
            left_week = matchup_limit * remaining_days     # per-day limit; today's adds not counted here
        else:
            left_week = max(matchup_limit - used_now, 0)
        if season_limit is not None and season_used is not None:
            left_season = max(season_limit - int(season_used), 0)
            left_week = left_season if left_week is None else min(left_week, left_season)
        rec = ((t.get("record") or {}).get("overall")) or {}
        team_info[t.get("id")] = {
            "wins": rec.get("wins"), "losses": rec.get("losses"), "ties": rec.get("ties"),
            "points_for": rec.get("pointsFor"), "points_against": rec.get("pointsAgainst"),
            "playoff_seed": t.get("playoffSeed"),
            "acq_used_this_week": used_now if tc else None,
            "acq_left_this_week": left_week,
            "acq_per_past_week": (sum(past) / len(past)) if (tc and past) else None,
            "acq_counter_found": bool(tc),
        }

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
                "matchup_acquisition_limit": matchup_limit,
                "limit_per_day": limit_per_day,
                **team_info.get(side.get("teamId"), {}),
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

    # ---- season lineup history (cached) -> starts, PPP, records ----------------
    current_sp = int(df["current_scoring_period"].iloc[0])

    def roster_source(sp):
        if args.from_json:
            if not args.roster_json_dir:
                raise RuntimeError("offline mode without --roster-json-dir")
            with open(os.path.join(args.roster_json_dir, f"espn_roster_day_{sp}.json"), encoding="utf-8") as fh:
                return json.load(fh)
        day = download_roster_day(sp)
        if args.save_raw:
            with open(os.path.join(BASE_DIR, f"espn_roster_day_{sp}.json"), "w", encoding="utf-8") as fh:
                json.dump(day, fh)
        return day

    starts, team_days = load_season_history(data, current_sp, roster_source)
    starts.to_csv(DAILY_STARTS_FILE, index=False)
    team_days.to_csv(TEAM_DAYS_FILE, index=False)

    sps = json.loads(df["scoring_period_ids"].iloc[0])
    completed_sps = sps[:int(df["first_sim_day_index"].iloc[0])]
    week = team_days[team_days["scoring_period"].isin(completed_sps)]
    have_all = len(completed_sps) == 0 or (
        len(starts) and set(completed_sps) <= set(starts["scoring_period"].unique()))
    if have_all and len(week):
        g = week.groupby("fantasy_team_id")
        st_map, lp_map = g["starts"].sum().to_dict(), g["lineup_points"].sum().to_dict()
        mism = week.assign(bad=(week["espn_points"] - week["lineup_points"]).abs() > 0.5).groupby("fantasy_team_id")["bad"].sum().to_dict()
        df["completed_starts"] = df["fantasy_team_id"].map(lambda t: int(st_map.get(t, 0)))
        df["lineup_points_check"] = df["fantasy_team_id"].map(lambda t: round(float(lp_map.get(t, 0.0)), 2))
        df["lineup_mismatch_days"] = df["fantasy_team_id"].map(lambda t: int(mism.get(t, 0)))
    elif have_all:
        df["completed_starts"] = 0
        df["lineup_points_check"] = 0.0
        df["lineup_mismatch_days"] = 0
    else:
        print("WARNING: missing lineups for this week's finished days; PPP to date unavailable.")
        df["completed_starts"] = np.nan
        df["lineup_points_check"] = np.nan
        df["lineup_mismatch_days"] = np.nan
    names = manager_names(data)
    df["manager"] = df["fantasy_team_id"].map(lambda t: names.get(t, ""))

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
    if "wins" in df.columns and df["wins"].notna().any():
        print()
        print("Standings:")
        st = df.drop_duplicates("fantasy_team_id").copy()
        st["pct"] = (st["wins"].fillna(0) + 0.5 * st["ties"].fillna(0)) / (
            st["wins"].fillna(0) + st["losses"].fillna(0) + st["ties"].fillna(0)).replace(0, 1)
        for _, r in st.sort_values(["pct", "points_for"], ascending=False).iterrows():
            print(f"  {r['fantasy_team']:<28} {int(r['wins'] or 0)}-{int(r['losses'] or 0)}-{int(r['ties'] or 0)}"
                  f"   PF {float(r['points_for'] or 0):,.1f}")
    lim = df["matchup_acquisition_limit"].iloc[0]
    print()
    if not df["acq_counter_found"].any():
        print("WARNING: ESPN returned no transaction counters; pickups will use the simulator's default limits.")
    else:
        print("Acquisition limit per", "day:" if df["limit_per_day"].iloc[0] else "week:",
              "unlimited" if pd.isna(lim) else int(lim))
        for _, r in df.drop_duplicates("fantasy_team_id").sort_values("fantasy_team").iterrows():
            left = r["acq_left_this_week"]
            print(f"  {r['fantasy_team']:<28} used {int(r['acq_used_this_week'] or 0)} this week, "
                  f"left {'unlimited' if pd.isna(left) else int(left)}")
    if (df["points_source"] != "pointsByScoringPeriod").any():
        print("\nWARNING: some teams used totalPoints fallback; run before today's games start for exact totals.")
    print("\nSaved:", os.path.basename(args.out))


if __name__ == "__main__":
    main()
