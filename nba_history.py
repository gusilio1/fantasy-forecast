"""
nba_history.py

NBA fantasy records since 2023-24: best single games, halves and quarters under
this league's scoring, regular season and playoffs.

    2023-24 to 2025-26   nba_fantasy_history_2023_2026.csv.gz (built once from NBA
                         play-by-play; totals match official box scores in 99.6% of
                         player-games, the rest differ by a late stat correction)
    2026-27 (this season) from ESPN's play-by-play for each finished game (validated
                         against ESPN box scores: 98% of player-games exact, quarters
                         agree with the NBA feed 99%+), cached in league_cache/pbp_espn/

FP = PTS + REB + 2*AST + 4*STL + 4*BLK - 2*TO + 2*FGM - FGA + 3PM
2023-24 games use that season's league scoring, which also had +2 per free throw made
and -1 per free throw attempted (column "scoring" = "old"; "fp_now" = today's scoring).
Halves are quarters 1+2 and 3+4 (overtime counts toward the game only).

Output: nba_records.json for the dashboard.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import requests

from espn_pbp_fantasy import STATS as ESPN_STATS, period_stats_espn, players_in_summary

BASE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE, "nba_fantasy_history_2023_2026.csv.gz")
CACHE_DIR = os.path.join(BASE, "league_cache", "pbp_espn")
OUT_FILE = os.path.join(BASE, "nba_records.json")
CURRENT_SEASON = "2026-27"
TOP_N = 25

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
HEADERS = {"User-Agent": "Mozilla/5.0"}
TEAM_FIXES = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def _get(url, params=None, attempts=3):
    for a in range(1, attempts + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (403, 404):
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * a)
    return None


def _espn_game_rows(summary: dict, gid: str) -> pd.DataFrame:
    """Player-game rows (totals + quarters) from one ESPN game summary."""
    ps = period_stats_espn(summary)
    if ps.empty:
        return pd.DataFrame()
    who = players_in_summary(summary)
    comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
    start = pd.to_datetime(comp.get("date"), errors="coerce", utc=True)
    day = str(start.tz_convert("America/New_York").date()) if pd.notna(start) else ""
    season_type = int((((summary.get("header") or {}).get("season") or {}).get("type")) or 2)
    teams = [TEAM_FIXES.get(t, t) for t in {v[1] for v in who.values()} if t]
    ps["q"] = np.where(ps["period"] <= 4, "q" + ps["period"].astype(str), "ot")
    wide = ps.pivot_table(index="espn_id", columns="q", values="fp", aggfunc="sum", fill_value=0).reset_index()
    for c in ("q1", "q2", "q3", "q4", "ot"):
        if c not in wide:
            wide[c] = 0
    tot = ps.groupby("espn_id", as_index=False)[ESPN_STATS + ["fp"]].sum()
    g = tot.merge(wide, on="espn_id")
    g["player"] = g["espn_id"].map(lambda i: who.get(i, ("", ""))[0])
    g["team"] = g["espn_id"].map(lambda i: TEAM_FIXES.get(who.get(i, ("", ""))[1], who.get(i, ("", ""))[1]))
    g["opponent"] = g["team"].map(lambda t: next((x for x in teams if x != t), ""))
    g["game_id"] = f"espn-{gid}"
    g["nba_id"] = "espn-" + g["espn_id"].astype(str)
    g["date"] = day
    g["playoffs"] = season_type == 3
    return g.drop(columns=["espn_id"])


def current_season_games() -> pd.DataFrame:
    """
    This season's finished games from ESPN's play-by-play (the same game summaries the
    box-score step downloads), one cached file per game in league_cache/pbp_espn/.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    box_path = os.path.join(BASE, "player_game_history_2027_current.csv")
    if not os.path.exists(box_path):
        print("No current-season box scores yet; this season's records skipped.")
        return pd.DataFrame()
    games = pd.read_csv(box_path, usecols=["game_id"], dtype={"game_id": str})["game_id"].dropna().unique()
    frames, fetched, failed = [], 0, 0
    for gid in sorted(games):
        path = os.path.join(CACHE_DIR, f"{gid}.csv")
        if os.path.exists(path):
            try:
                frames.append(pd.read_csv(path, dtype={"game_id": str, "nba_id": str}))
                continue
            except (pd.errors.EmptyDataError, ValueError):
                pass
        summary = _get(SUMMARY_URL, params={"event": gid})
        if not summary or not summary.get("plays"):
            failed += 1
            continue
        g = _espn_game_rows(summary, gid)
        if g.empty:
            failed += 1
            continue
        g.to_csv(path, index=False)
        frames.append(g)
        fetched += 1
        time.sleep(0.05)
    print(f"This season: {len(frames)} finished games ({fetched} downloaded this run"
          + (f", {failed} without play-by-play yet" if failed else "") + ")")
    if not frames:
        return pd.DataFrame()
    cur = pd.concat(frames, ignore_index=True)
    cur["season"] = CURRENT_SEASON
    return cur


def records(df: pd.DataFrame) -> dict:
    """
    Top lists under two scorings:
        played  each season's own league scoring (2023-24 had the old free-throw rule)
        today   today's scoring for every season
    Each entry also carries its value under the other scoring ("alt").
    """
    df = df.copy()
    for c in ("q1", "q2", "q3", "q4", "ot"):
        if f"{c}_now" not in df:
            df[f"{c}_now"] = df[c]
        df[f"{c}_now"] = df[f"{c}_now"].fillna(df[c])
    if "fp_now" not in df:
        df["fp_now"] = df["fp"]
    df["fp_now"] = df["fp_now"].fillna(df["fp"])
    df["scoring"] = df.get("scoring", "current")
    df["scoring"] = df["scoring"].fillna("current")

    def row(r, value, alt, part=None):
        return {"player": r.player, "team": r.team, "opp": r.opponent, "date": r.date, "season": r.season,
                "playoffs": bool(r.playoffs), "fp": int(value), "alt": int(alt), "part": part,
                "old": r.scoring == "old",
                "line": f"{int(r.pts)} pts, {int(r.reb)} reb, {int(r.ast)} ast"
                        + (f", {int(r.stl)} stl" if r.stl else "") + (f", {int(r.blk)} blk" if r.blk else "")
                        + (f", {int(r.ftm)}/{int(r.fta)} FT" if r.scoring == "old" and pd.notna(r.fta) else "")}

    def top(frame, col, alt_col, n=TOP_N, ascending=False):
        f = frame.sort_values([col, "date"], ascending=[ascending, True]).head(n)
        return [row(r, getattr(r, col), getattr(r, alt_col), getattr(r, "part", None)) for r in f.itertuples()]

    def scope(frame, now: bool):
        sfx, other = ("_now", "") if now else ("", "_now")
        g = frame.assign(v=frame["fp" + sfx], a=frame["fp" + other])
        halves = pd.concat([
            frame.assign(v=frame["q1" + sfx] + frame["q2" + sfx], a=frame["q1" + other] + frame["q2" + other], part="1st half"),
            frame.assign(v=frame["q3" + sfx] + frame["q4" + sfx], a=frame["q3" + other] + frame["q4" + other], part="2nd half")])
        quarters = pd.concat([frame.assign(v=frame[q + sfx], a=frame[q + other], part=f"Q{q[1]}")
                              for q in ("q1", "q2", "q3", "q4")])
        return {"game": top(g, "v", "a"), "half": top(halves, "v", "a"), "quarter": top(quarters, "v", "a"),
                "worst": top(g, "v", "a", n=10, ascending=True), "games": int(frame["game_id"].nunique())}

    out = {}
    for mode, now in (("played", False), ("today", True)):
        m = {"All seasons": scope(df, now)}
        for sn in sorted(df["season"].unique(), reverse=True):
            m[sn] = scope(df[df["season"] == sn], now)
        out[mode] = m
    return out


def main():
    hist = pd.read_csv(HISTORY_FILE, dtype={"game_id": str})
    cur = current_season_games()
    if len(cur):
        cur["scoring"] = "current"
        cur["fp_now"] = cur["fp"]
        df = pd.concat([hist, cur[hist.columns.intersection(cur.columns)]], ignore_index=True)
    else:
        df = hist
    df["playoffs"] = df["playoffs"].astype(str).str.lower().isin(["true", "1"])
    rec = records(df)
    payload = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "modes": rec,
               "seasons": [k for k in rec["played"] if k != "All seasons"]}
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    best, now = rec["played"]["All seasons"]["game"][0], rec["today"]["All seasons"]["game"][0]
    print(f"NBA records: {rec['played']['All seasons']['games']:,} games since 2023-24. Best game: "
          f"{best['player']} {best['fp']} FP ({best['date']}); under today's scoring: "
          f"{now['player']} {now['fp']} FP ({now['date']}).")


if __name__ == "__main__":
    main()
