"""
nba_history.py

NBA fantasy records since 2023-24: best single games, halves and quarters under
this league's scoring, regular season and playoffs.

    2023-24 to 2025-26   nba_fantasy_history_2023_2026.csv.gz (built once from NBA
                         play-by-play; totals match official box scores in 99.6% of
                         player-games, the rest differ by a late stat correction)
    2026-27 (this season) downloaded from the NBA's own play-by-play feed each run,
                         one file per finished game kept in league_cache/pbp/

FP = PTS + REB + 2*AST + 4*STL + 4*BLK - 2*TO + 2*FGM - FGA + 3PM
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

from pbp_fantasy import STATS, game_info, period_stats

BASE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE, "nba_fantasy_history_2023_2026.csv.gz")
CACHE_DIR = os.path.join(BASE, "league_cache", "pbp")
OUT_FILE = os.path.join(BASE, "nba_records.json")
CURRENT_SEASON = "2026-27"
SEASON_CODE = "26"            # NBA game ids: 002YY..... regular season, 004YY..... playoffs
TOP_N = 25

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{}.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"}
TEAM_FIXES = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def _get(url, attempts=3):
    for a in range(1, attempts + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (403, 404):
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * a)
    return None


def player_games_from_actions(actions: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game: totals plus fantasy points by quarter."""
    ps = period_stats(actions)
    gi = game_info(actions)
    ps["q"] = np.where(ps["period"] <= 4, "q" + ps["period"].astype(str), "ot")
    wide = ps.pivot_table(index=["game_id", "nba_id"], columns="q", values="fp",
                          aggfunc="sum", fill_value=0).reset_index()
    for c in ("q1", "q2", "q3", "q4", "ot"):
        if c not in wide:
            wide[c] = 0
    tot = ps.groupby(["game_id", "nba_id"], as_index=False)[STATS + ["fp"]].sum()
    return tot.merge(wide, on=["game_id", "nba_id"]).merge(gi, on=["game_id", "nba_id"])


def current_season_games() -> pd.DataFrame:
    """Finished games this season, one cached file per game."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    sched = _get(SCHEDULE_URL)
    if sched is None:
        print("WARNING: NBA schedule feed not reachable; this season's records use saved games only.")
        finished = []
    else:
        finished = []
        for day in (sched.get("leagueSchedule") or {}).get("gameDates") or []:
            for g in day.get("games") or []:
                gid = str(g.get("gameId") or "")
                if g.get("gameStatus") == 3 and gid[:3] in ("002", "004") and gid[3:5] == SEASON_CODE:
                    finished.append(gid)
    frames, fetched, failed = [], 0, 0
    have = {f[:-4] for f in os.listdir(CACHE_DIR) if f.endswith(".csv")}
    for gid in sorted(set(finished) | have):
        path = os.path.join(CACHE_DIR, f"{gid}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path, dtype={"game_id": str}))
            continue
        data = _get(PBP_URL.format(gid))
        acts = ((data or {}).get("game") or {}).get("actions")
        if not acts:
            failed += 1
            continue
        a = pd.DataFrame(acts)
        a["gameId"] = gid
        for c in ("assistPersonId", "blockPersonId", "stealPersonId", "teamTricode", "playerNameI", "shotResult"):
            if c not in a:
                a[c] = np.nan
        g = player_games_from_actions(a)
        g["playoffs"] = gid.startswith("004")
        g.to_csv(path, index=False)
        frames.append(g)
        fetched += 1
        time.sleep(0.1)
    print(f"This season: {len(frames)} finished games ({fetched} downloaded this run"
          + (f", {failed} not available yet" if failed else "") + ")")
    if not frames:
        return pd.DataFrame()
    cur = pd.concat(frames, ignore_index=True)
    cur["season"] = CURRENT_SEASON
    return cur


def attach_names(cur: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Full names: earlier seasons by NBA id, else ESPN box scores (same date/team/stat line)."""
    names = hist.dropna(subset=["player"]).groupby("nba_id")["player"].agg(lambda s: s.iloc[-1]).to_dict()
    cur["player"] = cur["nba_id"].map(names)
    espn_path = os.path.join(BASE, "player_game_history_2027_current.csv")
    if cur["player"].isna().any() and os.path.exists(espn_path):
        e = pd.read_csv(espn_path)
        need = {"date", "player_name", "team", "points", "rebounds", "assists"}
        if need <= set(e.columns):
            e = e.rename(columns={"points": "pts", "rebounds": "reb", "assists": "ast"})
            e["team"] = e["team"].astype(str).str.upper().replace(TEAM_FIXES)
            e["date"] = e["date"].astype(str).str[:10]
            key = ["date", "team", "pts", "reb", "ast"]
            e = e[~e.duplicated(key, keep=False)]
            m = cur[~cur.duplicated(key, keep=False)].merge(e[key + ["player_name"]], on=key)
            more = m.groupby("nba_id")["player_name"].agg(lambda s: s.mode().iat[0])
            miss = cur["player"].isna()
            cur.loc[miss, "player"] = cur.loc[miss, "nba_id"].map(more)
    cur["player"] = cur["player"].fillna(cur.get("short_name"))
    return cur


def records(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["h1"] = df["q1"] + df["q2"]
    df["h2"] = df["q3"] + df["q4"]

    def row(r, value, part=None):
        return {"player": r.player, "team": r.team, "opp": r.opponent, "date": r.date, "season": r.season,
                "playoffs": bool(r.playoffs), "fp": int(value), "part": part,
                "line": f"{int(r.pts)} pts, {int(r.reb)} reb, {int(r.ast)} ast"
                        + (f", {int(r.stl)} stl" if r.stl else "") + (f", {int(r.blk)} blk" if r.blk else "")}

    def top(frame, col, part_label=None, n=TOP_N, ascending=False):
        f = frame.sort_values([col, "date"], ascending=[ascending, True]).head(n)
        return [row(r, getattr(r, col), part_label(r) if part_label else None) for r in f.itertuples()]

    def scope(frame):
        halves = pd.concat([frame.assign(v=frame["h1"], part="1st half"),
                            frame.assign(v=frame["h2"], part="2nd half")])
        quarters = pd.concat([frame.assign(v=frame[q], part=f"Q{q[1]}") for q in ("q1", "q2", "q3", "q4")])
        return {
            "game": top(frame, "fp"),
            "half": top(halves, "v", lambda r: r.part),
            "quarter": top(quarters, "v", lambda r: r.part),
            "worst": top(frame, "fp", ascending=True, n=10),
            "games": int(frame["game_id"].nunique()),
        }

    out = {"All seasons": scope(df)}
    for s in sorted(df["season"].unique(), reverse=True):
        out[s] = scope(df[df["season"] == s])
    return out


def main():
    hist = pd.read_csv(HISTORY_FILE, dtype={"game_id": str})
    cur = current_season_games()
    if len(cur):
        cur = attach_names(cur, hist)
        df = pd.concat([hist, cur[hist.columns.intersection(cur.columns)]], ignore_index=True)
    else:
        df = hist
    df["playoffs"] = df["playoffs"].astype(str).str.lower().isin(["true", "1"])
    rec = records(df)
    payload = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "scopes": rec,
               "seasons": [k for k in rec if k != "All seasons"]}
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    best = rec["All seasons"]["game"][0]
    print(f"NBA records: {rec['All seasons']['games']:,} games since 2023-24. Best game: "
          f"{best['player']} {best['fp']} FP ({best['date']}).")


if __name__ == "__main__":
    main()
