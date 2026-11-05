"""
pbp_fantasy.py

Fantasy points per player per quarter from NBA play-by-play (cdn.nba.com "liveData"
format, the same format as the shufinskiy/nba_data "cdnnba" files).

League scoring: FP = PTS + REB + 2*AST + 4*STL + 4*BLK - 2*TO + 2*FGM - FGA + 3PM
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "tpm", "ftm"]


def fantasy_points(d: pd.DataFrame) -> pd.Series:
    return d["pts"] + d["reb"] + 2 * d["ast"] + 4 * d["stl"] + 4 * d["blk"] - 2 * d["tov"] \
        + 2 * d["fgm"] - d["fga"] + d["tpm"]


def _num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")


def period_stats(actions: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, period, player) with box-score counts and fantasy points."""
    a = actions
    game = a["gameId"].astype(str)
    period = _num(a["period"])
    typ = a["actionType"].astype(str).str.lower()
    made = a["shotResult"].astype(str).str.lower().eq("made")
    pid = _num(a["personId"])
    fg = typ.isin(["2pt", "3pt"]) & (pid > 0)
    three = typ.eq("3pt")
    ft = typ.eq("freethrow") & (pid > 0)

    parts = []

    def add(mask, who, **vals):
        m = mask & (who > 0)
        if not m.any():
            return
        df = pd.DataFrame({"game_id": game[m].values, "period": period[m].values, "nba_id": who[m].values})
        for k in STATS:
            v = vals.get(k, 0)
            df[k] = (v[m].values if isinstance(v, pd.Series) else v)
        parts.append(df)

    one = pd.Series(1, index=a.index)
    zero = pd.Series(0, index=a.index)
    add(fg, pid, fga=one, fgm=made.astype(int), tpm=(made & three).astype(int),
        pts=np.where(made, np.where(three, 3, 2), 0) * one)
    add(ft, pid, ftm=made.astype(int), pts=made.astype(int))
    add(typ.eq("rebound"), pid, reb=one)
    add(typ.eq("turnover"), pid, tov=one)
    if "assistPersonId" in a:
        add(fg & made, _num(a["assistPersonId"]), ast=one)
    if "blockPersonId" in a:
        add(fg & ~made, _num(a["blockPersonId"]), blk=one)
    if "stealPersonId" in a:
        add(typ.eq("turnover"), _num(a["stealPersonId"]), stl=one)
    out = pd.concat(parts, ignore_index=True)
    out = out.groupby(["game_id", "period", "nba_id"], as_index=False)[STATS].sum()
    out["fp"] = fantasy_points(out)
    return out


def game_info(actions: pd.DataFrame) -> pd.DataFrame:
    """Per (game, player): team, opponent, short name, US Eastern date."""
    a = actions[_num(actions["personId"]) > 0]
    t = pd.to_datetime(actions["timeActual"], errors="coerce", utc=True, format="mixed")
    start = t.groupby(actions["gameId"].astype(str)).min().dt.tz_convert("America/New_York").dt.date.astype(str)
    teams = actions.dropna(subset=["teamTricode"]).groupby(actions["gameId"].astype(str))["teamTricode"]\
        .agg(lambda s: sorted(set(s)))
    pl = a.groupby([a["gameId"].astype(str), _num(a["personId"])]).agg(
        team=("teamTricode", lambda s: s.mode().iat[0] if s.notna().any() else ""),
        short_name=("playerNameI", lambda s: s.dropna().mode().iat[0] if s.notna().any() else ""),
    ).reset_index()
    pl.columns = ["game_id", "nba_id", "team", "short_name"]
    pl["date"] = pl["game_id"].map(start)
    pl["opponent"] = [next((x for x in teams.get(g, []) if x != tm), "") for g, tm in zip(pl["game_id"], pl["team"])]
    return pl
