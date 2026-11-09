"""
build_dashboard.py

Turns the simulator's output files into one self-contained web page:
    site/index.html

Reads (same folder; optional files are skipped if missing):
    weekly_matchup_v1_8_team_summary.csv
    weekly_matchup_v1_8_matchup_win_probabilities.csv
    weekly_matchup_v1_8_player_summary.csv
    weekly_matchup_v1_8_run_meta.json
    weekly_matchup_v1_8_player_day_projections.csv
    weekly_matchup_v1_8_team_day_projections.csv
    weekly_matchup_v1_8_free_agents.csv
    weekly_matchup_v1_8_pickup_plan.csv
    weekly_matchup_v1_8_pickup_advice.csv
    fantasy_matchups.csv                                  (standings, managers)
    league_team_days.csv, league_daily_starts.csv         (PPP table, records)

Run after weekly_matchup_simulator_v3_2.py. Open site/index.html to preview.
"""

from __future__ import annotations

import json
import math

import numpy as np
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TAG = "v1_8"
SITE_DIR = os.path.join(BASE, "site")
LEAGUE_TITLE = "League forecast"
TOP_PLAYERS = 30
TOP_FREE_AGENTS = 40
MIN_STARTS_PPP_WEEK = 15     # a week needs this many starts to count for PPP records

# League history (History tab). Spreadsheet name -> the first name ESPN shows now, where
# they differ, so a manager's seasons join up. Names not listed are matched as they are.
NAME_ALIASES = {"Hammer": "Hamish"}
# This season's ESPN team name -> the name that manager has in the spreadsheets, for any
# manager whose ESPN first name differs from it. Example: {"Gustavo Giants": "Gus"}.
TEAM_HISTORY_NAMES = {
    "Gustavo Giants": "Gus",
    "PBNOJ": "Trent",
    "Daniel's Dangerous Team": "Burzomi",
    "YOUNG ANTS": "Hamish",
    "trembanyama": "Jay",
    "spida man": "Levi",
    "The new look Framptons": "Kaden",
    "Bez Mbengers and Mash": "Tom",
    "Souv's malakas": "Taeven",
    "Da Champ": "Will",
    "The Big KAT-astrophy": "Jackson",
    "Scoot N Shoot": "Ryder",
}
CURRENT_SEASON_LABEL = "2026-27"
LEAGUE_HISTORY_FILE = "league_history_2024_2026.csv"   # 2023-24 left out: 10-team league
NBA_RECORDS_FILE = "nba_records.json"
DAY_FULL = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
            "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}


def _p(name):
    return os.path.join(BASE, f"weekly_matchup_{TAG}_{name}")


def _read(path):
    try:
        return pd.read_csv(path, dtype={"player_id": str}) if os.path.exists(path) else pd.DataFrame()
    except pd.errors.EmptyDataError:   # e.g. no days left to simulate once the week is final
        return pd.DataFrame()


def _num(v, nd=1):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, nd)


def _str(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip()


def league_stats(standings_src: pd.DataFrame) -> dict:
    """PPP table (this week, last 3 weeks, season) and league records from finished days."""
    td = _read(os.path.join(BASE, "league_team_days.csv"))
    ds = _read(os.path.join(BASE, "league_daily_starts.csv"))
    if td.empty or standings_src.empty:
        return {"ppp": [], "records": [], "topGames": [], "weeksDone": 0}
    info = standings_src.drop_duplicates("fantasy_team_id").set_index("fantasy_team_id")
    name = {k: _str(v) for k, v in info["fantasy_team"].items()}
    mgr = {k: _str(v) for k, v in info["manager"].items()} if "manager" in info.columns else {}
    cur = int(standings_src["matchup_period"].iloc[0])
    td["starts"] = pd.to_numeric(td["starts"], errors="coerce").fillna(0)

    def ppp(frame):
        st = frame["starts"].sum()
        return (float(frame["espn_points"].sum() / st), int(st)) if st > 0 else (None, int(st))

    rows = []
    for tid, g in td.groupby("fantasy_team_id"):
        tw, tw_st = ppp(g[g["matchup_period"] == cur])
        lw, lw_st = ppp(g[g["matchup_period"] == cur - 1])
        # Before this week's first game, "last 3 weeks" means the 3 finished weeks before it.
        l3_end = cur if tw_st > 0 else cur - 1
        l3, l3_st = ppp(g[g["matchup_period"].between(l3_end - 2, l3_end)])
        se, se_st = ppp(g)
        rows.append({"team": name.get(tid, str(tid)), "manager": mgr.get(tid, ""),
                     "week": _num(tw), "weekStarts": tw_st, "lastWeek": _num(lw), "lastWeekStarts": lw_st,
                     "last3": _num(l3), "last3Starts": l3_st,
                     "season": _num(se), "seasonStarts": se_st})

    who = lambda tid: {"team": name.get(tid, str(tid)), "manager": mgr.get(tid, "")}
    records = []
    played_days = td[td["starts"] > 0]
    if len(played_days):
        hi = played_days.loc[played_days["espn_points"].idxmax()]
        lo = played_days.loc[played_days["espn_points"].idxmin()]
        records.append({"label": "Highest team day", "value": _num(hi["espn_points"]), "unit": "points",
                        **who(hi["fantasy_team_id"]), "when": _str(hi["date"])})
        records.append({"label": "Lowest team day", "value": _num(lo["espn_points"]), "unit": "points",
                        **who(lo["fantasy_team_id"]), "when": _str(lo["date"]),
                        "note": "Days with at least one player in the lineup"})
    weeks = (td[td["matchup_period"] < cur].groupby(["fantasy_team_id", "matchup_period"])
             .agg(points=("espn_points", "sum"), starts=("starts", "sum")).reset_index())
    if len(weeks):
        hi, lo = weeks.loc[weeks["points"].idxmax()], weeks.loc[weeks["points"].idxmin()]
        records.append({"label": "Highest team week", "value": _num(hi["points"]), "unit": "points",
                        **who(hi["fantasy_team_id"]), "when": f"Week {int(hi['matchup_period'])}"})
        records.append({"label": "Lowest team week", "value": _num(lo["points"]), "unit": "points",
                        **who(lo["fantasy_team_id"]), "when": f"Week {int(lo['matchup_period'])}"})
        wp = weeks[weeks["starts"] >= MIN_STARTS_PPP_WEEK].copy()
        if len(wp):
            wp["ppp"] = wp["points"] / wp["starts"]
            hi, lo = wp.loc[wp["ppp"].idxmax()], wp.loc[wp["ppp"].idxmin()]
            records.append({"label": "Best PPP week", "value": _num(hi["ppp"]), "unit": "PPP",
                            **who(hi["fantasy_team_id"]), "when": f"Week {int(hi['matchup_period'])}, {int(hi['starts'])} starts"})
            records.append({"label": "Worst PPP week", "value": _num(lo["ppp"]), "unit": "PPP",
                            **who(lo["fantasy_team_id"]), "when": f"Week {int(lo['matchup_period'])}, {int(lo['starts'])} starts"})
    top_games = []
    if len(ds):
        ds["played"] = ds["played"].astype(str).str.upper().isin({"TRUE", "1"})
        g = ds[ds["played"]].sort_values("points", ascending=False).head(10)
        for _, r in g.iterrows():
            top_games.append({"player": _str(r["player_name"]), "points": _num(r["points"]),
                              **who(r["fantasy_team_id"]), "when": _str(r.get("date"))})
        if top_games:
            t = top_games[0]
            records.insert(0, {"label": "Highest single game", "value": t["points"], "unit": "points",
                               "team": t["team"], "manager": t["manager"], "player": t["player"], "when": t["when"]})
    return {"ppp": rows, "records": records, "topGames": top_games,
            "weeksDone": int(weeks["matchup_period"].nunique()) if len(weeks) else 0, "currentWeek": cur}


def history_payload(fm: pd.DataFrame) -> dict:
    """NBA fantasy records since 2023-24 and the league's results since 2023-24 (spreadsheets + ESPN)."""
    out = {"nba": None, "league": None}
    try:
        with open(os.path.join(BASE, NBA_RECORDS_FILE), encoding="utf-8") as fh:
            out["nba"] = json.load(fh)
    except (OSError, ValueError):
        pass

    hist = _read(os.path.join(BASE, LEAGUE_HISTORY_FILE))
    alias = lambda n: NAME_ALIASES.get(_str(n), _str(n))
    rows = []
    for r in hist.to_dict("records"):
        rows.append({"season": r["season"], "week": int(r["week"]), "manager": alias(r["manager"]),
                     "opponent": alias(r["opponent"]), "points": float(r["points"]), "opp": float(r["opp_points"]),
                     "starts": float(r["starts"]) if pd.notna(r["starts"]) else None,
                     "playoffs": str(r["playoffs"]).lower() == "true", "long": str(r["long_week"]).lower() == "true"})

    # This season, from ESPN: every finished week.
    td = _read(os.path.join(BASE, "league_team_days.csv"))
    sched = _read(os.path.join(BASE, "league_schedule.csv"))
    team_now = {}
    if len(fm) and len(td) and len(sched):
        info = fm.drop_duplicates("fantasy_team_id")
        mgr = {_str(int(float(t))): (_str(m) if "manager" in info.columns and _str(m) else _str(n))
               for t, m, n in zip(info["fantasy_team_id"], info.get("manager", info["fantasy_team"]), info["fantasy_team"])}
        tname = {_str(int(float(t))): _str(n) for t, n in zip(info["fantasy_team_id"], info["fantasy_team"])}
        for t in list(mgr):
            key = {k.strip().lower(): v for k, v in TEAM_HISTORY_NAMES.items()}.get(tname.get(t, "").strip().lower())
            if key:
                mgr[t] = key
        for t, m in mgr.items():
            team_now[alias(m)] = tname.get(t, "")
        cur = int(fm["matchup_period"].iloc[0])
        done = int(fm["first_sim_day_index"].iloc[0]) >= 7
        td["tid"] = td["fantasy_team_id"].map(lambda v: _str(int(float(v))))
        td["starts"] = pd.to_numeric(td["starts"], errors="coerce").fillna(0)
        wk = td.groupby(["matchup_period", "tid"]).agg(points=("espn_points", "sum"), starts=("starts", "sum"))
        reg_weeks = int(sched.loc[sched["playoff_tier"].isna() | (sched["playoff_tier"].astype(str) == "NONE"),
                                  "matchup_period"].max() or 0)
        for m in sched.to_dict("records"):
            mp = int(m["matchup_period"])
            if not (mp < cur or (mp == cur and done)) or pd.isna(m["home_id"]) or pd.isna(m["away_id"]):
                continue
            h, a = _str(int(float(m["home_id"]))), _str(int(float(m["away_id"])))
            if (mp, h) not in wk.index or (mp, a) not in wk.index:
                continue
            for me, op in ((h, a), (a, h)):
                rows.append({"season": CURRENT_SEASON_LABEL, "week": mp, "manager": alias(mgr.get(me, me)),
                             "opponent": alias(mgr.get(op, op)), "points": float(wk.loc[(mp, me), "points"]),
                             "opp": float(wk.loc[(mp, op), "points"]), "starts": float(wk.loc[(mp, me), "starts"]),
                             "playoffs": bool(reg_weeks and mp > reg_weeks), "long": False})
    if not rows:
        return out
    w = pd.DataFrame(rows)
    w["result"] = np.where(w["points"] > w["opp"], "W", np.where(w["points"] < w["opp"], "L", "T"))
    w["ppp"] = w["points"] / w["starts"].where(w["starts"] > 0)
    w["margin"] = w["points"] - w["opp"]
    reg = w[~w["playoffs"]]
    # Week lengths vary (short NBA Cup weeks, long All-Star weeks). Score records only use
    # weeks of roughly normal length: 80-115% of the season's typical starts per team.
    wk_avg = w.groupby(["season", "week"])["starts"].transform("mean")
    typical = w.groupby(["season", "week"])["starts"].mean().groupby(level=0).median()
    ratio = wk_avg / w["season"].map(typical)
    normal = w[ratio.between(0.80, 1.15)]

    def team_rows(frame):
        res = []
        for m, g in frame.groupby("manager"):
            W, L, T = (g["result"] == "W").sum(), (g["result"] == "L").sum(), (g["result"] == "T").sum()
            st = g["starts"].sum()
            res.append({"manager": m, "team": team_now.get(m, ""), "seasons": int(g["season"].nunique()),
                        "w": int(W), "l": int(L), "t": int(T), "pct": _num((W + 0.5 * T) / max(len(g), 1), 3),
                        "pf": _num(g["points"].sum(), 0), "avg": _num(g["points"].mean(), 1),
                        "ppp": _num(g["points"].sum() / st if st else None, 2)})
        return sorted(res, key=lambda r: (-(r["pct"] or 0), -(r["pf"] or 0)))

    def rec(label, r, value, unit, note=None):
        return {"label": label, "value": _num(value, 2 if unit == "PPP" else 1), "unit": unit, "manager": r["manager"],
                "team": team_now.get(r["manager"], ""), "opponent": r["opponent"],
                "when": f"{r['season']}, week {int(r['week'])}" + (" (playoffs)" if r["playoffs"] else ""),
                "note": note}

    records = []
    if len(normal):
        hw = normal.loc[normal["points"].idxmax()]
        records.append(rec("Highest week", hw, hw["points"], "points", f"vs {hw['opponent']}, normal-length week"))
        big = w.loc[w["points"].idxmax()]
        if big.name not in normal.index:
            records.append(rec("Highest week (any length)", big, big["points"], "points",
                               f"vs {big['opponent']}, {'long' if ratio[big.name] > 1 else 'short'} week of {int(big['starts'])} starts"))
        records.append(rec("Lowest week", normal.loc[normal["points"].idxmin()], normal["points"].min(), "points"))
    pw = w[w["starts"] >= 20]
    if len(pw):
        records.append(rec("Best PPP week", pw.loc[pw["ppp"].idxmax()], pw["ppp"].max(), "PPP"))
        records.append(rec("Worst PPP week", pw.loc[pw["ppp"].idxmin()], pw["ppp"].min(), "PPP"))
    wins = normal[normal["margin"] > 0]
    if len(wins):
        b = wins.loc[wins["margin"].idxmax()]
        records.append(rec("Biggest win", b, b["margin"], "points", f"{_num(b['points'],0):,.0f} to {_num(b['opp'],0):,.0f} vs {b['opponent']}"))
        c = wins.loc[wins["margin"].idxmin()]
        records.append(rec("Closest win", c, c["margin"], "points", f"{_num(c['points'],0):,.0f} to {_num(c['opp'],0):,.0f} vs {c['opponent']}"))
    if len(normal):
        both = normal.assign(total=normal["points"] + normal["opp"])
        t = both.loc[both["total"].idxmax()]
        records.append(rec("Highest-scoring matchup", t, t["total"], "points", f"{t['manager']} {_num(t['points'],0):,.0f}, {t['opponent']} {_num(t['opp'],0):,.0f}"))

    w["length"] = np.where(ratio > 1.15, "Long week", np.where(ratio < 0.80, "Short week", ""))
    top_weeks = [{"manager": r["manager"], "team": team_now.get(r["manager"], ""), "points": _num(r["points"], 1),
                  "season": r["season"], "week": int(r["week"]), "opponent": r["opponent"], "won": r["result"] == "W",
                  "ppp": _num(r["ppp"], 2), "playoffs": bool(r["playoffs"]), "length": r["length"],
                  "starts": _num(r["starts"], 0)}
                 for r in w.sort_values("points", ascending=False).head(15).to_dict("records")]

    seasons = {}
    for sn, g in reg.groupby("season"):
        rows_s = team_rows(g)
        for i, r in enumerate(rows_s, 1):
            r["rank"] = i
        seasons[sn] = {"rows": rows_s, "weeks": int(g["week"].nunique())}

    h2h = {}
    for (m, o), g in w.groupby(["manager", "opponent"]):
        h2h.setdefault(m, {})[o] = [int((g["result"] == "W").sum()), int((g["result"] == "L").sum()),
                                    int((g["result"] == "T").sum())]
    out["league"] = {
        "allTime": team_rows(reg), "records": records, "topWeeks": top_weeks,
        "seasons": dict(sorted(seasons.items(), reverse=True)), "h2h": h2h,
        "seasonList": sorted(w["season"].unique(), reverse=True),
        "current": sorted(team_now.keys()),
    }
    return out


def season_payload(managers: dict) -> dict:
    sf = _read(os.path.join(BASE, "season_forecast.csv"))
    meta = {}
    try:
        with open(os.path.join(BASE, "season_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        pass
    if not len(sf):
        return {"meta": meta, "teams": [], "weeks": {}}
    n = int(meta.get("teams") or len(sf))
    teams = []
    for _, r in sf.iterrows():
        name = _str(r["fantasy_team"])
        teams.append({
            "name": name, "manager": managers.get(name, ""),
            "w": int(r["wins"]), "l": int(r["losses"]), "t": int(r["ties"]), "pf": _num(r["points_for"]),
            "pw": _num(r["proj_wins"], 1), "pl": _num(r["proj_losses"], 1), "pt": _num(r["proj_ties"], 1),
            "wlo": _num(r["wins_p10"], 0), "whi": _num(r["wins_p90"], 0), "ppf": _num(r["proj_points_for"], 0),
            "playoff": _num(r["playoff_prob"], 4), "bye": _num(r.get("bye_prob"), 4),
            "final": _num(r["final_prob"], 4), "title": _num(r["title_prob"], 4),
            "first": _num(r["first_place_prob"], 4), "last": _num(r["last_place_prob"], 4),
            "finish": _num(r["mean_finish"], 2),
            "seeds": [_num(r.get(f"seed_{k}"), 4) for k in range(1, n + 1)],
            "perWeek": _num(r.get("future_mean_per_week"), 0),
        })
    weeks = {}
    sw = _read(os.path.join(BASE, "season_weeks.csv"))
    for _, r in sw.iterrows():
        weeks.setdefault(_str(r["fantasy_team"]), []).append({
            "mp": int(r["matchup_period"]), "start": _str(r["start"]), "end": _str(r["end"]),
            "opp": _str(r["opponent"]), "mean": _num(r["mean"], 0), "oppMean": _num(r["opponent_mean"], 0),
            "win": _num(r["win_prob"], 4)})
    return {"meta": meta, "teams": teams, "weeks": weeks}


def check_payload() -> dict:
    try:
        with open(os.path.join(BASE, "model_check_report.json"), encoding="utf-8") as fh:
            rep = json.load(fh)
    except (OSError, ValueError):
        return {}
    snaps = _read(os.path.join(BASE, "history", "team_snapshots.csv"))
    ps = _read(os.path.join(BASE, "history", "player_snapshots.csv"))
    rep["saved"] = {"team_forecasts": int(len(snaps)), "player_forecasts": int(len(ps)),
                    "weeks": int(snaps["matchup_period"].nunique()) if len(snaps) else 0}
    return rep


def load_payload() -> dict:
    teams = _read(_p("team_summary.csv"))
    players = _read(_p("player_summary.csv"))
    wp = _read(_p("matchup_win_probabilities.csv"))
    pdays = _read(_p("player_day_projections.csv"))
    tdays = _read(_p("team_day_projections.csv"))
    fas = _read(_p("free_agents.csv"))
    plan = _read(_p("pickup_plan.csv"))
    advice = _read(_p("pickup_advice.csv"))
    standings_src = _read(os.path.join(BASE, "fantasy_matchups.csv"))
    meta = {}
    if os.path.exists(_p("run_meta.json")):
        with open(_p("run_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)

    eff = _read(_p("team_pickup_effect.csv"))
    names_by_id = {}
    if len(teams) and "fantasy_team_id" in teams.columns:
        names_by_id = {_str(int(float(t))): _str(n) for t, n in zip(teams["fantasy_team_id"], teams["fantasy_team"])}
    elif len(standings_src):
        names_by_id = {_str(int(float(t))): _str(n) for t, n in zip(standings_src["fantasy_team_id"], standings_src["fantasy_team"])}
    pickup_effect = {names_by_id.get(_str(int(float(r["fantasy_team_id"]))), ""): float(r["pickup_effect"])
                     for _, r in eff.iterrows()} if len(eff) else {}

    team_rows = [{
        "pickups": _num(pickup_effect.get(_str(r["fantasy_team"]))),
        "name": _str(r["fantasy_team"]),
        "final": _num(r["mean"], 0), "p10": _num(r["p10"], 0), "p90": _num(r["p90"], 0),
        "toDate": _num(r.get("actual_to_date", 0.0)),
        "startsToDate": _num(r.get("starts_to_date"), 0), "pppToDate": _num(r.get("ppp_to_date")),
        "startsRest": _num(r.get("starts_rest_mean"), 1), "pppRest": _num(r.get("ppp_rest_of_week")),
    } for _, r in teams.iterrows()]

    matchups = [{
        "a": _str(r["team"]), "b": _str(r["opponent"]),
        "aWin": _num(r["team_win_prob"], 4), "bWin": _num(r["opponent_win_prob"], 4),
        "tie": _num(r["tie_prob"], 4), "margin": _num(r["mean_margin"]),
        "marginLo": _num(r["margin_p10"]), "marginHi": _num(r["margin_p90"]),
    } for _, r in wp.iterrows()]

    plan_add = {_str(r["add_player_id"]): _str(r["fantasy_team"]) for _, r in plan.iterrows()} if len(plan) else {}
    plan_drop = {_str(r["drop_player_id"]): _str(r["fantasy_team"]) for _, r in plan.iterrows()} if len(plan) else {}

    rosters = {}
    for _, r in players.iterrows():
        team = _str(r["fantasy_team"])
        rosters.setdefault(team, []).append({
            "name": _str(r["player_name"]), "nba": _str(r["nba_team"]), "status": _str(r["injury_status"]),
            "slot": _str(r["lineup_slot"]), "games": int(r["games"]) if pd.notna(r["games"]) else 0,
            "proj": _num(r["mean_week_fp_active"], 0), "lo": _num(r["p10_active"], 0), "hi": _num(r["p90_active"], 0),
            "drop": plan_drop.get(_str(r["player_id"]), "") == team,
        })
    for team in rosters:
        rosters[team].sort(key=lambda x: -(x["proj"] or 0))

    next_day = meta.get("next_day")
    top_players, top_teams = [], []
    if next_day and len(pdays):
        nd = pdays[pdays["day"] == next_day].sort_values("expected_fp", ascending=False).head(TOP_PLAYERS * 3)
        for _, r in nd.iterrows():
            top_players.append({
                "name": _str(r["player_name"]), "nba": _str(r["nba_team"]), "opp": _str(r["opponent"]),
                "status": _str(r["injury_status"]), "pos": _str(r["eligible_positions"]),
                "owner": _str(r["fantasy_team"]) if _str(r["fantasy_status"]) == "ROSTERED" else "",
                "fa": _str(r["fantasy_status"]) == "FREE_AGENT",
                "proj": _num(r["expected_fp"]), "ifPlays": _num(r["fp_if_plays"]), "play": _num(r["play_prob"], 3),
            })
    if next_day and len(tdays):
        td = tdays[tdays["day"] == next_day].copy()
        # Current rosters only: a hypothetical pickup shouldn't count toward a day's projection.
        col = "expected_fp_roster" if "expected_fp_roster" in td.columns else "expected_fp"
        proj = {_str(r["fantasy_team"]): float(r[col]) for _, r in td.iterrows()}
        for name in [t["name"] for t in team_rows]:
            proj.setdefault(name, 0.0)          # teams with nobody playing that day
        top_teams = [{"name": k, "proj": _num(v)} for k, v in sorted(proj.items(), key=lambda kv: -kv[1])]

    free_agents = [{
        "name": _str(r["player_name"]), "nba": _str(r["nba_team"]), "pos": _str(r["eligible_positions"]),
        "status": _str(r["injury_status"]), "games": int(r["games_left"]),
        "rest": _num(r["expected_rest_of_week"]), "perGame": _num(r["fp_per_game"]),
        "next": _num(r["next_day_expected"]), "espn": _num(r.get("espn_projected_fp_avg")),
        "addBy": _str(r.get("planned_add_by")),
    } for _, r in fas.head(TOP_FREE_AGENTS).iterrows()]

    def moves(df):
        return [{"team": _str(r["fantasy_team"]), "day": _str(r["day"]), "add": _str(r["add_player"]),
                 "drop": _str(r["drop_player"]), "gain": _num(r["expected_gain"]),
                 "addRest": _num(r.get("add_expected_rest_of_week")),
                 "dropRest": _num(r.get("drop_expected_rest_of_week"))} for _, r in df.iterrows()]

    standings = []
    if len(standings_src) and "wins" in standings_src.columns:
        s = standings_src.drop_duplicates("fantasy_team_id").copy()
        for c in ("wins", "losses", "ties", "points_for", "points_against"):
            s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0)
        games = (s["wins"] + s["losses"] + s["ties"]).replace(0, 1)
        s["pct"] = (s["wins"] + 0.5 * s["ties"]) / games
        if "playoff_seed" in s.columns and pd.to_numeric(s["playoff_seed"], errors="coerce").notna().all():
            s = s.sort_values("playoff_seed")
        else:
            s = s.sort_values(["pct", "points_for"], ascending=False)
        win = {}
        for m in matchups:
            win[m["a"]], win[m["b"]] = m["aWin"], m["bWin"]
        for i, (_, r) in enumerate(s.iterrows(), 1):
            name = _str(r["fantasy_team"])
            standings.append({"rank": i, "name": name, "w": int(r["wins"]), "l": int(r["losses"]),
                              "t": int(r["ties"]), "pf": _num(r["points_for"], 0), "win": win.get(name)})

    stats = league_stats(standings_src)
    history = history_payload(standings_src)
    managers = {}
    if len(standings_src) and "manager" in standings_src.columns:
        managers = {_str(r["fantasy_team"]): _str(r["manager"]) for _, r in standings_src.iterrows()}

    last_week_ppp = {r["team"]: r.get("lastWeek") for r in (stats.get("ppp") or [])}
    for t in team_rows:
        t["pppLastWeek"] = last_week_ppp.get(t["name"])
    season, check = season_payload(managers), check_payload()
    by_name = {r["name"]: r for r in (season.get("teams") or [])}
    for st in standings:
        r = by_name.get(st["name"])
        if r:   # one record everywhere: the season forecast's (from ESPN's decided matchups)
            st.update({"playoff": r["playoff"], "w": r["w"], "l": r["l"], "t": r["t"], "pf": r["pf"]})
    if by_name:
        standings.sort(key=lambda x: (-((x["w"] + 0.5 * x["t"]) / max(x["w"] + x["l"] + x["t"], 1)), -(x["pf"] or 0)))
        for i, st in enumerate(standings, 1):
            st["rank"] = i

    return {
        "season": season, "check": check,
        "history": history,
        "title": LEAGUE_TITLE, "meta": meta, "dayFull": DAY_FULL, "stats": stats, "managers": managers,
        "teams": team_rows, "matchups": matchups, "rosters": rosters, "standings": standings,
        "topPlayers": top_players, "topTeams": top_teams, "freeAgents": free_agents,
        "plan": moves(plan), "advice": moves(advice) if len(advice) else [],
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap" rel="stylesheet">
<style>
:root{
  --band:#14213D; --band-ink:#FFFFFF; --band-muted:#B9C3D9;
  --bg:#F2F4F7; --surface:#FFFFFF; --ink:#14213D; --muted:#5A6479; --line:#DCE1E9; --row:#F7F8FA;
  --a:#2747D6; --a-soft:#E3E8FB; --b:#E8A317; --b-ink:#241800;
  --good:#11795A; --warn:#B45309; --bad:#B42318; --chip:#EEF1F6;
  --font:"Archivo",system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme: dark){
  :root{ --band:#0E1730; --band-ink:#F2F5FB; --band-muted:#9FAAC4;
         --bg:#111A2E; --surface:#18233C; --ink:#E9EDF6; --muted:#A3ADC4; --line:#2A3654; --row:#1C2843;
         --a:#7D90F2; --a-soft:#253263; --b:#F2B640; --b-ink:#211600;
         --good:#4CC59A; --warn:#F0A04B; --bad:#FF8A7A; --chip:#24304D; }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:400 15px/1.5 var(--font);font-stretch:100%}
.num{font-variant-numeric:tabular-nums lining-nums}
:focus-visible{outline:3px solid var(--a);outline-offset:2px;border-radius:4px}

/* Header band */
.band{background:var(--band);color:var(--band-ink)}
.band .inner{max-width:1240px;margin:0 auto;padding:22px 16px 0}
.top{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:12px 24px}
h1{margin:0;font-weight:800;font-stretch:118%;font-size:clamp(1.9rem,5vw,2.9rem);line-height:1.02;letter-spacing:-.01em}
.updated{margin:6px 0 0;color:var(--band-muted);font-size:.93rem;max-width:70ch}
.follow{display:flex;align-items:center;gap:8px;color:var(--band-muted);font-size:.9rem}
.follow select{font:500 .95rem var(--font);color:var(--ink);background:var(--surface);border:0;border-radius:6px;padding:7px 10px;min-width:12rem}

/* Tabs */
.tabs{display:flex;gap:2px;margin-top:18px;overflow-x:auto;scrollbar-width:none}
.tabs::-webkit-scrollbar{display:none}
.tab{appearance:none;border:0;background:transparent;color:var(--band-muted);font:600 .95rem var(--font);font-stretch:105%;
     padding:11px 14px 12px;border-radius:8px 8px 0 0;cursor:pointer;white-space:nowrap}
.tab:hover{color:var(--band-ink)}
.tab[aria-selected="true"]{background:var(--bg);color:var(--ink)}

/* Layout */
.wrap{max-width:1240px;margin:0 auto;padding:22px 16px 64px;display:grid;grid-template-columns:minmax(0,1fr);gap:22px}
@media (min-width:1000px){.wrap{grid-template-columns:minmax(0,1fr) 330px}}
.panel[hidden]{display:none}
h2{margin:0 0 4px;font-weight:700;font-stretch:110%;font-size:1.35rem;line-height:1.2}
.lede{margin:0 0 16px;color:var(--muted);max-width:72ch}
.note{color:var(--muted);font-size:.88rem;max-width:76ch;margin:12px 0 0}

/* Matchups */
.grid{display:grid;grid-template-columns:1fr;gap:14px}
@media (min-width:760px){.grid{grid-template-columns:1fr 1fr}}
.match{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 16px 14px}
.match.mine{border-color:var(--a);box-shadow:inset 0 3px 0 var(--a)}
.names{display:grid;grid-template-columns:1fr 1fr;gap:12px;font-weight:700;font-stretch:108%;font-size:1.05rem;line-height:1.2}
.names > span:last-child{text-align:right}
.names .small{font-weight:600;color:var(--muted);font-size:.92rem}
.tug{position:relative;display:flex;height:44px;margin:12px 0 12px;border-radius:6px;overflow:hidden;background:var(--chip)}
.tug .sa,.tug .sb{display:flex;align-items:center;font-weight:800;font-stretch:112%;font-size:1.35rem;white-space:nowrap;min-width:0;transition:flex-basis .5s ease}
.tug .sa{background:var(--a);color:#fff;padding-left:12px}
.tug .sb{background:var(--b);color:var(--b-ink);justify-content:flex-end;padding-right:12px}
.tug .mid{position:absolute;left:50%;top:0;bottom:0;width:2px;background:var(--surface);opacity:.85}
.facts{display:grid;grid-template-columns:auto 1fr auto;gap:5px 12px;align-items:baseline}
.facts .v{font-weight:600;font-size:1.02rem}
.facts .k{color:var(--muted);text-align:center;font-size:.84rem}
.facts .r{text-align:right}
.verdict{margin:12px 0 0;padding-top:10px;border-top:1px solid var(--line);color:var(--muted);font-size:.9rem}
.verdict b{color:var(--ink);font-weight:600}

/* Tables */
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px}
table{width:100%;border-collapse:collapse}
th,td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td.l,th.l{text-align:left}
thead th{font-weight:600;font-size:.82rem;color:var(--muted);background:var(--row);position:sticky;top:0}
tbody tr:last-child td{border-bottom:0}
tbody tr.mine td{background:var(--a-soft)}
td .sub{display:block;color:var(--muted);font-size:.82rem;white-space:nowrap}
td.big{font-weight:700;font-size:1.02rem}
.rank{color:var(--muted);width:2.2em}
.bar{position:relative;height:8px;border-radius:4px;background:var(--chip);min-width:90px}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px;background:var(--a)}

/* Chips */
.chip{display:inline-block;font-size:.76rem;font-weight:600;padding:1px 7px;border-radius:999px;background:var(--chip);color:var(--muted);margin-left:6px;vertical-align:1px}
.chip.out{background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad)}
.chip.q{background:color-mix(in srgb,var(--warn) 16%,transparent);color:var(--warn)}
.chip.fa{background:color-mix(in srgb,var(--good) 14%,transparent);color:var(--good)}
.chip.ir{background:var(--chip);color:var(--muted)}

/* Segmented filter */
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:0 0 12px;background:var(--surface)}
.seg button{appearance:none;border:0;background:transparent;font:500 .9rem var(--font);color:var(--muted);padding:7px 12px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--surface)}

/* Rosters */
details{background:var(--surface);border:1px solid var(--line);border-radius:10px;margin-bottom:10px;overflow:hidden}
summary{cursor:pointer;padding:12px 14px;font-weight:700;font-stretch:106%;display:flex;justify-content:space-between;gap:12px;list-style:none}
summary::-webkit-details-marker{display:none}
summary .meta{font-weight:400;color:var(--muted);font-size:.9rem}
details[open] summary{border-bottom:1px solid var(--line)}
details .tablewrap{border:0;border-radius:0}
.empty{background:var(--surface);border:1px dashed var(--line);border-radius:10px;padding:16px;color:var(--muted)}

/* Records */
.records{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;margin-bottom:22px}
.rec{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 14px 12px}
.rec .lab{color:var(--muted);font-size:.86rem;font-weight:600}
.rec .val{font-weight:800;font-stretch:112%;font-size:1.9rem;line-height:1.1;margin:4px 0 2px}
.rec .val small{font-weight:600;font-size:.85rem;color:var(--muted);font-stretch:100%;margin-left:4px}
.rec .who{font-weight:600}
.rec .when{color:var(--muted);font-size:.86rem}
th.sort{cursor:pointer;user-select:none}
th.sort[aria-sort="descending"]::after{content:" ↓"}
th.sort[aria-sort="ascending"]::after{content:" ↑"}

/* Season */
.seeds{display:flex;gap:2px}
.seeds i{display:block;width:9px;height:18px;border-radius:2px;background:var(--a)}
.seeds i.cut{margin-left:4px}
.key{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:.84rem;margin:10px 0 0}
.good{color:var(--good)} .bad{color:var(--bad)}
.subhead{margin-top:28px}

/* History */
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.controls{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;margin:4px 0 12px}
.controls select{font:500 .92rem var(--font);color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:7px 10px}
.controls .seg{margin:0}
.tag{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.02em;padding:1px 6px;border-radius:4px;background:var(--chip);color:var(--muted);margin-left:6px;vertical-align:1px}
.wl-w{color:var(--good);font-weight:700} .wl-l{color:var(--bad);font-weight:700}
td.muted{color:var(--muted)}

/* Standings rail */
aside .card{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}
aside h2{font-size:1.15rem;padding:14px 14px 4px}
aside .lede{padding:0 14px;margin-bottom:8px;font-size:.86rem}
aside table th,aside table td{padding:8px 6px;font-size:.9rem}
aside table td.team{padding-right:2px}
aside td.team{text-align:left;white-space:normal;line-height:1.25}
aside th.team{text-align:left}
td.wrapcell{white-space:normal;min-width:9rem}
td.wrapcell .sub{white-space:normal}
table.tight th,table.tight td{padding:10px 9px}
table.tight thead th{white-space:normal;vertical-align:bottom;line-height:1.25}
.only-sm{display:none}
@media (max-width:700px){ .opt, td .sub.opt{display:none} .only-sm, td .sub.only-sm{display:block} th,td{padding:9px 8px} td.wrapcell{min-width:7rem} #ppp-table th,#ppp-table td{padding:9px 6px} }
@media (min-width:1000px){aside .card{position:sticky;top:16px}}
@media (prefers-reduced-motion: reduce){.tug .sa,.tug .sb{transition:none}}
</style>
</head>
<body>
<header class="band">
  <div class="inner">
    <div class="top">
      <div>
        <h1 id="title"></h1>
        <p class="updated" id="updated"></p>
      </div>
      <label class="follow">My team
        <select id="follow"><option value="">Choose your team</option></select>
      </label>
    </div>
    <nav class="tabs" role="tablist" aria-label="Forecast sections" id="tabs"></nav>
  </div>
</header>

<div class="wrap">
  <main id="panels"></main>
  <aside aria-labelledby="h-standings">
    <div class="card">
      <h2 id="h-standings">Standings</h2>
      <p class="lede">Record before this week, each team's chance to win this week, and its chance to make the playoffs. PF is points for.</p>
      <div id="standings"></div>
    </div>
  </aside>
</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (v, d=0) => (v === null || v === undefined || Number.isNaN(v)) ? '–'
  : Number(v).toLocaleString('en-AU', {minimumFractionDigits:d, maximumFractionDigits:d});
const pct = v => (v === null || v === undefined) ? '–' : v >= .995 ? '>99%' : v < .005 ? '<1%' : Math.round(v*100) + '%';
// Before a team's first game of the week, show last week's PPP instead of nothing.
const pppNow = t => (t && t.startsToDate > 0) ? {v: t.pppToDate, last: false}
                                              : {v: t ? t.pppLastWeek : null, last: !!(t && t.pppLastWeek !== null && t.pppLastWeek !== undefined)};
const signed = (v, d=1) => (v === null || v === undefined) ? '–' : (v > 0 ? '+' : v < 0 ? '−' : '') + fmt(Math.abs(v), d);
const STATUS = {ACTIVE:'', OUT:'Out', DAY_TO_DAY:'Day-to-day', QUESTIONABLE:'Questionable', DOUBTFUL:'Doubtful',
                PROBABLE:'Probable', SUSPENSION:'Suspended', SUSPENDED:'Suspended', INJURY_RESERVE:'Injured reserve'};
function statusChip(s){
  const k = String(s || '').toUpperCase().replace(/[ -]/g, '_');
  if (!k || k === 'ACTIVE' || k === 'UNKNOWN' || k === 'NAN') return '';
  const label = STATUS[k] || k.charAt(0) + k.slice(1).toLowerCase().replace(/_/g, ' ');
  const cls = (k === 'OUT' || k.startsWith('SUSP') || k === 'INJURY_RESERVE') ? 'out' : 'q';
  return `<span class="chip ${cls}">${esc(label)}</span>`;
}
const day = d => (D.dayFull && D.dayFull[d]) || d || '';
const dateLabel = s => /^\d{4}-\d{2}-\d{2}$/.test(s || '')
  ? new Date(s + 'T12:00:00').toLocaleDateString('en-AU', {weekday:'short', day:'numeric', month:'short'}) : (s || '');
const team = Object.fromEntries(D.teams.map(t => [t.name, t]));
let mine = ''; try { mine = localStorage.getItem('myTeam') || ''; } catch (e) {}

/* ---------- Header ---------- */
function header(){
  const m = D.meta || {};
  $('#title').textContent = m.matchup_period ? `Week ${m.matchup_period} forecast` : D.title;
  const when = m.generated_utc ? new Date(m.generated_utc).toLocaleString('en-AU',
    {weekday:'short', day:'numeric', month:'short', hour:'numeric', minute:'2-digit'}) : 'unknown';
  const fd = (m.finished_days || []).map(day), sd = (m.simulated_days || []).map(day);
  const span = a => a.length > 2 ? `${a[0]} to ${a[a.length-1]}` : a.join(' and ');
  const done = fd.length ? `${span(fd)} ${fd.length > 1 ? 'are' : 'is'} final` : 'No days are final yet';
  const left = sd.length ? `${span(sd)} ${sd.length > 1 ? 'are' : 'is'} simulated ${fmt(m.sims)} times` : 'the week is complete';
  $('#updated').textContent = `Updated ${when}. ${done}; ${left}.`;
  const sel = $('#follow');
  D.teams.map(t => t.name).sort((a,b) => a.localeCompare(b)).forEach(n => {
    const o = document.createElement('option'); o.value = n; o.textContent = n; sel.appendChild(o);
  });
  sel.value = mine;
  sel.addEventListener('change', () => { mine = sel.value; try { localStorage.setItem('myTeam', mine); } catch (e) {} renderAll(); });
}

/* ---------- Panels ---------- */
const nd = (D.meta || {}).next_day;
const PANELS = [
  {id:'matchups', label:'Matchups', render: renderMatchups},
  {id:'season', label:'Season', render: renderSeason},
  {id:'teams', label:'Teams', render: renderTeams},
  {id:'players', label: nd ? `${day(nd)}'s top players` : 'Top players', render: renderTopPlayers},
  {id:'topteams', label: nd ? `${day(nd)}'s top teams` : 'Top teams', render: renderTopTeams},
  {id:'fa', label:'Free agents', render: renderFreeAgents},
  {id:'rosters', label:'Rosters', render: renderRosters},
  {id:'stats', label:'PPP and records', render: renderStats},
  {id:'history', label:'History', render: renderHistory},
  {id:'check', label:'Model check', render: renderCheck},
];
let current = 'matchups';
try { const t = localStorage.getItem('tab'); if (PANELS.some(p => p.id === t)) current = t; } catch (e) {}

function buildTabs(){
  const nav = $('#tabs'), main = $('#panels');
  PANELS.forEach(p => {
    const b = document.createElement('button');
    b.className = 'tab'; b.id = `tab-${p.id}`; b.setAttribute('role','tab'); b.setAttribute('aria-controls', `panel-${p.id}`);
    b.textContent = p.label;
    b.addEventListener('click', () => select(p.id, true));
    b.addEventListener('keydown', e => {
      const i = PANELS.findIndex(x => x.id === p.id);
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        const n = PANELS[(i + (e.key === 'ArrowRight' ? 1 : PANELS.length - 1)) % PANELS.length];
        select(n.id, true); document.getElementById(`tab-${n.id}`).focus();
      }
    });
    nav.appendChild(b);
    const s = document.createElement('section');
    s.className = 'panel'; s.id = `panel-${p.id}`; s.setAttribute('role','tabpanel'); s.setAttribute('aria-labelledby', `tab-${p.id}`);
    s.tabIndex = 0;
    main.appendChild(s);
  });
  select(current, false);
}
function select(id, remember){
  current = id;
  PANELS.forEach(p => {
    const on = p.id === id;
    const b = document.getElementById(`tab-${p.id}`);
    b.setAttribute('aria-selected', on); b.tabIndex = on ? 0 : -1;
    document.getElementById(`panel-${p.id}`).hidden = !on;
  });
  if (remember) { try { localStorage.setItem('tab', id); } catch (e) {} }
}

function renderMatchups(){
  if (!D.matchups.length) return `<h2>Matchups</h2><p class="empty">Matchups appear after the ESPN matchup download runs this week.</p>`;
  const weekDone = !((D.meta || {}).simulated_days || []).length;
  const ms = [...D.matchups].sort((x,y) => ((y.a===mine||y.b===mine) - (x.a===mine||x.b===mine)));
  const card = m => {
    const ta = team[m.a] || {}, tb = team[m.b] || {};
    const aw = Math.max(m.aWin || 0, 0), bw = Math.max(m.bWin || 0, 0), tot = aw + bw || 1;
    const aFav = aw >= bw, fav = aFav ? m.a : m.b;
    const lo = aFav ? m.marginLo : -m.marginHi, hi = aFav ? m.marginHi : -m.marginLo;
    const side = v => v >= 0 ? `winning by ${fmt(Math.abs(v))}` : `losing by ${fmt(Math.abs(v))}`;
    return `<article class="match${(m.a===mine||m.b===mine) ? ' mine' : ''}" aria-label="${esc(m.a)} versus ${esc(m.b)}">
      <div class="names"><span>${esc(m.a)}${aw < .24 ? ` <span class="small num">${pct(aw)}</span>` : ''}</span><span>${bw < .24 ? `<span class="small num">${pct(bw)}</span> ` : ''}${esc(m.b)}</span></div>
      <div class="tug" role="img" aria-label="${esc(m.a)} ${pct(aw)} to win, ${esc(m.b)} ${pct(bw)} to win">
        <div class="sa num" style="flex-basis:${aw/tot*100}%">${aw >= .24 ? pct(aw) : ''}</div>
        <div class="sb num" style="flex-basis:${bw/tot*100}%">${bw >= .24 ? pct(bw) : ''}</div>
        <div class="mid"></div>
      </div>
      <div class="facts num">
        <span class="v">${fmt(ta.toDate,1)}</span><span class="k">Points so far</span><span class="v r">${fmt(tb.toDate,1)}</span>
        <span class="v">${fmt(ta.final)}</span><span class="k">Projected final</span><span class="v r">${fmt(tb.final)}</span>
        ${(() => { const pa = pppNow(ta), pb = pppNow(tb), last = pa.last && pb.last;
          return `<span class="v">${fmt(pa.v,1)}</span><span class="k">${last ? 'PPP last week' : 'PPP so far'}</span><span class="v r">${fmt(pb.v,1)}</span>`; })()}
        <span class="v">${fmt(ta.pppRest,1)}</span><span class="k">PPP rest of week</span><span class="v r">${fmt(tb.pppRest,1)}</span>
      </div>
      <p class="verdict">${weekDone
        ? (Math.abs(m.margin || 0) < 0.05 ? `<b>Tied</b> at ${fmt(ta.toDate,1)}.` : `<b>${esc(fav)} won</b> by ${fmt(Math.abs(m.margin || 0),1)} points.`)
        : `<b>${esc(fav)} ${pct(Math.max(aw,bw))} to win</b>, by about ${fmt(Math.abs(m.margin || 0))} points. In 80% of simulations they finish between ${side(lo)} and ${side(hi)}.`}</p>
    </article>`;
  };
  return `<h2>Matchups</h2><p class="lede">${weekDone ? "This week is complete: these are ESPN's final scores. The next week's forecast starts once ESPN moves to the new week." : "Win chance is the share of simulations each team finished ahead. Finished days use ESPN's actual scores."}</p>
    <div class="grid">${ms.map(card).join('')}</div>
    <p class="note">PPP is points per player: fantasy points divided by games played by players in the active lineup.</p>`;
}

function renderTeams(){
  const win = {}; D.matchups.forEach(m => { win[m.a] = m.aWin; win[m.b] = m.bWin; });
  const rows = [...D.teams].sort((a,b) => (b.final||0) - (a.final||0));
  const hasPick = rows.some(t => t.pickups !== null && t.pickups !== undefined);
  return `<h2>Teams</h2><p class="lede">Every team's projected weekly total, ranked. Projections assume each manager makes their usual number of pickups${hasPick ? '; the From pickups column shows how many points those expected pickups add compared with keeping the current roster' : ''}.</p>
  <div class="tablewrap"><table class="num tight"><thead><tr>
    <th scope="col">Team</th><th scope="col">Projected</th>${hasPick ? '<th scope="col">From pickups</th>' : ''}<th scope="col" class="opt">80% range</th><th scope="col">Win chance</th>
    <th scope="col" class="opt">Points so far</th><th scope="col" class="opt">Starts so far</th><th scope="col">PPP so far</th>
    <th scope="col" class="opt">Starts left</th><th scope="col">PPP rest of week</th></tr></thead><tbody>${
    rows.map(t => `<tr class="${t.name===mine?'mine':''}"><td>${esc(t.name)}</td><td class="big">${fmt(t.final)}</td>${hasPick ? `<td>${t.pickups === null || t.pickups === undefined ? '–' : (Math.abs(t.pickups) < 0.5 ? '0' : signed(t.pickups, 0))}</td>` : ''}
      <td class="opt">${fmt(t.p10)} to ${fmt(t.p90)}</td><td>${pct(win[t.name])}</td><td class="opt">${fmt(t.toDate,1)}</td><td class="opt">${fmt(t.startsToDate)}</td>
      <td>${(() => { const p = pppNow(t); return p.last ? `${fmt(p.v,1)}<span class="sub">last wk</span>` : fmt(p.v,1); })()}</td><td class="opt">${fmt(t.startsRest,1)}</td><td>${fmt(t.pppRest,1)}</td></tr>`).join('')
  }</tbody></table></div>`;
}

let playerFilter = 'all';
function renderTopPlayers(){
  if (!D.topPlayers.length) return `<h2>Top players</h2><p class="empty">Daily projections appear once the week's remaining games are simulated.</p>`;
  const list = D.topPlayers.filter(p => playerFilter === 'all' ? true : playerFilter === 'fa' ? p.fa : !!p.owner).slice(0, 30);
  const btn = (k, label) => `<button type="button" data-filter="${k}" aria-pressed="${playerFilter===k}">${label}</button>`;
  return `<h2>${esc(day(nd))}'s top players</h2>
    <p class="lede">Projected fantasy points for ${esc(day(nd))}, including the chance each player sits out.</p>
    <div class="seg" role="group" aria-label="Show players">${btn('all','All players')}${btn('rostered','Rostered')}${btn('fa','Free agents')}</div>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="l">Player</th>
      <th scope="col" class="l opt">Fantasy team</th><th scope="col">Projected</th>
      <th scope="col" class="opt">If he plays</th><th scope="col" class="opt">Chance to play</th></tr></thead><tbody>${
      list.map((p,i) => `<tr class="${p.owner && p.owner===mine?'mine':''}"><td class="rank">${i+1}</td>
        <td class="l">${esc(p.name)}${statusChip(p.status)}<span class="sub">${esc(p.pos.replace(/\/UTIL$/,''))}, ${esc(p.nba)} vs ${esc(p.opp)}</span><span class="sub only-sm">${p.fa ? 'Free agent' : esc(p.owner)}</span></td>
        <td class="l wrapcell opt">${p.fa ? '<span class="chip fa" style="margin-left:0">Free agent</span>' : esc(p.owner)}</td>
        <td class="big">${fmt(p.proj,1)}</td><td class="opt">${fmt(p.ifPlays,1)}</td><td class="opt">${pct(p.play)}</td></tr>`).join('')
    }</tbody></table></div>`;
}

function renderTopTeams(){
  if (!D.topTeams.length) return `<h2>Top teams</h2><p class="empty">Daily team projections appear once the week's remaining games are simulated.</p>`;
  const max = Math.max(...D.topTeams.map(t => t.proj || 0), 1);
  const opp = {}; D.matchups.forEach(m => { opp[m.a] = m.b; opp[m.b] = m.a; });
  const by = Object.fromEntries(D.topTeams.map(t => [t.name, t.proj]));
  return `<h2>${esc(day(nd))}'s top teams</h2>
    <p class="lede">Each fantasy team's projected points for ${esc(day(nd))} alone, from its current roster with its best lineup. Free agents a team might pick up aren't included.</p>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="l">Team</th>
      <th scope="col" class="l" style="width:40%">Projected points</th><th scope="col">Points</th><th scope="col" class="l">Opponent</th><th scope="col">Opponent points</th></tr></thead><tbody>${
      D.topTeams.map((t,i) => `<tr class="${t.name===mine?'mine':''}"><td class="rank">${i+1}</td><td class="l">${esc(t.name)}</td>
        <td class="l"><div class="bar"><i style="width:${(t.proj||0)/max*100}%"></i></div></td>
        <td class="big">${fmt(t.proj)}</td><td class="l">${esc(opp[t.name] || '')}</td><td>${fmt(by[opp[t.name]])}</td></tr>`).join('')
    }</tbody></table></div>`;
}

function renderFreeAgents(){
  if (!D.freeAgents.length) return `<h2>Free agents</h2><p class="empty">No free-agent projections in this run.</p>`;
  return `<h2>Free agents</h2>
    <p class="lede">Unrostered players ranked by projected fantasy points over the rest of this week. Games, minutes, injuries and matchups are all simulated.</p>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="l">Player</th>
      <th scope="col">Games left</th><th scope="col">Rest of week</th><th scope="col" class="opt">Per game</th>
      <th scope="col" class="opt">${esc(day(nd))}</th><th scope="col" class="opt">ESPN avg</th></tr></thead><tbody>${
      D.freeAgents.map((p,i) => `<tr><td class="rank">${i+1}</td>
        <td class="l">${esc(p.name)}${statusChip(p.status)}<span class="sub">${esc(p.pos.replace(/\/UTIL$/,''))}, ${esc(p.nba)}</span></td>
        <td>${p.games}</td><td class="big">${fmt(p.rest,1)}</td><td class="opt">${fmt(p.perGame,1)}</td>
        <td class="opt">${fmt(p.next,1)}</td><td class="opt">${fmt(p.espn,1)}</td></tr>`).join('')
    }</tbody></table></div>`;
}

/* ---------- Season ---------- */
const ordinal = n => { const v = Math.round(n), s = ['th','st','nd','rd'], m = v % 100; return v + (s[(m-20)%10] || s[m] || s[0]); };
function renderSeason(){
  const S = D.season || {}, M = S.meta || {}, T = S.teams || [];
  if (!T.length) return `<h2>Season forecast</h2><p class="empty">${M.status === 'regular_season_over'
    ? 'The regular season is over.' : 'The season forecast appears after the next update.'}</p>`;
  const P = M.playoff_teams, byes = M.byes || 0, n = T.length;
  const tb = M.tiebreak === 'h2h' ? 'head-to-head record' : 'points for';
  const rec = t => `${t.w}-${t.l}${t.t ? '-' + t.t : ''}`;
  const prec = t => `${fmt(t.pw,1)}-${fmt(t.pl,1)}${t.pt >= .05 ? '-' + fmt(t.pt,1) : ''}`;
  let html = `<h2>Season forecast</h2>
    <p class="lede">The rest of the regular season played out ${fmt(M.sims)} times: this week from the full simulation, then ${M.weeks_left_after_this} more week${M.weeks_left_after_this === 1 ? '' : 's'} from a simpler weekly model. The top ${P} make the playoffs${byes ? `, and the top ${byes} get a first-round bye` : ''}. Ties in the standings go to ${tb}.</p>
    <div class="tablewrap"><table class="num tight"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="l">Team</th>
      <th scope="col" class="opt">Record</th><th scope="col">Projected</th><th scope="col">Playoffs</th>${byes ? '<th scope="col" class="opt">Bye</th>' : ''}
      <th scope="col" class="opt">Final</th><th scope="col">Title</th><th scope="col" class="opt">Avg finish</th>
      <th scope="col" class="l opt">Finishing position</th></tr></thead><tbody>${
      T.map((t,i) => `<tr class="${t.name===mine?'mine':''}"><td class="rank">${i+1}</td>
        <td class="l wrapcell">${who(t.name, t.manager)}</td>
        <td class="opt">${rec(t)}<span class="sub">${fmt(t.pf)} PF</span></td>
        <td>${prec(t)}<span class="sub opt">${fmt(t.wlo)} to ${fmt(t.whi)} wins</span></td>
        <td class="big">${pct(t.playoff)}</td>${byes ? `<td class="opt">${pct(t.bye)}</td>` : ''}
        <td class="opt">${pct(t.final)}</td><td class="big">${pct(t.title)}</td><td class="opt">${ordinal(t.finish)}</td>
        <td class="l opt"><div class="seeds" role="img" aria-label="${esc(t.name)} finishing-position chances: ${t.seeds.map((p,k) => `${ordinal(k+1)} ${pct(p)}`).join(', ')}">${
          t.seeds.map((p,k) => `<i class="${k === P ? 'cut' : ''}" style="opacity:${Math.max(0.06, Math.min(1, (p||0)*2.5)).toFixed(2)}" title="${ordinal(k+1)}: ${pct(p)}"></i>`).join('')}</div></td></tr>`).join('')
    }</tbody></table></div>
    <p class="note">Projected record includes this week. The finishing-position strip is darker where a finish is more likely; the gap marks the playoff line. Playoff rounds are played out the same way, using each team's projected scores for the playoff weeks.</p>`;
  const name = mine || T[0].name, wk = (S.weeks || {})[name] || [];
  html += `<h2 class="subhead">${esc(name)}: the weeks ahead</h2>`;
  if (!mine) html += `<p class="lede">Choose your team at the top of the page to see its schedule here.</p>`;
  html += wk.length ? `<div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="l">Week</th><th scope="col" class="l">Opponent</th>
      <th scope="col" class="opt">Projected</th><th scope="col" class="opt">Opponent</th><th scope="col">Win chance</th></tr></thead><tbody>${
      wk.slice(0, 8).map(w => `<tr><td class="l">Week ${w.mp}<span class="sub">${esc(dateLabel(w.start))} to ${esc(dateLabel(w.end))}</span></td>
        <td class="l wrapcell">${esc(w.opp)}<span class="sub only-sm">${fmt(w.mean)} vs ${fmt(w.oppMean)}</span></td><td class="opt">${fmt(w.mean)}</td><td class="opt">${fmt(w.oppMean)}</td>
        <td class="big ${w.win >= .6 ? 'good' : w.win <= .4 ? 'bad' : ''}">${pct(w.win)}</td></tr>`).join('')
    }</tbody></table></div>
    <p class="note">Future weeks use current rosters, each player's projected score, the NBA schedule (games per week matter) and ESPN's injury return dates. They're less certain the further out they are, since rosters and form change.</p>`
    : `<p class="empty">No future regular-season weeks.</p>`;
  return html;
}

/* ---------- History ---------- */
const shortDate = d => { const x = new Date(d + 'T12:00:00'); return isNaN(x) ? d : x.toLocaleDateString('en-AU', {day:'numeric', month:'short', year:'numeric'}); };
let histScope = 'All seasons', histCat = 'game', histMode = 'played', leagueSeason = null, h2hMgr = null;
const CAT_LABEL = {game:'Best games', half:'Best halves', quarter:'Best quarters', worst:'Worst games'};
function renderHistory(){
  const H = D.history || {}, N = H.nba, L = H.league;
  let html = `<h2>NBA fantasy records</h2>`;
  if (!N) html += `<p class="empty">NBA records appear after the next update.</p>`;
  else {
    const scopes = ['All seasons', ...(N.seasons || [])];
    const M = (N.modes || {})[histMode] || {};
    if (!M[histScope]) histScope = 'All seasons';
    const S = M[histScope] || {}, list = S[histCat] || [];
    const altLabel = histMode === 'played' ? `<span class="opt">Today's scoring</span><span class="only-sm">Today</span>`
                                           : `<span class="opt">At the time</span><span class="only-sm">Then</span>`;
    html += `<p class="lede">Every NBA player's games since 2023-24, regular season and playoffs. Until 2023-24 the league also gave +2 for each free throw made and −1 for each attempt. "At the time" scores each game the way the league scored it that season, and "Today's scoring" uses the current rules for every season. Each list shows the other score alongside for comparison. Halves are quarters 1–2 and 3–4; overtime counts toward the game only.</p>
      <div class="controls"><label class="sr-only" for="hist-scope">Season</label>
        <select id="hist-scope">${scopes.map(x => `<option${x === histScope ? ' selected' : ''}>${esc(x)}</option>`).join('')}</select>
        <div class="seg" role="group" aria-label="Scoring">${[['played','At the time'],['today',"Today's scoring"]].map(([k,v]) => `<button type="button" data-mode="${k}" aria-pressed="${k === histMode}">${v}</button>`).join('')}</div>
        <div class="seg" role="group" aria-label="Record type">${Object.entries(CAT_LABEL).map(([k,v]) => `<button type="button" data-cat="${k}" aria-pressed="${k === histCat}">${v}</button>`).join('')}</div></div>
      <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="l">Player</th>
        <th scope="col">FP</th><th scope="col">${altLabel}</th>${histCat === 'half' || histCat === 'quarter' ? '<th scope="col" class="l opt">Part</th>' : ''}
        <th scope="col" class="l opt">Game</th><th scope="col" class="l opt">Date</th></tr></thead><tbody>${
        list.map((r,i) => `<tr><td class="rank">${i+1}</td><td class="l">${esc(r.player)}<span class="sub">${esc(r.team)} vs ${esc(r.opp)}${histCat === 'half' || histCat === 'quarter' ? `<span class="only-sm">${esc(r.part)}</span>` : ''}${r.playoffs ? ' · Playoffs' : ''}${r.old ? ' · Old scoring' : ''}<span class="only-sm">${esc(shortDate(r.date))}</span></span></td>
          <td class="big">${fmt(r.fp)}</td><td class="${r.alt === r.fp ? 'muted' : (r.alt > r.fp ? 'wl-w' : 'wl-l')}">${r.alt === r.fp ? 'Same' : (r.alt > r.fp ? '+' : '−') + fmt(Math.abs(r.alt - r.fp)) + `<span class="sub">${fmt(r.alt)}</span>`}</td>${histCat === 'half' || histCat === 'quarter' ? `<td class="l opt">${esc(r.part)}</td>` : ''}
          <td class="l opt wrapcell">${esc(r.line)}</td><td class="l opt">${esc(dateLabel(r.date))}<span class="sub">${esc(r.season)}</span></td></tr>`).join('')
      }</tbody></table></div>
      <p class="note">${fmt(S.games)} NBA games. The second column shows how much more (green) or less (red) the same performance scores under the other rules. ${histCat === 'half' || histCat === 'quarter' ? 'The Game column is the full-game line. ' : ''}Built from NBA play-by-play: totals match official box scores in 99.6% of player-games; the rest differ by a late stat correction.</p>`;
  }

  html += `<h2 class="subhead">League history</h2>`;
  if (!L) return html + `<p class="empty">League history appears after the next update.</p>`;
  const nameCell = r => `${esc(r.manager)}${r.team ? `<span class="sub">${esc(r.team)}</span>` : '<span class="sub">Former manager</span>'}`;
  html += `<p class="lede">Every week since 2024-25, when the league went to 12 teams, from the league spreadsheets and, for this season, ESPN. Standings and averages are regular season only. Week-score records count weeks of about normal length, leaving out short NBA Cup weeks and long All-Star weeks; the any-length record is shown separately.</p>
    <div class="records">${L.records.map(r => `<div class="rec"><div class="lab">${esc(r.label)}</div>
      <div class="val num">${r.unit === 'PPP' ? fmt(r.value,2) : fmt(r.value,0)}<small>${r.unit === 'PPP' ? 'PPP' : 'pts'}</small></div>
      <div class="who">${esc(r.manager)}${r.team ? ` · ${esc(r.team)}` : ''}</div><div class="when">${esc(r.when)}${r.note ? ` · ${esc(r.note)}` : ` · vs ${esc(r.opponent)}`}</div></div>`).join('')}</div>

    <h2 class="subhead">All-time standings</h2>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="l">Manager</th>
      <th scope="col" class="opt">Seasons</th><th scope="col">Record</th><th scope="col">Win %</th><th scope="col" class="opt">Avg week</th><th scope="col">PPP</th></tr></thead><tbody>${
      L.allTime.map((r,i) => `<tr class="${r.team && r.team === mine ? 'mine' : ''}"><td class="rank">${i+1}</td><td class="l">${nameCell(r)}</td><td class="opt">${r.seasons}</td>
        <td>${r.w}-${r.l}${r.t ? '-' + r.t : ''}</td><td class="big">${fmt(r.pct*100,1)}%</td><td class="opt">${fmt(r.avg)}</td><td>${fmt(r.ppp,2)}</td></tr>`).join('')
    }</tbody></table></div>`;

  const seasons = Object.keys(L.seasons);
  if (!leagueSeason || !L.seasons[leagueSeason]) leagueSeason = seasons[0];
  const SS = L.seasons[leagueSeason];
  html += `<h2 class="subhead">Season standings</h2>
    <div class="controls"><label class="sr-only" for="league-season">Season</label>
      <select id="league-season">${seasons.map(x => `<option${x === leagueSeason ? ' selected' : ''}>${esc(x)}</option>`).join('')}</select>
      <span class="small">${SS.weeks} regular-season weeks</span></div>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="l">Manager</th>
      <th scope="col">Record</th><th scope="col">Points for</th><th scope="col" class="opt">Avg week</th><th scope="col">PPP</th></tr></thead><tbody>${
      SS.rows.map(r => `<tr class="${r.team && r.team === mine ? 'mine' : ''}"><td class="rank">${r.rank}</td><td class="l">${nameCell(r)}</td>
        <td>${r.w}-${r.l}${r.t ? '-' + r.t : ''}</td><td>${fmt(r.pf)}</td><td class="opt">${fmt(r.avg)}</td><td>${fmt(r.ppp,2)}</td></tr>`).join('')
    }</tbody></table></div>`;

  html += `<h2 class="subhead">Highest-scoring weeks</h2>
    <p class="lede">All weeks, any length. The long All-Star week (week 17) gives teams many more starts, so it's marked.</p>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="l">Manager</th>
      <th scope="col">Points</th><th scope="col" class="opt">PPP</th><th scope="col" class="l">Week</th></tr></thead><tbody>${
      L.topWeeks.map((r,i) => `<tr><td class="rank">${i+1}</td><td class="l">${esc(r.manager)}<span class="sub">vs ${esc(r.opponent)}, ${r.won ? 'won' : 'lost'}</span></td>
        <td class="big">${fmt(r.points)}</td><td class="opt">${fmt(r.ppp,2)}</td><td class="l">${esc(r.season)}<span class="sub">Week ${r.week}${r.playoffs ? ', playoffs' : ''}${r.length ? ` · ${esc(r.length)} (${fmt(r.starts)} starts)` : ''}</span></td></tr>`).join('')
    }</tbody></table></div>`;

  const mgrs = Object.keys(L.h2h).sort();
  const mineMgr = (L.allTime.find(r => r.team && r.team === mine) || {}).manager;
  if (!h2hMgr || !L.h2h[h2hMgr]) h2hMgr = mineMgr || mgrs[0];
  const opp = Object.entries(L.h2h[h2hMgr] || {}).sort((a,b) => (b[1][0]-b[1][1]) - (a[1][0]-a[1][1]));
  html += `<h2 class="subhead">Head to head</h2>
    <div class="controls"><label class="sr-only" for="h2h-mgr">Manager</label>
      <select id="h2h-mgr">${mgrs.map(x => `<option${x === h2hMgr ? ' selected' : ''}>${esc(x)}</option>`).join('')}</select>
      <span class="small">Every matchup in the spreadsheets and on ESPN, playoffs included</span></div>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="l">Opponent</th><th scope="col">Won</th><th scope="col">Lost</th><th scope="col" class="opt">Tied</th></tr></thead><tbody>${
      opp.map(([o, r]) => `<tr><td class="l">${esc(o)}</td><td class="${r[0] > r[1] ? 'wl-w' : ''}">${r[0]}</td><td class="${r[1] > r[0] ? 'wl-l' : ''}">${r[1]}</td><td class="opt">${r[2]}</td></tr>`).join('')
    }</tbody></table></div>`;
  return html;
}
function bindHistory(){
  const panel = document.getElementById('panel-history');
  const redraw = () => { panel.innerHTML = renderHistory(); bindHistory(); };
  const on = (id, fn) => { const el = document.getElementById(id); if (el) el.addEventListener('change', e => { fn(e.target.value); redraw(); }); };
  on('hist-scope', v => histScope = v);
  on('league-season', v => leagueSeason = v);
  on('h2h-mgr', v => h2hMgr = v);
  panel.querySelectorAll('.seg button[data-cat]').forEach(b => b.addEventListener('click', () => { histCat = b.dataset.cat; redraw(); }));
  panel.querySelectorAll('.seg button[data-mode]').forEach(b => b.addEventListener('click', () => { histMode = b.dataset.mode; redraw(); }));
}

/* ---------- Model check ---------- */
function renderCheck(){
  const C = D.check || {}, sv = C.saved || {}, cal = (C.calibration || {}).weekly || {};
  let html = `<h2>How accurate is the forecast?</h2>
    <p class="lede">Every update saves what was predicted. When a week ends, the predictions are checked against ESPN's actual results, and the simulation adjusts itself if it's been consistently off.</p>`;
  if (!C.weeks_checked) {
    html += `<p class="empty">Nothing to check yet. ${sv.team_forecasts ? `${fmt(sv.team_forecasts)} team forecasts and ${fmt(sv.player_forecasts)} player forecasts have been saved${C.history_since ? ' since ' + esc(dateLabel(C.history_since)) : ''}.` : 'Forecasts are saved from the next update.'} The first results appear once this week is finished${(C.players || {}).games ? '' : ', and player results once a day of saved forecasts is complete'}.</p>`;
  }
  const byDay = C.weekly_by_day || [], mon = byDay[0];
  const P = C.players;
  const cards = [];
  if (mon) {
    cards.push({lab:`Favourite won (forecast on ${day(mon.day)})`, val: mon.favourite_won === null ? '–' : Math.round(mon.favourite_won*100) + '%', who:`${mon.team_weeks/2|0} matchups checked`});
    cards.push({lab:`Average miss on weekly totals (${day(mon.day)})`, val: fmt(mon.mean_abs_error), small:'pts', who: mon.mean_error > 0 ? `Actual scores ran ${fmt(mon.mean_error)} above the forecast on average` : `Actual scores ran ${fmt(-mon.mean_error)} below the forecast on average`});
    cards.push({lab:'80% ranges that held the result', val: Math.round(mon.in_80_range*100) + '%', who:'Should be close to 80%'});
  }
  if (P && P.ours) cards.push({lab:'Average miss per player game', val: fmt(P.ours.mae,1), small:'pts', who: P.espn && P.espn.mae !== null ? `ESPN's projection: ${fmt(P.espn.mae,1)}` : `${fmt(P.games)} games checked`});
  if (cards.length) html += `<div class="records">${cards.map(c => `<div class="rec"><div class="lab">${esc(c.lab)}</div>
    <div class="val num">${c.val}${c.small ? `<small>${c.small}</small>` : ''}</div><div class="when">${esc(c.who)}</div></div>`).join('')}</div>`;

  if (byDay.length) html += `<h2 class="subhead">Weekly forecasts by day</h2>
    <p class="lede">The same week checked from each day's forecast. Later forecasts know more, so they should do better. The Brier score measures win chances: 0 is perfect and 0.25 is a coin flip.</p>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="l">Forecast made</th><th scope="col">Favourite won</th>
      <th scope="col">Brier score</th><th scope="col">Average miss</th><th scope="col" class="opt">Actual minus forecast</th><th scope="col">In 80% range</th></tr></thead><tbody>${
      byDay.map(r => `<tr><td class="l">${esc(day(r.day))}<span class="sub">${r.team_weeks} team-weeks</span></td>
        <td>${r.favourite_won === null ? '–' : Math.round(r.favourite_won*100) + '%'}</td><td>${fmt(r.brier,3)}</td>
        <td>${fmt(r.mean_abs_error)}</td><td class="opt">${signed(r.mean_error,0)}</td><td>${Math.round(r.in_80_range*100)}%</td></tr>`).join('')
    }</tbody></table></div>`;

  if ((C.reliability || []).length) html += `<h2 class="subhead">Do the win chances mean what they say?</h2>
    <p class="lede">All checked forecasts grouped by the win chance they gave. In a well-calibrated forecast, teams given about 70% win about 70% of the time.</p>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="l">Win chance given</th><th scope="col">Forecasts</th>
      <th scope="col">Average given</th><th scope="col">Actually won</th></tr></thead><tbody>${
      C.reliability.map(r => `<tr><td class="l">${esc(r.range)}</td><td>${r.n}</td><td>${pct(r.predicted)}</td><td class="big">${pct(r.actual)}</td></tr>`).join('')
    }</tbody></table></div>`;

  if (P && P.ours) {
    html += `<h2 class="subhead">Player projections</h2>
      <p class="lede">Each player's projected score for his next game against what he actually scored, for players in a fantasy lineup who played. Lower is better.</p>
      <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="l">Projection</th><th scope="col">Average miss</th>
        <th scope="col">Bias</th><th scope="col">Games</th></tr></thead><tbody>
        <tr><td class="l">This forecast</td><td class="big">${fmt(P.ours.mae,2)}</td><td>${signed(P.ours.bias,2)}</td><td>${fmt(P.ours.n)}</td></tr>
        ${P.espn && P.espn.n ? `<tr><td class="l">ESPN's projected average</td><td>${fmt(P.espn.mae,2)}</td><td>${signed(P.espn.bias,2)}</td><td>${fmt(P.espn.n)}</td></tr>` : ''}
        ${P.season_average && P.season_average.n ? `<tr><td class="l">Season average so far</td><td>${fmt(P.season_average.mae,2)}</td><td>${signed(P.season_average.bias,2)}</td><td>${fmt(P.season_average.n)}</td></tr>` : ''}
      </tbody></table></div><p class="note">Bias is projected minus actual: positive means projections ran high.</p>`;
  }
  if ((C.season_by_horizon || []).length) html += `<h2 class="subhead">Season model</h2>
    <p class="lede">Weekly totals forecast one or more weeks ahead, checked against the result.</p>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="l">Forecast</th><th scope="col">Team-weeks</th>
      <th scope="col">Average miss</th><th scope="col">Actual minus forecast</th><th scope="col">Brier score</th></tr></thead><tbody>${
      C.season_by_horizon.map(r => `<tr><td class="l">${r.weeks_ahead} week${r.weeks_ahead > 1 ? 's' : ''} ahead</td><td>${r.team_weeks}</td>
        <td>${fmt(r.mean_abs_error)}</td><td>${signed(r.mean_error,0)}</td><td>${fmt(r.brier,3)}</td></tr>`).join('')
    }</tbody></table></div>`;

  const PK = C.pickups;
  if (PK && PK.overall) {
    const o = PK.overall, better = (o.mae_with || 0) <= (o.mae_without || 0);
    html += `<h2 class="subhead">Do expected pickups help the forecast?</h2>
      <p class="lede">Every forecast is also run without pickups. Checked against the real results over ${fmt(o.team_weeks)} team-weeks, weekly totals were ${better ? 'closer with' : 'closer without'} expected pickups (average miss ${fmt(o.mae_with)} with, ${fmt(o.mae_without)} without).</p>
      <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="l">Forecast made</th><th scope="col">Miss, with pickups</th><th scope="col">Miss, without</th>
        <th scope="col" class="opt">Bias with</th><th scope="col" class="opt">Bias without</th><th scope="col">Brier with</th><th scope="col">Brier without</th></tr></thead><tbody>${
        PK.by_day.map(r => `<tr><td class="l">${esc(day(r.day))}<span class="sub">${r.team_weeks} team-weeks</span></td>
          <td class="${r.mae_with <= r.mae_without ? 'big' : ''}">${fmt(r.mae_with)}</td><td class="${r.mae_without < r.mae_with ? 'big' : ''}">${fmt(r.mae_without)}</td>
          <td class="opt">${signed(r.bias_with,0)}</td><td class="opt">${signed(r.bias_without,0)}</td><td>${fmt(r.brier_with,3)}</td><td>${fmt(r.brier_without,3)}</td></tr>`).join('')
      }</tbody></table></div><p class="note">Miss is the average gap between projected and actual weekly totals; bias is projected minus actual. Both versions here are before the learned corrections.</p>`;
  }
  const lvl = cal.bias || 0, spr = cal.spread || 1;
  html += `<h2 class="subhead">What the forecast has learned</h2>`;
  html += cal.active
    ? `<p class="lede">From ${fmt(cal.team_weeks)} checked team-weeks, the simulation now ${Math.abs(lvl) < .002 ? 'keeps its projected totals as they are' : `${lvl > 0 ? 'raises' : 'lowers'} projected remaining points by ${fmt(Math.abs(lvl)*100,1)}%`} and ${Math.abs(spr-1) < .01 ? 'keeps its ranges as they are' : `makes its ranges ${spr > 1 ? 'wider' : 'narrower'} (×${fmt(spr,2)})`}. Both corrections are deliberately cautious: they move only part of the way toward what the results suggest, and grow as more weeks are checked.</p>`
    : `<p class="lede">No corrections yet: ${esc(cal.reason || 'not enough finished weeks')}. Once enough results are in, the simulation will adjust the level and width of its weekly projections if they've been consistently off.</p>`;
  if ((C.player_biggest_misses || []).length) html += `<h2 class="subhead">Biggest player surprises</h2>
    <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="l">Player</th><th scope="col" class="l opt">Date</th>
      <th scope="col">Projected</th><th scope="col">Actual</th></tr></thead><tbody>${
      C.player_biggest_misses.map(r => `<tr><td class="l">${esc(r.player)}</td><td class="l opt">${esc(dateLabel(r.date))}</td>
        <td>${fmt(r.projected,1)}</td><td class="big">${fmt(r.actual,1)}</td></tr>`).join('')
    }</tbody></table></div>`;
  return html;
}

function renderRosters(){
  const names = Object.keys(D.rosters).sort((a,b) => (b===mine) - (a===mine) || a.localeCompare(b));
  return `<h2>Rosters</h2><p class="lede">Each player's projected fantasy points while in the active lineup for the rest of this week.</p>` +
    names.map(n => {
      const ps = D.rosters[n];
      const hurt = ps.filter(p => statusChip(p.status)).length;
      return `<details${n===mine?' open':''}><summary><span>${esc(n)}</span><span class="meta">${ps.length} players${hurt ? `, ${hurt} with an injury designation` : ''}</span></summary>
        <div class="tablewrap"><table class="num"><thead><tr><th scope="col">Player</th><th scope="col">NBA</th><th scope="col">Games left</th>
        <th scope="col">Projected points</th><th scope="col" class="opt">80% range</th></tr></thead><tbody>${
          ps.map(p => `<tr><td>${esc(p.name)}${statusChip(p.status)}${p.slot==='IR' ? '<span class="chip ir">IR</span>' : ''}${p.drop ? '<span class="chip">Expected drop</span>' : ''}</td>
            <td>${esc(p.nba)}</td><td>${p.games}</td><td class="big">${fmt(p.proj)}</td><td class="opt">${fmt(p.lo)} to ${fmt(p.hi)}</td></tr>`).join('')
        }</tbody></table></div></details>`;
    }).join('');
}

const who = (team, manager) => manager ? `${esc(manager)}<span class="sub">${esc(team)}</span>` : esc(team);
let pppSort = {key:'season', dir:-1};
function renderStats(){
  const S = D.stats || {};
  if (!(S.ppp || []).length) return `<h2>PPP and records</h2><p class="empty">Season stats appear after the ESPN download has collected finished days.</p>`;
  const rows = [...S.ppp].sort((a,b) => {
    const k = pppSort.key;
    if (k === 'name') return pppSort.dir * (a.manager || a.team).localeCompare(b.manager || b.team) * -1;
    return pppSort.dir * (((a[k] ?? -1) - (b[k] ?? -1)));
  });
  const th = (k, label) => `<th scope="col" class="sort${k==='name'?' l':''}" data-sort="${k}" tabindex="0"
      aria-sort="${pppSort.key===k ? (pppSort.dir < 0 ? 'descending' : 'ascending') : 'none'}">${label}</th>`;
  const best = k => Math.max(...S.ppp.map(r => r[k] ?? -Infinity));
  const cell = (v, n, k) => `<td class="${v !== null && v === best(k) ? 'big' : ''}">${fmt(v,1)}<span class="sub opt">${n} starts</span></td>`;
  const preWeek = S.ppp.every(r => !r.weekStarts) && S.ppp.some(r => r.lastWeekStarts);
  const wk = preWeek ? 'lastWeek' : 'week';
  let html = `<h2>Points per player</h2>
    <p class="lede">Fantasy points divided by games played by players in the active lineup, from ESPN's actual scores. ${preWeek ? '' : '"Last 3 weeks" includes this week so far.'}${preWeek ? " This week's games haven't started yet, so the first column shows last week and \"Last 3 weeks\" is the three weeks before this one." : ''}</p>
    <div class="tablewrap"><table class="num tight" id="ppp-table"><thead><tr>${th('name','Manager')}${th(wk, preWeek ? 'Last week' : 'This week')}${th('last3','Last 3 weeks')}${th('season','Season')}</tr></thead><tbody>${
      rows.map(r => `<tr class="${r.team===mine?'mine':''}"><td class="l wrapcell">${who(r.team, r.manager)}</td>
        ${cell(r[wk], r[wk + 'Starts'], wk)}${cell(r.last3, r.last3Starts, 'last3')}${cell(r.season, r.seasonStarts, 'season')}</tr>`).join('')
    }</tbody></table></div>
    <p class="note">The best value in each column is in bold. Click a column heading to sort.</p>`;
  const recs = S.records || [];
  html += `<h2 style="margin-top:28px">League records</h2>
    <p class="lede">From every finished day this season.${S.weeksDone ? '' : ' Weekly records start once the first full week is finished.'}</p>`;
  html += recs.length ? `<div class="records">${recs.map(r => `<div class="rec">
      <div class="lab">${esc(r.label)}</div>
      <div class="val num">${fmt(r.value,1)}<small>${esc(r.unit)}</small></div>
      <div class="who">${r.player ? esc(r.player) : esc(r.manager || r.team)}</div>
      <div class="when">${r.player ? esc(r.manager ? `${r.manager}, ${r.team}` : r.team) + '. ' : (r.manager ? esc(r.team) + '. ' : '')}${esc(dateLabel(r.when))}${r.note ? '. ' + esc(r.note) : ''}</div>
    </div>`).join('')}</div>` : '';
  if ((S.topGames || []).length) {
    html += `<h2>Highest-scoring games</h2><p class="lede">Single games by players in a fantasy lineup.</p>
      <div class="tablewrap"><table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="l">Player</th>
        <th scope="col" class="l">Manager</th><th scope="col" class="l opt">Date</th><th scope="col">Points</th></tr></thead><tbody>${
        S.topGames.map((g,i) => `<tr class="${g.team===mine?'mine':''}"><td class="rank">${i+1}</td><td class="l">${esc(g.player)}</td>
          <td class="l">${who(g.team, g.manager)}</td><td class="l opt">${esc(dateLabel(g.when))}</td><td class="big">${fmt(g.points,1)}</td></tr>`).join('')
      }</tbody></table></div>`;
  }
  return html;
}
function bindStats(){
  document.querySelectorAll('#ppp-table th.sort').forEach(h => {
    const go = () => {
      const k = h.dataset.sort;
      pppSort = {key:k, dir: pppSort.key === k ? -pppSort.dir : -1};
      document.getElementById('panel-stats').innerHTML = renderStats();
      bindStats();
    };
    h.addEventListener('click', go);
    h.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  });
}

function renderStandings(){
  const el = $('#standings');
  if (!D.standings.length) { el.innerHTML = `<p class="lede">Standings appear after the ESPN matchup download runs.</p>`; return; }
  const po = D.standings.some(s => s.playoff !== null && s.playoff !== undefined);
  el.innerHTML = `<table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="team">Team</th><th scope="col">W-L</th><th scope="col">${po ? 'Week' : 'This week'}</th>${po ? '<th scope="col">Playoffs</th>' : ''}</tr></thead><tbody>${
    D.standings.map(s => `<tr class="${s.name===mine?'mine':''}"><td class="rank">${s.rank}</td>
      <td class="team">${esc(s.name)}<span class="sub">${fmt(s.pf)} PF</span></td>
      <td>${s.w}-${s.l}${s.t ? '-' + s.t : ''}</td><td>${pct(s.win)}</td>${po ? `<td>${pct(s.playoff)}</td>` : ''}</tr>`).join('')
  }</tbody></table>`;
}

function renderAll(){
  PANELS.forEach(p => { document.getElementById(`panel-${p.id}`).innerHTML = p.render(); });
  document.querySelectorAll('#panel-players .seg button').forEach(b => b.addEventListener('click', () => {
    playerFilter = b.dataset.filter;
    document.getElementById('panel-players').innerHTML = renderTopPlayers();
    renderAllFilters();
  }));
  bindStats();
  bindHistory();
  renderStandings();
}
function renderAllFilters(){
  document.querySelectorAll('#panel-players .seg button').forEach(b => b.addEventListener('click', () => {
    playerFilter = b.dataset.filter;
    document.getElementById('panel-players').innerHTML = renderTopPlayers();
    renderAllFilters();
  }));
}

header();
buildTabs();
renderAll();
</script>
</body>
</html>
"""


def main():
    payload = load_payload()
    os.makedirs(SITE_DIR, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__TITLE__", payload["title"]).replace("__DATA__", data)
    out = os.path.join(SITE_DIR, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Dashboard written: {out} ({len(payload['matchups'])} matchups, {len(payload['teams'])} teams, "
          f"{len(payload['freeAgents'])} free agents, {len(payload['plan'])} expected moves)")


if __name__ == "__main__":
    main()
