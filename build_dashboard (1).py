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
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TAG = "v1_8"
SITE_DIR = os.path.join(BASE, "site")
LEAGUE_TITLE = "League forecast"
TOP_PLAYERS = 30
TOP_FREE_AGENTS = 40
MIN_STARTS_PPP_WEEK = 15     # a week needs this many starts to count for PPP records
DAY_FULL = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
            "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}


def _p(name):
    return os.path.join(BASE, f"weekly_matchup_{TAG}_{name}")


def _read(path):
    return pd.read_csv(path, dtype={"player_id": str}) if os.path.exists(path) else pd.DataFrame()


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
        l3, l3_st = ppp(g[g["matchup_period"].between(cur - 2, cur)])
        se, se_st = ppp(g)
        rows.append({"team": name.get(tid, str(tid)), "manager": mgr.get(tid, ""),
                     "week": _num(tw), "weekStarts": tw_st, "last3": _num(l3), "last3Starts": l3_st,
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

    team_rows = [{
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
        td = tdays[tdays["day"] == next_day].sort_values("expected_fp", ascending=False)
        top_teams = [{"name": _str(r["fantasy_team"]), "proj": _num(r["expected_fp"])} for _, r in td.iterrows()]

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
    managers = {}
    if len(standings_src) and "manager" in standings_src.columns:
        managers = {_str(r["fantasy_team"]): _str(r["manager"]) for _, r in standings_src.iterrows()}

    return {
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

/* Standings rail */
aside .card{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}
aside h2{font-size:1.15rem;padding:14px 14px 4px}
aside .lede{padding:0 14px;margin-bottom:8px;font-size:.86rem}
aside table th,aside table td{padding:8px 8px;font-size:.92rem}
aside td.team{text-align:left;white-space:normal;line-height:1.25}
aside th.team{text-align:left}
td.wrapcell{white-space:normal;min-width:9rem}
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
      <p class="lede">Record before this week, and each team's chance to win this week.</p>
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
  {id:'teams', label:'Teams', render: renderTeams},
  {id:'players', label: nd ? `${day(nd)}'s top players` : 'Top players', render: renderTopPlayers},
  {id:'topteams', label: nd ? `${day(nd)}'s top teams` : 'Top teams', render: renderTopTeams},
  {id:'fa', label:'Free agents', render: renderFreeAgents},
  {id:'rosters', label:'Rosters', render: renderRosters},
  {id:'stats', label:'PPP and records', render: renderStats},
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
        <span class="v">${fmt(ta.pppToDate,1)}</span><span class="k">PPP so far</span><span class="v r">${fmt(tb.pppToDate,1)}</span>
        <span class="v">${fmt(ta.pppRest,1)}</span><span class="k">PPP rest of week</span><span class="v r">${fmt(tb.pppRest,1)}</span>
      </div>
      <p class="verdict"><b>${esc(fav)} ${pct(Math.max(aw,bw))} to win</b>, by about ${fmt(Math.abs(m.margin || 0))} points. In 80% of simulations they finish between ${side(lo)} and ${side(hi)}.</p>
    </article>`;
  };
  return `<h2>Matchups</h2><p class="lede">Win chance is the share of simulations each team finished ahead. Finished days use ESPN's actual scores.</p>
    <div class="grid">${ms.map(card).join('')}</div>
    <p class="note">PPP is points per player: fantasy points divided by games played by players in the active lineup.</p>`;
}

function renderTeams(){
  const win = {}; D.matchups.forEach(m => { win[m.a] = m.aWin; win[m.b] = m.bWin; });
  const rows = [...D.teams].sort((a,b) => (b.final||0) - (a.final||0));
  return `<h2>Teams</h2><p class="lede">Every team's projected weekly total, ranked.</p>
  <div class="tablewrap"><table class="num tight"><thead><tr>
    <th scope="col">Team</th><th scope="col">Projected</th><th scope="col" class="opt">80% range</th><th scope="col">Win chance</th>
    <th scope="col" class="opt">Points so far</th><th scope="col" class="opt">Starts so far</th><th scope="col">PPP so far</th>
    <th scope="col" class="opt">Starts left</th><th scope="col">PPP rest of week</th></tr></thead><tbody>${
    rows.map(t => `<tr class="${t.name===mine?'mine':''}"><td>${esc(t.name)}</td><td class="big">${fmt(t.final)}</td>
      <td class="opt">${fmt(t.p10)} to ${fmt(t.p90)}</td><td>${pct(win[t.name])}</td><td class="opt">${fmt(t.toDate,1)}</td><td class="opt">${fmt(t.startsToDate)}</td>
      <td>${fmt(t.pppToDate,1)}</td><td class="opt">${fmt(t.startsRest,1)}</td><td>${fmt(t.pppRest,1)}</td></tr>`).join('')
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
    <p class="lede">Each fantasy team's projected points for ${esc(day(nd))} alone, with its best lineup and expected pickups.</p>
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
  let html = `<h2>Points per player</h2>
    <p class="lede">Fantasy points divided by games played by players in the active lineup, from ESPN's actual scores. "Last 3 weeks" includes this week so far.</p>
    <div class="tablewrap"><table class="num tight" id="ppp-table"><thead><tr>${th('name','Manager')}${th('week','This week')}${th('last3','Last 3 weeks')}${th('season','Season')}</tr></thead><tbody>${
      rows.map(r => `<tr class="${r.team===mine?'mine':''}"><td class="l wrapcell">${who(r.team, r.manager)}</td>
        ${cell(r.week, r.weekStarts, 'week')}${cell(r.last3, r.last3Starts, 'last3')}${cell(r.season, r.seasonStarts, 'season')}</tr>`).join('')
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
  el.innerHTML = `<table class="num"><thead><tr><th scope="col" class="rank">#</th><th scope="col" class="team">Team</th><th scope="col">W-L</th><th scope="col">This week</th></tr></thead><tbody>${
    D.standings.map(s => `<tr class="${s.name===mine?'mine':''}"><td class="rank">${s.rank}</td>
      <td class="team">${esc(s.name)}<span class="sub">${fmt(s.pf)} points for</span></td>
      <td>${s.w}-${s.l}${s.t ? '-' + s.t : ''}</td><td>${pct(s.win)}</td></tr>`).join('')
  }</tbody></table>`;
}

function renderAll(){
  PANELS.forEach(p => { document.getElementById(`panel-${p.id}`).innerHTML = p.render(); });
  document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
    playerFilter = b.dataset.filter;
    document.getElementById('panel-players').innerHTML = renderTopPlayers();
    renderAllFilters();
  }));
  bindStats();
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
