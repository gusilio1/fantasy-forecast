"""
build_full_nba_projection_universe.py

Builds a complete NBA simulation universe, separate from the fantasy-relevant
current_player_projections_v3.csv pool.

Run AFTER:
    rebuild_current_season_history_2027.py
    build_current_player_projections_v3.py

Output:
    all_nba_player_projections_2027.csv

Design:
- Existing V3 players keep their validated projection unchanged.
- Missing current NBA roster players are added from ESPN's public team rosters.
- If a missing player has 2026-27 minutes, his expected minutes and per-exposure
  rates are estimated from those games (recent minutes weighted more heavily).
- If he has not played yet, he receives a deliberately low-minute deep-bench
  fallback based on low-minute V3 players. These rows exist primarily so the
  NBA rotation model has somewhere realistic to put deep-rotation minutes.
"""
import os, time
import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
V3_FILE = os.path.join(BASE, "current_player_projections_v3.csv")
CURRENT_FILE = os.path.join(BASE, "player_game_history_2027_current.csv")
OUT_FILE = os.path.join(BASE, "all_nba_player_projections_2027.csv")

TEAM_IDS = {
    "ATL":1,"BOS":2,"NOP":3,"CHI":4,"CLE":5,"DAL":6,"DEN":7,"DET":8,
    "GSW":9,"HOU":10,"IND":11,"LAC":12,"LAL":13,"MIA":14,"MIL":15,
    "MIN":16,"BKN":17,"NYK":18,"ORL":19,"PHI":20,"PHX":21,"POR":22,
    "SAC":23,"SAS":24,"OKC":25,"UTA":26,"WAS":27,"TOR":28,"MEM":29,"CHA":30,
}

def norm_id(x):
    s = str(x).strip()
    return s[:-2] if s.endswith('.0') else s

def fetch_roster(team, team_id):
    urls = [
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster",
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team}/roster",
    ]
    last = None
    for url in urls:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status()
            data = r.json()
            athletes = data.get("athletes", []) or []
            flat = []
            for item in athletes:
                if isinstance(item, dict) and item.get("id"):
                    flat.append(item)
                elif isinstance(item, dict):
                    for a in item.get("items", []) or []:
                        if isinstance(a, dict) and a.get("id"):
                            flat.append(a)
            if flat:
                return flat
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not fetch {team} roster: {last}")

def recent_weighted_minutes(g):
    g = g.sort_values("date")
    mins = pd.to_numeric(g["minutes"], errors="coerce").fillna(0).to_numpy(float)
    if len(mins) == 0: return 0.0
    k = min(5, len(mins))
    recent = mins[-k:]
    w = np.arange(1, k + 1, dtype=float)
    recent_mean = float(np.average(recent, weights=w))
    season_mean = float(np.mean(mins))
    return 0.70 * recent_mean + 0.30 * season_mean

def rate_per_exposure(g, col):
    mins = pd.to_numeric(g["minutes"], errors="coerce").fillna(0).sum()
    exp = mins / 48.0 * 100.0
    val = pd.to_numeric(g[col], errors="coerce").fillna(0).sum()
    return float(val / exp) if exp > 0 else np.nan

def weighted_pct(g, made, att, fallback):
    m = pd.to_numeric(g[made], errors="coerce").fillna(0).sum()
    a = pd.to_numeric(g[att], errors="coerce").fillna(0).sum()
    return float(m/a) if a > 0 else float(fallback)

for f in (V3_FILE, CURRENT_FILE):
    if not os.path.exists(f):
        raise FileNotFoundError(os.path.basename(f))

v3 = pd.read_csv(V3_FILE, dtype={"player_id":str})
v3["player_id"] = v3["player_id"].map(norm_id)
cur = pd.read_csv(CURRENT_FILE, dtype={"player_id":str})
cur["player_id"] = cur["player_id"].map(norm_id)
cur["date"] = pd.to_datetime(cur["date"], errors="coerce")

ready = v3[v3.get("projection_ready", True).astype(str).str.upper().isin({"TRUE","1"})].copy()
low = ready[(pd.to_numeric(ready["expected_minutes"], errors="coerce") >= 3) &
            (pd.to_numeric(ready["expected_minutes"], errors="coerce") <= 12)].copy()
if low.empty: low = ready.copy()

model_cols = ["fga_rate","fta_rate","ast_rate","tov_rate","stl_rate","blk_rate",
              "three_share","two_pct","three_pct","ft_pct","oreb_capture","dreb_capture"]
fallback = {c: float(pd.to_numeric(low[c], errors="coerce").median()) for c in model_cols}

roster_rows = []
for team, tid in TEAM_IDS.items():
    athletes = fetch_roster(team, tid)
    for a in athletes:
        roster_rows.append({
            "player_id": norm_id(a.get("id")),
            "player_name": a.get("displayName") or a.get("fullName") or "UNKNOWN",
            "nba_team": team,
            "injury_status": str((a.get("status") or {}).get("type") or a.get("status") or "ACTIVE").upper(),
        })
    print(f"{team}: {len(athletes)} roster players")
    time.sleep(0.03)

rosters = pd.DataFrame(roster_rows).drop_duplicates("player_id")
known = set(v3["player_id"])
rows = []

# Keep every V3-ready player who is on a current NBA roster, but use current team from roster.
for _, rr in rosters.iterrows():
    pid = rr.player_id
    if pid in known:
        x = v3[v3.player_id == pid].iloc[0].to_dict()
        x["nba_team"] = rr.nba_team
        if not str(x.get("injury_status", "")).strip() or str(x.get("injury_status")).lower() == "nan":
            x["injury_status"] = rr.injury_status
        x["universe_source"] = "FANTASY_V3"
        rows.append(x)
        continue

    g = cur[cur.player_id == pid].copy()
    if len(g):
        expected_minutes = max(2.0, min(30.0, recent_weighted_minutes(g)))
        fga = rate_per_exposure(g, "fga")
        fta = rate_per_exposure(g, "fta")
        ast = rate_per_exposure(g, "assists")
        tov = rate_per_exposure(g, "turnovers")
        stl = rate_per_exposure(g, "steals")
        blk = rate_per_exposure(g, "blocks")
        oreb = rate_per_exposure(g, "offensive_rebounds")
        dreb = rate_per_exposure(g, "defensive_rebounds")
        fga_sum = pd.to_numeric(g.fga, errors="coerce").fillna(0).sum()
        tpa_sum = pd.to_numeric(g.three_pa, errors="coerce").fillna(0).sum()
        three_share = float(tpa_sum/fga_sum) if fga_sum > 0 else fallback["three_share"]
        two_att = (pd.to_numeric(g.fga, errors="coerce").fillna(0)-pd.to_numeric(g.three_pa, errors="coerce").fillna(0)).clip(lower=0)
        two_made = (pd.to_numeric(g.fgm, errors="coerce").fillna(0)-pd.to_numeric(g.three_pm, errors="coerce").fillna(0)).clip(lower=0)
        two_pct = float(two_made.sum()/two_att.sum()) if two_att.sum()>0 else fallback["two_pct"]
        three_pct = weighted_pct(g,"three_pm","three_pa",fallback["three_pct"])
        ft_pct = weighted_pct(g,"ftm","fta",fallback["ft_pct"])
        vals = {"fga_rate":fga,"fta_rate":fta,"ast_rate":ast,"tov_rate":tov,"stl_rate":stl,"blk_rate":blk,
                "three_share":three_share,"two_pct":two_pct,"three_pct":three_pct,"ft_pct":ft_pct,
                # capture-rate reconstruction needs team miss opportunities; for deep players use conservative fallback
                "oreb_capture": fallback["oreb_capture"], "dreb_capture": fallback["dreb_capture"]}
        for c in model_cols:
            if not np.isfinite(vals[c]): vals[c] = fallback[c]
        source = "CURRENT_SEASON_DEEP"
        residuals = repr([0.0])
    else:
        expected_minutes = 4.0
        vals = fallback.copy()
        source = "DEEP_BENCH_FALLBACK"
        residuals = repr([0.0])

    rows.append({
        "player_id":pid,"player_name":rr.player_name,"nba_team":rr.nba_team,
        "injury_status":rr.injury_status,"expected_minutes":expected_minutes,
        **vals,"minutes_residuals":residuals,"projection_ready":True,
        "projection_note":"OK","prior_source":source,"rebound_source":"DEEP_FALLBACK",
        "universe_source":source,"current_games":len(g),
    })

out = pd.DataFrame(rows)
out["player_id"] = out["player_id"].map(norm_id)
out = out.drop_duplicates("player_id", keep="first").reset_index(drop=True)
out.to_csv(OUT_FILE, index=False)

print("\n" + "="*100)
print("FULL NBA SIMULATION UNIVERSE")
print("="*100)
print("Players:", len(out))
print(out["universe_source"].value_counts(dropna=False).to_string())
print("\nPlayers per team:")
print(out.groupby("nba_team").size().sort_index().to_string())
print("\nSaved:", os.path.basename(OUT_FILE))
