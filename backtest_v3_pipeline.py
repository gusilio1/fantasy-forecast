"""
backtest_v3_pipeline.py

Validates the PRODUCTION rotation pipeline steps that sit around the V3 model
(coach-evidence blend, nightly rotation cut, role floors/ceilings, minimum
minutes) on historical games the model never saw.

Requires (produced by train_injury_rotation_v3.py):
    injury_rotation_v3_model_trainonly.npz
    injury_rotation_v3_roster_features.pkl

Evaluation:
    TEST period  = game days on/after the model's training split day
    For every team-game: all players with history are the roster; health
    absences are known; coach evidence uses that team's previous 12 games.
    Scored on every available player with history (actual minutes 0 if DNP),
    and separately on the V2-style candidate pool.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import injury_rotation_v3_features as F

BASE = os.path.dirname(os.path.abspath(__file__))
KEYCOLS = ["history_season", "game_id", "team"]

model = F.load_model(os.path.join(BASE, "injury_rotation_v3_model_trainonly.npz"))
split_day = pd.Timestamp(str(np.load(os.path.join(BASE, "injury_rotation_v3_model_trainonly.npz"))["split_day"][0]))
a = pd.read_pickle(os.path.join(BASE, "injury_rotation_v3_roster_features.pkl"))

import sys
# Optional: python backtest_v3_pipeline.py 3   -> emulate early season (only 3 prior team games)
LOOKBACK = int(sys.argv[1]) if len(sys.argv) > 1 else 12
STACK = dict(use_evidence=True, use_selection=True, use_guardrails=True)
VARIANTS = {
    "P0 model only": dict(),
    "P1 + coach evidence": dict(use_evidence=True),
    "P2 + nightly rotation cut": dict(use_selection=True),
    "P3 + floors/ceilings": dict(use_guardrails=True),
    "P4 + min 2 minutes": dict(min_minutes=2.0),
    "P5 evidence + floors/ceilings": dict(use_evidence=True, use_guardrails=True),
    "P6 all (V2.5-style stack)": dict(STACK),
    "P7 stack, no evidence": dict(use_selection=True, use_guardrails=True),
    "P8 stack, per-player evidence k=1": dict(STACK, evidence_player_k=1.0),
    "P9 stack, per-player evidence k=2": dict(STACK, evidence_player_k=2.0),
    "P10 stack, per-player evidence k=4": dict(STACK, evidence_player_k=4.0),
}

t0 = time.time()
# Past games per team, chronological: who played and their minutes.
games_by_team = {}
order = a[["history_season", "game_id", "team", "game_day"]].drop_duplicates().sort_values(["game_day", "game_id"])
grp = {k: g for k, g in a.groupby(KEYCOLS, sort=False)}
for _, row in order.iterrows():
    key = (row["history_season"], row["game_id"], row["team"])
    g = grp[key]
    played = g[g["played_bool"]]
    games_by_team.setdefault(row["team"], []).append({
        "key": key, "game_day": row["game_day"],
        "appeared": set(played["player_id"]),
        "minutes": list(zip(played["player_id"], played["actual_minutes"])),
    })

rows = []
for team, games in games_by_team.items():
    for gi, game in enumerate(games):
        if game["game_day"] < split_day:
            continue
        r = grp[game["key"]]
        r = r[r["exp_minutes"].notna()].reset_index(drop=True)
        if len(r) < 8:
            continue
        av = ~r["health_absence"].to_numpy(bool)
        cand = av & ((r["actual_minutes"].to_numpy(float) > 0) | (r["exp_minutes"].to_numpy(float) >= 5))
        if cand.sum() < 5:
            continue
        past = games[max(0, gi - LOOKBACK):gi]
        base = {
            "season": game["key"][0], "game_id": game["key"][1], "team": team,
            "important_out": int((~av & (r["exp_minutes"].to_numpy(float) >= F.IMPORTANT_MINUTES)).sum()),
        }
        preds = {}
        for name, kw in VARIANTS.items():
            preds[name], _ = F.predict_team_minutes(r, av, model, past_games=past, **kw)
        for i in np.flatnonzero(av):
            row = dict(base, player_id=r["player_id"].iloc[i], actual=float(r["actual_minutes"].iloc[i]),
                       exp=float(r["exp_minutes"].iloc[i]), candidate=bool(cand[i]))
            for name in VARIANTS:
                row[name] = float(preds[name][i])
            rows.append(row)

res = pd.DataFrame(rows)
print(f"[{time.time()-t0:.0f}s] Scored {res[['season','game_id','team']].drop_duplicates().shape[0]} "
      f"held-out team-games, {len(res)} available-player rows")

def table(df, label):
    out = {"subset": label, "rows": len(df)}
    for name in VARIANTS:
        out[name] = float(np.mean(np.abs(df[name] - df["actual"])))
    return out

summary = pd.DataFrame([
    table(res, "all available players"),
    table(res[res["candidate"]], "candidate pool (played or exp>=5)"),
    table(res[res["important_out"] >= 1], "injury games, all available"),
    table(res[res["important_out"] >= 2], "2+ important out, all available"),
    table(res[res["exp"] >= 26], "players with exp >= 26"),
    table(res[res["exp"] < 10], "players with exp < 10"),
])
pd.set_option("display.width", 250)
print("\nPIPELINE BACKTEST — MAE (minutes), held-out test period")
print(summary.set_index("subset").round(3).T.to_string())
out_name = "injury_rotation_v3_pipeline_backtest.csv" if LOOKBACK == 12 else f"injury_rotation_v3_pipeline_backtest_lookback{LOOKBACK}.csv"
summary.to_csv(os.path.join(BASE, out_name), index=False)
print(f"\nLookback: {LOOKBACK} prior team games | Saved: {out_name}")
