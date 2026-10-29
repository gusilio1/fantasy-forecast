"""
backtest_matchup_context.py

Does the real-life matchup help predict fantasy points? Validates the code in
matchup_context.py (the same code the simulator uses) on held-out games.

PACE     game-specific possessions from both teams' recent pace
         (vs the simulator's fixed 101.55 for every game)
DEFENSE  opponent factors per stat (FGA, FTA, AST, TOV, STL, BLK, OREB, DREB,
         3PA share, 2P%, 3P%, FT%)

Leak-free design:
  * every rate and factor uses games on strictly earlier calendar days
  * minutes are held at the actual minutes played, isolating the matchup effect
  * defense factors are residuals against each opponent player's own rates
  * shrinkage k (including "no adjustment") and decay half-life are chosen on the
    TRAIN period only; TEST = the last 40% of 2025-26 game days

Inputs:  player_game_model_history.csv
Outputs: matchup_context_settings.json  (chosen settings; read by the simulator)
         matchup_context_backtest_summary.csv, matchup_context_backtest_stats.csv
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import matchup_context as MC

BASE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(BASE, "player_game_model_history.csv")
K_GRID = [0.0, 5.0, 15.0, 40.0, 100.0, np.inf]       # inf = no adjustment
HALF_LIFE_GRID = [15.0, 40.0]
TEST_FRACTION_OF_LAST_SEASON = 0.40
ACTUAL = {"FGA": "fga", "FTA": "fta", "AST": "assists", "TOV": "turnovers", "STL": "steals",
          "BLK": "blocks", "OREB": "offensive_rebounds", "DREB": "defensive_rebounds",
          "3PA": "three_pa", "3PM": "three_pm", "FGM": "fgm", "FTM": "ftm", "PTS": "points",
          "REB": "rebounds", "FP": "fantasy_points"}

m = MC.add_player_rates(MC.load_history(HIST))
fp_check = MC.fantasy_points(m["points"], m["rebounds"], m["assists"], m["steals"], m["blocks"],
                             m["turnovers"], m["fgm"], m["fga"], m["three_pm"])
print(f"Fantasy-point formula matches stored fantasy_points: {np.isclose(fp_check, m['fantasy_points']).mean():.1%}")
tg = MC.team_game_table(m)

last_days = np.sort(m.loc[m["history_season"] == m["history_season"].max(), "day"].unique())
split_day = last_days[int(len(last_days) * (1 - TEST_FRACTION_OF_LAST_SEASON))]
print(f"Train: game days before {pd.Timestamp(split_day).date()} | Test: {pd.Timestamp(split_day).date()} onward")

pg = m[m["has_rate"]].merge(tg[["game_id", "team", "poss"]], on=["game_id", "team"], how="left")
pg["is_test"] = pg["day"] >= split_day
key = ["game_id", "player_id"]


def rates_of(frame):
    return {s: frame[f"rate_{s}"].to_numpy(float) for s in MC.STATS}


per_hl = {}
pace_rows = []
for hl in HALF_LIFE_GRID:
    st, _, _ = MC.team_states(tg, hl)
    own = st.rename(columns={c: f"own_{c}" for c in st.columns if c not in ("game_id", "team")})
    opp = st.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in st.columns if c not in ("game_id", "team")}})
    x = pg.merge(own, on=["game_id", "team"], how="left").merge(opp, on=["game_id", "opponent"], how="left")
    per_hl[hl] = x
    games = x.drop_duplicates(["game_id"])
    for k in K_GRID:
        ph = (games["own_L"] + MC.pace_term(games["own_pace_dev"], games["own_pace_w"], k)
              + MC.pace_term(games["opp_pace_dev"], games["opp_pace_w"], k)) if np.isfinite(k) \
            else pd.Series(MC.FIXED_POSSESSIONS, index=games.index)
        for split in (False, True):
            sel = games["is_test"] == split
            pace_rows.append({"half_life": hl, "k": k, "test": split,
                              "MAE": float(np.mean(np.abs(ph[sel] - games["poss"][sel])))})
pace_res = pd.DataFrame(pace_rows)
best = pace_res[~pace_res.test].sort_values("MAE").iloc[0]
pace_hl, pace_k = float(best.half_life), float(best.k)
print(f"\nPace chosen on train: half-life {pace_hl:.0f} games, k = {pace_k}")
print(pace_res.pivot_table(index=["half_life", "k"], columns="test", values="MAE")
      .rename(columns={False: "train MAE", True: "test MAE"}).round(3).to_string())

# Defense: per stat, choose (half-life, k) on train by that stat's own error.
TARGET = {"3PA_SHARE": "3PA", "2P_PCT": "FGM", "3P_PCT": "3PM", "FT_PCT": "FTM"}
def_choice = {}
for s in MC.STATS:
    best_s = None
    for hl, x in per_hl.items():
        tr = x[~x["is_test"]]
        for k in K_GRID:
            fac = {t: np.ones(len(tr)) for t in MC.STATS}
            fac[s] = MC.shrink_factor(tr[f"opp_dA_{s}"], tr[f"opp_dE_{s}"], tr["opp_def_w"], k)
            pr = MC.predict(rates_of(tr), tr["minutes"], np.full(len(tr), MC.FIXED_POSSESSIONS), fac)
            t = TARGET.get(s, s)
            e = float(np.mean(np.abs(pr[t] - tr[ACTUAL[t]].to_numpy(float))))
            if best_s is None or e < best_s[0] - 1e-12:
                best_s = (e, hl, k)
    def_choice[s] = {"half_life": best_s[1], "k": best_s[2]}
print("\nDefense chosen on train (k = inf means 'no adjustment'):")
for s, c in def_choice.items():
    print(f"  {s:<10} half-life {c['half_life']:>4.0f}  k = {c['k']}")

# Held-out comparison.
test = per_hl[pace_hl][per_hl[pace_hl]["is_test"]].reset_index(drop=True)


def poss_for(frame, use):
    if not use:
        return np.full(len(frame), MC.FIXED_POSSESSIONS)
    return (frame["own_L"] + MC.pace_term(frame["own_pace_dev"], frame["own_pace_w"], pace_k)
            + MC.pace_term(frame["opp_pace_dev"], frame["opp_pace_w"], pace_k)).to_numpy(float)


def factors_for(frame, use):
    fac = {s: np.ones(len(frame)) for s in MC.STATS}
    if not use:
        return fac
    for s, c in def_choice.items():
        if not np.isfinite(c["k"]):
            continue
        xx = frame[key].merge(per_hl[c["half_life"]][key + [f"opp_dA_{s}", f"opp_dE_{s}", "opp_def_w"]],
                              on=key, how="left")
        fac[s] = MC.shrink_factor(xx[f"opp_dA_{s}"], xx[f"opp_dE_{s}"], xx["opp_def_w"], c["k"])
    return fac


variants = {"Fixed pace, no defense (current)": (False, False), "Pace": (True, False),
            "Defense": (False, True), "Pace + defense": (True, True)}
preds = {n: MC.predict(rates_of(test), test["minutes"], poss_for(test, up), factors_for(test, ud))
         for n, (up, ud) in variants.items()}
act = test["fantasy_points"].to_numpy(float)

rng = np.random.default_rng(7)
ug, inv = np.unique(test["game_id"].to_numpy(), return_inverse=True)
base_ae = np.abs(preds["Fixed pace, no defense (current)"]["FP"] - act)
rows = []
for n, pr in preds.items():
    err = pr["FP"] - act
    d = test[["game_id", "team"]].assign(p=pr["FP"], a=act).groupby(["game_id", "team"])[["p", "a"]].sum()
    diff = base_ae - np.abs(err)
    by_game, cnt = np.bincount(inv, weights=diff), np.bincount(inv)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(ug), len(ug))
        boots.append(by_game[idx].sum() / cnt[idx].sum())
    rows.append({"variant": n, "player_games": len(err), "FP_MAE": float(np.mean(np.abs(err))),
                 "FP_RMSE": float(np.sqrt(np.mean(err ** 2))), "FP_bias": float(np.mean(err)),
                 "team_FP_MAE": float(np.mean(np.abs(d.p - d.a))),
                 "gain_95pct_CI": f"{np.percentile(boots, 2.5):+.4f} to {np.percentile(boots, 97.5):+.4f}"})
summary = pd.DataFrame(rows)
summary.insert(4, "MAE_gain_vs_current", summary["FP_MAE"].iloc[0] - summary["FP_MAE"])

per_stat = pd.DataFrame([{"stat": s, **{n: float(np.mean(np.abs(pr[s] - test[ACTUAL[s]].to_numpy(float))))
                                         for n, pr in preds.items()}}
                         for s in ["FGA", "3PA", "FTA", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FP"]])

pd.set_option("display.width", 220)
print("\n" + "=" * 110)
print("HELD-OUT TEST — player-game fantasy points (minutes held at actual)")
print("=" * 110)
print(summary.round(4).to_string(index=False))
print("\nPer-stat MAE on held-out games:")
print(per_stat.round(4).to_string(index=False))

summary.to_csv(os.path.join(BASE, "matchup_context_backtest_summary.csv"), index=False)
per_stat.to_csv(os.path.join(BASE, "matchup_context_backtest_stats.csv"), index=False)
settings = {
    "validated_on": f"held-out games from {pd.Timestamp(split_day).date()}",
    "pace": {"half_life": pace_hl, "k": pace_k if np.isfinite(pace_k) else None},
    "defense": {s: {"half_life": c["half_life"], "k": c["k"] if np.isfinite(c["k"]) else None}
                for s, c in def_choice.items()},
    "heldout": summary.drop(columns=["gain_95pct_CI"]).to_dict(orient="records"),
}
with open(os.path.join(BASE, "matchup_context_settings.json"), "w") as fh:
    json.dump(settings, fh, indent=2)
print("\nSaved: matchup_context_settings.json and backtest summaries")
