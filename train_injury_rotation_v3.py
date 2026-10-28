"""
train_injury_rotation_v3.py

Leakage-safe training + backtest for Injury Rotation V3, with a like-for-like
comparison against V2 on the SAME held-out games.

Inputs (same folder):
    player_availability_all.csv
    player_game_model_history.csv
    injury_rotation_v2_model.npz        (only to benchmark the current production model)
    injury_rotation_v3_features.py

Outputs:
    injury_rotation_v3_model.npz              production model (used by the simulator)
    injury_rotation_v3_model_trainonly.npz    same model fitted on the train period only
    injury_rotation_v3_roster_features.pkl    roster features for backtest_v3_pipeline.py
    injury_rotation_v3_backtest_summary.csv
    injury_rotation_v3_backtest_by_role.csv
    injury_rotation_v3_baseline_check.csv
    injury_rotation_v3_test_results.csv
    injury_rotation_v3_coefficients.csv

Runtime: about 4-5 minutes.

Split: the first 70% of injury team-games (by date) train, the last 30% test.
Every hyper-parameter (deflation strength, ridge penalty, shrinkage, and the
choice between F and G) is chosen on the training period only.

Variants compared on identical held-out team-games:
    A  proportional baseline from expected minutes          (V2's baseline)
    B  proportional baseline from healthy-state minutes
    C  saved production V2 model applied to leak-free features
    D  V2 feature set retrained on leak-free features (injury games only)
    E  V3 features, injury games only, healthy baseline
    F  V3 + return features, ALL team-games, expected-minutes baseline
    G  V3 + return features, ALL team-games, healthy baseline   <- exported

Run backtest_v3_pipeline.py afterwards to validate the production steps that
wrap the model (coach evidence, nightly rotation cut, floors/ceilings).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import injury_rotation_v3_features as F

BASE = os.path.dirname(os.path.abspath(__file__))
AVAIL_FILE = os.path.join(BASE, "player_availability_all.csv")
MODEL_FILE = os.path.join(BASE, "player_game_model_history.csv")
V2_MODEL_FILE = os.path.join(BASE, "injury_rotation_v2_model.npz")
OUT_MODEL = os.path.join(BASE, "injury_rotation_v3_model.npz")

TRAIN_FRACTION = 0.70
RIDGE_GRID = [0.1, 1.0, 10.0, 50.0, 100.0, 300.0]
SHRINK_GRID = [0.00, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.00]
ALPHA_GRID = [0.0, 0.25, 0.50, 0.75, 1.00]
KEYCOLS = ["history_season", "game_id", "team"]


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def rmse(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(np.mean(d * d)))


def fit_scaler(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    mu[~np.isfinite(mu)] = 0.0
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    return mu, sd


def transform(X, mu, sd):
    return (np.where(np.isfinite(X), X, mu) - mu) / sd


def ridge_fit(X, y, alpha):
    X1 = np.column_stack([np.ones(len(X)), X])
    eye = np.eye(X1.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(X1.T @ X1 + alpha * eye, X1.T @ y)


def ridge_predict(X, beta):
    return beta[0] + X @ beta[1:]


def renormalize_within_pool(df, pred_col, out_col, base_col):
    """V2 evaluation rule: rescale predictions to the candidate pool's baseline total."""
    parts = []
    for _, g in df.groupby(KEYCOLS, sort=False):
        g = g.copy()
        x = np.maximum(g[pred_col].to_numpy(float), 0.0)
        tot = g[base_col].sum()
        if x.sum() > 0:
            x *= tot / x.sum()
        g[out_col] = x
        parts.append(g)
    return pd.concat(parts)


# =============================================================================
# 1. LOAD + LEAK-FREE PLAYER FEATURES
# =============================================================================
t0 = time.time()
avail, model = F.load_training_history(AVAIL_FILE, MODEL_FILE)
histories = F.build_player_histories(model)

pre = F.pregame_features(avail, histories)
rates = F.starter_played_rates(avail)
a = pd.concat([avail, pre, rates], axis=1)

actual = model[["game_id", "player_id", "minutes"]].rename(columns={"minutes": "actual_minutes"})
actual = actual.drop_duplicates(["game_id", "player_id"])
a = a.merge(actual, on=["game_id", "player_id"], how="left")
a["actual_minutes"] = a["actual_minutes"].fillna(0.0)

leak_check = a[(a["actual_minutes"] > 0) & a["recent1_minutes"].notna()]
same_game_share = float(np.isclose(leak_check["recent1_minutes"], leak_check["actual_minutes"]).mean())
print(f"[{time.time()-t0:5.0f}s] Leak-free features built. "
      f"recent1 == actual minutes in {same_game_share:.1%} of played rows "
      f"(chance-level coincidences only; V2 data: 91.4%).")

# =============================================================================
# 2. HEALTHY-STATE MINUTES FOR EACH DEFLATION STRENGTH
# =============================================================================
lost_tg = F.team_game_lost_minutes(a, "exp_minutes")
lost_model_rows = F.attach_lost_to_played_games(model, a, lost_tg)
lost_by_pid = F.lost_by_player_rows(model, histories, lost_model_rows)
healthy = {alpha: F.healthy_expected_minutes(a, histories, lost_by_pid, alpha) for alpha in ALPHA_GRID}
print(f"[{time.time()-t0:5.0f}s] Healthy-state minutes built for alpha grid {ALPHA_GRID}.")

a["important_absence"] = a["health_absence"] & a["exp_minutes"].notna() & (a["exp_minutes"] >= F.IMPORTANT_MINUTES)

# =============================================================================
# 3. CHRONOLOGICAL SPLIT (by qualifying injury team-games, as in V2)
# =============================================================================
tg = (a.groupby(KEYCOLS, sort=False)
        .agg(game_day=("game_day", "first"),
             n_exp=("exp_minutes", lambda s: int(s.notna().sum())),
             n_important=("important_absence", "sum"))
        .reset_index())
qual = tg[(tg["n_important"] > 0) & (tg["n_exp"] >= 8)].sort_values(["game_day"] + KEYCOLS)
split_day = qual["game_day"].iloc[int(len(qual) * TRAIN_FRACTION)]
print(f"Train period: game days < {split_day.date()} | test period: >= {split_day.date()}")

# =============================================================================
# 4. BASELINE CHECK ON *ALL* TEAM-GAMES (healthy vs raw expected minutes)
# =============================================================================
def baseline_rows(alpha):
    rows = []
    hcol = healthy[alpha]
    a["_h"] = hcol
    for keys, r in a.groupby(KEYCOLS, sort=False):
        r = r[r["exp_minutes"].notna()]
        if len(r) < 8:
            continue
        av = ~r["health_absence"].to_numpy(bool)
        pred = F.normalize_240(r["_h"].to_numpy(float), av)
        cand = av & ((r["actual_minutes"].to_numpy(float) > 0) | (r["exp_minutes"].to_numpy(float) >= 5))
        rows.append(pd.DataFrame({
            "game_day": r["game_day"].to_numpy()[cand],
            "player_id": r["player_id"].to_numpy()[cand],
            "history_season": keys[0], "game_id": keys[1], "team": keys[2],
            "pred": pred[cand],
            "actual": r["actual_minutes"].to_numpy(float)[cand],
        }))
    return pd.concat(rows, ignore_index=True)

alpha_scores = []
base_tables = {}
for alpha in ALPHA_GRID:
    bt = baseline_rows(alpha)
    base_tables[alpha] = bt
    tr = bt[bt["game_day"] < split_day]
    te = bt[bt["game_day"] >= split_day]
    alpha_scores.append({"alpha": alpha, "train_MAE": mae(tr["pred"], tr["actual"]),
                         "test_MAE": mae(te["pred"], te["actual"]),
                         "train_RMSE": rmse(tr["pred"], tr["actual"]),
                         "test_RMSE": rmse(te["pred"], te["actual"])})
alpha_scores = pd.DataFrame(alpha_scores)
best_alpha_defl = float(alpha_scores.sort_values("train_MAE")["alpha"].iloc[0])
print(f"[{time.time()-t0:5.0f}s] Deflation strength chosen on TRAIN period only: alpha = {best_alpha_defl}")
print(alpha_scores.round(4).to_string(index=False))

# Return games: an important player plays after >= 2 straight health absences.
seq = a.sort_values(["player_id", "game_day", "game_id"])
ret_keys = set()
for pid, g in seq.groupby("player_id", sort=False):
    streak = 0
    for _, row in g.iterrows():
        if row["health_absence"]:
            streak += 1
            continue
        if streak >= 2 and row["played_bool"] and (row["exp_minutes"] or 0) >= F.IMPORTANT_MINUTES:
            ret_keys.add((row["history_season"], row["game_id"], row["team"]))
        streak = 0

def keyed(df):
    return list(zip(df["history_season"], df["game_id"], df["team"]))

b0 = base_tables[0.0]
bh = base_tables[best_alpha_defl]
te0 = b0[b0["game_day"] >= split_day]
teh = bh[bh["game_day"] >= split_day]
r0 = te0[[k in ret_keys for k in keyed(te0)]]
rh = teh[[k in ret_keys for k in keyed(teh)]]
baseline_check = pd.DataFrame([
    {"subset": "all test team-games", "rows": len(te0),
     "expected_minutes_baseline_MAE": mae(te0["pred"], te0["actual"]),
     "healthy_baseline_MAE": mae(teh["pred"], teh["actual"])},
    {"subset": "test games where an important player RETURNS (>=2 missed)", "rows": len(r0),
     "expected_minutes_baseline_MAE": mae(r0["pred"], r0["actual"]),
     "healthy_baseline_MAE": mae(rh["pred"], rh["actual"])},
])
print("\nBASELINE CHECK (test period)")
print(baseline_check.round(3).to_string(index=False))

# =============================================================================
# 5. TEAM-GAME DATASET (ALL team-games) WITH V2 / V3 / RETURN FEATURES
# =============================================================================
a["healthy_exp_minutes"] = healthy[best_alpha_defl]
a["healthy_exp_alpha0"] = healthy[0.0]
a = pd.concat([a, F.missed_streak_features(a)], axis=1)
qual_keys = set(map(tuple, qual[KEYCOLS].to_numpy()))

parts = []
for keys, roster in a.groupby(KEYCOLS, sort=False):
    r = roster[roster["exp_minutes"].notna()].copy()
    if len(r) < 8:
        continue
    av = ~r["health_absence"].to_numpy(bool)
    cand = av & ((r["actual_minutes"].to_numpy(float) > 0) | (r["exp_minutes"].to_numpy(float) >= 5))
    if cand.sum() < 5:
        continue
    r["baseline_pred_exp"] = F.normalize_240(r["exp_minutes"].to_numpy(float), av)
    r["baseline_pred_healthy"] = F.normalize_240(r["healthy_exp_minutes"].to_numpy(float), av)

    frames = {}
    for tag, bcol, hcol in [("exp", "baseline_pred_exp", "healthy_exp_alpha0"),
                            ("hlt", "baseline_pred_healthy", "healthy_exp_minutes")]:
        x = r.copy()
        x["baseline_pred"] = x[bcol]
        x["healthy_exp_minutes"] = x[hcol]
        frames[tag] = F.team_state_features(x, av)

    c = r.loc[cand, KEYCOLS + ["game_day", "player_id", "player_name", "actual_minutes",
                               "exp_minutes", "baseline_pred_exp", "baseline_pred_healthy"]].copy()
    for f in F.V3R_FEATURES:
        c[f"exp__{f}"] = frames["exp"].loc[cand, f].to_numpy()
        c[f"hlt__{f}"] = frames["hlt"].loc[cand, f].to_numpy()
    c["qualifying"] = keys in qual_keys
    c["returning_game"] = keys in ret_keys
    c["important_absent_count"] = frames["exp"]["important_absent_count"].iloc[0]
    parts.append(c)

data = pd.concat(parts, ignore_index=True)
is_train = data["game_day"] < split_day
train_all = data[is_train].copy()
test = data[~is_train].copy()
print(f"\n[{time.time()-t0:5.0f}s] All team-games: train {train_all[KEYCOLS].drop_duplicates().shape[0]}, "
      f"test {test[KEYCOLS].drop_duplicates().shape[0]} (injury-qualifying test: "
      f"{test.loc[test.qualifying, KEYCOLS].drop_duplicates().shape[0]})")


def fit_variant(train, prefix, feats, base_col):
    """Ridge + shrinkage chosen by inner chronological validation inside `train`."""
    cols = [f"{prefix}__{f}" for f in feats]
    games = train[KEYCOLS + ["game_day"]].drop_duplicates().sort_values(["game_day"] + KEYCOLS)
    inner_day = games["game_day"].iloc[int(len(games) * 0.80)]
    fit_mask = (train["game_day"] < inner_day).to_numpy()
    y = (train["actual_minutes"] - train[base_col]).to_numpy(float)
    X_raw = train[cols].to_numpy(float)
    mu_i, sd_i = fit_scaler(X_raw[fit_mask])
    Xf, Xv = transform(X_raw[fit_mask], mu_i, sd_i), transform(X_raw[~fit_mask], mu_i, sd_i)
    base_v = train[base_col].to_numpy(float)[~fit_mask]
    act_v = train["actual_minutes"].to_numpy(float)[~fit_mask]
    best = None
    for alpha in RIDGE_GRID:
        beta = ridge_fit(Xf, y[fit_mask], alpha)
        corr = ridge_predict(Xv, beta)
        for s in SHRINK_GRID:
            score = mae(base_v + s * corr, act_v)
            if best is None or (score, alpha, s) < best:
                best = (score, alpha, s)
    score, alpha, shrink = best
    mu, sd = fit_scaler(X_raw)
    beta = ridge_fit(transform(X_raw, mu, sd), y, alpha)
    return {"prefix": prefix, "cols": cols, "features": feats, "base_col": base_col,
            "mu": mu, "sd": sd, "beta": beta, "ridge_alpha": alpha, "shrink": shrink,
            "inner_val_MAE": score}


def apply_fit(fit, df):
    X = transform(df[fit["cols"]].to_numpy(float), fit["mu"], fit["sd"])
    return fit["shrink"] * ridge_predict(X, fit["beta"])


train_inj = train_all[train_all["qualifying"]]
fits = {
    "D": fit_variant(train_inj, "exp", F.V2_FEATURES, "baseline_pred_exp"),
    "E": fit_variant(train_inj, "hlt", F.V3_FEATURES, "baseline_pred_healthy"),
    "F": fit_variant(train_all, "exp", F.V3R_FEATURES, "baseline_pred_exp"),
    "G": fit_variant(train_all, "hlt", F.V3R_FEATURES, "baseline_pred_healthy"),
}

# C: saved production V2 on leak-free features (applies only to injury games).
v2 = F.load_model(V2_MODEL_FILE)
v2_frame = test[[f"exp__{f}" for f in v2["features"]]].copy()
v2_frame.columns = v2["features"]
test["corr_C"] = np.where(test["qualifying"], v2["shrink"] * F.predict_correction(v2_frame, v2), 0.0)
test["base_C"] = test["baseline_pred_exp"]

for v, fit in fits.items():
    corr = apply_fit(fit, test)
    if v in ("D", "E"):          # injury-only models: baseline elsewhere
        corr = np.where(test["qualifying"], corr, 0.0)
    test[f"corr_{v}"] = corr
    test[f"base_{v}"] = test[fit["base_col"]]

test["pred_A"] = test["baseline_pred_exp"]
test["pred_B"] = test["baseline_pred_healthy"]
for v in ["C", "D", "E", "F", "G"]:
    test[f"raw_{v}"] = test[f"base_{v}"] + test[f"corr_{v}"]
    test = renormalize_within_pool(test, f"raw_{v}", f"pred_{v}", f"base_{v}")

LABELS = {
    "A": "A proportional, expected minutes",
    "B": "B proportional, healthy minutes",
    "C": "C CURRENT V2 (leak-free inputs)",
    "D": "D V2 retrained leak-free",
    "E": "E V3 injury-games, healthy base",
    "F": "F V3+returns all-games, exp base",
    "G": "G V3+returns all-games, healthy base",
}
SUBSETS = {
    "all test team-games": test,
    "injury games (1+ important out)": test[test["qualifying"]],
    "2+ important absences": test[test["important_absent_count"] >= 2],
    "3+ important absences": test[test["important_absent_count"] >= 3],
    "important player returning": test[test["returning_game"]],
}
rows = []
for sname, df in SUBSETS.items():
    for v, label in LABELS.items():
        rows.append({"subset": sname, "variant": label, "rows": len(df),
                     "MAE": mae(df[f"pred_{v}"], df["actual_minutes"]),
                     "RMSE": rmse(df[f"pred_{v}"], df["actual_minutes"])})
summary = pd.DataFrame(rows)

test["role_bucket"] = pd.cut(test["exp_minutes"], bins=[0, 10, 18, 26, 32, 48],
                             labels=["<10", "10-18", "18-26", "26-32", "32+"], include_lowest=True)
by_role = pd.DataFrame([
    {"role_bucket": str(rb), "variant": LABELS[v], "rows": len(g),
     "MAE": mae(g[f"pred_{v}"], g["actual_minutes"])}
    for rb, g in test.groupby("role_bucket", observed=True) for v in LABELS
])

# Pick the production variant by TRAINING-period inner validation only (F vs G).
choice = min(["F", "G"], key=lambda v: fits[v]["inner_val_MAE"])
fin = fits[choice]
print(f"Production variant chosen by inner validation (training data only): {choice} "
      f"(F {fits['F']['inner_val_MAE']:.4f} vs G {fits['G']['inner_val_MAE']:.4f})")

# Train-period-only copy of the chosen model, for leak-free pipeline backtests.
np.savez_compressed(
    os.path.join(BASE, "injury_rotation_v3_model_trainonly.npz"),
    model_version=np.asarray(["injury_rotation_v3_trainonly"], dtype=str),
    features=np.asarray(fin["features"], dtype=str),
    scaler_mean=fin["mu"], scaler_sd=fin["sd"], coefficients=fin["beta"],
    best_alpha=np.asarray([float(fin["ridge_alpha"])]),
    best_shrinkage=np.asarray([float(fin["shrink"])]),
    deflation_alpha=np.asarray([best_alpha_defl if fin["base_col"] == "baseline_pred_healthy" else 0.0]),
    split_day=np.asarray([str(split_day.date())]),
)
a.to_pickle(os.path.join(BASE, "injury_rotation_v3_roster_features.pkl"))

# =============================================================================
# 6. FINAL MODEL ON ALL TEAM-GAMES, THEN EXPORT
# =============================================================================
X_all = data[fin["cols"]].to_numpy(float)
y_all = (data["actual_minutes"] - data[fin["base_col"]]).to_numpy(float)
mu_f, sd_f = fit_scaler(X_all)
beta_f = ridge_fit(transform(X_all, mu_f, sd_f), y_all, fin["ridge_alpha"])
uses_healthy = fin["base_col"] == "baseline_pred_healthy"

np.savez_compressed(
    OUT_MODEL,
    model_version=np.asarray(["injury_rotation_v3"], dtype=str),
    features=np.asarray(fin["features"], dtype=str),
    scaler_mean=mu_f, scaler_sd=sd_f, coefficients=beta_f,
    best_alpha=np.asarray([float(fin["ridge_alpha"])]),
    best_shrinkage=np.asarray([float(fin["shrink"])]),
    deflation_alpha=np.asarray([best_alpha_defl if uses_healthy else 0.0]),
    training_rows=np.asarray([len(data)], dtype=int),
    heldout_mae=np.asarray([mae(test[f"pred_{choice}"], test["actual_minutes"])]),
)
chk = F.load_model(OUT_MODEL)
assert chk["features"] == list(fin["features"]) and np.allclose(chk["beta"], beta_f)

coef = pd.DataFrame({"feature": ["INTERCEPT"] + list(fin["features"]), "coefficient_standardized": beta_f})
coef = coef.reindex(coef["coefficient_standardized"].abs().sort_values(ascending=False).index)

summary.to_csv(os.path.join(BASE, "injury_rotation_v3_backtest_summary.csv"), index=False)
by_role.to_csv(os.path.join(BASE, "injury_rotation_v3_backtest_by_role.csv"), index=False)
pd.concat([baseline_check, alpha_scores.assign(subset="alpha grid")], ignore_index=True).to_csv(
    os.path.join(BASE, "injury_rotation_v3_baseline_check.csv"), index=False)
coef.to_csv(os.path.join(BASE, "injury_rotation_v3_coefficients.csv"), index=False)
test.drop(columns=[c for c in test.columns if c.startswith(("exp__", "hlt__"))]).to_csv(
    os.path.join(BASE, "injury_rotation_v3_test_results.csv"), index=False)

pd.set_option("display.width", 220)
print("\n" + "=" * 110)
print("INJURY ROTATION V3 — LEAK-FREE HELD-OUT COMPARISON (MAE, minutes)")
print("=" * 110)
print(summary.pivot(index="variant", columns="subset", values="MAE")
      .reindex(columns=list(SUBSETS)).round(3).to_string())
print("\nRows per subset:", {k: len(v) for k, v in SUBSETS.items()})
print("\nBY TEAMMATE ROLE (all test team-games, MAE)")
print(by_role.pivot(index="variant", columns="role_bucket", values="MAE").round(3).to_string())
for v, fit in fits.items():
    print(f"{v}: ridge={fit['ridge_alpha']} shrink={fit['shrink']} inner-val MAE={fit['inner_val_MAE']:.4f}")
print("\nTop coefficients of exported model:")
print(coef.head(15).round(3).to_string(index=False))
print(f"\nSaved: {os.path.basename(OUT_MODEL)} variant {choice} (final fit on {len(data)} rows) [{time.time()-t0:.0f}s]")
