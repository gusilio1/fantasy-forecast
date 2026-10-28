"""
injury_rotation_v3_features.py

Shared, leakage-safe feature construction for Injury Rotation V3.

Used by BOTH:
    train_injury_rotation_v3.py          (historical training / backtest)
    weekly_matchup_simulator_v3_0.py     (live weekly simulation)

so the features the model is trained on are built by exactly the same code
that builds them at prediction time.

KEY FIXES VERSUS V2
-------------------
1. Leak-free history cutoff.
   V2 compared calendar-day dates from player_game_model_history.csv
   ("2024-10-22" -> midnight UTC) with tip-off timestamps from
   player_availability_all.csv ("2024-10-22T23:30Z"). Midnight is always
   earlier than tip-off, so every game's OWN minutes entered its "pre-game"
   features (recent1_minutes == actual minutes in 91% of played rows).
   V3 maps every game_id to one calendar game day and uses strictly earlier
   game days only.

2. Healthy-state minutes.
   A player's history is "deflated" for the teammate minutes that were missing
   when he played:   healthy-equivalent minutes = minutes * (240 - a*L) / 240
   where L = expected minutes of health-absent teammates in that game. This is
   the exact inverse of the proportional 240-minute baseline, so injury-boosted
   minutes are not counted twice when the injured player returns. The strength
   `a` is chosen on training data only (see train_injury_rotation_v3.py).

3. Multi-absence features.
   Every health-absent teammate contributes (position-family minutes lost,
   minute-weighted shot/assist/rebound/block volume lost, starter minutes lost),
   not only the single highest-minute "primary" absence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_MINUTES = 240.0
IMPORTANT_MINUTES = 20.0
MIN_HISTORY_GAMES = 5
RECENT_N = 5
MIN_DEFLATION_FACTOR = 0.60

HEALTH_CATEGORIES = {"INJURY", "REST"}

RATE_COLS = {
    "fga": "fga",
    "ast": "assists",
    "dreb": "defensive_rebounds",
    "blk": "blocks",
}

# Original V2 feature list (kept so V2 can be re-evaluated on clean features).
V2_FEATURES = [
    "baseline_pred", "exp_minutes", "recent1_minutes", "recent3_minutes",
    "recent5_minutes", "minutes_trend", "team_minutes_rank", "starter_rate5",
    "played_rate5", "same_family", "adjacent_family", "cross_guard_big",
    "important_minutes_lost", "total_health_minutes_lost",
    "important_absent_count", "health_absent_count", "primary_absent_minutes",
    "primary_absent_fga_rate", "primary_absent_ast_rate",
    "primary_absent_dreb_rate", "primary_absent_blk_rate", "player_fga_rate",
    "player_ast_rate", "player_dreb_rate", "player_blk_rate", "is_guard",
    "is_wing", "is_big", "absent_is_guard", "absent_is_wing", "absent_is_big",
]

# V3 = V2 structure on healthy baselines + every-absence features.
V3_EXTRA_FEATURES = [
    "healthy_exp_minutes",
    "history_injury_boost",
    "lost_guard_minutes",
    "lost_wing_minutes",
    "lost_big_minutes",
    "same_family_lost_minutes",
    "adjacent_family_lost_minutes",
    "cross_family_lost_minutes",
    "second_absent_minutes",
    "lost_starter_minutes",
    "lost_fga_volume",
    "lost_ast_volume",
    "lost_dreb_volume",
    "lost_blk_volume",
    "fga_x_lost_fga",
    "ast_x_lost_ast",
    "dreb_x_lost_dreb",
    "blk_x_lost_blk",
    "baseline_share_x_lost",
]
V3_FEATURES = V2_FEATURES + V3_EXTRA_FEATURES

# Return-from-absence features (player's own layoff + teammates coming back).
RETURN_FEATURES = [
    "own_missed_streak",
    "own_log_missed_streak",
    "own_games_since_return",
    "own_returning_flag",
    "returning_teammate_minutes",
    "returning_teammate_count",
]
V3R_FEATURES = V3_FEATURES + RETURN_FEATURES
MAX_STREAK = 30
MAX_SINCE_RETURN = 10


def missed_streak_features(avail: pd.DataFrame) -> pd.DataFrame:
    """
    For each availability row, using STRICTLY EARLIER team games only:
        own_missed_streak       consecutive health absences immediately before
        own_games_since_return  games played since the last >=2-game absence
                                (MAX_SINCE_RETURN if none recently)
    """
    streak = np.zeros(len(avail))
    since = np.full(len(avail), float(MAX_SINCE_RETURN))
    order = avail.sort_values(["player_id", "game_day", "game_id"])
    pos_of = {idx: p for p, idx in enumerate(avail.index)}
    for _, g in order.groupby("player_id", sort=False):
        cur_streak, cur_since = 0, MAX_SINCE_RETURN
        for idx, health_abs, played in zip(g.index, g["health_absence"].to_numpy(bool),
                                           g["played_bool"].to_numpy(bool)):
            p = pos_of[idx]
            streak[p] = min(cur_streak, MAX_STREAK)
            since[p] = cur_since
            if health_abs:
                cur_streak += 1
            elif played:
                if cur_streak >= 2:
                    cur_since = 0
                else:
                    cur_since = min(cur_since + 1, MAX_SINCE_RETURN)
                cur_streak = 0
    return pd.DataFrame({"own_missed_streak": streak, "own_games_since_return": since},
                        index=avail.index)


# =============================================================================
# SMALL HELPERS
# =============================================================================

def norm_id(value) -> str:
    s = str(value).strip()
    return s[:-2] if s.endswith(".0") else s


def pos_family(pos) -> str:
    p = str(pos or "").upper().strip()
    if p in {"PG", "SG", "G"}:
        return "GUARD"
    if p in {"SF", "PF", "F", "GF", "G-F", "F-G"}:
        return "WING_FORWARD"
    if p in {"C", "FC", "F-C", "C-F"}:
        return "BIG"
    return "UNKNOWN"


def adjacent_family(a, b) -> float:
    return float({a, b} in ({"GUARD", "WING_FORWARD"}, {"WING_FORWARD", "BIG"}))


def cross_guard_big(a, b) -> float:
    return float({a, b} == {"GUARD", "BIG"})


def normalize_240(values, available_mask):
    x = np.asarray(values, dtype=float).copy()
    m = np.asarray(available_mask, dtype=bool)
    x[~m] = 0.0
    x[m] = np.maximum(np.nan_to_num(x[m], nan=0.0), 0.0)
    total = x[m].sum()
    if total > 0:
        x[m] *= TEAM_MINUTES / total
    return x


def expected_from_series(values: np.ndarray) -> float:
    """0.70 * recency-weighted last-5 + 0.30 * all-history mean (V2 formula)."""
    a = np.asarray(values, dtype=float)
    if len(a) < MIN_HISTORY_GAMES:
        return np.nan
    r = a[-RECENT_N:]
    w = np.arange(1, len(r) + 1, dtype=float)
    return float(0.70 * np.average(r, weights=w) + 0.30 * np.mean(a))


def deflation_factor(lost_minutes, alpha: float):
    f = (TEAM_MINUTES - alpha * np.asarray(lost_minutes, dtype=float)) / TEAM_MINUTES
    return np.clip(f, MIN_DEFLATION_FACTOR, 1.0)


# =============================================================================
# LOADING
# =============================================================================

def load_training_history(avail_path: str, model_path: str):
    """Load availability + played-game history with one calendar game day per game."""
    avail = pd.read_csv(avail_path, dtype={"game_id": str, "player_id": str, "team_id": str},
                        low_memory=False)
    model = pd.read_csv(model_path, dtype={"game_id": str, "player_id": str}, low_memory=False)

    for df in (avail, model):
        df["game_id"] = df["game_id"].map(norm_id)
        df["player_id"] = df["player_id"].map(norm_id)

    model["game_day"] = pd.to_datetime(model["date"], errors="coerce").dt.normalize()
    day_by_game = model.drop_duplicates("game_id").set_index("game_id")["game_day"]

    # Availability tip-offs are UTC; US evening games roll into the next UTC day.
    # Prefer the model file's calendar day; fall back to the US/Eastern date.
    tip = pd.to_datetime(avail["date"], errors="coerce", utc=True)
    fallback_day = tip.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    avail["game_day"] = avail["game_id"].map(day_by_game).fillna(fallback_day)
    avail["tipoff"] = tip

    avail["position_family"] = avail["position"].map(pos_family)
    avail["absence_category"] = avail["absence_category"].fillna("")
    avail["health_absence"] = avail["absence_category"].isin(HEALTH_CATEGORIES)

    def truthy(v):
        return str(v).strip().upper() in {"TRUE", "1", "YES"}

    avail["starter_bool"] = avail["starter"].map(truthy)
    avail["played_bool"] = avail["played"].map(truthy)
    return avail, model


# =============================================================================
# PER-PLAYER PRE-GAME FEATURES (strictly earlier game days)
# =============================================================================

class PlayerHistory:
    """Sorted played-game arrays for one player with prefix sums for fast cutoffs."""

    def __init__(self, g: pd.DataFrame):
        g = g.sort_values(["game_day", "game_id"])
        self.days = g["game_day"].to_numpy("datetime64[ns]")
        self.game_ids = g["game_id"].to_numpy(str)
        self.minutes = pd.to_numeric(g["minutes"], errors="coerce").to_numpy(float)
        self.rate_num = {}
        self.rate_den = {}
        self.rate_valid = {}
        exp_col = g["player_possession_exposure"] if "player_possession_exposure" in g.columns else None
        for key, col in RATE_COLS.items():
            if exp_col is None or col not in g.columns:
                self.rate_valid[key] = np.zeros(len(g), dtype=bool)
                self.rate_num[key] = np.zeros(len(g))
                self.rate_den[key] = np.zeros(len(g))
                continue
            num = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            den = pd.to_numeric(exp_col, errors="coerce").to_numpy(float)
            valid = np.isfinite(num) & np.isfinite(den) & (den > 0)
            self.rate_valid[key] = valid
            self.rate_num[key] = np.where(valid, num, 0.0)
            self.rate_den[key] = np.where(valid, den, 0.0)

    def prefix_len(self, cutoff_day) -> int:
        return int(np.searchsorted(self.days, np.datetime64(cutoff_day, "ns"), side="left"))

    def rate(self, key: str, k: int) -> float:
        valid = self.rate_valid[key][:k]
        if valid.sum() < MIN_HISTORY_GAMES:
            return np.nan
        num = self.rate_num[key][:k][valid]
        den = self.rate_den[key][:k][valid]
        rr = num[-RECENT_N:].sum() / den[-RECENT_N:].sum()
        cr = num.sum() / den.sum()
        return float(0.70 * rr + 0.30 * cr)


def build_player_histories(model: pd.DataFrame) -> dict:
    return {pid: PlayerHistory(g) for pid, g in model.groupby("player_id", sort=False)}


def pregame_features(avail: pd.DataFrame, histories: dict) -> pd.DataFrame:
    """V2 player features for each availability row, using strictly earlier game days."""
    out = {c: np.full(len(avail), np.nan) for c in [
        "exp_minutes", "recent1_minutes", "recent3_minutes", "recent5_minutes",
        "minutes_trend", "player_fga_rate", "player_ast_rate",
        "player_dreb_rate", "player_blk_rate", "_prefix_len",
    ]}
    pids = avail["player_id"].to_numpy(str)
    days = avail["game_day"].to_numpy("datetime64[ns]")
    for i in range(len(avail)):
        h = histories.get(pids[i])
        if h is None or np.isnat(days[i]):
            continue
        k = h.prefix_len(days[i])
        out["_prefix_len"][i] = k
        if k < MIN_HISTORY_GAMES:
            continue
        mins = h.minutes[:k]
        mins = mins[np.isfinite(mins)]
        if len(mins) < MIN_HISTORY_GAMES:
            continue
        r1 = float(np.mean(mins[-1:]))
        r5 = float(np.mean(mins[-5:]))
        out["exp_minutes"][i] = expected_from_series(mins)
        out["recent1_minutes"][i] = r1
        out["recent3_minutes"][i] = float(np.mean(mins[-3:]))
        out["recent5_minutes"][i] = r5
        out["minutes_trend"][i] = r1 - r5
        for key in RATE_COLS:
            out[f"player_{key}_rate"][i] = h.rate(key, k)
    return pd.DataFrame(out, index=avail.index)


def starter_played_rates(avail: pd.DataFrame) -> pd.DataFrame:
    """Last-5 starter / played rates from strictly earlier game days (V2 logic)."""
    starter5 = np.full(len(avail), np.nan)
    played5 = np.full(len(avail), np.nan)
    order = avail.sort_values(["player_id", "game_day", "game_id"])
    for _, g in order.groupby("player_id", sort=False):
        past_s, past_p = [], []
        for _, dg in g.groupby("game_day", sort=True):
            if len(past_s) >= MIN_HISTORY_GAMES:
                s_val = float(np.mean(past_s[-5:]))
                p_val = float(np.mean(past_p[-5:]))
                for j in dg.index:
                    pos = avail.index.get_loc(j)
                    starter5[pos] = s_val
                    played5[pos] = p_val
            past_s.extend(dg["starter_bool"].astype(float).tolist())
            past_p.extend(dg["played_bool"].astype(float).tolist())
    return pd.DataFrame({"starter_rate5": starter5, "played_rate5": played5}, index=avail.index)


# =============================================================================
# HEALTHY-STATE MINUTES
# =============================================================================

def team_game_lost_minutes(avail: pd.DataFrame, exp_col: str = "exp_minutes") -> pd.Series:
    """Expected minutes of health-absent players per (game_id, team)."""
    x = avail[avail["health_absence"]]
    return x.groupby(["game_id", "team"])[exp_col].sum(min_count=1).fillna(0.0)


def attach_lost_to_played_games(model: pd.DataFrame, avail: pd.DataFrame,
                                lost: pd.Series) -> np.ndarray:
    """Lost teammate minutes for each played-game row in the model history."""
    team_of = avail.set_index(["game_id", "player_id"])["team"]
    team_of = team_of[~team_of.index.duplicated()]
    keys = pd.MultiIndex.from_arrays([model["game_id"], model["player_id"]])
    teams = team_of.reindex(keys).to_numpy()
    lk = pd.MultiIndex.from_arrays([model["game_id"].to_numpy(), teams])
    return lost.reindex(lk).fillna(0.0).to_numpy(float)


def healthy_expected_minutes(avail: pd.DataFrame, histories: dict,
                             lost_by_model_row: dict, alpha: float) -> np.ndarray:
    """Healthy-state expected minutes per availability row (strictly earlier days)."""
    out = np.full(len(avail), np.nan)
    pids = avail["player_id"].to_numpy(str)
    days = avail["game_day"].to_numpy("datetime64[ns]")
    deflated = {
        pid: h.minutes * deflation_factor(lost_by_model_row[pid], alpha)
        for pid, h in histories.items()
    }
    for i in range(len(avail)):
        h = histories.get(pids[i])
        if h is None or np.isnat(days[i]):
            continue
        k = h.prefix_len(days[i])
        if k < MIN_HISTORY_GAMES:
            continue
        d = deflated[pids[i]][:k]
        d = d[np.isfinite(d)]
        out[i] = expected_from_series(d)
    return out


def lost_by_player_rows(model: pd.DataFrame, histories: dict, lost_array: np.ndarray) -> dict:
    """Re-key per-model-row lost minutes into each PlayerHistory's sorted order."""
    tmp = pd.DataFrame({
        "player_id": model["player_id"].to_numpy(),
        "game_id": model["game_id"].to_numpy(),
        "lost": lost_array,
    })
    by_pid = {}
    for pid, g in tmp.groupby("player_id", sort=False):
        h = histories[pid]
        m = dict(zip(g["game_id"], g["lost"]))
        by_pid[pid] = np.asarray([m.get(gid, 0.0) for gid in h.game_ids], float)
    return by_pid


# =============================================================================
# TEAM-STATE FEATURES (one roster in one game/state)
# =============================================================================

def team_state_features(r: pd.DataFrame, available: np.ndarray,
                        baseline_col: str = "baseline_pred") -> pd.DataFrame:
    """
    Add V2 + V3 team-state features to roster `r` for one availability state.

    Required columns in r:
        exp_minutes, healthy_exp_minutes, position_family, starter_rate5,
        player_{fga,ast,dreb,blk}_rate
    `available` is the health-availability mask (True = can play).
    Absent players are those with available == False.
    """
    r = r.copy()
    avail_mask = np.asarray(available, dtype=bool)
    exp = r["exp_minutes"].to_numpy(float)
    hexp = r["healthy_exp_minutes"].to_numpy(float)
    fam = r["position_family"].astype(str).to_numpy()

    absent = ~avail_mask
    important = absent & np.isfinite(exp) & (exp >= IMPORTANT_MINUTES)

    # ---- V2 primary-absence block (identical definitions to V2) ------------
    r["team_minutes_rank"] = pd.Series(exp, index=r.index).rank(method="average", ascending=False)
    r["important_minutes_lost"] = float(np.nansum(exp[important]))
    r["total_health_minutes_lost"] = float(np.nansum(exp[absent]))
    r["important_absent_count"] = float(important.sum())
    r["health_absent_count"] = float(absent.sum())

    if important.any():
        pi = np.flatnonzero(important)[np.argmax(exp[important])]
        pfam = fam[pi]
        r["primary_absent_minutes"] = float(exp[pi])
        for key in RATE_COLS:
            r[f"primary_absent_{key}_rate"] = float(r[f"player_{key}_rate"].iloc[pi])
    else:
        pfam = "NONE"
        r["primary_absent_minutes"] = 0.0
        for key in RATE_COLS:
            r[f"primary_absent_{key}_rate"] = np.nan

    r["same_family"] = (fam == pfam).astype(float)
    r["adjacent_family"] = [adjacent_family(pfam, f) for f in fam]
    r["cross_guard_big"] = [cross_guard_big(pfam, f) for f in fam]
    r["is_guard"] = (fam == "GUARD").astype(float)
    r["is_wing"] = (fam == "WING_FORWARD").astype(float)
    r["is_big"] = (fam == "BIG").astype(float)
    r["absent_is_guard"] = float(pfam == "GUARD")
    r["absent_is_wing"] = float(pfam == "WING_FORWARD")
    r["absent_is_big"] = float(pfam == "BIG")

    # ---- V3 every-absence block --------------------------------------------
    lost_h = np.where(absent & np.isfinite(hexp), hexp, 0.0)
    r["history_injury_boost"] = exp - hexp
    r["lost_guard_minutes"] = float(lost_h[fam == "GUARD"].sum())
    r["lost_wing_minutes"] = float(lost_h[fam == "WING_FORWARD"].sum())
    r["lost_big_minutes"] = float(lost_h[fam == "BIG"].sum())
    by_fam = {
        "GUARD": r["lost_guard_minutes"].iloc[0],
        "WING_FORWARD": r["lost_wing_minutes"].iloc[0],
        "BIG": r["lost_big_minutes"].iloc[0],
    }
    same = np.array([by_fam.get(f, 0.0) for f in fam])
    adj = np.array([
        sum(v for g, v in by_fam.items() if adjacent_family(f, g)) for f in fam
    ])
    cross = np.array([
        sum(v for g, v in by_fam.items() if cross_guard_big(f, g)) for f in fam
    ])
    r["same_family_lost_minutes"] = same
    r["adjacent_family_lost_minutes"] = adj
    r["cross_family_lost_minutes"] = cross

    sorted_lost = np.sort(lost_h[absent])[::-1]
    r["second_absent_minutes"] = float(sorted_lost[1]) if len(sorted_lost) > 1 else 0.0

    srate = np.nan_to_num(r["starter_rate5"].to_numpy(float), nan=0.0)
    r["lost_starter_minutes"] = float((lost_h * srate).sum())

    for key in RATE_COLS:
        prate = np.nan_to_num(r[f"player_{key}_rate"].to_numpy(float), nan=0.0)
        vol = float((lost_h * prate).sum())
        r[f"lost_{key}_volume"] = vol
        r[f"{key}_x_lost_{key}"] = r[f"player_{key}_rate"].to_numpy(float) * vol

    total_lost = float(lost_h.sum())
    r["baseline_share_x_lost"] = r[baseline_col].to_numpy(float) * total_lost / TEAM_MINUTES

    # ---- return-from-absence block (only if streak columns are present) ----
    if "own_missed_streak" in r.columns:
        st = np.nan_to_num(r["own_missed_streak"].to_numpy(float), nan=0.0)
        r["own_log_missed_streak"] = np.log1p(st)
        r["own_returning_flag"] = ((st >= 2) & avail_mask).astype(float)
        returning = (st >= 2) & avail_mask
        r["returning_teammate_minutes"] = float(np.nansum(np.where(returning, exp, 0.0)))
        r["returning_teammate_count"] = float(returning.sum())
        # A player's own return should not count as a returning teammate.
        r["returning_teammate_minutes"] -= np.where(returning, np.nan_to_num(exp), 0.0)
        r["returning_teammate_count"] -= returning.astype(float)
    return r


# =============================================================================
# MODEL APPLICATION
# =============================================================================

def predict_correction(feature_frame: pd.DataFrame, model: dict) -> np.ndarray:
    feats = model["features"]
    X = feature_frame.reindex(columns=feats).to_numpy(float)
    X = np.where(np.isfinite(X), X, model["mu"])
    Xs = (X - model["mu"]) / model["sd"]
    return model["beta"][0] + Xs @ model["beta"][1:]


def load_model(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    m = {
        "features": [str(x) for x in z["features"]],
        "mu": np.asarray(z["scaler_mean"], float),
        "sd": np.asarray(z["scaler_sd"], float),
        "beta": np.asarray(z["coefficients"], float),
        "shrink": float(z["best_shrinkage"][0]),
    }
    m["deflation_alpha"] = float(z["deflation_alpha"][0]) if "deflation_alpha" in z.files else 0.0
    m["version"] = str(z["model_version"][0]) if "model_version" in z.files else "unknown"
    return m


# =============================================================================
# PRODUCTION ROTATION PIPELINE (shared by pipeline backtest and simulator)
# =============================================================================

STAR_SOFT_CEILING = 37.0
ROTATION_SOFT_CEILING = 35.0
FRINGE_SOFT_CEILING = 28.0
MAX_EXPECTED_MINUTES = 37.0
EMERGENCY_EXPECTED_MINUTES = 44.0
PHYSICAL_MINUTES_LIMIT = 48.0


def sustainable_minutes_ceiling(baseline_minutes: float) -> float:
    """Role-aware ordinary workload ceiling (ported from V2.5)."""
    m = float(max(baseline_minutes, 0.0))
    if m >= 32.0:
        return min(STAR_SOFT_CEILING, m + 3.0)
    if m >= 28.0:
        return min(MAX_EXPECTED_MINUTES, m + 4.0)
    if m >= 20.0:
        return min(ROTATION_SOFT_CEILING, m + 7.0)
    if m >= 10.0:
        return min(30.0, m + 6.0)
    if m >= 5.0:
        return min(FRINGE_SOFT_CEILING, m + 4.0)
    return min(18.0, m + 3.0)


def role_floor(baseline_minutes: float) -> float:
    m = float(max(baseline_minutes, 0.0))
    if m >= 34.0:
        return m - 2.0
    if m >= 30.0:
        return m - 3.0
    if m >= 26.0:
        return m - 4.0
    return 0.0


def coach_state_evidence(past_games, pids, availability, expected, lookback_games=12):
    """
    Coach behaviour in recent games, weighted by how closely each game's
    availability pattern matches the simulated state (ported from V2.5).

    past_games: chronological list of {"appeared": set(pid), "minutes": [(pid, mins)]}
    Returns (observed_minutes_by_pid, meta).
    """
    games = list(past_games)[-lookback_games:]
    if not games:
        return {}, {"games_used": 0, "effective_games": 0.0, "best_match": 0.0}
    target = {pid: bool(availability.get(pid, True)) for pid in pids}
    key_pids = [pid for pid in pids if expected.get(pid, 0.0) >= 12.0]
    if len(key_pids) < 5:
        key_pids = list(pids)
    weighted = {}
    best_match = 0.0
    total_w = 0.0
    for idx, game in enumerate(games):
        appeared = game["appeared"]
        agree = denom = 0.0
        for pid in key_pids:
            imp = max(expected.get(pid, 0.0), 8.0)
            denom += imp
            if (pid in appeared) == target.get(pid, True):
                agree += imp
        sim = agree / denom if denom > 0 else 0.0
        best_match = max(best_match, sim)
        w = max(sim, 0.05) ** 6 * (0.55 + 0.45 * (idx + 1) / max(len(games), 1))
        if w < 0.015:
            continue
        total_w += w
        for pid, m in game["minutes"]:
            if target.get(pid, False) and m is not None and np.isfinite(m):
                weighted.setdefault(pid, []).append((float(m), w))
    observed = {
        pid: float(sum(m * w for m, w in v) / sum(w for _, w in v))
        for pid, v in weighted.items() if sum(w for _, w in v) > 0
    }
    player_weight = {pid: float(sum(w for _, w in v)) for pid, v in weighted.items()}
    return observed, {"games_used": len(games), "effective_games": total_w, "best_match": best_match,
                      "player_weight": player_weight}


def evidence_strength(meta) -> float:
    eff = float(meta.get("effective_games", 0.0))
    match = float(meta.get("best_match", 0.0))
    s = eff / (eff + 1.5) if eff > 0 else 0.0
    return float(np.clip(s * match ** 2, 0.0, 0.95))


def nightly_rotation_size(past_games, lookback_games=8) -> int:
    games = list(past_games)[-lookback_games:]
    counts = [sum(1 for _, m in g["minutes"] if m is not None and np.isfinite(m) and m >= 4.0)
              for g in games]
    if not counts:
        return 10
    return int(np.clip(round(float(np.median(counts))), 8, 11))


def balance_to_240(target, avail, floors, caps, prior):
    """Iteratively balance to exactly 240 within floors/caps (emergency headroom if needed)."""
    target = np.asarray(target, float).copy()
    caps = np.asarray(caps, float).copy()
    emergency = None
    for _ in range(200):
        diff = TEAM_MINUTES - float(target.sum())
        if abs(diff) < 1e-8:
            break
        if diff > 0:
            capacity = np.where(avail, np.maximum(caps - target, 0.0), 0.0)
            if capacity.sum() <= 1e-9:
                n = int(avail.sum())
                if n <= 0:
                    break
                need = TEAM_MINUTES / n
                ceiling = min(PHYSICAL_MINUTES_LIMIT,
                              max(MAX_EXPECTED_MINUTES, min(EMERGENCY_EXPECTED_MINUTES, need + 1.0)))
                if np.where(avail, ceiling, 0.0).sum() < TEAM_MINUTES - 1e-9:
                    ceiling = PHYSICAL_MINUTES_LIMIT
                caps = np.maximum(caps, np.where(avail, ceiling, 0.0))
                emergency = ceiling
                if np.where(avail, caps, 0.0).sum() < TEAM_MINUTES - 1e-9:
                    break
                continue
            w = np.where(capacity > 0, np.sqrt(np.maximum(prior, 0.25)), 0.0)
            target += np.minimum(diff * w / w.sum(), capacity)
        else:
            removable = np.where(avail, np.maximum(target - floors, 0.0), 0.0)
            if removable.sum() <= 1e-9:
                removable = np.where(avail, np.maximum(target, 0.0), 0.0)
            if removable.sum() <= 1e-9:
                break
            target -= min(-diff, removable.sum()) * removable / removable.sum()
            target = np.maximum(target, 0.0)
    return target, emergency


def predict_team_minutes(
    r: pd.DataFrame,
    available,
    model: dict,
    *,
    past_games=None,
    use_selection: bool = False,
    use_guardrails: bool = False,
    use_evidence: bool = False,
    min_minutes: float = 0.0,
    evidence_player_k: float = 0.0,
):
    """
    Whole-team expected minutes for one availability state.

    r must hold every feature column the model needs except the team-state block
    (exp_minutes, healthy_exp_minutes, recent*, rates, starter/played rates,
    position_family, own_missed_streak, own_games_since_return).

    Steps (each optional step is validated in backtest_v3_pipeline.py):
        1. proportional healthy baseline over available players
        2. V3 learned correction for this availability state
        3. [use_evidence]   blend with matching-state coach evidence
        4. [use_selection]  keep only tonight's likely rotation
        5. [use_guardrails] role floors / sustainable ceilings
        6. renormalise to exactly 240
    """
    avail = np.asarray(available, dtype=bool)
    out = np.zeros(len(r))
    meta = {"evidence_strength": 0.0, "emergency_ceiling": None, "rotation_size": int(avail.sum())}
    if avail.sum() == 0:
        return out, meta
    hexp = np.nan_to_num(r["healthy_exp_minutes"].to_numpy(float), nan=0.0)
    baseline = normalize_240(hexp, avail)
    x = r.copy()
    x["baseline_pred"] = baseline
    x = team_state_features(x, avail)
    corr = float(model["shrink"]) * predict_correction(x, model)
    pred = np.where(avail, np.maximum(baseline + corr, 0.0), 0.0)
    pred = normalize_240(pred, avail)

    pids = [norm_id(p) for p in r["player_id"]]
    observed = {}
    if (use_evidence or use_selection) and past_games is not None:
        expd = {pid: float(e) for pid, e in zip(pids, np.nan_to_num(r["exp_minutes"].to_numpy(float)))}
        availability = {pid: bool(a) for pid, a in zip(pids, avail)}
        observed, emeta = coach_state_evidence(past_games, pids, availability, expd)
        if use_evidence:
            s = evidence_strength(emeta)
            meta["evidence_strength"] = s
            pw = emeta.get("player_weight", {})
            for i, pid in enumerate(pids):
                if avail[i] and pid in observed:
                    # Per-player shrink: evidence from only one or two games of
                    # this player counts for less (k = 0 reproduces V2.5).
                    w = pw.get(pid, 0.0)
                    si = s * (w / (w + evidence_player_k) if evidence_player_k > 0 else 1.0)
                    pred[i] = (1.0 - si) * pred[i] + si * observed[pid]
            pred = normalize_240(pred, avail)

    sel = avail.copy()
    if use_selection and past_games is not None and avail.sum() > 5:
        n = min(nightly_rotation_size(past_games), int(avail.sum()))
        score = np.array([2.0 * observed.get(pid, 0.0) + pred[i] for i, pid in enumerate(pids)])
        order = [i for i in np.argsort(-score) if avail[i]]
        sel = np.zeros(len(r), dtype=bool)
        sel[order[:n]] = True
        meta["rotation_size"] = n
        pred = normalize_240(np.where(sel, pred, 0.0), sel)

    if min_minutes > 0:
        keep = sel & (pred >= min_minutes)
        if keep.sum() >= 5:
            sel = keep
            pred = normalize_240(np.where(sel, pred, 0.0), sel)

    if use_guardrails:
        floors = np.array([role_floor(h) if sel[i] else 0.0 for i, h in enumerate(hexp)])
        caps = np.array([sustainable_minutes_ceiling(h) if sel[i] else 0.0 for i, h in enumerate(hexp)])
        caps = np.maximum(caps, floors)
        target = np.clip(pred, floors, caps)
        target = np.where(sel, target, 0.0)
        pred, emergency = balance_to_240(target, sel, floors, caps, hexp)
        meta["emergency_ceiling"] = emergency

    out = np.where(sel, pred, 0.0)
    return out, meta


# =============================================================================
# LIVE (PRODUCTION) PLAYER FRAME AS OF A CUTOFF DAY
# =============================================================================

LIVE_HEALTH_ABSENT_MIN_EXP = 12.0   # absent current-season player with exp >= this = health absence


def build_live_player_frame(avail_path, model_path, current_path, universe: pd.DataFrame,
                            cutoff_day, alpha: float, extra_positions: dict | None = None,
                            healthy_now: dict | None = None):
    """
    Pre-game features for every player in `universe` as of `cutoff_day`
    (games on or after cutoff_day are never used).

    universe: DataFrame with player_id, nba_team, expected_minutes (projection),
              one row per NBA player in the simulation universe.

    healthy_now: optional {player_id: True if currently fully healthy (ACTIVE)}.

    Returns (frame indexed by player_id, past_games_by_team, diagnostics).
    The current season has no DNP rows, so a universe player missing from a team
    box score is treated as a HEALTH absence if his PROJECTED minutes (current
    role) are >= 12, otherwise as a coach decision.

    Stale-history guard: a currently healthy player who has not appeared in any
    of his team's 2+ games this season is out of the rotation; his history
    (often from a previous team) is replaced by his projection minutes.
    """
    cutoff = pd.Timestamp(cutoff_day).normalize()
    avail, model = load_training_history(avail_path, model_path)
    avail = avail[avail["game_day"] < cutoff].copy()
    model = model[model["game_day"] < cutoff].copy()

    uni = universe.copy()
    uni["player_id"] = uni["player_id"].map(norm_id)
    uni = uni.drop_duplicates("player_id").set_index("player_id")
    proj_exp = pd.to_numeric(uni["expected_minutes"], errors="coerce").fillna(0.0)

    # ---------------- current season box scores ----------------------------
    cur = pd.read_csv(current_path, dtype={"game_id": str, "player_id": str})
    cur["player_id"] = cur["player_id"].map(norm_id)
    cur["game_id"] = cur["game_id"].map(norm_id)
    cur["game_day"] = pd.to_datetime(cur["date"], errors="coerce").dt.normalize()
    cur = cur[cur["game_day"] < cutoff].copy()
    cur["minutes"] = pd.to_numeric(cur["minutes"], errors="coerce")
    cur["team"] = cur["nba_team"].astype(str).str.upper()

    # ---------------- historical pre-game exp for avail-season deflation -----
    hist_hist = build_player_histories(model)
    pre_av = pregame_features(avail, hist_hist)
    avail = pd.concat([avail, pre_av[["exp_minutes"]]], axis=1)
    lost_hist = team_game_lost_minutes(avail, "exp_minutes")
    lost_hist_rows = attach_lost_to_played_games(model, avail, lost_hist)

    # ---------------- combined played-game history --------------------------
    cur_model = pd.DataFrame({
        "game_id": cur["game_id"], "player_id": cur["player_id"], "game_day": cur["game_day"],
        "minutes": cur["minutes"],
        "fga": cur.get("fga"), "assists": cur.get("assists"),
        "defensive_rebounds": cur.get("defensive_rebounds"), "blocks": cur.get("blocks"),
        "player_possession_exposure": np.nan,   # no possessions in current file -> rates use past seasons
    })
    combined = pd.concat([
        model[["game_id", "player_id", "game_day", "minutes", "fga", "assists",
               "defensive_rebounds", "blocks", "player_possession_exposure"]],
        cur_model,
    ], ignore_index=True)
    histories = build_player_histories(combined)

    # Live exp minutes (formula) — needed to infer current-season absences.
    live_rows = pd.DataFrame({"player_id": uni.index, "game_day": cutoff})
    live_pre = pregame_features(live_rows, histories)
    live_pre.index = uni.index
    live_exp = live_pre["exp_minutes"].fillna(proj_exp)

    # Current-season team games, roster = universe players of that team.
    roster_by_team = {t: list(g.index) for t, g in uni.groupby(uni["nba_team"].astype(str).str.upper())}
    cur_events = []          # per-player event rows for streak / played / starter rates
    cur_lost = {}            # (game_id, team) -> lost minutes for deflation
    past_games = {}
    for (gid, team), g in cur.sort_values(["game_day", "game_id"]).groupby(["game_id", "team"], sort=False):
        day = g["game_day"].iloc[0]
        appeared = set(g["player_id"])
        mins = dict(zip(g["player_id"], g["minutes"]))
        starters = set(g.sort_values("minutes", ascending=False)["player_id"].head(5))
        lost = 0.0
        for pid in roster_by_team.get(team, []):
            if pid in appeared:
                cur_events.append((pid, day, gid, False, True, pid in starters))
            else:
                health = float(proj_exp.get(pid, 0.0)) >= LIVE_HEALTH_ABSENT_MIN_EXP
                cur_events.append((pid, day, gid, health, False, False))
                if health:
                    lost += float(live_exp.get(pid, 0.0))
        cur_lost[(gid, team)] = lost
        past_games.setdefault(team, []).append({
            "game_day": day, "appeared": appeared,
            "minutes": [(p, float(m)) for p, m in mins.items() if np.isfinite(m)],
        })

    # ---------------- healthy-state minutes (deflated combined history) ------
    lost_cur_rows = np.asarray([cur_lost.get((g, t), 0.0) for g, t in zip(cur["game_id"], cur["team"])], float)
    lost_combined = np.concatenate([lost_hist_rows, lost_cur_rows])
    lost_by_pid = lost_by_player_rows(combined, histories, lost_combined)
    healthy = healthy_expected_minutes(live_rows, histories, lost_by_pid, alpha)

    # ---------------- streak / starter / played rates ------------------------
    ev_hist = avail[["player_id", "game_day", "game_id", "health_absence", "played_bool", "starter_bool"]]
    ev_cur = pd.DataFrame(cur_events, columns=["player_id", "game_day", "game_id",
                                               "health_absence", "played_bool", "starter_bool"])
    ev = pd.concat([ev_hist, ev_cur], ignore_index=True)
    ev = ev[ev["player_id"].isin(uni.index)].sort_values(["player_id", "game_day", "game_id"])
    state = {}
    for pid, g in ev.groupby("player_id", sort=False):
        streak, since = 0, MAX_SINCE_RETURN
        for h, p in zip(g["health_absence"].to_numpy(bool), g["played_bool"].to_numpy(bool)):
            if h:
                streak += 1
            elif p:
                since = 0 if streak >= 2 else min(since + 1, MAX_SINCE_RETURN)
                streak = 0
        tail = g.tail(5)
        state[pid] = {
            "own_missed_streak": float(min(streak, MAX_STREAK)),
            "own_games_since_return": float(since),
            "starter_rate5": float(tail["starter_bool"].astype(float).mean()) if len(g) >= MIN_HISTORY_GAMES else np.nan,
            "played_rate5": float(tail["played_bool"].astype(float).mean()) if len(g) >= MIN_HISTORY_GAMES else np.nan,
        }

    # ---------------- position family ----------------------------------------
    last_pos = avail.sort_values("game_day").groupby("player_id")["position"].last()
    fam = {pid: pos_family(p) for pid, p in last_pos.items()}
    for pid, p in (extra_positions or {}).items():
        if fam.get(pid, "UNKNOWN") == "UNKNOWN":
            fam[pid] = pos_family(p)

    # ---------------- assemble ----------------------------------------------
    frame = live_pre.copy()
    frame["has_history"] = frame["exp_minutes"].notna()
    frame["healthy_exp_minutes"] = healthy
    frame["projection_minutes"] = proj_exp
    # Players with < 5 career games: fall back to the projection file.
    nohist = ~frame["has_history"]
    frame.loc[nohist, "exp_minutes"] = proj_exp[nohist]
    frame.loc[nohist, "healthy_exp_minutes"] = proj_exp[nohist]
    for c in ["recent1_minutes", "recent3_minutes", "recent5_minutes"]:
        frame.loc[nohist, c] = proj_exp[nohist]
    frame.loc[nohist, "minutes_trend"] = 0.0
    frame["healthy_exp_minutes"] = frame["healthy_exp_minutes"].fillna(frame["exp_minutes"])

    # Stale-history guard (see docstring).
    team_games = {t: len(v) for t, v in past_games.items()}
    appeared_any = set(cur["player_id"])
    stale = pd.Series([
        (pid not in appeared_any)
        and team_games.get(str(uni.at[pid, "nba_team"]).upper(), 0) >= 2
        and bool((healthy_now or {}).get(pid, False))
        and bool(frame.at[pid, "has_history"])
        for pid in frame.index
    ], index=frame.index)
    for col in ["exp_minutes", "healthy_exp_minutes", "recent1_minutes", "recent3_minutes", "recent5_minutes"]:
        frame.loc[stale, col] = proj_exp[stale]
    frame.loc[stale, "minutes_trend"] = 0.0
    frame["stale_history_replaced"] = stale
    for col in ["own_missed_streak", "own_games_since_return", "starter_rate5", "played_rate5"]:
        default = {"own_missed_streak": 0.0, "own_games_since_return": float(MAX_SINCE_RETURN)}.get(col, np.nan)
        frame[col] = [state.get(pid, {}).get(col, default) for pid in frame.index]
    frame["position_family"] = [fam.get(pid, "UNKNOWN") for pid in frame.index]
    frame["nba_team"] = uni["nba_team"].astype(str).str.upper()
    frame = frame.drop(columns=["_prefix_len"], errors="ignore")

    diag = {
        "players": len(frame),
        "with_history": int(frame["has_history"].sum()),
        "projection_fallback": int(nohist.sum()),
        "stale_history_replaced": int(stale.sum()),
        "current_team_games": int(sum(len(v) for v in past_games.values())),
        "unknown_position": int((frame["position_family"] == "UNKNOWN").sum()),
    }
    return frame, past_games, diag
