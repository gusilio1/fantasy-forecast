"""
build_current_player_projections_v2.py

Production current-player projection builder for the 2026-27 fantasy season.

Uses:
    player_pool.csv
    player_game_history_2026.csv          -> 2025-26 NBA prior
    player_game_history_2027_current.csv  -> 2026-27 NBA games to date
    player_game_model_history.csv         -> empirical minutes residuals

Output:
    current_player_projections_v2.csv

Key V2 changes
--------------
1. Correct season architecture:
       2025-26 prior + 2026-27 current season

2. Established players:
       validated dynamic current-season weighting for minutes
       stat-specific equivalent-game prior strengths for rates

3. Sparse / rookie players:
       current NBA data is used immediately.
       ESPN projected fantasy average is retained as prior context.
       For stat composition, a comparable-player prior is built from
       established 2025-26 players with similar ESPN projected FP where possible.
       As current games accumulate, own NBA data increasingly takes over.

4. Non-three-point shooters:
       zero/near-zero 3PA share is valid.
       3P% no longer makes the entire projection fail when projected 3PA = 0.

5. Rebound capture:
       player_game_model_history.csv does not contain the opportunity columns
       used in the standalone rebound validation, so capture rates are estimated
       from historical missed-shot environments reconstructed by game/team.

This file builds HEALTHY baseline projections.
Availability + V2 injury-minute redistribution happen later.
"""

import ast
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

POOL_FILE = os.path.join(BASE, "player_pool.csv")
PRIOR_FILE = os.path.join(BASE, "player_game_history_2026.csv")
CURRENT_FILE = os.path.join(BASE, "player_game_history_2027_current.csv")
MODEL_HISTORY_FILE = os.path.join(BASE, "player_game_model_history.csv")
OUTPUT_FILE = os.path.join(BASE, "current_player_projections_v2.csv")

RATE_PRIOR_GAMES = {
    "FGA": 5,
    "FTA": 10,
    "AST": 5,
    "OREB": 15,
    "DREB": 10,
    "STL": 10,
    "BLK": 10,
    "TO": 15,
}

SHOOTING_PRIOR_GAMES = 15
ROOKIE_COMPARABLE_COUNT = 25

STAT_COLS = [
    "MIN", "PTS", "FGM", "FGA", "3PM", "3PA",
    "FTM", "FTA", "REB", "OREB", "DREB",
    "AST", "STL", "BLK", "TO", "FP",
]


def find_col(df, names):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def standardise(df):
    candidates = {
        "PLAYER_ID": ["player_id"],
        "PLAYER_NAME": ["player_name"],
        "DATE": ["date"],
        "GAME_ID": ["game_id"],
        "TEAM": ["nba_team", "team"],
        "MIN": ["minutes"],
        "PTS": ["points"],
        "FGM": ["fgm"],
        "FGA": ["fga"],
        "3PM": ["three_pm"],
        "3PA": ["three_pa"],
        "FTM": ["ftm"],
        "FTA": ["fta"],
        "REB": ["rebounds"],
        "OREB": ["offensive_rebounds"],
        "DREB": ["defensive_rebounds"],
        "AST": ["assists"],
        "STL": ["steals"],
        "BLK": ["blocks"],
        "TO": ["turnovers"],
        "FP": ["fantasy_points"],
    }

    rename = {}
    optional = {"GAME_ID", "TEAM"}

    for target, names in candidates.items():
        c = find_col(df, names)
        if c is None:
            if target in optional:
                continue
            raise ValueError(
                f"Missing required {target} column. Columns found: {list(df.columns)}"
            )
        rename[c] = target

    x = df.rename(columns=rename).copy()
    x["PLAYER_ID"] = (
        x["PLAYER_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    x["DATE"] = pd.to_datetime(x["DATE"], errors="coerce")

    for c in STAT_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    return x.dropna(subset=["PLAYER_ID", "DATE", "MIN"])


def weighted_last5(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]

    if len(a) == 0:
        return np.nan

    if len(a) < 5:
        weights = np.arange(1, len(a) + 1, dtype=float)
        return float(np.average(a, weights=weights))

    return float(np.average(a[-5:], weights=[1, 2, 2, 3, 4]))


def healthy_minutes(prior_minutes, current_df, fallback_minutes):
    """
    Validated current-season minutes architecture.

    0 games: prior/fallback
    1-2: deliberately conservative early-season blend
    3-4: ~20 prior / 40 current / 40 recent
    5+:  ~10 prior / 25 current / 65 recent
    """
    n = len(current_df)

    if n == 0:
        return fallback_minutes

    cur_mean = float(current_df["MIN"].mean())
    recent = weighted_last5(current_df["MIN"].to_numpy())

    if not np.isfinite(fallback_minutes):
        fallback_minutes = cur_mean

    if n <= 2:
        return (
            0.50 * fallback_minutes
            + 0.25 * cur_mean
            + 0.25 * recent
        )

    if n <= 4:
        return (
            0.20 * fallback_minutes
            + 0.40 * cur_mean
            + 0.40 * recent
        )

    return (
        0.10 * fallback_minutes
        + 0.25 * cur_mean
        + 0.65 * recent
    )


def add_exposure(df):
    x = df.copy()
    # Projection backtest's stable per-possession exposure basis.
    x["EXPOSURE"] = x["MIN"] / 48.0 * 100.0
    return x


def aggregate_comparable_prior(comps):
    """
    Pool comparable established-player games into a prior pseudo-player.
    """
    if comps.empty:
        return None

    result = {}
    result["MIN"] = float(comps["MIN"].mean())

    exp = comps["EXPOSURE"].sum()
    if exp <= 0:
        return None

    for stat in RATE_PRIOR_GAMES:
        result[f"{stat}_RATE"] = float(comps[stat].sum() / exp)

    result["THREE_SHARE"] = (
        float(comps["3PA"].sum() / comps["FGA"].sum())
        if comps["FGA"].sum() > 0 else 0.0
    )

    result["THREE_PCT"] = (
        float(comps["3PM"].sum() / comps["3PA"].sum())
        if comps["3PA"].sum() > 0 else 0.36
    )

    two_pa = (comps["FGA"] - comps["3PA"]).clip(lower=0)
    two_pm = (comps["FGM"] - comps["3PM"]).clip(lower=0)

    result["TWO_PCT"] = (
        float(two_pm.sum() / two_pa.sum())
        if two_pa.sum() > 0 else 0.52
    )

    result["FT_PCT"] = (
        float(comps["FTM"].sum() / comps["FTA"].sum())
        if comps["FTA"].sum() > 0 else 0.78
    )

    return result


def make_established_prior_table(prior, pool):
    """
    Player-level 2025-26 prior table, including historical fantasy average.
    """
    rows = []

    for pid, g in prior.groupby("PLAYER_ID"):
        if len(g) < 10:
            continue

        exp = g["EXPOSURE"].sum()
        if exp <= 0:
            continue

        row = {
            "PLAYER_ID": pid,
            "PLAYER_NAME": g["PLAYER_NAME"].iloc[-1],
            "GAMES": len(g),
            "MIN": float(g["MIN"].mean()),
            "FP_AVG": float(g["FP"].mean()),
        }

        for stat in RATE_PRIOR_GAMES:
            row[f"{stat}_RATE"] = float(g[stat].sum() / exp)

        row["THREE_SHARE"] = (
            float(g["3PA"].sum() / g["FGA"].sum())
            if g["FGA"].sum() > 0 else 0.0
        )

        row["THREE_PCT"] = (
            float(g["3PM"].sum() / g["3PA"].sum())
            if g["3PA"].sum() > 0 else 0.36
        )

        two_pa = (g["FGA"] - g["3PA"]).clip(lower=0)
        two_pm = (g["FGM"] - g["3PM"]).clip(lower=0)

        row["TWO_PCT"] = (
            float(two_pm.sum() / two_pa.sum())
            if two_pa.sum() > 0 else 0.52
        )

        row["FT_PCT"] = (
            float(g["FTM"].sum() / g["FTA"].sum())
            if g["FTA"].sum() > 0 else 0.78
        )

        rows.append(row)

    return pd.DataFrame(rows)


def comparable_prior_for_player(player_row, established, prior_games):
    """
    Use ESPN projected FP to choose established-player comparables.
    If ESPN projection is missing, fall back to broad league prior.
    """
    espn_fp = pd.to_numeric(
        pd.Series([player_row.get("projected_fp_avg")]),
        errors="coerce",
    ).iloc[0]

    if len(established) == 0:
        return None, "NO_COMPARABLES"

    if np.isfinite(espn_fp):
        temp = established.copy()
        temp["DIST"] = (temp["FP_AVG"] - espn_fp).abs()
        ids = temp.nsmallest(
            min(ROOKIE_COMPARABLE_COUNT, len(temp)),
            "DIST",
        )["PLAYER_ID"]

        comps = prior_games[
            prior_games["PLAYER_ID"].isin(set(ids))
        ]

        return aggregate_comparable_prior(comps), "ESPN_FP_COMPARABLES"

    return aggregate_comparable_prior(prior_games), "LEAGUE_PRIOR"


def player_prior_dict(pg):
    if pg is None or len(pg) == 0:
        return None

    exp = pg["EXPOSURE"].sum()
    if exp <= 0:
        return None

    result = {
        "MIN": float(pg["MIN"].mean()),
    }

    for stat in RATE_PRIOR_GAMES:
        result[f"{stat}_RATE"] = float(pg[stat].sum() / exp)

    result["THREE_SHARE"] = (
        float(pg["3PA"].sum() / pg["FGA"].sum())
        if pg["FGA"].sum() > 0 else 0.0
    )

    # IMPORTANT: if player essentially never shoots threes, 3P% is irrelevant.
    result["THREE_PCT"] = (
        float(pg["3PM"].sum() / pg["3PA"].sum())
        if pg["3PA"].sum() > 0 else 0.36
    )

    two_pa = (pg["FGA"] - pg["3PA"]).clip(lower=0)
    two_pm = (pg["FGM"] - pg["3PM"]).clip(lower=0)

    result["TWO_PCT"] = (
        float(two_pm.sum() / two_pa.sum())
        if two_pa.sum() > 0 else 0.52
    )

    result["FT_PCT"] = (
        float(pg["FTM"].sum() / pg["FTA"].sum())
        if pg["FTA"].sum() > 0 else 0.78
    )

    return result


def blend_rate(prior_rate, current_df, stat, strength_games, prior_min):
    if len(current_df) == 0:
        return float(prior_rate)

    current_exp = current_df["EXPOSURE"].sum()
    if current_exp <= 0:
        return float(prior_rate)

    prior_exp_per_game = max(prior_min / 48.0 * 100.0, 1e-6)
    strength = strength_games * prior_exp_per_game

    return float(
        (prior_rate * strength + current_df[stat].sum())
        / (strength + current_exp)
    )


def blend_probability(
    prior_p,
    current_successes,
    current_trials,
    strength_games,
    prior_trials_per_game,
):
    prior_p = float(np.clip(prior_p, 0.0, 1.0))
    current_successes = float(current_successes)
    current_trials = float(current_trials)

    strength = max(
        strength_games * max(prior_trials_per_game, 0.25),
        0.25,
    )

    return float(
        (prior_p * strength + current_successes)
        / (strength + current_trials)
    )


def reconstruct_rebound_capture(history):
    """
    Reconstruct missed-FG opportunities from player game rows.

    OREB opportunity:
        own team missed FGs * MIN/48
    DREB opportunity:
        opponent team missed FGs * MIN/48

    If GAME_ID/TEAM aren't available, return empty table and later use
    conservative rate-based fallback.
    """
    if "GAME_ID" not in history.columns or "TEAM" not in history.columns:
        return pd.DataFrame()

    h = history.copy()
    h["MISSED_FG"] = (h["FGA"] - h["FGM"]).clip(lower=0)

    team = (
        h.groupby(["GAME_ID", "TEAM"], as_index=False)
        .agg(team_missed_fg=("MISSED_FG", "sum"))
    )

    teams_per_game = (
        team.groupby("GAME_ID")["TEAM"]
        .apply(list)
        .to_dict()
    )

    missed_lookup = {
        (str(r.GAME_ID), str(r.TEAM)): float(r.team_missed_fg)
        for r in team.itertuples()
    }

    rows = []

    for r in h.itertuples():
        game_id = str(r.GAME_ID)
        team_name = str(r.TEAM)
        teams = teams_per_game.get(game_id, [])

        opponents = [t for t in teams if str(t) != team_name]
        if not opponents:
            continue

        opp = str(opponents[0])

        own_miss = missed_lookup.get((game_id, team_name), np.nan)
        opp_miss = missed_lookup.get((game_id, opp), np.nan)

        if not np.isfinite(own_miss) or not np.isfinite(opp_miss):
            continue

        rows.append({
            "PLAYER_ID": r.PLAYER_ID,
            "OREB": r.OREB,
            "DREB": r.DREB,
            "OREB_OPP": own_miss * r.MIN / 48.0,
            "DREB_OPP": opp_miss * r.MIN / 48.0,
        })

    if not rows:
        return pd.DataFrame()

    x = pd.DataFrame(rows)

    return (
        x.groupby("PLAYER_ID", as_index=False)
        .agg(
            OREB=("OREB", "sum"),
            DREB=("DREB", "sum"),
            OREB_OPP=("OREB_OPP", "sum"),
            DREB_OPP=("DREB_OPP", "sum"),
        )
    )


def empirical_minutes_residuals(model_history, player_id):
    g = model_history[
        model_history["player_id"] == player_id
    ].sort_values("date")

    if len(g) < 6:
        return []

    mins = pd.to_numeric(
        g["minutes"], errors="coerce"
    ).to_numpy(float)
    mins = mins[np.isfinite(mins)]

    residuals = []

    for i in range(5, len(mins)):
        expected = weighted_last5(mins[:i])
        if np.isfinite(expected):
            residuals.append(float(mins[i] - expected))

    if not residuals:
        return []

    arr = np.asarray(residuals, dtype=float)
    arr -= arr.mean()

    return [round(float(x), 4) for x in arr]


def safe(x, lo, hi, fallback):
    if x is None or not np.isfinite(x):
        return float(fallback)
    return float(np.clip(x, lo, hi))


print()
print("=" * 100)
print("BUILDING 2026-27 CURRENT PLAYER PROJECTIONS V2")
print("=" * 100)

for f in [POOL_FILE, PRIOR_FILE, CURRENT_FILE, MODEL_HISTORY_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"Required file not found: {os.path.basename(f)}"
        )

pool = pd.read_csv(POOL_FILE)
prior = add_exposure(standardise(pd.read_csv(PRIOR_FILE)))
current = add_exposure(standardise(pd.read_csv(CURRENT_FILE)))

model_history = pd.read_csv(
    MODEL_HISTORY_FILE,
    dtype={"player_id": str, "game_id": str},
)
model_history["player_id"] = (
    model_history["player_id"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)
model_history["date"] = pd.to_datetime(
    model_history["date"], errors="coerce"
)

pool["player_id"] = (
    pool["player_id"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)

prior_groups = {
    pid: g.sort_values("DATE").reset_index(drop=True)
    for pid, g in prior.groupby("PLAYER_ID")
}

current_groups = {
    pid: g.sort_values("DATE").reset_index(drop=True)
    for pid, g in current.groupby("PLAYER_ID")
}

established = make_established_prior_table(prior, pool)
rebound_capture = reconstruct_rebound_capture(prior)

rebound_lookup = {}
if len(rebound_capture):
    for r in rebound_capture.itertuples():
        rebound_lookup[str(r.PLAYER_ID)] = {
            "oreb_capture": (
                r.OREB / r.OREB_OPP
                if r.OREB_OPP > 0 else np.nan
            ),
            "dreb_capture": (
                r.DREB / r.DREB_OPP
                if r.DREB_OPP > 0 else np.nan
            ),
        }

rows = []

for _, p in pool.iterrows():
    pid = p["player_id"]

    pg = prior_groups.get(pid)
    cg = current_groups.get(pid)

    if cg is None:
        cg = current.iloc[0:0].copy()

    prior_source = "OWN_2025_26"
    prior_dict = None

    if pg is not None and len(pg) >= 10:
        prior_dict = player_prior_dict(pg)
    else:
        prior_dict, prior_source = comparable_prior_for_player(
            p, established, prior
        )

    if prior_dict is None:
        rows.append({
            "player_id": pid,
            "player_name": p.get("player_name"),
            "nba_team": p.get("nba_team"),
            "fantasy_team_id": p.get("fantasy_team_id"),
            "fantasy_team": p.get("fantasy_team"),
            "roster_status": p.get("roster_status"),
            "lineup_slot": p.get("lineup_slot"),
            "eligible_slot_ids": p.get("eligible_slot_ids"),
            "eligible_positions": p.get("eligible_positions"),
            "injury_status": p.get("injury_status"),
            "projection_ready": False,
            "projection_note": "NO_PRIOR_AVAILABLE",
        })
        continue

    fallback_min = prior_dict["MIN"]

    # For rookies/sparse players with current NBA games, own current minutes
    # begin taking over immediately.
    expected_minutes = healthy_minutes(
        prior_dict["MIN"],
        cg,
        fallback_min,
    )
    expected_minutes = safe(
        expected_minutes, 0.0, 48.0, fallback_min
    )

    rates = {}

    for stat, strength in RATE_PRIOR_GAMES.items():
        rates[stat] = blend_rate(
            prior_dict[f"{stat}_RATE"],
            cg,
            stat,
            strength,
            prior_dict["MIN"],
        )

    prior_fga_per_game = (
        prior_dict["FGA_RATE"]
        * (prior_dict["MIN"] / 48.0 * 100.0)
    )
    prior_3pa_per_game = (
        prior_fga_per_game * prior_dict["THREE_SHARE"]
    )

    cur_fga = cg["FGA"].sum()
    cur_3pa = cg["3PA"].sum()

    three_share = blend_probability(
        prior_dict["THREE_SHARE"],
        cur_3pa,
        cur_fga,
        SHOOTING_PRIOR_GAMES,
        max(prior_fga_per_game, 0.25),
    )

    # If expected 3PA volume is essentially zero, 3P% is irrelevant to the
    # simulation. Keep a harmless league-ish fallback instead of failing.
    projected_3pa_rate = rates["FGA"] * three_share

    if projected_3pa_rate < 1e-5:
        three_pct = 0.36
    else:
        three_pct = blend_probability(
            prior_dict["THREE_PCT"],
            cg["3PM"].sum(),
            cg["3PA"].sum(),
            SHOOTING_PRIOR_GAMES,
            max(prior_3pa_per_game, 0.25),
        )

    prior_two_share = 1.0 - prior_dict["THREE_SHARE"]
    prior_2pa_per_game = (
        prior_fga_per_game * prior_two_share
    )

    cur_2pa = (cg["FGA"] - cg["3PA"]).clip(lower=0)
    cur_2pm = (cg["FGM"] - cg["3PM"]).clip(lower=0)

    two_pct = blend_probability(
        prior_dict["TWO_PCT"],
        cur_2pm.sum(),
        cur_2pa.sum(),
        SHOOTING_PRIOR_GAMES,
        max(prior_2pa_per_game, 0.25),
    )

    prior_fta_per_game = (
        prior_dict["FTA_RATE"]
        * (prior_dict["MIN"] / 48.0 * 100.0)
    )

    ft_pct = blend_probability(
        prior_dict["FT_PCT"],
        cg["FTM"].sum(),
        cg["FTA"].sum(),
        SHOOTING_PRIOR_GAMES,
        max(prior_fta_per_game, 0.25),
    )

    # Rebound capture. Established players use reconstructed opportunity data.
    rc = rebound_lookup.get(pid, {})

    oreb_capture = rc.get("oreb_capture", np.nan)
    dreb_capture = rc.get("dreb_capture", np.nan)

    # Sparse-player fallback from per-exposure rate. This is temporary until
    # current-season opportunity history is sufficiently large.
    if not np.isfinite(oreb_capture):
        oreb_capture = rates["OREB"] * 2.0

    if not np.isfinite(dreb_capture):
        dreb_capture = rates["DREB"] * 2.0

    residuals = empirical_minutes_residuals(
        model_history, pid
    )

    values = [
        expected_minutes,
        rates["FGA"], rates["FTA"], rates["AST"],
        rates["TO"], rates["STL"], rates["BLK"],
        three_share, two_pct, three_pct, ft_pct,
        oreb_capture, dreb_capture,
    ]

    ready = all(np.isfinite(x) for x in values)

    rows.append({
        "player_id": pid,
        "player_name": p.get("player_name"),
        "nba_team": p.get("nba_team"),
        "fantasy_team_id": p.get("fantasy_team_id"),
        "fantasy_team": p.get("fantasy_team"),
        "roster_status": p.get("roster_status"),
        "lineup_slot": p.get("lineup_slot"),
        "eligible_slot_ids": p.get("eligible_slot_ids"),
        "eligible_positions": p.get("eligible_positions"),
        "injury_status": p.get("injury_status"),
        "espn_projected_fp_avg": p.get("projected_fp_avg"),
        "prior_source": prior_source,
        "prior_games": len(pg) if pg is not None else 0,
        "current_games": len(cg),
        "expected_minutes": expected_minutes,
        "fga_rate": rates["FGA"],
        "fta_rate": rates["FTA"],
        "ast_rate": rates["AST"],
        "tov_rate": rates["TO"],
        "stl_rate": rates["STL"],
        "blk_rate": rates["BLK"],
        "three_share": safe(three_share, 0.0, 1.0, 0.35),
        "two_pct": safe(two_pct, 0.0, 1.0, 0.52),
        "three_pct": safe(three_pct, 0.0, 1.0, 0.36),
        "ft_pct": safe(ft_pct, 0.0, 1.0, 0.78),
        "oreb_capture": safe(
            oreb_capture, 0.0, 0.95, 0.08
        ),
        "dreb_capture": safe(
            dreb_capture, 0.0, 0.95, 0.18
        ),
        "minutes_residuals": repr(residuals),
        "projection_ready": ready,
        "projection_note": "OK" if ready else "MISSING_MODEL_INPUT",
    })

out = pd.DataFrame(rows)
out.to_csv(OUTPUT_FILE, index=False)

print("Players in player_pool:", len(pool))
print("Projection rows:", len(out))
print(
    "Ready for possession simulation:",
    int(out["projection_ready"].sum())
)
print(
    "Not ready:",
    int((~out["projection_ready"]).sum())
)

print()
print("PRIOR SOURCE SUMMARY")
print("-" * 100)
print(
    out["prior_source"]
    .value_counts(dropna=False)
    .to_string()
)

print()
print("ROSTERED PLAYERS NOT READY")
print("-" * 100)

nr = out[
    (out["roster_status"] == "ROSTERED")
    & (~out["projection_ready"])
]

if len(nr):
    print(
        nr[
            [
                "fantasy_team",
                "player_name",
                "nba_team",
                "prior_source",
                "prior_games",
                "current_games",
                "projection_note",
            ]
        ].to_string(index=False)
    )
else:
    print("None.")

print()
print("ROSTERED ROOKIES / SPARSE-PRIOR PLAYERS")
print("-" * 100)

rook = out[
    (out["roster_status"] == "ROSTERED")
    & (out["prior_source"] != "OWN_2025_26")
].copy()

if len(rook):
    print(
        rook[
            [
                "fantasy_team",
                "player_name",
                "nba_team",
                "prior_source",
                "current_games",
                "expected_minutes",
                "espn_projected_fp_avg",
                "projection_ready",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
else:
    print("None.")

print()
print("FOUR PREVIOUSLY FAILED ESTABLISHED PLAYERS")
print("-" * 100)

check_names = {
    "Rudy Gobert",
    "Daniel Gafford",
    "Ivica Zubac",
    "Jalen Duren",
}

check = out[out["player_name"].isin(check_names)]

print(
    check[
        [
            "player_name",
            "expected_minutes",
            "three_share",
            "three_pct",
            "projection_ready",
            "projection_note",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print()
print("CURRENT-SEASON LEADERS IN MODEL")
print("-" * 100)

leaders = out[
    (out["roster_status"] == "ROSTERED")
    & (out["projection_ready"])
].sort_values(
    "expected_minutes",
    ascending=False,
)

print(
    leaders[
        [
            "fantasy_team",
            "player_name",
            "nba_team",
            "current_games",
            "expected_minutes",
            "prior_source",
        ]
    ]
    .head(25)
    .round(3)
    .to_string(index=False)
)

print()
print("=" * 100)
print("FILE SAVED")
print("=" * 100)
print(os.path.basename(OUTPUT_FILE))
