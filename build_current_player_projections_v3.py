"""
build_current_player_projections_v3.py

Production current-player projection builder for the 2026-27 fantasy season.

Run AFTER:
    fantasy_sim.py
    rebuild_current_season_history_2027.py

Required files:
    player_pool.csv
    player_game_history_2025.csv          -> 2024-25 NBA history
    player_game_history_2026.csv          -> 2025-26 NBA history
    player_game_history_2027_current.csv  -> 2026-27 NBA games to date
    player_game_model_history.csv
    build_current_player_projections_v2.py

Output:
    current_player_projections_v3.csv

V3 change
---------
Corrected prior hierarchy:

1. Player has >=10 games in 2025-26
       -> OWN_2025_26

2. Player has <10 games in 2025-26 but >=10 games in 2024-25
       -> OWN_2024_25_PLUS_2025_26
          Uses the player's own 2024-25 history plus any available 2025-26
          games. This prevents established NBA players with an injury-shortened
          2025-26 season from being treated like rookies.

3. Player lacks meaningful own NBA history
       -> ESPN_FP_COMPARABLES
          Same V2 rookie/new-player fallback.

Current 2026-27 games then progressively take over using the already-validated
V2 current-season weighting.

This file intentionally does NOT add opponent defensive rating to player rates.
The matchup-context V1/V2 backtests did not improve held-out MAE.
"""

import importlib.util
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

V2_FILE = os.path.join(BASE, "build_current_player_projections_v2.py")
POOL_FILE = os.path.join(BASE, "player_pool.csv")
HIST_2024_25_FILE = os.path.join(BASE, "player_game_history_2025.csv")
HIST_2025_26_FILE = os.path.join(BASE, "player_game_history_2026.csv")
CURRENT_FILE = os.path.join(BASE, "player_game_history_2027_current.csv")
MODEL_HISTORY_FILE = os.path.join(BASE, "player_game_model_history.csv")
OUTPUT_FILE = os.path.join(BASE, "current_player_projections_v3.csv")

MIN_MEANINGFUL_HISTORY_GAMES = 10


# ---------------------------------------------------------------------------
# Load V2 helper functions without running V2's production build section.
# ---------------------------------------------------------------------------

if not os.path.exists(V2_FILE):
    raise FileNotFoundError(
        "build_current_player_projections_v2.py must be in the same folder. "
        "V3 deliberately reuses its already-tested helper functions."
    )

v2_source = open(V2_FILE, "r", encoding="utf-8").read()

marker = 'print()\nprint("=" * 100)\nprint("BUILDING 2026-27 CURRENT PLAYER PROJECTIONS V2")'
cut = v2_source.find(marker)

if cut == -1:
    raise RuntimeError(
        "Could not locate the V2 build-section marker. "
        "Your V2 file may differ from the validated version."
    )

helper_source = v2_source[:cut]

helper_namespace = {
    "__file__": V2_FILE,
    "__name__": "_projection_v2_helpers",
}
exec(compile(helper_source, V2_FILE, "exec"), helper_namespace)

standardise = helper_namespace["standardise"]
add_exposure = helper_namespace["add_exposure"]
healthy_minutes = helper_namespace["healthy_minutes"]
make_established_prior_table = helper_namespace["make_established_prior_table"]
comparable_prior_for_player = helper_namespace["comparable_prior_for_player"]
player_prior_dict = helper_namespace["player_prior_dict"]
blend_rate = helper_namespace["blend_rate"]
blend_probability = helper_namespace["blend_probability"]
reconstruct_rebound_capture = helper_namespace["reconstruct_rebound_capture"]
empirical_minutes_residuals = helper_namespace["empirical_minutes_residuals"]
safe = helper_namespace["safe"]
RATE_PRIOR_GAMES = helper_namespace["RATE_PRIOR_GAMES"]
SHOOTING_PRIOR_GAMES = helper_namespace["SHOOTING_PRIOR_GAMES"]


def choose_prior(pid, player_row, hist_2024_25_groups, hist_2025_26_groups,
                 established_2025_26, hist_2025_26):
    """
    V3 prior hierarchy.

    For established players with sparse 2025-26 data, concatenate their own
    2024-25 and 2025-26 games. This is intentionally simple and transparent:
    own NBA history is preferred over generic comparables.

    Returns:
        prior_dict
        prior_source
        games_2024_25
        games_2025_26
    """
    g25 = hist_2024_25_groups.get(pid)
    g26 = hist_2025_26_groups.get(pid)

    n25 = 0 if g25 is None else len(g25)
    n26 = 0 if g26 is None else len(g26)

    # 1. Normal established-player case: most recent completed season.
    if n26 >= MIN_MEANINGFUL_HISTORY_GAMES:
        return player_prior_dict(g26), "OWN_2025_26", n25, n26

    # 2. Established player whose 2025-26 season was sparse.
    if n25 >= MIN_MEANINGFUL_HISTORY_GAMES:
        own_parts = [g25]
        if n26 > 0:
            own_parts.append(g26)

        own_history = (
            pd.concat(own_parts, ignore_index=True)
            .sort_values("DATE")
            .reset_index(drop=True)
        )

        return (
            player_prior_dict(own_history),
            "OWN_2024_25_PLUS_2025_26",
            n25,
            n26,
        )

    # 3. Not enough meaningful own NBA history -> rookie/new-player fallback.
    prior_dict, source = comparable_prior_for_player(
        player_row,
        established_2025_26,
        hist_2025_26,
    )

    return prior_dict, source, n25, n26


print()
print("=" * 100)
print("BUILDING 2026-27 CURRENT PLAYER PROJECTIONS V3")
print("=" * 100)

required = [
    POOL_FILE,
    HIST_2024_25_FILE,
    HIST_2025_26_FILE,
    CURRENT_FILE,
    MODEL_HISTORY_FILE,
    V2_FILE,
]

for f in required:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"Required file not found: {os.path.basename(f)}"
        )

pool = pd.read_csv(POOL_FILE)

hist_2024_25 = add_exposure(
    standardise(pd.read_csv(HIST_2024_25_FILE))
)
hist_2025_26 = add_exposure(
    standardise(pd.read_csv(HIST_2025_26_FILE))
)
current = add_exposure(
    standardise(pd.read_csv(CURRENT_FILE))
)

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

hist_2024_25_groups = {
    pid: g.sort_values("DATE").reset_index(drop=True)
    for pid, g in hist_2024_25.groupby("PLAYER_ID")
}

hist_2025_26_groups = {
    pid: g.sort_values("DATE").reset_index(drop=True)
    for pid, g in hist_2025_26.groupby("PLAYER_ID")
}

current_groups = {
    pid: g.sort_values("DATE").reset_index(drop=True)
    for pid, g in current.groupby("PLAYER_ID")
}

# Comparables remain based on established 2025-26 players, exactly as V2.
established_2025_26 = make_established_prior_table(
    hist_2025_26, pool
)

# Rebound capture hierarchy mirrors the prior hierarchy:
# prefer 2025-26 capture, then 2024-25 capture, then V2 rate fallback.
rc26 = reconstruct_rebound_capture(hist_2025_26)
rc25 = reconstruct_rebound_capture(hist_2024_25)


def capture_lookup(table):
    lookup = {}
    if table is None or len(table) == 0:
        return lookup

    for r in table.itertuples():
        lookup[str(r.PLAYER_ID)] = {
            "oreb_capture": (
                r.OREB / r.OREB_OPP
                if r.OREB_OPP > 0 else np.nan
            ),
            "dreb_capture": (
                r.DREB / r.DREB_OPP
                if r.DREB_OPP > 0 else np.nan
            ),
        }
    return lookup


rc26_lookup = capture_lookup(rc26)
rc25_lookup = capture_lookup(rc25)

rows = []

for _, p in pool.iterrows():
    pid = p["player_id"]
    cg = current_groups.get(pid)

    if cg is None:
        cg = current.iloc[0:0].copy()

    prior_dict, prior_source, n25, n26 = choose_prior(
        pid,
        p,
        hist_2024_25_groups,
        hist_2025_26_groups,
        established_2025_26,
        hist_2025_26,
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
            "espn_projected_fp_avg": p.get("projected_fp_avg"),
            "prior_source": prior_source,
            "prior_games_2024_25": n25,
            "prior_games_2025_26": n26,
            "current_games": len(cg),
            "projection_ready": False,
            "projection_note": "NO_PRIOR_AVAILABLE",
        })
        continue

    fallback_min = prior_dict["MIN"]

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

    # Rebound capture: use recent completed season if available;
    # otherwise the player's own 2024-25 capture before generic fallback.
    rc = rc26_lookup.get(pid)
    rebound_source = "OWN_2025_26"

    if rc is None and prior_source == "OWN_2024_25_PLUS_2025_26":
        rc = rc25_lookup.get(pid)
        rebound_source = "OWN_2024_25"

    if rc is None:
        rc = {}
        rebound_source = "RATE_FALLBACK"

    oreb_capture = rc.get("oreb_capture", np.nan)
    dreb_capture = rc.get("dreb_capture", np.nan)

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
        "prior_games_2024_25": n25,
        "prior_games_2025_26": n26,
        "current_games": len(cg),
        "rebound_source": rebound_source,
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
                "prior_games_2024_25",
                "prior_games_2025_26",
                "current_games",
                "projection_note",
            ]
        ].to_string(index=False)
    )
else:
    print("None.")

print()
print("ROSTERED PLAYERS USING OLDER OWN-NBA PRIOR")
print("-" * 100)

older = out[
    (out["roster_status"] == "ROSTERED")
    & (out["prior_source"] == "OWN_2024_25_PLUS_2025_26")
].copy()

if len(older):
    print(
        older[
            [
                "fantasy_team",
                "player_name",
                "nba_team",
                "prior_games_2024_25",
                "prior_games_2025_26",
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
print("ROSTERED PLAYERS STILL USING ESPN COMPARABLES")
print("-" * 100)

comps = out[
    (out["roster_status"] == "ROSTERED")
    & (out["prior_source"].isin(["ESPN_FP_COMPARABLES", "LEAGUE_PRIOR"]))
].copy()

if len(comps):
    print(
        comps[
            [
                "fantasy_team",
                "player_name",
                "nba_team",
                "prior_games_2024_25",
                "prior_games_2025_26",
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
print("KEY ESTABLISHED-PLAYER CHECK")
print("-" * 100)

check_names = {
    "Jayson Tatum",
    "Damian Lillard",
    "Tyrese Haliburton",
    "Fred VanVleet",
    "Kyrie Irving",
    "Walker Kessler",
    "Rudy Gobert",
    "Daniel Gafford",
    "Ivica Zubac",
    "Jalen Duren",
}

check = out[out["player_name"].isin(check_names)].copy()

if len(check):
    print(
        check[
            [
                "player_name",
                "prior_source",
                "prior_games_2024_25",
                "prior_games_2025_26",
                "current_games",
                "expected_minutes",
                "rebound_source",
                "projection_ready",
            ]
        ]
        .sort_values("player_name")
        .round(3)
        .to_string(index=False)
    )
else:
    print("None found.")

print()
print("=" * 100)
print("FILE SAVED")
print("=" * 100)
print(os.path.basename(OUTPUT_FILE))

print()
print(
    "Paste PRIOR SOURCE SUMMARY, ROSTERED PLAYERS USING OLDER OWN-NBA PRIOR, "
    "ROSTERED PLAYERS STILL USING ESPN COMPARABLES, and KEY ESTABLISHED-PLAYER CHECK."
)
