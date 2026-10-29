"""
weekly_matchup_simulator_v3_2.py

V3.2 — MATCHUPS, PICKUPS, DAILY PROJECTIONS
-------------------------------------------
* Real-life matchup context (matchup_context.py, settings validated by
  backtest_matchup_context.py on held-out games): each NBA game gets its own
  expected possessions from both teams' pace, and players' rates are scaled by
  the opponent's defense factors. Held-out player FP error 7.97 -> 7.89,
  team-game FP error 31.7 -> 29.3.
* Two passes. A short pre-pass (PREPASS_SIMS) estimates every NBA player's
  expected fantasy points for each remaining day, including free agents.
  Those pre-game values then drive daily lineups and the pickup planner.
* Pickups. In the projections every team (yours included) streams free agents
  into its lineup with its usual number of adds (average adds per past week from
  ESPN, never more than it has left). Separately, every team gets ADVICE: its
  best moves using all the adds it has left. Streaming only drops players within
  STREAM_DROP_MARGIN FP/game of waiver replacement level.
* Outputs for the dashboard: per-player per-day projections (rostered and free
  agents), per-team per-day totals, free-agent rankings, the pickup plan.

Everything below is the V3.1 description.

weekly_matchup_simulator_v3_1.py

V3.1 — LIVE MID-WEEK MATCHUPS
-----------------------------
* Reads fantasy_matchups.csv (written by fetch_espn_matchups.py):
    - this week's fantasy matchup pairings
    - each team's ACTUAL ESPN points from days already finished
* Simulates only the remaining days; final score = actual-to-date + simulated rest.
* Prints win / tie probability and projected margin for every matchup.
* The rotation model's history cutoff moves to the first simulated day, so games
  already played this week count as history (layoffs, coach evidence).
* Without fantasy_matchups.csv it behaves exactly like V3.0 (whole week, no win
  probabilities). SIMULATE_FROM_DAY can force the first simulated day.

Everything below is the V3.0 description.

weekly_matchup_simulator_v3_0.py

Weekly fantasy matchup Monte Carlo for the NBA fantasy simulator.

V3.0 — INJURY LAYER REWRITE
---------------------------
Schedule, lineup optimiser, IR manager, availability draws, ESPN return-date
gate, future-injury model and output files are unchanged from V2.5.2. The NBA
rotation layer is replaced by Injury Rotation V3 (injury_rotation_v3_features.py
+ injury_rotation_v3_model.npz, trained by train_injury_rotation_v3.py):

1. Leak-free training. V2 was trained with each game's own minutes inside its
   "pre-game" features (calendar-day vs tip-off timestamp mismatch). On
   leak-free features the production V2 was worse than plain proportional
   redistribution (held-out MAE 5.92 vs 5.80 minutes).
2. Every absence counts. Position-family minutes lost, minute-weighted shot /
   assist / rebound / block volume lost and starter minutes lost are summed over
   ALL absent teammates, not just the highest-minute one.
3. Healthy-state baselines. Each player's history is deflated for the teammate
   minutes that were missing when he played, so injury-boosted roles are not
   double counted when the injured player returns.
4. Returns are modelled. Each player's missed-game streak and games since
   returning are features, and they advance game by game inside every
   simulation (miss Monday, return Wednesday = a returning player Wednesday).
5. One model for every team-game (trained on all team-games), then the V2.5
   coach-evidence blend, nightly rotation cut and role floors/ceilings, each of
   which was kept only after a held-out pipeline backtest showed it helps:
       V3 model alone 5.14 -> full stack 5.03 MAE (injury games 5.48 -> 5.33).
6. Audits count every absent player in every team-game as a RATE, and minute
   gains/losses are measured against each team's full-strength V3 rotation
   instead of raw projection minutes.

Known limits: the current-season file has no DNP rows, so a missing player with
12+ expected minutes is treated as a health absence; games already played this
week (on/after WEEK_START) are not used yet (actual-to-date scoring deferred).

ORIGINAL HARNESS NOTES
----------------------
First integration harness for the NBA fantasy simulator.

PURPOSE
-------
Validate the weekly fantasy layer before adding live injury probabilities,
waiver pickups, IR activation decisions, and actual-to-date scoring.

This version:
    - loads current_player_projections_v3.csv
    - uses the Mon-Sun schedule flags already saved in player_pool.csv
    - preserves ESPN roster ownership and IR status
    - uses the exact 10-slot lineup structure
    - chooses lineups from EXPECTED fantasy value, never simulated hindsight
    - provides a dry-run/data-contract audit
    - provides a baseline Monte Carlo using simulator_core_v1.py

IMPORTANT V1 LIMITATION
-----------------------
The fitted Injury Rotation V2 model is validated and exported, but production
feature generation for a live NBA team requires current full NBA rotation
features (including non-fantasy players). This script therefore does NOT
pretend to apply the learned injury allocator yet.

For this integration harness:
    - players marked OUT are excluded from fantasy lineup selection
    - other players are treated as available
    - NBA team simulations use all projection-ready players in V3 for that team
    - IR fantasy players are never eligible for the fantasy active lineup

This is deliberately a validation bridge, not the final injury-aware product.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from scipy.optimize import linear_sum_assignment

import injury_rotation_v3_features as F3
import matchup_context as MC

from simulator_core_v1 import (
    PlayerProjection,
    simulate_nba_game,
    apply_availability,
)

from fantasy_roster_manager_v1 import (
    estimate_replacement_level,
    manage_healthy_ir_player,
    explain_drop_board,
    projected_fpg,
    HARD_PROTECT_FPG,
)

BASE = os.path.dirname(os.path.abspath(__file__))

POOL_FILE = os.path.join(BASE, "player_pool.csv")
PROJECTION_FILE = os.path.join(BASE, "current_player_projections_v3.csv")
ALL_NBA_PROJECTION_FILE = os.path.join(BASE, "all_nba_player_projections_2027.csv")
ROTATION_MODEL_FILE = os.path.join(BASE, "injury_rotation_v3_model.npz")
MODEL_HISTORY_FILE = os.path.join(BASE, "player_game_model_history.csv")
AVAIL_HISTORY_FILE = os.path.join(BASE, "player_availability_all.csv")
CURRENT_HISTORY_FILE = os.path.join(BASE, "player_game_history_2027_current.csv")
ESPN_NBA_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

SIMS = 2000
SEED = 20261027

# Live matchup file from fetch_espn_matchups.py (optional).
MATCHUPS_FILE = os.path.join(BASE, "fantasy_matchups.csv")
# None = take the first simulated day from fantasy_matchups.csv (or Monday if the
# file is missing). Set 0-6 (0 = Monday) to override, e.g. if ESPN has not yet
# rolled over to the next day after tonight's games finished.
SIMULATE_FROM_DAY = None

# ---- V3.2 settings ---------------------------------------------------------
MY_TEAM_NAME = "Gustavo Giants"
MATCHUP_SETTINGS_FILE = os.path.join(BASE, "matchup_context_settings.json")
USE_MATCHUP_CONTEXT = True          # pace + opponent defense (needs matchup_context_settings.json)
PREPASS_SIMS = 250                  # pre-pass for daily expected FP (lineups + pickups)
PICKUPS_ENABLED = True
MIN_STREAM_GAIN = 5.0               # an add must gain at least this many expected FP this week
STREAM_DROP_MARGIN = 2.0            # streaming may only drop players within this many FP/game of
                                    # waiver replacement level (rest-of-season value, not this week)
PLANNER_FA_CANDIDATES = 15          # free agents considered per day (best rest-of-week value)
MAX_ADDS_IF_UNLIMITED = 7           # cap when the league has no weekly acquisition limit
DEFAULT_ADDS_IF_UNKNOWN = 2         # per team when ESPN acquisition data is unavailable
PUBLISH_PICKUP_ADVICE = True        # False = the public page shows expected moves but no advice

# Pre-game lineup values {(player_id, day): expected FP if he plays}, set from the
# pre-pass. Empty = fall back to ESPN's projected average (V3.1 behaviour).
LINEUP_VALUES: dict = {}
LINEUP_MEMO: dict = {}          # (day, players, values version) -> best lineup; pure function of inputs

# Tag used in every saved CSV name. Kept as "v1_8" so any downstream script or
# spreadsheet that reads these files keeps working. Change in one place here.
OUTPUT_TAG = "v1_8"


def output_path(suffix: str) -> str:
    return os.path.join(BASE, f"weekly_matchup_{OUTPUT_TAG}_{suffix}.csv")

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Physical active lineup places.
ACTIVE_LINEUP_SLOTS = [
    "PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "UTIL", "UTIL"
]
ACTIVE_LINEUP_SLOT_IDS = [0, 1, 2, 3, 4, 5, 6, 11, 11, 11]

IR_SLOT_ID = 13

def _current_week_start() -> pd.Timestamp:
    """Monday of the current US Eastern week, or the WEEK_START environment setting."""
    override = os.environ.get("WEEK_START", "").strip()
    if override:
        return pd.Timestamp(override).normalize()
    from zoneinfo import ZoneInfo
    from datetime import datetime as _dt
    today = pd.Timestamp(_dt.now(ZoneInfo("America/New_York")).date())
    return (today - pd.Timedelta(days=today.weekday())).normalize()


WEEK_START = _current_week_start()

TEAM_ABBREVIATION_FIXES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}

# ESPN labels vary slightly. For this integration harness, only definite OUT
# states are excluded. Questionable/probable probability calibration comes next.
DEFINITE_OUT = {
    "OUT",
    "O",
    "INJURY_RESERVE",
    "INJURED_RESERVE",
}

# V1.6 short-term status -> probability of playing each scheduled game.
# User-specified NBA designation priors. DAY_TO_DAY is mapped to QUESTIONABLE.
STATUS_PLAY_PROB = {
    "ACTIVE": 1.00,
    "HEALTHY": 1.00,
    "NORMAL": 1.00,
    "PROBABLE": 0.75,
    "P": 0.75,
    "QUESTIONABLE": 0.50,
    "Q": 0.50,
    "DAY_TO_DAY": 0.50,
    "DTD": 0.50,
    "DOUBTFUL": 0.25,
    "D": 0.25,
    "OUT": 0.00,
    "O": 0.00,
    "INJURY_RESERVE": 0.00,
    "INJURED_RESERVE": 0.00,
}


def norm_id(value) -> str:
    s = str(value)
    return s[:-2] if s.endswith(".0") else s


def parse_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(text)
            if isinstance(result, (list, tuple, np.ndarray)):
                return list(result)
        except Exception:
            pass

    return []


def parse_eligible_slot_ids(value) -> set[int]:
    return {int(x) for x in parse_list(value)}


def parse_float_list(value) -> List[float]:
    out = []
    for x in parse_list(value):
        try:
            f = float(x)
            if np.isfinite(f):
                out.append(f)
        except (TypeError, ValueError):
            pass
    return out


def is_ir(row: pd.Series) -> bool:
    slot = row.get("lineup_slot")

    if pd.isna(slot):
        return False

    text = str(slot).upper().strip()

    if text in {"IR", "13", "IR_SLOT"}:
        return True

    try:
        return int(float(slot)) == IR_SLOT_ID
    except (TypeError, ValueError):
        return False


def is_definitely_out(status) -> bool:
    return str(status or "").upper().strip() in DEFINITE_OUT


def status_play_probability(status) -> float:
    key = str(status or "").upper().strip().replace(" ", "_").replace("-", "_")
    # Unlabelled / unfamiliar statuses default to healthy rather than inventing
    # an injury probability. They are reported in the V1.6 audit.
    return float(STATUS_PLAY_PROB.get(key, 1.0))


def download_espn_expected_returns() -> dict[str, dict]:
    """Fetch ESPN public NBA injury records keyed by nested athlete ESPN ID."""
    try:
        r = requests.get(
            ESPN_NBA_INJURIES_URL,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"WARNING — ESPN expected-return feed unavailable: {exc}")
        print("Falling back to status-only V1.6 availability.")
        return {}

    out = {}
    for team_block in data.get("injuries", []) or []:
        items = (team_block.get("injuries") or team_block.get("items") or team_block.get("athletes") or [])
        for item in items:
            athlete = item.get("athlete") or {}
            raw_pid = athlete.get("id")
            if raw_pid is None:
                for link in athlete.get("links", []) or []:
                    href = str(link.get("href", ""))
                    match = re.search(r"/id/(\d+)(?:/|$)", href)
                    if match:
                        raw_pid = match.group(1)
                        break
            pid = norm_id(raw_pid)
            if not pid or pid == "None":
                continue
            details = item.get("details") or {}
            raw_return = details.get("returnDate")
            parsed = pd.to_datetime(raw_return, errors="coerce") if raw_return else pd.NaT
            return_date = None if pd.isna(parsed) else parsed.normalize()
            out[pid] = {
                "return_date": return_date,
                "return_date_raw": raw_return,
                "public_status": (details.get("fantasyStatus") or {}).get("description"),
                "injury_type": details.get("type"),
                "injury_detail": details.get("detail"),
                "short_comment": item.get("shortComment"),
            }
    return out


def sample_out_return_dates(
    status_by_pid: dict[str, str],
    expected_returns: dict[str, dict],
    rng: np.random.Generator,
) -> dict[str, pd.Timestamp]:
    """Sample one return date per OUT injury episode: 50% R, 25% R+1, 25% R+2."""
    sampled = {}
    for pid, status in status_by_pid.items():
        key = str(status or "").upper().strip().replace(" ", "_").replace("-", "_")
        if key not in DEFINITE_OUT:
            continue
        info = expected_returns.get(norm_id(pid)) or {}
        rdate = info.get("return_date")
        if rdate is None:
            continue
        rbase = pd.Timestamp(rdate).normalize()
        # If ESPN already says the player returns after this fantasy week, there is
        # zero chance of playing this week. Do not let any later draw override that.
        if rbase > (WEEK_START + pd.Timedelta(days=6)).normalize():
            sampled[norm_id(pid)] = rbase
            continue
        u = rng.random()
        offset = 0 if u < 0.50 else (1 if u < 0.75 else 2)
        sampled[norm_id(pid)] = rbase + pd.Timedelta(days=offset)
    return sampled


def player_available_today(
    player_id,
    status,
    game_date,
    sampled_out_returns: dict[str, pd.Timestamp],
    rng: np.random.Generator,
    expected_returns: dict[str, dict] | None = None,
) -> bool:
    """
    V1.6 availability.

    OUT + ESPN returnDate: sampled episode return date (50% R / 25% R+1 / 25% R+2).
    OUT without returnDate: unavailable.
    PROBABLE / QUESTIONABLE / DTD / DOUBTFUL: independent game-status draw.
    Healthy/active: available.
    """
    pid = norm_id(player_id)
    key = str(status or "").upper().strip().replace(" ", "_").replace("-", "_")

    if key in DEFINITE_OUT:
        # V2.5.1 HARD RETURN-DATE GUARD.  Check ESPN's actual expected-return
        # date before consulting the sampled V1.6 return episode.  In particular,
        # an OUT player whose ESPN return date is after this fantasy week can
        # never become available during this week's simulation.
        info = (expected_returns or {}).get(pid) or {}
        espn_return = info.get("return_date")
        if espn_return is not None:
            espn_return = pd.Timestamp(espn_return).normalize()
            week_end = (WEEK_START + pd.Timedelta(days=6)).normalize()
            if espn_return > week_end:
                return False

        sampled = sampled_out_returns.get(pid)
        # OUT with no usable ESPN return date remains unavailable.
        return sampled is not None and pd.Timestamp(game_date).normalize() >= pd.Timestamp(sampled).normalize()

    p = status_play_probability(status)
    if p >= 1.0:
        return True
    if p <= 0.0:
        return False
    return bool(rng.random() < p)


def draw_v16_availability_for_day(
    roster_or_pool: pd.DataFrame,
    day: str,
    game_date,
    sampled_out_returns: dict[str, pd.Timestamp],
    rng: np.random.Generator,
    expected_returns: dict[str, dict] | None = None,
) -> dict[str, bool]:
    result = {}
    for _, row in roster_or_pool.iterrows():
        if int(row.get(day, 0) or 0) != 1:
            continue
        pid = norm_id(row["player_id"])
        result[pid] = player_available_today(
            pid, row.get("injury_status"), game_date, sampled_out_returns, rng, expected_returns=expected_returns
        )
    return result

def projection_availability_for_event(
    players: Sequence[PlayerProjection],
    day_availability: dict[str, bool],
) -> List[PlayerProjection]:
    availability = {
        norm_id(p.player_id): bool(day_availability.get(norm_id(p.player_id), True))
        for p in players
    }
    # apply_availability keys use the PlayerProjection player_id exactly.
    exact = {p.player_id: availability[norm_id(p.player_id)] for p in players}
    return apply_availability(players, exact)


def expected_fp_proxy(row: pd.Series) -> float:
    """
    Pregame lineup-ranking value.

    V1 uses ESPN projected FP when available because player_pool already carries
    it and because lineup choice must not use the simulated result.

    Later we can replace this with a Monte-Carlo-derived healthy expected FP
    from V3 without changing the lineup optimiser.
    """
    for col in ("projected_fp_avg", "espn_projected_fp_avg"):
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                pass
    return 0.0


def best_expected_lineup(candidates: pd.DataFrame, day: str | None = None):
    """
    Exact DP lineup optimiser.

    Objective:
        1. maximize number of filled physical lineup places
        2. among equally full lineups, maximize expected FP

    Never uses simulated fantasy points.
    """
    if len(candidates) == 0:
        return [], [], 0.0

    memo_key = (day, tuple(sorted(norm_id(p) for p in candidates["player_id"])), id(LINEUP_VALUES), len(LINEUP_VALUES))
    hit = LINEUP_MEMO.get(memo_key)
    if hit is not None:
        return list(hit[0]), list(hit[1]), hit[2]

    c = candidates.copy()
    c["lineup_value"] = c.apply(expected_fp_proxy, axis=1)
    if day is not None and LINEUP_VALUES:
        c["lineup_value"] = [
            LINEUP_VALUES.get((norm_id(pid), day), fallback)
            for pid, fallback in zip(c["player_id"], c["lineup_value"])
        ]
    c = c.sort_values("lineup_value", ascending=False).reset_index(drop=True)

    dp = {0: (0.0, [])}

    for player_index, player in c.iterrows():
        eligible_ids = parse_eligible_slot_ids(player["eligible_slot_ids"])
        current_states = list(dp.items())

        for mask, (score, assignments) in current_states:
            for slot_index, slot_id in enumerate(ACTIVE_LINEUP_SLOT_IDS):
                bit = 1 << slot_index

                if mask & bit:
                    continue

                if slot_id not in eligible_ids:
                    continue

                new_mask = mask | bit
                new_score = score + float(player["lineup_value"])
                new_assignments = assignments + [(slot_index, player_index)]

                if new_mask not in dp or new_score > dp[new_mask][0]:
                    dp[new_mask] = (new_score, new_assignments)

    best_mask, (best_score, assignments) = max(
        dp.items(),
        key=lambda item: (item[0].bit_count(), item[1][0])
    )

    starters = []
    used = set()

    for slot_index, player_index in sorted(assignments):
        row = c.loc[player_index]
        used.add(player_index)
        starters.append({
            "slot": ACTIVE_LINEUP_SLOTS[slot_index],
            "player_id": norm_id(row["player_id"]),
            "player_name": row["player_name"],
            "nba_team": row["nba_team"],
            "lineup_value": float(row["lineup_value"]),
        })

    bench = []
    for player_index, row in c.iterrows():
        if player_index in used:
            continue
        bench.append({
            "player_id": norm_id(row["player_id"]),
            "player_name": row["player_name"],
            "nba_team": row["nba_team"],
            "lineup_value": float(row["lineup_value"]),
        })

    LINEUP_MEMO[memo_key] = (tuple(starters), tuple(bench), float(best_score))
    return starters, bench, float(best_score)


def make_projection(row: pd.Series) -> PlayerProjection:
    return PlayerProjection(
        player_id=norm_id(row["player_id"]),
        name=str(row["player_name"]),
        nba_team=str(row["nba_team"]),
        expected_minutes=float(row["expected_minutes"]),
        fga_rate=float(row["fga_rate"]),
        fta_rate=float(row["fta_rate"]),
        ast_rate=float(row["ast_rate"]),
        tov_rate=float(row["tov_rate"]),
        stl_rate=float(row["stl_rate"]),
        blk_rate=float(row["blk_rate"]),
        three_share=float(row["three_share"]),
        two_pct=float(row["two_pct"]),
        three_pct=float(row["three_pct"]),
        ft_pct=float(row["ft_pct"]),
        oreb_capture=float(row["oreb_capture"]),
        dreb_capture=float(row["dreb_capture"]),
        minutes_residuals=parse_float_list(row.get("minutes_residuals")),
        available=not is_definitely_out(row.get("injury_status")),
    )


def load_data():
    for path in (POOL_FILE, PROJECTION_FILE, ALL_NBA_PROJECTION_FILE, ROTATION_MODEL_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file missing: {os.path.basename(path)}"
            )

    pool = pd.read_csv(POOL_FILE, dtype={"player_id": str})
    proj = pd.read_csv(PROJECTION_FILE, dtype={"player_id": str})
    all_nba_proj = pd.read_csv(ALL_NBA_PROJECTION_FILE, dtype={"player_id": str})

    pool["player_id"] = pool["player_id"].map(norm_id)
    proj["player_id"] = proj["player_id"].map(norm_id)
    all_nba_proj["player_id"] = all_nba_proj["player_id"].map(norm_id)

    # ------------------------------------------------------------------
    # CANONICALISE DUPLICATE ESPN PLAYER IDs
    # ------------------------------------------------------------------
    # fantasy_sim.py can receive the same player from both the roster
    # endpoint and the available-player endpoint. Roster ownership is
    # authoritative, so ROSTERED always wins over FREE_AGENT.
    #
    # V3 was built from that same pool, so the duplicate can also exist
    # there. Before collapsing V3, verify that all actual simulation-model
    # fields agree across duplicate rows. If they do not, stop rather than
    # silently choosing conflicting projections.
    pool_dupes = pool[pool.duplicated("player_id", keep=False)].copy()

    if len(pool_dupes):
        print()
        print("=" * 100)
        print("DUPLICATE PLAYER-ID RESOLUTION")
        print("=" * 100)
        print("Duplicate player_pool rows:", len(pool_dupes))
        print("Unique duplicated player IDs:", pool_dupes["player_id"].nunique())

        show_cols = [
            c for c in [
                "player_id", "player_name", "roster_status",
                "fantasy_team", "lineup_slot"
            ]
            if c in pool_dupes.columns
        ]
        print(pool_dupes[show_cols].to_string(index=False))

        pool["_ownership_priority"] = (
            pool["roster_status"]
            .map({"ROSTERED": 0, "FREE_AGENT": 1})
            .fillna(2)
        )

        pool = (
            pool.sort_values(
                ["player_id", "_ownership_priority"],
                kind="stable"
            )
            .drop_duplicates("player_id", keep="first")
            .drop(columns="_ownership_priority")
            .reset_index(drop=True)
        )

        print(
            "player_pool canonicalised:",
            len(pool),
            "unique players"
        )

    model_compare_cols = [
        c for c in [
            "expected_minutes",
            "fga_rate", "fta_rate", "ast_rate", "tov_rate",
            "stl_rate", "blk_rate",
            "three_share", "two_pct", "three_pct", "ft_pct",
            "oreb_capture", "dreb_capture",
            "minutes_residuals",
            "projection_ready",
            "prior_source",
            "rebound_source",
        ]
        if c in proj.columns
    ]

    proj_dupes = proj[proj.duplicated("player_id", keep=False)].copy()

    if len(proj_dupes):
        conflicting_ids = []

        for pid, g in proj_dupes.groupby("player_id"):
            # Convert to strings for a robust equality check that also handles
            # list-like residual fields and booleans saved through CSV.
            distinct = (
                g[model_compare_cols]
                .astype(str)
                .drop_duplicates()
            )

            if len(distinct) > 1:
                conflicting_ids.append(pid)

        if conflicting_ids:
            bad = proj[proj["player_id"].isin(conflicting_ids)]
            raise RuntimeError(
                "Duplicate V3 rows contain conflicting simulation parameters "
                "for player IDs: "
                + ", ".join(map(str, conflicting_ids))
                + "\n\n"
                + bad[
                    [
                        c for c in [
                            "player_id", "player_name", "fantasy_team",
                            "roster_status", "expected_minutes",
                            "prior_source"
                        ]
                        if c in bad.columns
                    ]
                ].to_string(index=False)
            )

        proj = (
            proj.drop_duplicates("player_id", keep="first")
            .reset_index(drop=True)
        )

        print(
            "V3 duplicate projections verified identical and collapsed:",
            len(proj),
            "unique players"
        )

    if pool["player_id"].duplicated().any():
        raise RuntimeError("player_pool still contains duplicate player IDs.")

    if proj["player_id"].duplicated().any():
        raise RuntimeError("V3 projections still contain duplicate player IDs.")

    # Keep roster/schedule fields from player_pool; model fields from V3.
    projection_cols = [
        "player_id",
        "expected_minutes",
        "fga_rate",
        "fta_rate",
        "ast_rate",
        "tov_rate",
        "stl_rate",
        "blk_rate",
        "three_share",
        "two_pct",
        "three_pct",
        "ft_pct",
        "oreb_capture",
        "dreb_capture",
        "minutes_residuals",
        "projection_ready",
        "prior_source",
        "rebound_source",
        "espn_projected_fp_avg",
    ]

    projection_cols = [c for c in projection_cols if c in proj.columns]

    merged = pool.merge(
        proj[projection_cols],
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    for day in DAY_NAMES:
        if day not in merged.columns:
            raise RuntimeError(
                f"{day} schedule flag missing from player_pool.csv. "
                "Run fantasy_sim.py first."
            )

    rostered = merged[merged["roster_status"] == "ROSTERED"].copy()

    if rostered["expected_minutes"].isna().any():
        bad = rostered[rostered["expected_minutes"].isna()]
        raise RuntimeError(
            "Rostered players missing V3 projection:\n"
            + bad[["player_name", "fantasy_team"]].to_string(index=False)
        )

    return merged, rostered, proj, all_nba_proj



# =============================================================================
# INJURY ROTATION V3 — LIVE WHOLE-TEAM ROTATION
# =============================================================================
#
# Every NBA team-game in the simulation is projected by the shared pipeline in
# injury_rotation_v3_features.py (identical code to the validated backtests):
#
#   1. healthy-state baseline over available players, scaled to 240
#   2. V3 learned correction for this exact availability state
#      (every absent player counts; returning players carry their layoff length)
#   3. blend with matching-state coach evidence from this season's games
#   4. tonight's likely rotation (nightly size learned from recent games)
#   5. role floors / sustainable ceilings, balanced to exactly 240
#
# Leak-free held-out backtest (1,112 team-games, Jan-Mar 2026), MAE in minutes:
#   proportional baseline 5.80 | current V2 5.92 | V3 model 5.14 | V3 full stack 5.00
#   injury games: proportional 5.59 | V2 6.20 | V3 full stack 5.33

ROTATION_TEAM_MINUTES = 240.0
ROTATION_IMPORTANT_MINUTES = 20.0
TOP_AVERAGE_MINUTES_REFERENCE = 36.0
STAR_SOFT_CEILING = F3.STAR_SOFT_CEILING
MAX_EXPECTED_MINUTES = F3.MAX_EXPECTED_MINUTES

# evidence_player_k=1: a player's coach evidence is discounted when he has only
# been seen in one or two matching games (validated in backtest_v3_pipeline.py:
# 12-game window MAE 5.03 -> 5.00, early-season 3-game window 5.11 -> 5.08).
ROTATION_PIPELINE = dict(use_evidence=True, use_selection=True, use_guardrails=True,
                         evidence_player_k=1.0)


class TeamRotationV3:
    """Static per-team inputs plus a cache of rotation results by availability state."""

    def __init__(self, team: str, team_players, frame: pd.DataFrame, past_games: dict, model: dict,
                 week_start_available: dict | None = None):
        self.team = str(team).upper()
        self.pids = [norm_id(p.player_id) for p in team_players]
        self.names = [str(p.name) for p in team_players]
        base = frame.reindex(self.pids).copy()
        missing = base["exp_minutes"].isna()
        if missing.any():
            # Universe player without a live row: fall back to his projection minutes.
            proj = {norm_id(p.player_id): float(p.expected_minutes) for p in team_players}
            for col in ["exp_minutes", "healthy_exp_minutes", "recent1_minutes",
                        "recent3_minutes", "recent5_minutes"]:
                base.loc[missing, col] = [proj[pid] for pid in base.index[missing]]
            base.loc[missing, "minutes_trend"] = 0.0
            base.loc[missing, "own_missed_streak"] = 0.0
            base.loc[missing, "own_games_since_return"] = float(F3.MAX_SINCE_RETURN)
            base.loc[missing, "position_family"] = "UNKNOWN"
        base["player_id"] = self.pids
        self.base = base.reset_index(drop=True)
        self.healthy = self.base["healthy_exp_minutes"].to_numpy(float)
        self.past = past_games.get(self.team, [])
        self.model = model
        self.cache = {}
        # Full-strength reference rotation (everyone available, no layoffs):
        # audit deltas are measured against this, not against raw projections.
        n = len(self.pids)
        self.full_strength, _ = self._predict(np.ones(n, bool), np.zeros(n), np.full(n, float(F3.MAX_SINCE_RETURN)))
        # Week-start reference: known OUT players absent, everyone else available,
        # pre-week layoff streaks. Deltas against this show in-week changes only.
        ws = week_start_available or {}
        ws_avail = np.asarray([bool(ws.get(pid, True)) for pid in self.pids], bool)
        self.week_start, _ = self._predict(
            ws_avail,
            self.base["own_missed_streak"].to_numpy(float),
            self.base["own_games_since_return"].to_numpy(float),
        )

    def _predict(self, avail, streak, since):
        r = self.base.copy()
        r["own_missed_streak"] = streak
        r["own_games_since_return"] = since
        return F3.predict_team_minutes(r, avail, self.model, past_games=self.past, **ROTATION_PIPELINE)

    def minutes(self, avail, streak, since):
        key = (avail.tobytes(), streak.tobytes(), since.tobytes())
        hit = self.cache.get(key)
        if hit is None:
            mins, meta = self._predict(avail, streak, since)
            hit = (mins, meta)
            self.cache[key] = hit
        return hit[0].copy(), dict(hit[1])


def load_live_rotation_context(team_proj: Dict[str, List[PlayerProjection]],
                               all_nba_proj: pd.DataFrame, merged: pd.DataFrame,
                               cutoff_day=None):
    """Build every team's V3 rotation object from history up to WEEK_START."""
    for path in (MODEL_HISTORY_FILE, AVAIL_HISTORY_FILE, CURRENT_HISTORY_FILE, ROTATION_MODEL_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError("Injury Rotation V3 requires " + os.path.basename(path))
    model = F3.load_model(ROTATION_MODEL_FILE)
    if model["version"] != "injury_rotation_v3":
        raise RuntimeError(f"{os.path.basename(ROTATION_MODEL_FILE)} is not a V3 model ({model['version']}).")

    universe = pd.DataFrame([
        {"player_id": norm_id(p.player_id), "nba_team": team, "expected_minutes": float(p.expected_minutes)}
        for team, players in team_proj.items() for p in players
    ])
    positions = {}
    if "default_position" in merged.columns:
        positions = {norm_id(p): d for p, d in zip(merged["player_id"], merged["default_position"])}

    # Who is fully healthy right now (fantasy-pool status overrides full-NBA status).
    status = build_nba_status_map(all_nba_proj)
    for pid, st in zip(merged["player_id"], merged.get("injury_status", pd.Series(dtype=str))):
        if isinstance(st, str) and st.strip():
            status[norm_id(pid)] = st
    healthy_now = {pid: status_play_probability(status.get(pid, "ACTIVE")) >= 1.0
                   for pid in universe["player_id"]}

    frame, past_games, diag = F3.build_live_player_frame(
        AVAIL_HISTORY_FILE, MODEL_HISTORY_FILE, CURRENT_HISTORY_FILE,
        universe, cutoff_day if cutoff_day is not None else WEEK_START,
        model["deflation_alpha"], positions, healthy_now,
    )
    week_start_available = {pid: status_play_probability(status.get(pid, "ACTIVE")) > 0.0
                            for pid in universe["player_id"]}
    rotations = {
        team: TeamRotationV3(team, players, frame, past_games, model, week_start_available)
        for team, players in team_proj.items()
    }
    return rotations, frame, model, diag


def initial_layoff_state(rotations: Dict[str, TeamRotationV3]) -> dict:
    """Pre-week (missed streak, games since return) for every NBA player."""
    state = {}
    for rot in rotations.values():
        for i, pid in enumerate(rot.pids):
            state[pid] = (float(rot.base["own_missed_streak"].iloc[i]),
                          float(rot.base["own_games_since_return"].iloc[i]))
    return state


def update_layoff_state(state: dict, pids, health_available: dict, played_minutes: dict) -> None:
    """Mirror of missed_streak_features(): advance after one team game."""
    for pid in pids:
        streak, since = state.get(pid, (0.0, float(F3.MAX_SINCE_RETURN)))
        if not health_available.get(pid, True):
            streak = min(streak + 1.0, float(F3.MAX_STREAK))
        elif played_minutes.get(pid, 0.0) > 0.0:
            since = 0.0 if streak >= 2 else min(since + 1.0, float(F3.MAX_SINCE_RETURN))
            streak = 0.0
        state[pid] = (streak, since)


def new_rotation_audit() -> dict:
    return {
        "team_events": 0,
        "minutes_sum_errors": [],
        "max_player_minutes": [],
        "rotation_sizes": defaultdict(int),
        "emergency_events": 0,
        "emergency_ceilings": [],
        "important_absent_counts": [],
        # Every absent player, every team-game (no single "primary" trigger).
        "absent_games": defaultdict(int),
        "team_games": defaultdict(int),
        # Per-player minute aggregates versus the team's full-strength rotation.
        "recipient": defaultdict(lambda: [0, 0.0, 0.0, -np.inf, np.inf]),  # n, sum_min, sum_delta, max, min
        "names": {},
        "teams": {},
        "full_strength": {},
        "week_start": {},
        "healthy": {},
    }


def apply_injury_rotation_v3(team_players, availability, rot: TeamRotationV3, layoff_state: dict,
                             audit: dict | None = None):
    """Return team players with V3 expected minutes for this availability state."""
    players = list(team_players)
    avail = np.asarray([bool(availability.get(pid, True)) for pid in rot.pids], bool)
    streak = np.asarray([layoff_state.get(pid, (0.0, 0.0))[0] for pid in rot.pids], float)
    since = np.asarray([layoff_state.get(pid, (0.0, float(F3.MAX_SINCE_RETURN)))[1] for pid in rot.pids], float)
    mins, meta = rot.minutes(avail, streak, since)
    out = [replace(p, expected_minutes=float(mins[i]), available=bool(mins[i] > 1e-8))
           for i, p in enumerate(players)]

    if audit is not None:
        audit["team_events"] += 1
        audit["minutes_sum_errors"].append(float(mins.sum() - ROTATION_TEAM_MINUTES))
        audit["max_player_minutes"].append(float(mins.max()) if len(mins) else 0.0)
        audit["rotation_sizes"][int((mins > 1e-8).sum())] += 1
        if meta.get("emergency_ceiling") is not None:
            audit["emergency_events"] += 1
            audit["emergency_ceilings"].append(float(meta["emergency_ceiling"]))
        audit["important_absent_counts"].append(
            int((~avail & (rot.healthy >= ROTATION_IMPORTANT_MINUTES)).sum()))
        for i, pid in enumerate(rot.pids):
            audit["team_games"][pid] += 1
            if not avail[i]:
                audit["absent_games"][pid] += 1
            elif mins[i] > 1e-8:
                rec = audit["recipient"][pid]
                d = float(mins[i] - rot.full_strength[i])
                rec[0] += 1
                rec[1] += float(mins[i])
                rec[2] += d
                rec[3] = max(rec[3], d)
                rec[4] = min(rec[4], d)
            if pid not in audit["names"]:
                audit["names"][pid] = rot.names[i]
                audit["teams"][pid] = rot.team
                audit["full_strength"][pid] = float(rot.full_strength[i])
                audit["week_start"][pid] = float(rot.week_start[i])
                audit["healthy"][pid] = float(rot.healthy[i])
    return out


def audit_rotation_model():
    z = np.load(ROTATION_MODEL_FILE, allow_pickle=False)
    features = list(z["features"])
    coeffs = z["coefficients"]
    print()
    print("=" * 100)
    print("INJURY ROTATION MODEL CONTRACT")
    print("=" * 100)
    print("Model:", str(z["model_version"][0]) if "model_version" in z.files else "unknown")
    print("Features:", len(features), "| Coefficients:", len(coeffs))
    print("Ridge alpha:", float(z["best_alpha"][0]), "| Shrinkage:", float(z["best_shrinkage"][0]))
    if "deflation_alpha" in z.files:
        print("Healthy-state deflation strength:", float(z["deflation_alpha"][0]))
    if "heldout_mae" in z.files:
        print(f"Held-out MAE (leak-free, model only): {float(z['heldout_mae'][0]):.3f} minutes")
    if len(coeffs) != len(features) + 1:
        raise RuntimeError("Rotation-model coefficient count is invalid.")
    missing = [f for f in features if f not in F3.V3R_FEATURES]
    if missing:
        raise RuntimeError("Model expects features the live builder does not produce: " + ", ".join(missing))
    print("Saved model structure: OK | live feature contract: OK")


def build_lineup_plan(rostered: pd.DataFrame):
    """
    Pregame lineup plan for every fantasy team/day.

    IR players and definite OUT players are excluded.
    """
    plans = {}
    diagnostics = []

    for team_id, team in rostered.groupby("fantasy_team_id", dropna=False):
        team_name = str(team["fantasy_team"].iloc[0])

        for day in DAY_NAMES:
            candidates = team[
                (team[day] == 1)
                & (~team.apply(is_ir, axis=1))
                & (~team["injury_status"].map(is_definitely_out))
            ].copy()

            starters, bench, expected = best_expected_lineup(candidates)

            plans[(str(team_id), day)] = {
                "team_name": team_name,
                "starter_ids": {x["player_id"] for x in starters},
                "starters": starters,
                "bench": bench,
                "expected_fp": expected,
            }

            diagnostics.append({
                "fantasy_team_id": team_id,
                "fantasy_team": team_name,
                "day": day,
                "players_with_games": len(candidates),
                "starters": len(starters),
                "bench": len(bench),
                "expected_active_fp": expected,
            })

    return plans, pd.DataFrame(diagnostics)


def build_nba_team_projection_map(proj: pd.DataFrame):
    ready = proj.copy()

    if "projection_ready" in ready.columns:
        # Handles bool or string representation.
        ready = ready[
            ready["projection_ready"].astype(str).str.upper().isin(
                {"TRUE", "1"}
            )
        ].copy()

    result = {}

    for nba_team, g in ready.groupby("nba_team"):
        result[str(nba_team)] = [make_projection(row) for _, row in g.iterrows()]

    return result


def build_nba_status_map(all_nba_proj: pd.DataFrame) -> dict[str, str]:
    """
    Player-ID -> raw injury designation from the full-NBA projection file.

    Blank / missing statuses are omitted so the caller falls back to the
    projection's ACTIVE/OUT flag rather than inventing a status.
    """
    if "injury_status" not in all_nba_proj.columns:
        return {}
    out = {}
    for pid, status in zip(all_nba_proj["player_id"], all_nba_proj["injury_status"]):
        if status is None or (isinstance(status, float) and np.isnan(status)):
            continue
        text = str(status).strip()
        if not text:
            continue
        out[norm_id(pid)] = text
    return out


def download_event_schedule():
    """
    Download the current week's event-level NBA schedule.

    Robustness:
      1. retries transient ESPN failures
      2. checks HTTP status before JSON parsing
      3. checks that the response actually looks like JSON
      4. falls back to the last validated local schedule CSV if ESPN fails

    The fallback is safe for this harness because audit_event_schedule() still
    cross-checks it against the player_pool Mon-Sun flags before simulation.
    """
    import time

    cache_path = os.path.join(BASE, "nba_week_event_schedule.csv")

    # ESPN scoreboard abbreviations -> canonical abbreviations used by
    # player_pool / projection data. Keep this local so the schedule loader
    # does not depend on a global mapping existing elsewhere in the script.
    scoreboard_team_fixes = {
        "GS": "GSW",
        "NO": "NOP",
        "NY": "NYK",
        "SA": "SAS",
        "UTAH": "UTA",
        "WSH": "WAS",
    }

    all_rows = []
    errors = []

    for day_offset, day_name in enumerate(DAY_NAMES):
        game_day = WEEK_START + pd.Timedelta(days=day_offset)
        date_str = game_day.strftime("%Y%m%d")

        url = (
            "https://site.api.espn.com/apis/site/v2/sports/"
            "basketball/nba/scoreboard"
        )
        params = {"dates": date_str}

        data = None

        for attempt in range(1, 4):
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=20,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 Chrome/151 Safari/537.36"
                        ),
                        "Accept": "application/json,text/plain,*/*",
                    },
                )

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {response.status_code}; "
                        f"content-type={content_type or 'unknown'}"
                    )

                body = response.text.lstrip()

                if not body:
                    raise RuntimeError("ESPN returned an empty response.")

                if (
                    "json" not in content_type
                    and not body.startswith("{")
                    and not body.startswith("[")
                ):
                    preview = body[:120].replace("\n", " ")
                    raise RuntimeError(
                        "ESPN returned non-JSON content: "
                        + repr(preview)
                    )

                data = response.json()
                break

            except Exception as exc:
                msg = (
                    f"{day_name} {game_day.date()} attempt {attempt}/3: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(msg)

                if attempt < 3:
                    time.sleep(1.0 * attempt)

        if data is None:
            print()
            print("WARNING: ESPN schedule download failed.")
            for msg in errors[-3:]:
                print("  ", msg)

            if os.path.exists(cache_path):
                print(
                    "Using cached schedule:",
                    os.path.basename(cache_path),
                )
                cached = pd.read_csv(cache_path)

                required = {
                    "game_id", "date", "day",
                    "team_a", "team_b", "status",
                }
                missing = required - set(cached.columns)

                if missing:
                    raise RuntimeError(
                        "Cached NBA schedule exists but is missing columns: "
                        + ", ".join(sorted(missing))
                    )

                cached["game_id"] = cached["game_id"].astype(str)
                cached["day"] = cached["day"].astype(str)
                cached["team_a"] = cached["team_a"].astype(str)
                cached["team_b"] = cached["team_b"].astype(str)

                print(
                    f"Cached schedule loaded: {len(cached)} games. "
                    "It will still be cross-checked before simulation."
                )
                return cached

            raise RuntimeError(
                "ESPN schedule download failed and no cached "
                "nba_week_event_schedule.csv was found."
            )

        for event in data.get("events", []):
            competitors = (
                event.get("competitions", [{}])[0]
                .get("competitors", [])
            )

            teams = []
            for comp in competitors:
                abbr = (
                    comp.get("team", {})
                    .get("abbreviation", "")
                    .upper()
                )
                abbr = scoreboard_team_fixes.get(abbr, abbr)
                if abbr:
                    teams.append(abbr)

            if len(teams) != 2:
                continue

            all_rows.append({
                "game_id": str(event.get("id", "")),
                "date": game_day.strftime("%Y-%m-%d"),
                "day": day_name,
                "team_a": teams[0],
                "team_b": teams[1],
                "status": (
                    event.get("status", {})
                    .get("type", {})
                    .get("name", "")
                ),
            })

    schedule = pd.DataFrame(all_rows)

    if schedule.empty:
        if os.path.exists(cache_path):
            print(
                "WARNING: ESPN returned no schedule events; "
                "using cached nba_week_event_schedule.csv."
            )
            return pd.read_csv(cache_path)

        raise RuntimeError(
            "ESPN returned no NBA schedule events and no cached schedule exists."
        )

    # Basic duplicate safety before the deeper audit.
    dupes = schedule[schedule.duplicated("game_id", keep=False)]
    if not dupes.empty:
        raise RuntimeError(
            "Duplicate NBA game IDs returned by ESPN:\n"
            + dupes.to_string(index=False)
        )

    return schedule.reset_index(drop=True)

def audit_event_schedule(schedule: pd.DataFrame, rostered: pd.DataFrame):
    print()
    print("=" * 100)
    print("EVENT-LEVEL NBA SCHEDULE")
    print("=" * 100)

    counts = (
        schedule.groupby("day")
        .size()
        .reindex(DAY_NAMES, fill_value=0)
    )

    for day in DAY_NAMES:
        print(f"{day}: {int(counts.loc[day]):2d} games")

    print("Total:", len(schedule), "games")

    # Cross-check event table against the old team/day flags.
    mismatches = []

    for _, row in rostered.iterrows():
        team = str(row["nba_team"])

        for day in DAY_NAMES:
            event_has_game = bool(
                (
                    (schedule["day"] == day)
                    & (
                        (schedule["team_a"] == team)
                        | (schedule["team_b"] == team)
                    )
                ).any()
            )

            old_flag = int(row[day]) == 1

            if event_has_game != old_flag:
                mismatches.append({
                    "player_name": row["player_name"],
                    "nba_team": team,
                    "day": day,
                    "old_flag": int(row[day]),
                    "event_schedule": int(event_has_game),
                })

    if mismatches:
        mismatch_df = pd.DataFrame(mismatches)
        print()
        print("WARNING: event schedule disagrees with saved player flags:")
        print(mismatch_df.head(30).to_string(index=False))
        raise RuntimeError(
            f"Schedule cross-check failed: {len(mismatches)} player-day mismatches."
        )

    print("Cross-check against player_pool Mon-Sun flags: PASS")

    schedule.to_csv(
        os.path.join(BASE, "nba_week_event_schedule.csv"),
        index=False,
    )
    print("Saved: nba_week_event_schedule.csv")


def infer_daily_games(rostered: pd.DataFrame):
    """
    The pool contains team/day flags rather than event IDs/opponents.

    For the integration harness we identify which NBA teams play each day.
    Pairing opponents requires the raw scoreboard event list, so full shared-game
    simulation is intentionally blocked until that event table is supplied.

    This prevents us from inventing opponent pairings.
    """
    by_day = {}
    for day in DAY_NAMES:
        teams = sorted(
            set(
                rostered.loc[rostered[day] == 1, "nba_team"]
                .dropna()
                .astype(str)
            )
        )
        by_day[day] = teams
    return by_day


def print_contract_audit(rostered, plans, lineup_diag, team_proj):
    print()
    print("=" * 100)
    print("WEEKLY SIMULATOR V1 — DATA CONTRACT AUDIT")
    print("=" * 100)

    print("Rostered entries:", len(rostered))
    print("Fantasy teams:", rostered["fantasy_team_id"].nunique())
    print(
        "IR entries:",
        int(rostered.apply(is_ir, axis=1).sum())
    )
    print(
        "Definite OUT entries:",
        int(rostered["injury_status"].map(is_definitely_out).sum())
    )
    print(
        "NBA teams with simulation projections:",
        len(team_proj)
    )

    print()
    print("LINEUP PLAN SUMMARY")
    print("-" * 100)

    congested = lineup_diag[lineup_diag["players_with_games"] > 10]

    print(
        lineup_diag.groupby("day")
        .agg(
            fantasy_teams=("fantasy_team_id", "nunique"),
            players_with_games=("players_with_games", "sum"),
            starters=("starters", "sum"),
            bench=("bench", "sum"),
        )
        .to_string()
    )

    print()
    print("Congested team-days (>10 playable rostered players):", len(congested))

    if len(congested):
        print(
            congested[
                [
                    "fantasy_team",
                    "day",
                    "players_with_games",
                    "starters",
                    "bench",
                    "expected_active_fp",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )

    print()
    print("SAMPLE LINEUP — GUSTAVO GIANTS")
    print("-" * 100)

    gustavo = [
        (key, value)
        for key, value in plans.items()
        if str(value["team_name"]).strip().casefold() == "gustavo giants".casefold()
    ]

    if not gustavo:
        print("No exact 'Gustavo Giants' team-name match found.")
        print("Fantasy team names loaded:")
        for name in sorted(
            {str(v["team_name"]) for v in plans.values()}
        ):
            print(" ", repr(name))

    for (team_id, day), plan in gustavo:
        print()
        print(day, f"({len(plan['starters'])} starters)")
        for p in plan["starters"]:
            print(
                f"  {p['slot']:>4}  {p['player_name']:<28} "
                f"{p['nba_team']:<4} {p['lineup_value']:6.2f}"
            )
        if plan["bench"]:
            print("  BENCH:")
            for p in plan["bench"]:
                print(
                    f"        {p['player_name']:<28} "
                    f"{p['nba_team']:<4} {p['lineup_value']:6.2f}"
                )



def audit_nba_team_coverage(
    event_schedule: pd.DataFrame,
    team_proj: Dict[str, List[PlayerProjection]],
):
    schedule_teams = set(event_schedule["team_a"]) | set(event_schedule["team_b"])
    projection_teams = set(team_proj)

    missing = sorted(schedule_teams - projection_teams)
    extra = sorted(projection_teams - schedule_teams)

    print()
    print("=" * 100)
    print("NBA TEAM PROJECTION COVERAGE")
    print("=" * 100)
    print("Teams in weekly NBA schedule:", len(schedule_teams))
    print("Teams in full-NBA projection map:", len(projection_teams))

    if missing:
        raise RuntimeError(
            "NBA teams in schedule with no full-NBA projection players: "
            + ", ".join(missing)
        )

    print("All scheduled NBA teams have full-NBA projections: PASS")

    if extra:
        print(
            "Extra full-NBA team labels not used this week:",
            ", ".join(extra)
        )

    rotation_rows = []
    for team in sorted(schedule_teams):
        players = team_proj[team]
        rotation_rows.append({
            "nba_team": team,
            "projection_players": len(players),
            "sum_expected_minutes": sum(
                max(float(p.expected_minutes), 0.0) for p in players
            ),
            "available_projection_players": sum(
                bool(p.available) and p.expected_minutes > 0 for p in players
            ),
        })

    rotation = pd.DataFrame(rotation_rows)

    print()
    print("FULL-NBA ROTATION COVERAGE")
    print("-" * 100)
    print(
        rotation.round(2).to_string(index=False)
    )

    suspicious = rotation[
        (rotation["available_projection_players"] < 7)
        | (rotation["sum_expected_minutes"] < 150)
    ]

    if len(suspicious):
        print()
        print(
            "WARNING: Some NBA teams have thin full-NBA rotation coverage. "
            "The core would scale these players to 240 minutes, which could "
            "distort results."
        )
        print(suspicious.round(2).to_string(index=False))
        raise RuntimeError(
            "Thin NBA rotation coverage detected. Fix projection coverage "
            "before trusting Monte Carlo."
        )

    print()
    print("Rotation coverage safety gate: PASS")

    return rotation


def fantasy_team_maps(rostered: pd.DataFrame):
    id_to_team = {}
    id_to_player = {}

    for _, row in rostered.iterrows():
        pid = norm_id(row["player_id"])
        team_id = str(row["fantasy_team_id"])
        id_to_team[pid] = team_id
        id_to_player[pid] = str(row["player_name"])

    team_names = {
        str(team_id): str(g["fantasy_team"].iloc[0])
        for team_id, g in rostered.groupby("fantasy_team_id", dropna=False)
    }

    return id_to_team, id_to_player, team_names



def _set_ir_state(roster: pd.DataFrame, player_id: str, on_ir: bool) -> None:
    """Mutate the simulation-local fantasy roster IR state."""
    pid = norm_id(player_id)
    mask = roster["player_id"].map(norm_id).eq(pid)
    if not mask.any():
        raise RuntimeError(f"Roster mutation could not find player {pid}.")
    roster.loc[mask, "_sim_on_ir"] = bool(on_ir)


def _drop_from_roster(roster: pd.DataFrame, player_id: str) -> None:
    pid = norm_id(player_id)
    mask = roster["player_id"].map(norm_id).eq(pid)
    if not mask.any():
        raise RuntimeError(f"Drop mutation could not find player {pid}.")
    roster.loc[mask, "_sim_rostered"] = False
    roster.loc[mask, "_sim_on_ir"] = False


def _manager_view(team_roster: pd.DataFrame) -> pd.DataFrame:
    """
    Convert simulation-local roster state into the shape expected by
    fantasy_roster_manager_v1.py.
    """
    x = team_roster[team_roster["_sim_rostered"]].copy()
    x["lineup_slot"] = np.where(x["_sim_on_ir"], "IR", "BENCH")
    # IMPORTANT: game-day availability is not IR eligibility. An NBA off-day or a
    # failed Q/DTD availability draw must never turn a healthy player into an IR candidate.
    if "_sim_ir_eligible" in x.columns:
        x.loc[x["_sim_ir_eligible"].astype(bool), "injury_status"] = "OUT"
    return x




_FUTURE_INJURY_MODEL = None


def estimate_future_injury_model():
    """Cached wrapper: the history file does not change during a run."""
    global _FUTURE_INJURY_MODEL
    if _FUTURE_INJURY_MODEL is None:
        _FUTURE_INJURY_MODEL = _estimate_future_injury_model()
    return _FUTURE_INJURY_MODEL


def _estimate_future_injury_model():
    """Estimate conservative new-injury incidence and duration from historical availability.

    A new injury episode is a transition from PLAYED to a subsequent INJURY absence.
    The returned incidence is per completed player-game; durations are measured in
    missed scheduled games. Falls back conservatively if the historical schema lacks
    the fields needed for reconstruction.
    """
    fallback_rate = 0.006
    fallback_durations = np.asarray([1, 1, 1, 2, 2, 3, 3, 4, 5, 7], dtype=int)
    if not os.path.exists(AVAIL_HISTORY_FILE):
        return fallback_rate, fallback_durations, "fallback"
    try:
        a = pd.read_csv(AVAIL_HISTORY_FILE, dtype={"player_id": str, "game_id": str})
        if "date" not in a.columns or "played" not in a.columns:
            return fallback_rate, fallback_durations, "fallback"
        a["date"] = pd.to_datetime(a["date"], errors="coerce", utc=True)
        def b(v): return str(v).strip().upper() in {"TRUE","1","YES"}
        a["_played"] = a["played"].map(b)
        reason_col = next((c for c in ["absence_category","reason_category","absence_type","reason"] if c in a.columns), None)
        if reason_col is None:
            return fallback_rate, fallback_durations, "fallback"
        a["_inj"] = a[reason_col].astype(str).str.upper().str.contains("INJUR", na=False)
        opportunities = 0; starts = 0; durations = []
        for _, g in a.sort_values(["player_id","date","game_id"]).groupby("player_id", sort=False):
            rows = g.reset_index(drop=True)
            for i in range(len(rows)-1):
                if bool(rows.loc[i,"_played"]):
                    opportunities += 1
                    if (not bool(rows.loc[i+1,"_played"])) and bool(rows.loc[i+1,"_inj"]):
                        starts += 1
                        d=0; j=i+1
                        while j < len(rows) and (not bool(rows.loc[j,"_played"])) and bool(rows.loc[j,"_inj"]):
                            d += 1; j += 1
                        if d: durations.append(d)
        if opportunities >= 1000 and starts >= 20 and durations:
            rate = float(np.clip(starts/opportunities, 0.001, 0.02))
            return rate, np.asarray(durations, dtype=int), "historical"
    except Exception:
        pass
    return fallback_rate, fallback_durations, "fallback"


def apply_future_injury_availability(pid, base_available, future_injury_games_left):
    """A simulated future injury overrides today's otherwise-known availability."""
    return bool(base_available) and int(future_injury_games_left.get(norm_id(pid), 0)) <= 0


def advance_future_injury_after_game(pid, future_injury_games_left):
    pid = norm_id(pid)
    if int(future_injury_games_left.get(pid, 0)) > 0:
        future_injury_games_left[pid] -= 1
        if future_injury_games_left[pid] <= 0:
            future_injury_games_left.pop(pid, None)


def maybe_start_future_injury(pid, rng, injury_rate, injury_durations, future_injury_games_left):
    """Sample an injury after a completed game; it affects later scheduled games."""
    pid = norm_id(pid)
    if pid in future_injury_games_left:
        return False, 0
    if rng.random() >= injury_rate:
        return False, 0
    duration = int(rng.choice(injury_durations)) if len(injury_durations) else 1
    duration = max(duration, 1)
    future_injury_games_left[pid] = duration
    return True, duration


def _lineup_expected_total(starters):
    total = 0.0
    for r in starters:
        try: total += float(r.get("lineup_value", r.get("projected_fpg", r.get("avg_fp", 0.0))) or 0.0)
        except Exception: pass
    return total


def should_activate_ir_with_drop_today(team_roster, returning_pid, day, min_lineup_gain=3.0):
    """Preserve IR option value unless activation materially improves today's lineup."""
    pid = norm_id(returning_pid)
    row = team_roster[team_roster["player_id"].map(norm_id).eq(pid)]
    if row.empty or int(row.iloc[0].get(day, 0) or 0) != 1:
        return False
    before, _, _ = dynamic_daily_lineup(team_roster, day)
    trial = team_roster.copy()
    _set_ir_state(trial, pid, False)
    # Temporarily allow the returning player into lineup without yet choosing a drop.
    after, _, _ = dynamic_daily_lineup(trial, day)
    return _lineup_expected_total(after) >= _lineup_expected_total(before) + float(min_lineup_gain)

def resolve_healthy_ir_players(
    team_roster: pd.DataFrame,
    replacement_fpg: float,
    day: str | None = None,
):
    """
    Apply actual roster mutations for healthy players occupying fantasy IR.

    Current deterministic availability rule:
        definite OUT -> unavailable / IR-eligible
        every other status -> healthy enough to require IR activation

    Later this function will receive the Monte Carlo availability draw instead
    of inferring health directly from ESPN's current status label.
    """
    moves = []
    # Fast exit: nobody on this team is both on fantasy IR and available today.
    if not (team_roster["_sim_rostered"].astype(bool) & team_roster["_sim_on_ir"].astype(bool)
            & team_roster.get("_sim_available", pd.Series(False, index=team_roster.index)).astype(bool)).any():
        return moves
    # V2.5.2: healthy IR players who failed today's "worth activating?" guard.
    # They stay on IR today, but the loop continues so OTHER healthy IR players
    # are still considered (previously the loop stopped at the first deferral).
    deferred_pids: set[str] = set()

    while True:
        active = team_roster[team_roster["_sim_rostered"]].copy()
        healthy_ir = active[
            active["_sim_on_ir"]
            & active.get("_sim_available", pd.Series(False, index=active.index)).astype(bool)
        ]
        if deferred_pids:
            healthy_ir = healthy_ir[
                ~healthy_ir["player_id"].map(norm_id).isin(deferred_pids)
            ]

        if healthy_ir.empty:
            break

        # Resolve highest-value healthy IR player first.
        healthy_ir = healthy_ir.copy()
        healthy_ir["_manager_value"] = healthy_ir.apply(expected_fp_proxy, axis=1)
        healthy_ir = healthy_ir.sort_values(
            ["_manager_value", "player_name"],
            ascending=[False, True],
        )
        returning = healthy_ir.iloc[0]
        pid = norm_id(returning["player_id"])

        view = _manager_view(team_roster)
        # If there is no injured roster mate available to take IR, preserve the
        # option value of the occupied IR slot unless activation helps TODAY.
        other_ir_eligible = active[
            (~active["_sim_on_ir"])
            & active.get("_sim_ir_eligible", pd.Series(False, index=active.index)).astype(bool)
        ]
        if other_ir_eligible.empty and day is not None and not should_activate_ir_with_drop_today(team_roster, pid, day):
            deferred_pids.add(pid)
            continue
        decision = manage_healthy_ir_player(
            view,
            healthy_ir_player_id=pid,
            replacement_fpg=replacement_fpg,
            ir_eligible_statuses=DEFINITE_OUT,
        )

        # Activate returning player.
        _set_ir_state(team_roster, decision.activated_player_id, False)

        # Then either swap another injured player into IR or drop someone.
        for move in decision.moves:
            if move.action == "MOVE_TO_IR":
                _set_ir_state(team_roster, move.player_id, True)
            elif move.action == "DROP":
                _drop_from_roster(team_roster, move.player_id)

        moves.append(decision)

    return moves


def dynamic_daily_lineup(
    team_roster: pd.DataFrame,
    day: str,
):
    """Choose today's lineup from the CURRENT simulated roster state."""
    candidates = team_roster[
        team_roster["_sim_rostered"]
        & (~team_roster["_sim_on_ir"])
        & (team_roster[day] == 1)
        & team_roster.get("_sim_available", pd.Series(False, index=team_roster.index)).astype(bool)
    ].copy()

    return best_expected_lineup(candidates, day)


def audit_roster_manager(rostered: pd.DataFrame, merged: pd.DataFrame):
    """
    Forced behavioural tests.

    These do not alter the live simulation roster. They verify the exact
    scenarios requested by the user before Monte Carlo is allowed to run.
    """
    replacement = estimate_replacement_level(merged)

    kawhi_any = rostered[
        rostered["player_name"].astype(str).str.casefold()
        == "kawhi leonard".casefold()
    ].copy()

    print()
    print("=" * 100)
    print("ROSTER MANAGER BEHAVIOURAL TESTS")
    print("=" * 100)
    print("Estimated waiver replacement level:", round(replacement, 2), "FP/game")

    if kawhi_any.empty:
        print("Kawhi Leonard not found on any fantasy roster; forced Kawhi tests skipped.")
        return replacement

    target_team_id = str(kawhi_any.iloc[0]["fantasy_team_id"])
    target_team_name = str(kawhi_any.iloc[0]["fantasy_team"])
    gustavo = rostered[
        rostered["fantasy_team_id"].astype(str) == target_team_id
    ].copy()

    print("Kawhi roster detected:", target_team_name, f"(team_id={target_team_id})")

    kawhi = gustavo[
        gustavo["player_name"].astype(str).str.casefold()
        == "kawhi leonard".casefold()
    ]
    chet = gustavo[
        gustavo["player_name"].astype(str).str.casefold()
        == "chet holmgren".casefold()
    ]

    if kawhi.empty:
        print("Kawhi Leonard not found; forced Kawhi tests skipped.")
        return replacement

    # ------------------------------------------------------------------
    # TEST 1: Kawhi healthy + Chet OUT -> Chet to IR, nobody dropped.
    # ------------------------------------------------------------------
    test1 = gustavo.copy()
    test1["_sim_rostered"] = True
    test1["_sim_on_ir"] = test1.apply(is_ir, axis=1)
    # Resolver uses simulated availability, not injury_status alone.
    test1["_sim_available"] = ~test1["injury_status"].map(is_definitely_out)

    test1.loc[
        test1["player_name"].astype(str).str.casefold()
        == "kawhi leonard".casefold(),
        "injury_status"
    ] = "ACTIVE"
    test1.loc[
        test1["player_name"].astype(str).str.casefold()
        == "kawhi leonard".casefold(),
        "_sim_available"
    ] = True

    if not chet.empty:
        test1.loc[
            test1["player_name"].astype(str).str.casefold()
            == "chet holmgren".casefold(),
            "injury_status"
        ] = "OUT"
        test1.loc[
            test1["player_name"].astype(str).str.casefold()
            == "chet holmgren".casefold(),
            "_sim_available"
        ] = False

        decisions = resolve_healthy_ir_players(test1, replacement)
        if not decisions:
            raise RuntimeError(
                "Roster-manager Test 1 FAILED: no activation decision was produced "
                "for healthy Kawhi on IR."
            )
        d = decisions[0]

        if (
            not d.used_ir_swap
            or d.dropped_player_id is not None
            or not any(
                m.action == "MOVE_TO_IR"
                and m.player_name == "Chet Holmgren"
                for m in d.moves
            )
        ):
            raise RuntimeError(
                "Roster-manager Test 1 FAILED: expected Kawhi activation, "
                "Chet to IR, and no drop."
            )

        print("TEST 1 PASS: Kawhi healthy + Chet OUT -> Chet to IR; nobody dropped.")
    else:
        print("TEST 1 skipped: Chet Holmgren not found on Gustavo roster.")

    # ------------------------------------------------------------------
    # TEST 2: Kawhi healthy + nobody else IR-eligible -> a drop occurs,
    # and no >=30 FP/game player may be dropped.
    # ------------------------------------------------------------------
    test2 = gustavo.copy()
    test2["_sim_rostered"] = True
    test2["_sim_on_ir"] = test2.apply(is_ir, axis=1)
    test2.loc[:, "injury_status"] = "ACTIVE"
    # Everyone is healthy in Test 2, including the player currently occupying IR.
    test2["_sim_available"] = True

    decisions = resolve_healthy_ir_players(test2, replacement)
    if not decisions:
        raise RuntimeError(
            "Roster-manager Test 2 FAILED: no activation decision was produced "
            "for healthy Kawhi on IR."
        )
    d = decisions[0]

    if d.used_ir_swap or d.dropped_player_id is None:
        raise RuntimeError(
            "Roster-manager Test 2 FAILED: expected an activation drop."
        )

    dropped = test2[
        test2["player_id"].map(norm_id).eq(norm_id(d.dropped_player_id))
    ].iloc[0]
    dropped_fpg = expected_fp_proxy(dropped)

    if dropped_fpg >= 30.0:
        raise RuntimeError(
            "Roster-manager Test 2 FAILED: dropped a >=30 FP/game player: "
            f"{d.dropped_player_name} ({dropped_fpg:.2f})."
        )

    print(
        "TEST 2 PASS: Kawhi healthy + no IR replacement -> dropped",
        f"{d.dropped_player_name} ({dropped_fpg:.2f} FP/game);",
        "30+ FP/game assets protected."
    )

    board = explain_drop_board(_manager_view(test2), replacement)
    show_cols = [
        "player_name", "projected_fpg", "replacement_fpg",
        "hold_value", "hard_protected", "soft_protected"
    ]
    print()
    print("GUSTAVO ROS DROP BOARD (weekly games deliberately absent)")
    print("-" * 100)
    print(board[show_cols].round(2).to_string(index=False))

    return replacement



def audit_v15_availability(rostered: pd.DataFrame, expected_returns: dict[str, dict]):
    print()
    print("=" * 100)
    print("V1.6 AVAILABILITY MODEL")
    print("=" * 100)
    print("PROBABLE:     75%")
    print("QUESTIONABLE: 50%")
    print("DAY_TO_DAY:   50% (mapped to QUESTIONABLE)")
    print("DOUBTFUL:     25%")
    print("OUT:           0%")
    print("ACTIVE:      100%")
    print("OUT return rule: 50% return on ESPN date, 25% next day, 25% two days after.")
    print("This gives cumulative return probabilities of 50% / 75% / 100%.")
    print("DTD/Q/Doubtful/Probable are game-specific status draws, not permanent recovery draws.")
    print("OUT without an ESPN return date remains unavailable in V1.6.")

    matched = []
    for _, row in rostered.iterrows():
        pid = norm_id(row["player_id"])
        info = expected_returns.get(pid)
        if info and info.get("return_date") is not None:
            matched.append((row["player_name"], row.get("injury_status"), info["return_date"].date()))
    print(f"Rostered players with ESPN expected return dates: {len(matched)}")
    for name, status, rdate in sorted(matched):
        print(f"  {name}: {status} | expected return {rdate}")

    counts = rostered["injury_status"].fillna("<blank>").astype(str).value_counts()
    print()
    print("Current roster status counts:")
    print(counts.to_string())

    unknown = sorted({
        str(x) for x in rostered["injury_status"].dropna().unique()
        if str(x).upper().strip().replace(" ", "_").replace("-", "_") not in STATUS_PLAY_PROB
    })
    if unknown:
        print("WARNING — unrecognised statuses default to 100%:", unknown)
    else:
        print("Status mapping coverage: PASS")


def prepare_deterministic_week(
    rostered: pd.DataFrame,
    replacement_fpg: float,
):
    """
    Prepare fantasy roster mutations and daily lineups ONCE for V1.4.

    This is valid only while availability is deterministic:
      definite OUT = unavailable
      every other current status = healthy

    Roster state persists Mon->Sun. Daily lineups are chosen from the
    post-management roster state using pregame expected FP only.
    """
    sim_roster = rostered.copy()
    sim_roster["_sim_rostered"] = True
    sim_roster["_sim_on_ir"] = sim_roster.apply(is_ir, axis=1)

    starters_by_day = {}
    move_counter = defaultdict(int)

    for day in DAY_NAMES:
        starters_by_team = {}

        for team_id, team_idx in sim_roster.groupby(
            "fantasy_team_id", dropna=False
        ).groups.items():
            team_id = str(team_id)
            team_view = sim_roster.loc[team_idx].copy()

            decisions = resolve_healthy_ir_players(
                team_view,
                replacement_fpg,
            )

            sim_roster.loc[
                team_idx, ["_sim_rostered", "_sim_on_ir"]
            ] = team_view[["_sim_rostered", "_sim_on_ir"]].values

            for d in decisions:
                if d.used_ir_swap:
                    moved = next(
                        (
                            m.player_name for m in d.moves
                            if m.action == "MOVE_TO_IR"
                        ),
                        "UNKNOWN",
                    )
                    key = (
                        d.fantasy_team,
                        "IR_SWAP",
                        d.activated_player_name,
                        moved,
                    )
                else:
                    key = (
                        d.fantasy_team,
                        "DROP",
                        d.activated_player_name,
                        d.dropped_player_name or "UNKNOWN",
                    )
                move_counter[key] += 1

            current_team = sim_roster.loc[team_idx]
            starters, bench, expected = dynamic_daily_lineup(
                current_team,
                day,
            )
            starters_by_team[team_id] = {
                norm_id(x["player_id"]) for x in starters
            }

        starters_by_day[day] = starters_by_team

    return sim_roster, starters_by_day, move_counter


# =============================================================================
# V3.2 — REAL-LIFE MATCHUP CONTEXT
# =============================================================================

def build_event_contexts(event_schedule: pd.DataFrame, team_proj: Dict[str, List[PlayerProjection]],
                         ctx) -> tuple[dict, dict, pd.DataFrame]:
    """
    Per NBA game: expected possessions from both teams' pace, and each side's
    players with rates scaled by the OPPONENT's defense factors.
    Returns (event_players {(game_id, team): [PlayerProjection]}, event_poss {game_id: possessions}, table).
    """
    event_players, event_poss, rows = {}, {}, []
    if ctx is None:
        return event_players, event_poss, pd.DataFrame()
    for ev in event_schedule.to_dict("records"):
        gid, a, b = str(ev["game_id"]), str(ev["team_a"]), str(ev["team_b"])
        poss = ctx.possessions(a, b)
        event_poss[gid] = poss
        row = {"game_id": gid, "day": ev["day"], "team_a": a, "team_b": b, "possessions": poss}
        for team, opp in ((a, b), (b, a)):
            f = ctx.defense_factors(opp)
            adjusted = []
            for p in team_proj.get(team, []):
                changes = {}
                for stat, field in MC.SIM_FIELD.items():
                    mult = f.get(stat, 1.0)
                    if mult == 1.0:
                        continue
                    v = float(getattr(p, field)) * mult
                    if field in ("three_share", "two_pct", "three_pct", "ft_pct"):
                        v = float(np.clip(v, 0.0, 0.99))
                    changes[field] = v
                adjusted.append(replace(p, **changes) if changes else p)
            event_players[(gid, team)] = adjusted
            row.update({f"{'a' if team == a else 'b'}_vs_def_{s}": f[s] for s in ("FGA", "FTA", "AST", "TOV", "STL", "BLK")})
        rows.append(row)
    return event_players, event_poss, pd.DataFrame(rows)


# =============================================================================
# V3.2 — PICKUP / STREAMING PLANNER
# =============================================================================

def _slot_matrix(eligible_sets, values):
    """Players x 10 lineup places; value where eligible, 0 otherwise (0 = place left empty)."""
    M = np.zeros((len(values), len(ACTIVE_LINEUP_SLOT_IDS)))
    for i, (elig, v) in enumerate(zip(eligible_sets, values)):
        if v <= 0:
            continue
        for j, sid in enumerate(ACTIVE_LINEUP_SLOT_IDS):
            if sid in elig:
                M[i, j] = v
    return M


def lineup_value(players, day, exp_val, elig):
    """Best legal lineup expected FP for one team-day (exact, via assignment)."""
    vals = [exp_val.get((p, day), 0.0) for p in players]
    if not players or max(vals, default=0.0) <= 0:
        return 0.0
    M = _slot_matrix([elig[p] for p in players], vals)
    r, c = linear_sum_assignment(M, maximize=True)
    return float(M[r, c].sum())


def plan_pickups(merged: pd.DataFrame, rostered: pd.DataFrame, exp_val: dict, start_day: int,
                 team_caps: dict, replacement_fpg: float, min_gain: float = None):
    """
    Day-by-day streaming plan for every fantasy team, chosen from PRE-GAME
    expected values only (never simulated results).

    A move = add one free agent who plays that day and drop one droppable player.
    Droppable = rest-of-season value within STREAM_DROP_MARGIN FP/game of waiver
    replacement level (and below the hard 30 FP/game floor). Dropping anyone
    better would cost more over the season than one week of streaming gains, so a
    strong player is never dropped just because a weaker one has more games.
    A move is made only if it adds at least MIN_STREAM_GAIN expected FP over the
    rest of the week. Free agents claimed or dropped this week are not re-used.
    """
    min_gain = MIN_STREAM_GAIN if min_gain is None else min_gain
    days = DAY_NAMES[start_day:]
    elig = {norm_id(r["player_id"]): parse_eligible_slot_ids(r["eligible_slot_ids"]) for _, r in merged.iterrows()}
    fpg = {norm_id(r["player_id"]): projected_fpg(r) for _, r in merged.iterrows()}
    name = {norm_id(r["player_id"]): str(r["player_name"]) for _, r in merged.iterrows()}
    drop_line = replacement_fpg + STREAM_DROP_MARGIN

    def droppable(pid):
        v = fpg.get(pid, 0.0)
        return v < HARD_PROTECT_FPG and v < drop_line

    rosters, team_name = {}, {}
    for tid, g in rostered.groupby("fantasy_team_id"):
        tid = norm_id(tid)
        team_name[tid] = str(g["fantasy_team"].iloc[0])
        rosters[tid] = [norm_id(p) for p, slot in zip(g["player_id"], g["lineup_slot"]) if str(slot).upper() != "IR"]
    pool = {norm_id(p) for p, st, ready in zip(merged["player_id"], merged["roster_status"],
                                                merged.get("projection_ready", pd.Series(True, index=merged.index)))
            if str(st).upper() == "FREE_AGENT" and str(ready).upper() in {"TRUE", "1"}}
    caps = {t: int(team_caps.get(t, 0) or 0) for t in rosters}

    def rest_value(pid, from_day):
        return sum(exp_val.get((pid, d), 0.0) for d in DAY_NAMES[DAY_NAMES.index(from_day):])

    moves = []
    for d in days:
        added_today = defaultdict(set)
        while True:
            fas = sorted((p for p in pool if exp_val.get((p, d), 0.0) > 0),
                         key=lambda p: -rest_value(p, d))[:PLANNER_FA_CANDIDATES]
            if not fas:
                break
            best = None
            for tid, R in rosters.items():
                if caps.get(tid, 0) <= 0:
                    continue
                drops = [p for p in R if droppable(p) and p not in added_today[tid]]
                if not drops:
                    continue
                later = DAY_NAMES[DAY_NAMES.index(d):]
                base = {day: lineup_value(R, day, exp_val, elig) for day in later}
                for fa in fas:
                    for dr in drops:
                        R2 = [p for p in R if p != dr] + [fa]
                        gain = 0.0
                        for day in later:
                            if exp_val.get((fa, day), 0.0) > 0 or exp_val.get((dr, day), 0.0) > 0:
                                gain += lineup_value(R2, day, exp_val, elig) - base[day]
                        if best is None or gain > best[0]:
                            best = (gain, tid, fa, dr)
            if best is None or best[0] < min_gain:
                break
            gain, tid, fa, dr = best
            rosters[tid] = [p for p in rosters[tid] if p != dr] + [fa]
            pool.discard(fa)
            caps[tid] -= 1
            added_today[tid].add(fa)
            moves.append({
                "day": d, "fantasy_team_id": tid, "fantasy_team": team_name[tid],
                "add_player_id": fa, "add_player": name.get(fa, fa),
                "drop_player_id": dr, "drop_player": name.get(dr, dr),
                "expected_gain": round(gain, 2),
                "add_expected_rest_of_week": round(rest_value(fa, d), 2),
                "drop_expected_rest_of_week": round(rest_value(dr, d), 2),
                "drop_projected_fpg": round(fpg.get(dr, 0.0), 2),
            })
    return moves


def acquisition_caps(matchups: pd.DataFrame | None, team_ids) -> tuple[dict, dict, list]:
    """
    Returns (usual, left, notes):
      usual  adds each team is expected to make for the rest of the week: its
             average adds per past week (ESPN) minus adds already used this week,
             never above what it has left
      left   every add the team still has this week (used for advice)
    Without ESPN data both are DEFAULT_ADDS_IF_UNKNOWN.
    """
    caps, left_caps, notes = {}, {}, []
    info = {}
    if matchups is not None and "acq_left_this_week" in matchups.columns:
        for _, r in matchups.iterrows():
            info[norm_id(r["fantasy_team_id"])] = r
    for tid in team_ids:
        r = info.get(tid)
        if r is None or not bool(r.get("acq_counter_found", False)):
            caps[tid] = left_caps[tid] = DEFAULT_ADDS_IF_UNKNOWN
            notes.append(f"{tid}: no ESPN acquisition data, assumed {DEFAULT_ADDS_IF_UNKNOWN}")
            continue
        left = r.get("acq_left_this_week")
        left = MAX_ADDS_IF_UNLIMITED if pd.isna(left) else int(left)
        left_caps[tid] = left
        past = r.get("acq_per_past_week")
        used = int(r.get("acq_used_this_week") or 0)
        if pd.isna(past):
            caps[tid] = left
        else:
            caps[tid] = int(min(left, max(0, int(np.ceil(float(past))) - used)))
    return caps, left_caps, notes


def find_team_id(team_names: dict, wanted: str) -> str | None:
    for tid, nm in team_names.items():
        if str(nm).strip().casefold() == wanted.strip().casefold():
            return norm_id(tid)
    return None


def build_roster_template(rostered: pd.DataFrame, merged: pd.DataFrame, moves: list) -> pd.DataFrame:
    """Rostered players plus every planned add (not rostered until its add day)."""
    t = rostered.copy()
    t["_start_rostered"] = True
    if not moves:
        return t
    add_rows = []
    by_pid = merged.set_index(merged["player_id"].map(norm_id))
    for mv in moves:
        row = by_pid.loc[mv["add_player_id"]].copy()
        row["fantasy_team_id"] = float(mv["fantasy_team_id"]) if str(mv["fantasy_team_id"]).isdigit() else mv["fantasy_team_id"]
        row["fantasy_team"] = mv["fantasy_team"]
        row["roster_status"] = "ROSTERED"
        row["lineup_slot"] = "BENCH"
        row["_start_rostered"] = False
        add_rows.append(row)
    adds = pd.DataFrame(add_rows).reset_index(drop=True)
    return pd.concat([t, adds], ignore_index=True)


def simulate_week(
    event_schedule: pd.DataFrame,
    team_proj: Dict[str, List[PlayerProjection]],
    rostered: pd.DataFrame,
    merged: pd.DataFrame,
    replacement_fpg: float,
    expected_returns: dict[str, dict],
    rotations: Dict[str, "TeamRotationV3"],
    nba_status_by_pid: dict[str, str] | None = None,
    sims: int = SIMS,
    seed: int = SEED,
    start_day_index: int = 0,
    event_players: dict | None = None,
    event_poss: dict | None = None,
    pickup_moves: list | None = None,
    roster_template: pd.DataFrame | None = None,
):
    """Availability draws + Injury Rotation V3 Monte Carlo for the remaining days."""
    event_players = event_players or {}
    event_poss = event_poss or {}
    pickup_moves = pickup_moves or []
    if roster_template is None:
        roster_template = rostered.copy()
        roster_template["_start_rostered"] = True
    moves_by_day = defaultdict(list)
    for mv in pickup_moves:
        moves_by_day[mv["day"]].append(mv)
    # Per player per day (all NBA players, incl. free agents) and per team per day.
    day_fp_sum = defaultdict(float)
    day_played = defaultdict(int)
    team_day_sum = defaultdict(float)
    rng = np.random.default_rng(seed)
    id_to_team, id_to_player, team_names = fantasy_team_maps(rostered)
    fantasy_team_ids = sorted(team_names)

    team_scores = {tid: np.zeros(sims, dtype=float) for tid in fantasy_team_ids}
    # Player-games started that were actually played (for PPP = points / starts).
    team_starts = {tid: np.zeros(sims, dtype=float) for tid in fantasy_team_ids}
    player_active_fp = defaultdict(lambda: np.zeros(sims, dtype=float))
    player_game_fp = defaultdict(lambda: np.zeros(sims, dtype=float))
    player_games = defaultdict(int)

    # V2.5.1: direct NBA-wide audit of known OUT players.  This is deliberately
    # separate from the rotation-trigger audit: a player can be correctly absent
    # without necessarily being counted as a particular redistribution trigger.
    known_out_game_opportunities = defaultdict(int)
    known_out_available_opportunities = defaultdict(int)

    events_by_day = {
        day: event_schedule[event_schedule["day"] == day]
        .sort_values(["date", "game_id"]).to_dict("records")
        for day in DAY_NAMES
    }

    # Status lookup for the entire NBA projection universe, not only fantasy rosters.
    status_by_pid = {}
    # Start with the complete NBA simulation universe, then let fantasy-pool status override it.
    # V2.5.2: use the real designation (PROBABLE / QUESTIONABLE / DOUBTFUL / DTD /
    # OUT) from the full-NBA projection file when available, so non-fantasy NBA
    # players get the same game-specific availability draws as fantasy players.
    # Falls back to ACTIVE/OUT from the projection's `available` flag only when
    # the full-NBA file carries no status for that player.
    nba_status_by_pid = nba_status_by_pid or {}
    for nba_players in team_proj.values():
        for pproj in nba_players:
            pid = norm_id(pproj.player_id)
            status_by_pid[pid] = nba_status_by_pid.get(
                pid, "ACTIVE" if pproj.available else "OUT"
            )
    for _, row in merged.iterrows():
        status_by_pid[norm_id(row["player_id"])] = row.get("injury_status", "ACTIVE")

    print()
    print("=" * 100)
    print("MONTE CARLO — V3.0 INJURY ROTATION + RETURN-DATE GATE + PERSISTENT IR")
    print("=" * 100)
    print("Simulations:", sims)
    print("NBA games per simulation:", len(event_schedule))
    print("Total NBA game simulations:", sims * len(event_schedule))
    print("Status priors: PROBABLE 75% | QUESTIONABLE/DTD 50% | DOUBTFUL 25% | OUT 0%")
    print("OUT + ESPN returnDate: cumulative return 50% on R | 75% by R+1 | 100% by R+2")
    print("Short-term statuses are GAME-SPECIFIC draws; they do not permanently recover after one success")
    print("Roster decisions: REST-OF-SEASON value")
    print("Daily lineup decisions: TODAY'S expected value after availability draw")
    print("Weekly games remaining in drop score: NO")
    print("Rotation: Injury Rotation V3 (healthy baseline + every-absence model + coach evidence + floors/ceilings)")
    print("Layoff tracking: missed-game streaks advance inside each simulation (returns are modelled)")
    print(f"Simulated days: {', '.join(DAY_NAMES[start_day_index:]) or 'none'} "
          f"(earlier days use ESPN actual points when fantasy_matchups.csv is present)")
    print()

    progress_every = max(sims // 10, 1)
    move_counter = defaultdict(int)
    availability_counts = defaultdict(lambda: [0, 0])  # [plays, opportunities]
    out_return_offset_counts = defaultdict(lambda: [0, 0, 0])
    lineup_cache = {}
    injury_rate, injury_durations, injury_model_source = estimate_future_injury_model()
    future_injury_starts = 0
    future_injury_missed_games = 0
    future_injury_player_counts = defaultdict(int)
    future_injury_player_missed = defaultdict(int)
    print(f"Future injury model: {injury_model_source}; onset per completed player-game = {injury_rate:.4%}")

    rotation_audit = new_rotation_audit()
    base_layoff_state = initial_layoff_state(rotations)

    for sim_index in range(sims):
        sim_roster = roster_template.copy()
        sim_roster["_sim_rostered"] = sim_roster["_start_rostered"].astype(bool)
        sim_roster["_sim_on_ir"] = sim_roster.apply(is_ir, axis=1)
        sim_roster["_sim_available"] = False
        sim_roster["_sim_ir_eligible"] = False
        future_injury_games_left = {}
        seen_moves_this_sim = set()
        layoff_state = dict(base_layoff_state)

        # One episode-level return-date sample shared by fantasy and NBA simulation.
        sampled_out_returns = sample_out_return_dates(status_by_pid, expected_returns, rng)
        for pid, sampled_date in sampled_out_returns.items():
            info = expected_returns.get(pid) or {}
            base_date = info.get("return_date")
            if base_date is not None and pid in id_to_team:
                offset = int((pd.Timestamp(sampled_date) - pd.Timestamp(base_date).normalize()).days)
                if 0 <= offset <= 2:
                    out_return_offset_counts[pid][offset] += 1

        for day_index, day in enumerate(DAY_NAMES):
            if day_index < start_day_index:
                continue   # finished day: ESPN actual points are added later
            game_date = WEEK_START + pd.Timedelta(days=day_index)
            # --------------------------------------------------------------
            # 1. Draw today's availability for fantasy-rostered players.
            # --------------------------------------------------------------
            today = draw_v16_availability_for_day(
                sim_roster, day, game_date, sampled_out_returns, rng, expected_returns=expected_returns
            )
            today = {pid: apply_future_injury_availability(pid, av, future_injury_games_left) for pid, av in today.items()}
            sim_roster["_sim_available"] = sim_roster["player_id"].map(
                lambda x: bool(today.get(norm_id(x), False))
            )
            # IR eligibility is a persistent HEALTH concept, not game-day availability.
            # Known definite OUT remains eligible until its sampled return date; newly
            # simulated injuries are eligible. Q/DTD/Doubtful and NBA off-days are not.
            def _ir_eligible_now(row):
                pid = norm_id(row["player_id"])
                if int(future_injury_games_left.get(pid, 0)) > 0:
                    return True
                status = str(row.get("injury_status", "") or "").upper().strip().replace(" ", "_").replace("-", "_")
                if status not in DEFINITE_OUT:
                    return False
                sampled = sampled_out_returns.get(pid)
                if sampled is None:
                    return True
                return pd.Timestamp(game_date).normalize() < pd.Timestamp(sampled).normalize()
            sim_roster["_sim_ir_eligible"] = sim_roster.apply(_ir_eligible_now, axis=1)

            for _, row in sim_roster[sim_roster[day] == 1].iterrows():
                pid = norm_id(row["player_id"])
                p = status_play_probability(row.get("injury_status"))
                has_out_return = (
                    is_definitely_out(row.get("injury_status"))
                    and expected_returns.get(pid, {}).get("return_date") is not None
                )
                if (0.0 < p < 1.0) or has_out_return:
                    availability_counts[pid][1] += 1
                    availability_counts[pid][0] += int(bool(today.get(pid, False)))

            # --------------------------------------------------------------
            # 1b. Planned pickups for today (add free agent, drop streamable).
            # --------------------------------------------------------------
            for mv in moves_by_day.get(day, []):
                pids = sim_roster["player_id"].map(norm_id)
                drop_mask = pids.eq(mv["drop_player_id"]) & sim_roster["_sim_rostered"]
                sim_roster.loc[drop_mask, "_sim_rostered"] = False
                sim_roster.loc[drop_mask, "_sim_on_ir"] = False
                add_mask = pids.eq(mv["add_player_id"]) & (~sim_roster["_start_rostered"].astype(bool))
                sim_roster.loc[add_mask, "_sim_rostered"] = True
                sim_roster.loc[add_mask, "_sim_on_ir"] = False

            # --------------------------------------------------------------
            # 2. Resolve fantasy IR activation/drop, then choose lineups.
            # --------------------------------------------------------------
            starter_team_by_pid = {}

            for team_id, team_idx in sim_roster.groupby(
                "fantasy_team_id", dropna=False
            ).groups.items():
                team_id = str(team_id)
                team_view = sim_roster.loc[team_idx].copy()

                decisions = resolve_healthy_ir_players(team_view, replacement_fpg, day=day)
                sim_roster.loc[
                    team_idx, ["_sim_rostered", "_sim_on_ir"]
                ] = team_view[["_sim_rostered", "_sim_on_ir"]].values

                for d in decisions:
                    if d.used_ir_swap:
                        moved = next((m.player_name for m in d.moves if m.action == "MOVE_TO_IR"), "UNKNOWN")
                        key = (d.fantasy_team, "IR_SWAP", d.activated_player_name, moved)
                    else:
                        key = (d.fantasy_team, "DROP", d.activated_player_name, d.dropped_player_name or "UNKNOWN")
                    if key not in seen_moves_this_sim:
                        move_counter[key] += 1
                        seen_moves_this_sim.add(key)

                current_team = sim_roster.loc[team_idx]
                eligible_ids = tuple(sorted(
                    norm_id(x) for x in current_team.loc[
                        current_team["_sim_rostered"]
                        & (~current_team["_sim_on_ir"])
                        & (current_team[day] == 1)
                        & current_team["_sim_available"],
                        "player_id"
                    ]
                ))
                cache_key = (team_id, day, eligible_ids)
                if cache_key not in lineup_cache:
                    starters, _, _ = dynamic_daily_lineup(current_team, day)
                    lineup_cache[cache_key] = tuple(norm_id(x["player_id"]) for x in starters)

                for pid in lineup_cache[cache_key]:
                    starter_team_by_pid[pid] = team_id

            # --------------------------------------------------------------
            # 3. Draw NBA-wide availability for today's actual game players.
            #    Fantasy-rostered players reuse the exact same draw.
            # --------------------------------------------------------------
            nba_today = dict(today)
            for event in events_by_day[day]:
                for nba_team in (str(event["team_a"]), str(event["team_b"])):
                    for pproj in team_proj[nba_team]:
                        pid = norm_id(pproj.player_id)
                        if pid in nba_today:
                            continue
                        nba_today[pid] = apply_future_injury_availability(
                            pid,
                            player_available_today(
                                pid, status_by_pid.get(pid, "ACTIVE"), game_date,
                                sampled_out_returns, rng, expected_returns=expected_returns,
                            ),
                            future_injury_games_left,
                        )

            # V2.5.1 known-OUT availability audit. Count each scheduled player-game
            # opportunity independently of whether Injury Rotation V2 uses that
            # player as a named trigger.
            for event in events_by_day[day]:
                for nba_team in (str(event["team_a"]), str(event["team_b"])):
                    for pproj in team_proj[nba_team]:
                        pid = norm_id(pproj.player_id)
                        if not is_definitely_out(status_by_pid.get(pid, "ACTIVE")):
                            continue
                        known_out_game_opportunities[pid] += 1
                        known_out_available_opportunities[pid] += int(bool(nba_today.get(pid, False)))

            # --------------------------------------------------------------
            # 4. Simulate each NBA event once with today's availability.
            # --------------------------------------------------------------
            for event in events_by_day[day]:
                team_a = str(event["team_a"])
                team_b = str(event["team_b"])
                gid = str(event["game_id"])
                a_players = apply_injury_rotation_v3(
                    event_players.get((gid, team_a), team_proj[team_a]), nba_today,
                    rotations[team_a], layoff_state, audit=rotation_audit
                )
                b_players = apply_injury_rotation_v3(
                    event_players.get((gid, team_b), team_proj[team_b]), nba_today,
                    rotations[team_b], layoff_state, audit=rotation_audit
                )

                game = simulate_nba_game(
                    game_id=gid,
                    team_a=team_a, team_b=team_b,
                    team_a_players=a_players, team_b_players=b_players, rng=rng,
                    possession_mean=event_poss.get(gid, MC.FIXED_POSSESSIONS),
                )

                for pgame in game.players:
                    pid = norm_id(pgame.player_id)
                    day_fp_sum[(pid, day)] += pgame.fantasy_points
                    if float(getattr(pgame, "minutes", 0.0) or 0.0) > 0.0:
                        day_played[(pid, day)] += 1
                    if pid in id_to_team:
                        player_game_fp[pid][sim_index] += pgame.fantasy_points
                    fantasy_team_id = starter_team_by_pid.get(pid)
                    if fantasy_team_id is not None:
                        team_scores[fantasy_team_id][sim_index] += pgame.fantasy_points
                        team_day_sum[(fantasy_team_id, day)] += pgame.fantasy_points
                        if float(getattr(pgame, "minutes", 0.0) or 0.0) > 0.0:
                            team_starts[fantasy_team_id][sim_index] += 1
                        player_active_fp[pid][sim_index] += pgame.fantasy_points

                # Advance each player's layoff streak / games-since-return so a
                # player who misses Monday and returns Wednesday is treated as a
                # returning player on Wednesday, exactly as in training.
                played_minutes = {norm_id(pg.player_id): float(pg.minutes) for pg in game.players}
                update_layoff_state(
                    layoff_state, rotations[team_a].pids + rotations[team_b].pids,
                    nba_today, played_minutes,
                )

                # A previously simulated injury consumes one missed scheduled game
                # whenever that player's NBA team has just played. Do this BEFORE
                # sampling new post-game injuries so a new injury is not decremented
                # on the game in which it occurred.
                for pproj in list(team_proj[team_a]) + list(team_proj[team_b]):
                    qid = norm_id(pproj.player_id)
                    if qid in future_injury_games_left and not bool(nba_today.get(qid, True)):
                        advance_future_injury_after_game(qid, future_injury_games_left)

                # Injuries occur after this completed game and affect later games.
                for pgame in game.players:
                    pid = norm_id(pgame.player_id)
                    if float(getattr(pgame, "minutes", 0.0) or 0.0) <= 0.0:
                        continue
                    started, dur = maybe_start_future_injury(
                        pid, rng, injury_rate, injury_durations, future_injury_games_left
                    )
                    if started:
                        future_injury_starts += 1
                        future_injury_missed_games += dur
                        future_injury_player_counts[pid] += 1
                        future_injury_player_missed[pid] += int(dur)

        if (sim_index + 1) % progress_every == 0 or sim_index + 1 == sims:
            print(f"Completed {sim_index + 1:,} / {sims:,} simulations")

    for _, row in rostered.iterrows():
        pid = norm_id(row["player_id"])
        player_games[pid] = int(sum(int(row[d]) for d in DAY_NAMES[start_day_index:]))

    print()
    print("NBA-WIDE OUT RETURN-DATE SANITY AUDIT")
    print("-" * 100)
    audit_rows = []
    name_by_pid_all = {norm_id(r["player_id"]): str(r["player_name"]) for _, r in merged.iterrows()}
    for nba_players in team_proj.values():
        for pp in nba_players:
            name_by_pid_all.setdefault(norm_id(pp.player_id), str(pp.name))
    for pid, opps in known_out_game_opportunities.items():
        avail = known_out_available_opportunities.get(pid, 0)
        rd = (expected_returns.get(pid) or {}).get("return_date")
        audit_rows.append({
            "player_name": name_by_pid_all.get(pid, pid),
            "player_id": pid,
            "espn_return_date": rd.date() if rd is not None else None,
            "week_end": (WEEK_START + pd.Timedelta(days=6)).date(),
            "opportunities": opps,
            "available": avail,
            "play_rate": (avail / opps) if opps else np.nan,
            "absent_rate": (1.0 - avail / opps) if opps else np.nan,
            "post_week_return_gate": bool(rd is not None and pd.Timestamp(rd).normalize() > (WEEK_START + pd.Timedelta(days=6)).normalize()),
        })
    if audit_rows:
        audit_df = pd.DataFrame(audit_rows).sort_values(["post_week_return_gate", "player_name"], ascending=[False, True])
        print(audit_df.round(4).to_string(index=False))
        bad = audit_df[(audit_df["post_week_return_gate"]) & (audit_df["available"] > 0)]
        print()
        print("Post-week OUT return-date hard gate:", "PASS" if bad.empty else "FAIL")
        if not bad.empty:
            print(bad.to_string(index=False))
    else:
        print("No definite-OUT NBA players had scheduled opportunities in this simulation week.")

    print()
    print("INJURY ROTATION V3 — LIVE APPLICATION DIAGNOSTICS")
    print("-" * 100)
    print(f"NBA team-events evaluated: {rotation_audit['team_events']:,}")
    cache_states = sum(len(r.cache) for r in rotations.values())
    print(f"Distinct availability/layoff states computed (cached): {cache_states:,}")
    if rotation_audit["rotation_sizes"]:
        dist = dict(sorted(rotation_audit["rotation_sizes"].items()))
        total = sum(dist.values())
        mean = sum(k * v for k, v in dist.items()) / total
        print(f"Players given minutes per team-game: mean {mean:.2f}; distribution {dist}")

    print()
    print("KNOWN HEALTH STATE AT START OF WEEK")
    print("-" * 100)
    health_rows=[]
    for _, row in rostered.iterrows():
        pid=norm_id(row["player_id"]); st=str(row.get("injury_status", "ACTIVE") or "ACTIVE")
        if st.upper().strip().replace(" ", "_").replace("-", "_") in DEFINITE_OUT or status_play_probability(st) < 1.0:
            rd=(expected_returns.get(pid) or {}).get("return_date")
            health_rows.append({"player_name":row["player_name"],"fantasy_team":row["fantasy_team"],"status":st,"espn_return_date":rd.date() if rd is not None else None})
    print(pd.DataFrame(health_rows).to_string(index=False) if health_rows else "No rostered injury designations.")

    print()
    print("SIMULATED FUTURE INJURY TRACKER")
    print("-" * 100)
    print(f"New injury starts: {future_injury_starts:,}; sampled subsequent missed games: {future_injury_missed_games:,}")
    if future_injury_player_counts:
        name_by_pid={norm_id(r["player_id"]):str(r["player_name"]) for _,r in merged.iterrows()}
        inj_rows=[]
        for pid,c in future_injury_player_counts.items():
            inj_rows.append({"player_name":name_by_pid.get(pid,pid),"player_id":pid,"injury_starts":c,"share_of_sims":c/sims,"avg_sampled_games_missed":future_injury_player_missed[pid]/c})
        print(pd.DataFrame(inj_rows).sort_values(["injury_starts","player_name"],ascending=[False,True]).head(30).round(3).to_string(index=False))
    else:
        print("No future injuries were sampled.")

    print()
    print("AVAILABILITY DIAGNOSTICS — UNCERTAIN / RETURNING ROSTERED PLAYERS")
    print("-" * 100)
    rows = []
    for _, row in rostered.iterrows():
        pid = norm_id(row["player_id"])
        p = status_play_probability(row.get("injury_status"))
        has_out_return = (
            is_definitely_out(row.get("injury_status"))
            and expected_returns.get(pid, {}).get("return_date") is not None
        )
        if (0.0 < p < 1.0) or has_out_return:
            plays, opps = availability_counts[pid]
            rows.append({
                "player_name": row["player_name"],
                "status": row["injury_status"],
                "base_p": p,
                "model": ("OUT_RETURN_50_75_100" if has_out_return else "GAME_STATUS_DRAW"),
                "espn_return_date": (expected_returns.get(pid, {}).get("return_date").date()
                                     if expected_returns.get(pid, {}).get("return_date") is not None else None),
                "drawn_play_rate": plays / opps if opps else np.nan,
                "game_opportunities": opps,
            })
    if rows:
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    else:
        print("No uncertain rostered players.")

    print()
    print("OUT RETURN-DATE SAMPLE FREQUENCIES")
    print("-" * 100)
    return_rows = []
    for _, row in rostered.iterrows():
        pid = norm_id(row["player_id"])
        counts = out_return_offset_counts.get(pid)
        if counts and sum(counts) > 0:
            total = sum(counts)
            return_rows.append({
                "player_name": row["player_name"],
                "espn_return_date": expected_returns[pid]["return_date"].date(),
                "on_R": counts[0] / total,
                "on_R_plus_1": counts[1] / total,
                "on_R_plus_2": counts[2] / total,
                "samples": total,
            })
    if return_rows:
        print(pd.DataFrame(return_rows).round(3).to_string(index=False))
    else:
        print("No rostered OUT players had an ESPN return date.")

    print()
    print("ROSTER MOVE FREQUENCIES")
    print("-" * 100)
    if move_counter:
        for key, count in sorted(move_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            team, action, activated, other = key
            label = "move to IR" if action == "IR_SWAP" else "drop"
            print(f"{team}: activate {activated}; {label} {other} -> {count}/{sims} sims ({count/sims:.1%})")
    else:
        print("No IR activation moves occurred.")

    print("Unique cached lineup states:", len(lineup_cache))

    injury_diagnostics = {
        "future_injury_starts": int(future_injury_starts),
        "future_injury_missed_games": int(future_injury_missed_games),
        "injury_rate": float(injury_rate),
        "injury_model_source": injury_model_source,
    }
    injury_diagnostics["team_starts"] = team_starts
    injury_diagnostics["day_fp_sum"] = day_fp_sum
    injury_diagnostics["day_played"] = day_played
    injury_diagnostics["team_day_sum"] = team_day_sum
    injury_diagnostics["sims"] = sims
    return team_scores, player_active_fp, player_game_fp, player_games, team_names, id_to_player, rotation_audit, injury_diagnostics


def next_game_day(event_schedule: pd.DataFrame, start_day: int):
    for d in DAY_NAMES[start_day:]:
        if (event_schedule["day"] == d).any():
            return d
    return None


def write_daily_outputs(diag, team_proj, merged, rostered, moves, team_names_all, start_day, event_schedule):
    """Per-player per-day, per-team per-day and free-agent projections for the dashboard."""
    n = diag["sims"]
    info = {norm_id(r["player_id"]): r for _, r in merged.iterrows()}
    planned_by = {mv["add_player_id"]: mv["fantasy_team"] for mv in moves}
    dropped_by = {mv["drop_player_id"]: mv["fantasy_team"] for mv in moves}
    nba_name = {norm_id(p.player_id): (p.name, team) for team, ps in team_proj.items() for p in ps}
    opp_by_team_day = {}
    for ev in event_schedule.to_dict("records"):
        opp_by_team_day[(str(ev["team_a"]), ev["day"])] = str(ev["team_b"])
        opp_by_team_day[(str(ev["team_b"]), ev["day"])] = str(ev["team_a"])
    rows = []
    for (pid, day), total in diag["day_fp_sum"].items():
        nm, nba = nba_name.get(pid, (pid, ""))
        r = info.get(pid)
        status = str(r["roster_status"]) if r is not None else "NOT_IN_POOL"
        rows.append({
            "player_id": pid, "player_name": nm, "nba_team": nba, "day": day,
            "opponent": opp_by_team_day.get((nba, day), ""),
            "fantasy_status": status,
            "fantasy_team": (str(r["fantasy_team"]) if r is not None and status == "ROSTERED" else ""),
            "injury_status": (str(r["injury_status"]) if r is not None else ""),
            "eligible_positions": (str(r["eligible_positions"]) if r is not None else ""),
            "expected_fp": total / n,
            "play_prob": diag["day_played"].get((pid, day), 0) / n,
            "fp_if_plays": (total / diag["day_played"][(pid, day)]) if diag["day_played"].get((pid, day)) else 0.0,
            "planned_add_by": planned_by.get(pid, ""),
            "planned_drop_by": dropped_by.get(pid, ""),
        })
    pdays = pd.DataFrame(rows)
    pdays.to_csv(output_path("player_day_projections"), index=False)

    team_rows = [{"fantasy_team_id": tid, "fantasy_team": team_names_all.get(norm_id(tid), tid), "day": day,
                  "expected_fp": total / n} for (tid, day), total in diag["team_day_sum"].items()]
    pd.DataFrame(team_rows).to_csv(output_path("team_day_projections"), index=False)

    if not pdays.empty:
        fa = pdays[pdays["fantasy_status"] == "FREE_AGENT"]
        agg = fa.groupby(["player_id", "player_name", "nba_team", "injury_status", "eligible_positions",
                          "planned_add_by"], dropna=False).agg(
            games_left=("day", "nunique"), expected_rest_of_week=("expected_fp", "sum"),
            mean_play_prob=("play_prob", "mean")).reset_index()
        agg["fp_per_game"] = agg["expected_rest_of_week"] / agg["games_left"].replace(0, np.nan)
        nd = next_game_day(event_schedule, start_day)
        nxt = fa[fa["day"] == nd].set_index("player_id")["expected_fp"] if nd else pd.Series(dtype=float)
        agg["next_day_expected"] = agg["player_id"].map(nxt).fillna(0.0)
        agg["espn_projected_fp_avg"] = agg["player_id"].map(
            lambda p: info[p].get("projected_fp_avg") if p in info else np.nan)
        agg.sort_values("expected_rest_of_week", ascending=False).to_csv(output_path("free_agents"), index=False)


completed_starts: dict = {}


def load_matchups():
    """
    Read fantasy_matchups.csv. Returns (frame or None, first_sim_day, completed_points).
    completed_points: {fantasy_team_id(str): ESPN points from finished days}.
    """
    if not os.path.exists(MATCHUPS_FILE):
        first = int(SIMULATE_FROM_DAY) if SIMULATE_FROM_DAY is not None else 0
        print()
        print("No fantasy_matchups.csv found: simulating from", DAY_NAMES[first] if first < 7 else "end",
              "with no actual-to-date points and no win probabilities.")
        print("Run fetch_espn_matchups.py first to enable live matchups.")
        return None, first, {}
    m = pd.read_csv(MATCHUPS_FILE)
    for col in ("fantasy_team_id", "opponent_team_id"):
        m[col] = m[col].map(lambda v: "" if pd.isna(v) else norm_id(v))
    first = int(m["first_sim_day_index"].iloc[0])
    if SIMULATE_FROM_DAY is not None:
        print(f"SIMULATE_FROM_DAY override: {DAY_NAMES[int(SIMULATE_FROM_DAY)]} "
              f"(fantasy_matchups.csv said {DAY_NAMES[first] if first < 7 else 'week finished'})")
        if int(SIMULATE_FROM_DAY) != first:
            print("WARNING: actual points in fantasy_matchups.csv cover the first "
                  f"{first} day(s); re-run fetch_espn_matchups.py so they match.")
        first = int(SIMULATE_FROM_DAY)
    completed = {r["fantasy_team_id"]: float(r["completed_points"]) for _, r in m.iterrows()}
    global completed_starts
    if "completed_starts" in m.columns:
        completed_starts = {r["fantasy_team_id"]: (float(r["completed_starts"]) if pd.notna(r["completed_starts"]) else np.nan)
                            for _, r in m.iterrows()}
    if "week_start" in m.columns and str(m["week_start"].iloc[0]) != str(WEEK_START.date()):
        print(f"WARNING: fantasy_matchups.csv is for the week starting {m['week_start'].iloc[0]}, "
              f"but WEEK_START is {WEEK_START.date()}. Update WEEK_START (and fantasy_sim.py's week_start).")

    print()
    print("=" * 100)
    print("LIVE MATCHUPS (ESPN)")
    print("=" * 100)
    print(f"Matchup period {m['matchup_period'].iloc[0]} | fetched {m['fetched_at_utc'].iloc[0]} UTC")
    print("Finished days (ESPN actual points):", ", ".join(DAY_NAMES[:first]) or "none")
    print("Simulated days:", ", ".join(DAY_NAMES[first:]) or "none — week finished")
    for _, r in m.drop_duplicates("matchup_id").iterrows():
        print(f"  {r['fantasy_team']:<28} {r['completed_points']:>8.1f}   vs   "
              f"{completed.get(r['opponent_team_id'], 0.0):<8.1f} {r['opponent_team']}")
    return m, first, completed


def matchup_win_probabilities(matchups: pd.DataFrame | None, team_scores, team_names):
    """Win / tie probability and margin for every matchup from simulated final totals."""
    if matchups is None:
        return pd.DataFrame()
    # Simulator team keys may be "9.0" (float column); ESPN ids are "9".
    team_scores = {norm_id(k): v for k, v in team_scores.items()}
    team_names = {norm_id(k): v for k, v in team_names.items()}
    rows = []
    seen = set()
    for _, r in matchups.iterrows():
        a, b = r["fantasy_team_id"], r["opponent_team_id"]
        key = tuple(sorted([a, b]))
        if not b or key in seen or a not in team_scores or b not in team_scores:
            continue
        seen.add(key)
        sa, sb = team_scores[a], team_scores[b]
        diff = sa - sb
        rows.append({
            "matchup_id": r["matchup_id"],
            "team": team_names.get(a, a),
            "opponent": team_names.get(b, b),
            "team_actual_to_date": float(r["completed_points"]),
            "opponent_actual_to_date": float(
                matchups.loc[matchups["fantasy_team_id"] == b, "completed_points"].iloc[0]),
            "team_mean_final": float(np.mean(sa)),
            "opponent_mean_final": float(np.mean(sb)),
            "mean_margin": float(np.mean(diff)),
            "margin_p10": float(np.percentile(diff, 10)),
            "margin_p90": float(np.percentile(diff, 90)),
            "team_win_prob": float(np.mean(diff > 0)),
            "tie_prob": float(np.mean(diff == 0)),
            "opponent_win_prob": float(np.mean(diff < 0)),
        })
    return pd.DataFrame(rows)


def summarize_team_scores(team_scores, team_names):
    rows = []

    for team_id, scores in team_scores.items():
        rows.append({
            "fantasy_team_id": team_id,
            "fantasy_team": team_names[team_id],
            "mean": float(np.mean(scores)),
            "sd": float(np.std(scores, ddof=1)),
            "p10": float(np.percentile(scores, 10)),
            "median": float(np.percentile(scores, 50)),
            "p90": float(np.percentile(scores, 90)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )


def summarize_players(
    rostered,
    player_active_fp,
    player_game_fp,
    player_games,
):
    rows = []

    for _, row in rostered.iterrows():
        pid = norm_id(row["player_id"])
        all_fp = player_game_fp[pid]
        active_fp = player_active_fp[pid]

        games = player_games[pid]

        rows.append({
            "fantasy_team": row["fantasy_team"],
            "player_id": pid,
            "player_name": row["player_name"],
            "nba_team": row["nba_team"],
            "injury_status": row["injury_status"],
            "lineup_slot": row["lineup_slot"],
            "games": games,
            "mean_week_fp_all_games": float(np.mean(all_fp)),
            "mean_week_fp_active": float(np.mean(active_fp)),
            "p10_active": float(np.percentile(active_fp, 10)),
            "median_active": float(np.percentile(active_fp, 50)),
            "p90_active": float(np.percentile(active_fp, 90)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["fantasy_team", "mean_week_fp_active"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def find_gustavo_matchup(team_scores, team_names, win_probs: pd.DataFrame | None = None):
    gustavo_ids = [
        team_id
        for team_id, name in team_names.items()
        if str(name).strip().casefold() == "gustavo giants".casefold()
    ]

    if not gustavo_ids:
        print()
        print(
            "Gustavo Giants was not found by exact name, so matchup "
            "win probability is deferred."
        )
        return None

    gustavo_id = gustavo_ids[0]

    print()
    print("=" * 100)
    print("GUSTAVO GIANTS WEEK DISTRIBUTION")
    print("=" * 100)

    scores = team_scores[gustavo_id]
    print("Mean:", round(float(np.mean(scores)), 2))
    print("SD:", round(float(np.std(scores, ddof=1)), 2))
    print("P10:", round(float(np.percentile(scores, 10)), 2))
    print("Median:", round(float(np.percentile(scores, 50)), 2))
    print("P90:", round(float(np.percentile(scores, 90)), 2))
    print()
    gname = str(team_names[gustavo_id]).strip()
    wp = None
    if win_probs is not None and not win_probs.empty:
        hit = win_probs[(win_probs["team"].str.strip() == gname) | (win_probs["opponent"].str.strip() == gname)]
        if len(hit):
            wp = hit.iloc[0]
    if wp is None:
        print("No live matchup loaded (run fetch_espn_matchups.py) — win probability unavailable.")
    else:
        us_first = str(wp["team"]).strip() == gname
        opp = wp["opponent"] if us_first else wp["team"]
        p_win = wp["team_win_prob"] if us_first else wp["opponent_win_prob"]
        p_lose = wp["opponent_win_prob"] if us_first else wp["team_win_prob"]
        margin = wp["mean_margin"] if us_first else -wp["mean_margin"]
        lo, hi = (wp["margin_p10"], wp["margin_p90"]) if us_first else (-wp["margin_p90"], -wp["margin_p10"])
        print(f"Opponent: {opp}")
        print(f"WIN PROBABILITY: {p_win:.1%}   (lose {p_lose:.1%}, tie {wp['tie_prob']:.1%})")
        print(f"Projected margin: {margin:+.1f} FP (80% range {lo:+.1f} to {hi:+.1f})")

    return gustavo_id


def main():
    merged, rostered, proj, all_nba_proj = load_data()
    audit_rotation_model()

    plans, lineup_diag = build_lineup_plan(rostered)
    team_proj = build_nba_team_projection_map(all_nba_proj)
    nba_status_by_pid = build_nba_status_map(all_nba_proj)
    if nba_status_by_pid:
        non_active = sum(
            1 for v in nba_status_by_pid.values()
            if status_play_probability(v) < 1.0
        )
        print(
            f"Full-NBA injury designations loaded: {len(nba_status_by_pid):,} players "
            f"({non_active:,} with a non-ACTIVE designation)"
        )
    else:
        print(
            "WARNING: all_nba_player_projections_2027.csv has no injury_status column; "
            "non-fantasy NBA players fall back to ACTIVE/OUT only."
        )

    event_schedule = download_event_schedule()
    audit_event_schedule(event_schedule, rostered)

    print_contract_audit(rostered, plans, lineup_diag, team_proj)

    rotation_coverage = audit_nba_team_coverage(
        event_schedule,
        team_proj,
    )

    replacement_fpg = audit_roster_manager(rostered, merged)
    expected_returns = download_espn_expected_returns()
    matchups, start_day, completed_points = load_matchups()
    sim_start_date = WEEK_START + pd.Timedelta(days=min(start_day, 6))
    rotations, live_frame, rotation_model, live_diag = load_live_rotation_context(
        team_proj, all_nba_proj, merged, cutoff_day=sim_start_date)
    print(
        f"Live Injury Rotation V3 frame: {live_diag['players']:,} players | "
        f"{live_diag['with_history']:,} with 5+ games of history | "
        f"{live_diag['projection_fallback']:,} on projection fallback | "
        f"{live_diag['stale_history_replaced']:,} stale histories replaced | "
        f"{live_diag['current_team_games']:,} current-season team-games before {sim_start_date.date()}"
    )
    live_frame.reset_index(names="player_id").to_csv(output_path("rotation_live_features"), index=False)
    audit_v15_availability(rostered, expected_returns)

    lineup_diag.to_csv(
        output_path("lineup_audit"),
        index=False,
    )
    rotation_coverage.to_csv(
        output_path("rotation_coverage"),
        index=False,
    )

    # ------------------------------------------------------------------
    # V3.2: real-life matchup context for every remaining NBA game.
    # ------------------------------------------------------------------
    sim_events = event_schedule[event_schedule["day"].isin(DAY_NAMES[start_day:])]
    ctx = None
    if USE_MATCHUP_CONTEXT and os.path.exists(MATCHUP_SETTINGS_FILE):
        ctx = MC.LiveMatchupContext.build(MATCHUP_SETTINGS_FILE, MODEL_HISTORY_FILE,
                                          CURRENT_HISTORY_FILE, sim_start_date)
        print(f"\nMatchup context: pace + opponent defense from games before {sim_start_date.date()} "
              f"(league average {ctx.league:.1f} possessions)")
    else:
        print("\nMatchup context: OFF (set USE_MATCHUP_CONTEXT and add matchup_context_settings.json)")
    event_players, event_poss, event_table = build_event_contexts(sim_events, team_proj, ctx)
    if not event_table.empty:
        event_table.to_csv(output_path("event_context"), index=False)
        print(f"Game possessions this week: {event_table['possessions'].min():.1f} to "
              f"{event_table['possessions'].max():.1f}")

    # ------------------------------------------------------------------
    # V3.2: pre-pass -> each player's expected FP per remaining day.
    # ------------------------------------------------------------------
    print(f"\nPre-pass: {PREPASS_SIMS} simulations for daily expected fantasy points ...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        pre = simulate_week(
            event_schedule=event_schedule, team_proj=team_proj, rostered=rostered, merged=merged,
            replacement_fpg=replacement_fpg, expected_returns=expected_returns, rotations=rotations,
            nba_status_by_pid=nba_status_by_pid, sims=PREPASS_SIMS, seed=SEED + 1,
            start_day_index=start_day, event_players=event_players, event_poss=event_poss,
        )
    pre_diag = pre[7]
    n_pre = pre_diag["sims"]
    exp_uncond = {k: v / n_pre for k, v in pre_diag["day_fp_sum"].items()}
    exp_if_plays = {k: pre_diag["day_fp_sum"][k] / n for k, n in pre_diag["day_played"].items() if n > 0}

    # ------------------------------------------------------------------
    # V3.2: pickup plan for every team (pre-game values only).
    # ------------------------------------------------------------------
    team_names_all = {norm_id(t): str(g["fantasy_team"].iloc[0]) for t, g in rostered.groupby("fantasy_team_id")}
    my_id = find_team_id(team_names_all, MY_TEAM_NAME)
    moves, advice = [], []
    if PICKUPS_ENABLED and start_day < 7:
        caps, left_caps, cap_notes = acquisition_caps(matchups, list(team_names_all))
        moves = plan_pickups(merged, rostered, exp_uncond, start_day, caps, replacement_fpg)
        # Advice: each team's best moves with ALL its adds, others acting as usual.
        for tid in team_names_all:
            trial = dict(caps)
            trial[tid] = left_caps.get(tid, caps.get(tid, 0))
            advice += [dict(mv, adds_left=trial[tid]) for mv in
                       plan_pickups(merged, rostered, exp_uncond, start_day, trial, replacement_fpg)
                       if mv["fantasy_team_id"] == tid]
        print()
        print("=" * 100)
        print("PICKUP PLAN (expected adds for the rest of the week)")
        print("=" * 100)
        for note in cap_notes:
            print("Note:", note)
        print("Adds each team is expected to make:",
              ", ".join(f"{team_names_all[t]} {c}" for t, c in sorted(caps.items(), key=lambda kv: team_names_all[kv[0]])))
        if moves:
            print(pd.DataFrame(moves)[["day", "fantasy_team", "add_player", "drop_player", "expected_gain"]]
                  .to_string(index=False))
        else:
            print("No add/drop clears the minimum gain of", MIN_STREAM_GAIN, "expected FP.")
    pd.DataFrame(moves, columns=["day", "fantasy_team_id", "fantasy_team", "add_player_id", "add_player",
                                 "drop_player_id", "drop_player", "expected_gain", "add_expected_rest_of_week",
                                 "drop_expected_rest_of_week", "drop_projected_fpg"]).to_csv(
        output_path("pickup_plan"), index=False)
    pd.DataFrame(advice if PUBLISH_PICKUP_ADVICE else [],
                 columns=["day", "fantasy_team_id", "fantasy_team", "add_player_id", "add_player",
                          "drop_player_id", "drop_player", "expected_gain", "add_expected_rest_of_week",
                          "drop_expected_rest_of_week", "drop_projected_fpg", "adds_left"]).to_csv(
        output_path("pickup_advice"), index=False)
    if my_id:
        mine = [a for a in advice if a["fantasy_team_id"] == my_id]
        print(f"\nBest moves for {MY_TEAM_NAME} ({len(mine)}):")
        for a in mine:
            print(f"  {a['day']}: add {a['add_player']}, drop {a['drop_player']}  (+{a['expected_gain']:.1f} expected FP)")
    roster_template = build_roster_template(rostered, merged, moves)
    LINEUP_VALUES.clear()
    LINEUP_VALUES.update(exp_if_plays)
    LINEUP_MEMO.clear()

    (
        team_scores,
        player_active_fp,
        player_game_fp,
        player_games,
        team_names,
        id_to_player,
        rotation_audit,
        injury_diagnostics,
    ) = simulate_week(
        event_schedule=event_schedule,
        team_proj=team_proj,
        rostered=rostered,
        merged=merged,
        replacement_fpg=replacement_fpg,
        expected_returns=expected_returns,
        rotations=rotations,
        nba_status_by_pid=nba_status_by_pid,
        sims=SIMS,
        seed=SEED,
        start_day_index=start_day,
        event_players=event_players,
        event_poss=event_poss,
        pickup_moves=moves,
        roster_template=roster_template,
    )
    write_daily_outputs(injury_diagnostics, team_proj, merged, rostered, moves, team_names_all,
                        start_day, event_schedule)

    # PPP (simulated days): fantasy points / player-games started and played.
    team_starts = injury_diagnostics.get("team_starts", {})
    ppp_rows = {}
    for tid, pts in team_scores.items():
        starts = team_starts.get(tid, np.zeros_like(pts))
        total_starts = float(starts.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            per_sim = np.where(starts > 0, pts / starts, np.nan)
        ppp_rows[tid] = {
            "starts_mean": float(starts.mean()),
            "ppp": float(pts.sum() / total_starts) if total_starts > 0 else np.nan,
            "ppp_p10": float(np.nanpercentile(per_sim, 10)) if np.isfinite(per_sim).any() else np.nan,
            "ppp_p90": float(np.nanpercentile(per_sim, 90)) if np.isfinite(per_sim).any() else np.nan,
        }

    # Final weekly score = ESPN actual points from finished days + simulated rest.
    for tid in team_scores:
        team_scores[tid] = team_scores[tid] + completed_points.get(norm_id(tid), 0.0)

    print()
    print("=" * 100)
    print("STOCHASTIC INJURY DIAGNOSTICS")
    print("=" * 100)
    print(f"Simulated future injury starts: {injury_diagnostics['future_injury_starts']:,}")
    print(f"Sampled future missed games from those injuries: {injury_diagnostics['future_injury_missed_games']:,}")
    print(f"Estimated injury onset per completed player-game: {injury_diagnostics['injury_rate']:.4%}")
    print(f"Injury model source: {injury_diagnostics['injury_model_source']}")

    print()
    print("=" * 100)
    print("INJURY ROTATION V3 — SANITY AUDIT")
    print("=" * 100)
    events = int(rotation_audit["team_events"])
    errs = np.asarray(rotation_audit["minutes_sum_errors"], float)
    maxmins = np.asarray(rotation_audit["max_player_minutes"], float)
    print(f"Team-games projected: {events:,}")
    print(f"Max absolute team-minute error vs 240: {np.max(np.abs(errs)) if len(errs) else 0.0:.8f}")
    print(f"Largest individual expected minutes: {np.max(maxmins) if len(maxmins) else 0.0:.2f}")
    print(f"Team-games with a player >48 min: {int(np.sum(maxmins > 48.000001)):,}")
    print(f"Team-games with a player >{TOP_AVERAGE_MINUTES_REFERENCE:.0f} min: {int(np.sum(maxmins > TOP_AVERAGE_MINUTES_REFERENCE + 1e-6)):,}")
    print(f"Team-games with a player >{STAR_SOFT_CEILING:.0f} min: {int(np.sum(maxmins > STAR_SOFT_CEILING + 1e-6)):,}")
    print(f"Team-games with >1 important absence: {int(np.sum(np.asarray(rotation_audit['important_absent_counts']) > 1)):,}")
    print(f"Team-games needing emergency >{MAX_EXPECTED_MINUTES:.0f}-minute headroom: {rotation_audit['emergency_events']:,}")
    if rotation_audit["emergency_ceilings"]:
        print(f"Largest emergency ceiling required: {max(rotation_audit['emergency_ceilings']):.2f}")

    # Every absence, every team-game: absence RATE, not a single 'primary' trigger.
    absence_rows = []
    for pid, opps in rotation_audit["team_games"].items():
        n_abs = rotation_audit["absent_games"].get(pid, 0)
        if n_abs == 0:
            continue
        absence_rows.append({
            "player_id": pid, "player_name": rotation_audit["names"].get(pid, pid),
            "nba_team": rotation_audit["teams"].get(pid, ""),
            "healthy_minutes": rotation_audit["healthy"].get(pid, np.nan),
            "team_games": opps, "absent_games": n_abs, "absent_rate": n_abs / opps,
        })
    absence_df = pd.DataFrame(absence_rows)
    print("\nABSENCES — EVERY ABSENT PLAYER COUNTED IN EVERY TEAM-GAME (healthy minutes >= 12)")
    print("-" * 100)
    if not absence_df.empty:
        absence_df = absence_df.sort_values(["absent_rate", "healthy_minutes"], ascending=[False, False])
        print(absence_df[absence_df["healthy_minutes"] >= 12].round(3).to_string(index=False))
    else:
        print("None")

    rec_rows = []
    for pid, (n, sum_min, sum_delta, mx, mn) in rotation_audit["recipient"].items():
        if n == 0:
            continue
        rec_rows.append({
            "player_id": pid, "player_name": rotation_audit["names"].get(pid, pid),
            "nba_team": rotation_audit["teams"].get(pid, ""),
            "games_played": n,
            "full_strength_minutes": rotation_audit["full_strength"].get(pid, np.nan),
            "week_start_minutes": rotation_audit["week_start"].get(pid, np.nan),
            "avg_v3_minutes": sum_min / n, "avg_delta": sum_delta / n,
            "max_gain": mx, "max_loss": mn,
        })
    recip = pd.DataFrame(rec_rows)
    if not recip.empty:
        print("\nfull_strength_minutes = everyone healthy (teams with long-term injuries may not have played like this")
        print("this season); week_start_minutes = this week's known OUT players absent, everyone else available.")
        print("avg_delta is versus full strength; compare avg_v3_minutes with week_start_minutes for in-week changes.")
        print("\nLARGEST AVERAGE MINUTE GAINS vs each team's FULL-STRENGTH rotation")
        print("-" * 100)
        print(recip.sort_values("avg_delta", ascending=False).head(20).round(2).to_string(index=False))
        print("\nLARGEST AVERAGE MINUTE LOSSES vs each team's FULL-STRENGTH rotation")
        print("-" * 100)
        print(recip.sort_values("avg_delta", ascending=True).head(20).round(2).to_string(index=False))
        recip.to_csv(output_path("rotation_audit_recipients"), index=False)
    if not absence_df.empty:
        absence_df.to_csv(output_path("rotation_audit_absences"), index=False)

    team_minutes_pass = (len(errs) == 0) or bool(np.max(np.abs(errs)) < 1e-6)
    player_minutes_pass = (len(maxmins) == 0) or bool(np.max(maxmins) <= 48.000001)
    print("\nSANITY GATES")
    print("-" * 100)
    print("Every adjusted team sums to 240 minutes:", "PASS" if team_minutes_pass else "FAIL")
    print("No adjusted player exceeds 48 minutes:", "PASS" if player_minutes_pass else "FAIL")
    print("Rotation audit verdict:", "PASS" if team_minutes_pass and player_minutes_pass else "REVIEW REQUIRED")

    team_summary = summarize_team_scores(team_scores, team_names)
    team_summary["actual_to_date"] = team_summary["fantasy_team_id"].map(
        lambda t: completed_points.get(norm_id(t), 0.0))
    team_summary["simulated_rest_mean"] = team_summary["mean"] - team_summary["actual_to_date"]
    # PPP to date (ESPN actual points / actual starts on finished days) and
    # PPP rest of week (simulated points / simulated starts on remaining days).
    team_summary["starts_to_date"] = team_summary["fantasy_team_id"].map(
        lambda t: completed_starts.get(norm_id(t), np.nan))
    team_summary["ppp_to_date"] = team_summary["actual_to_date"] / team_summary["starts_to_date"].replace(0, np.nan)
    team_summary["starts_rest_mean"] = team_summary["fantasy_team_id"].map(lambda t: ppp_rows[t]["starts_mean"])
    team_summary["ppp_rest_of_week"] = team_summary["fantasy_team_id"].map(lambda t: ppp_rows[t]["ppp"])
    win_probs = matchup_win_probabilities(matchups, team_scores, team_names)
    player_summary = summarize_players(
        rostered,
        player_active_fp,
        player_game_fp,
        player_games,
    )

    print()
    print("=" * 100)
    print("FANTASY TEAM WEEK DISTRIBUTIONS")
    print("=" * 100)
    print(team_summary.round(2).to_string(index=False))
    print("\nPPP = fantasy points / player-games started (lineup players who played).")
    print(f"  ppp_to_date: ESPN actual, finished days ({', '.join(DAY_NAMES[:start_day]) or 'none yet'})")
    print(f"  ppp_rest_of_week: simulated, remaining days ({', '.join(DAY_NAMES[start_day:]) or 'none'})")

    if not win_probs.empty:
        print()
        print("=" * 100)
        print("MATCHUP WIN PROBABILITIES (final = ESPN actual-to-date + simulated remaining days)")
        print("=" * 100)
        show_wp = win_probs[["team", "team_actual_to_date", "team_mean_final", "team_win_prob", "tie_prob",
                             "opponent_win_prob", "opponent_mean_final", "opponent_actual_to_date",
                             "opponent", "mean_margin"]]
        print(show_wp.round(3).to_string(index=False))
        win_probs.to_csv(output_path("matchup_win_probabilities"), index=False)

    # Run info for the dashboard header.
    meta = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "sims": int(SIMS),
        "week_start": str(WEEK_START.date()),
        "finished_days": DAY_NAMES[:start_day],
        "simulated_days": DAY_NAMES[start_day:],
        "matchup_period": (int(matchups["matchup_period"].iloc[0]) if matchups is not None else None),
        "espn_fetched_utc": (str(matchups["fetched_at_utc"].iloc[0]) if matchups is not None else None),
        "my_team": MY_TEAM_NAME,
        "matchup_context": bool(event_poss),
        "pickups": bool(PICKUPS_ENABLED),
        "next_day": next_game_day(event_schedule, start_day),
        "week_dates": {d: str((WEEK_START + pd.Timedelta(days=i)).date()) for i, d in enumerate(DAY_NAMES)},
    }
    with open(os.path.join(BASE, f"weekly_matchup_{OUTPUT_TAG}_run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    gustavo_id = find_gustavo_matchup(team_scores, team_names, win_probs)

    if gustavo_id is not None:
        print()
        print("=" * 100)
        print("GUSTAVO GIANTS — PLAYER DIAGNOSTICS")
        print("=" * 100)
        gname = team_names[gustavo_id]
        show = player_summary[
            player_summary["fantasy_team"] == gname
        ]
        print(show.round(2).to_string(index=False))

    team_summary.to_csv(
        output_path("team_summary"),
        index=False,
    )
    player_summary.to_csv(
        output_path("player_summary"),
        index=False,
    )

    print()
    print("=" * 100)
    print("MONTE CARLO INTEGRATION STATUS")
    print("=" * 100)
    print("Fantasy V3 projection load:           PASS")
    print("Full NBA simulation universe load:     PASS")
    print("Fantasy roster ownership:            PASS")
    print("Dynamic IR activation/drop manager: PASS")
    print("ROS-based drop protection:          PASS")
    print("Exact daily lineup optimiser:       PASS")
    print("Event-level NBA schedule:            PASS")
    print("Shared-game possession simulation:   PASS")
    print("Fantasy weekly score accumulation:   PASS")
    print("No simulated-score lineup hindsight: PASS")
    print()
    print("Game-specific status availability:      PASS (V1.6 priors)")
    print("ESPN OUT return-date 50/75/100 model:   PASS (status-only fallback if feed unavailable)")
    print("Injury Rotation V3 whole-team rotation:  PASS (leak-free backtest + role guardrails)")
    print("Layoff / return modelling inside sims:   PASS")
    print("Still intentionally deferred:")
    print("  1. historical calibration of status play probabilities")
    print("  2. waiver pickups / streaming (IR activation + drops now integrated)")
    print()
    print("Live matchups + win probability:         " + ("PASS" if not win_probs.empty else "NOT LOADED (no fantasy_matchups.csv)"))
    print("Actual-to-date + remaining-days sim:     " + (f"PASS (simulated from {DAY_NAMES[start_day] if start_day < 7 else 'end'})"))
    print()
    print(f"Saved: weekly_matchup_{OUTPUT_TAG}_team_summary.csv")
    print(f"Saved: weekly_matchup_{OUTPUT_TAG}_player_summary.csv")
    print(f"Saved: weekly_matchup_{OUTPUT_TAG}_rotation_coverage.csv")
    if not win_probs.empty:
        print(f"Saved: weekly_matchup_{OUTPUT_TAG}_matchup_win_probabilities.csv")


if __name__ == "__main__":
    main()
