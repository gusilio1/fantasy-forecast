"""
fantasy_roster_manager_v1.py

Long-term roster-management layer for the NBA fantasy simulator.

Core principle
--------------
Roster decisions optimize REST-OF-SEASON asset value.
Daily lineup decisions optimize TODAY'S expected fantasy points.

Therefore:
    - remaining games this fantasy week NEVER enter the drop score
    - a strong player is not dropped just because a weaker player has more
      games left this week
    - when a healthy IR player must be activated, another IR-eligible injured
      player is moved into IR first
    - only if no IR swap is possible do we consider dropping a player
    - high-value players receive hard protection from ordinary activation drops

This module is intentionally independent of the Monte Carlo game engine so it
can later be used for:
    IR activation
    drops
    waiver pickups
    streaming decisions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

# User preference: a genuine ~30 FP/game player should not be sacrificed merely
# to solve a short-term roster crunch.
HARD_PROTECT_FPG = 30.0

# Additional protection relative to the waiver pool.
# Example: replacement level 20 -> players >= 26 receive soft protection.
SOFT_PROTECT_ABOVE_REPLACEMENT = 6.0

# Health labels that are definitely IR-eligible in our current V1 logic.
# DAY_TO_DAY / QUESTIONABLE are deliberately NOT assumed IR-eligible here.
DEFAULT_IR_ELIGIBLE_STATUSES = {
    "OUT",
    "INJURED",
    "SUSPENSION",
    "SUSPENDED",
}


# =============================================================================
# DATA OBJECTS
# =============================================================================

@dataclass
class RosterValue:
    player_id: str
    player_name: str
    fantasy_team: str
    projected_fpg: float
    replacement_fpg: float
    hold_value: float
    hard_protected: bool
    soft_protected: bool


@dataclass
class RosterMove:
    action: str
    player_id: str
    player_name: str
    reason: str


@dataclass
class IRDecision:
    fantasy_team: str
    activated_player_id: str
    activated_player_name: str
    moves: List[RosterMove]
    dropped_player_id: Optional[str]
    dropped_player_name: Optional[str]
    used_ir_swap: bool
    warning: Optional[str] = None


# =============================================================================
# BASIC HELPERS
# =============================================================================

def norm_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value)
    return s[:-2] if s.endswith(".0") else s


def clean_status(value) -> str:
    return str(value or "").strip().upper()


def projected_fpg(row: pd.Series) -> float:
    """
    Long-term per-game quality estimate.

    V1 uses the best available projection column. This is deliberately PER GAME,
    not remaining-week total points.
    """
    for col in (
        "projected_fp_avg",
        "espn_projected_fp_avg",
        "expected_fp",
        "projection_fp_avg",
    ):
        if col in row and pd.notna(row[col]):
            try:
                value = float(row[col])
                if np.isfinite(value):
                    return max(value, 0.0)
            except (TypeError, ValueError):
                pass
    return 0.0


def is_ir_slot(row: pd.Series) -> bool:
    value = row.get("lineup_slot")
    if pd.isna(value):
        return False
    text = str(value).strip().upper()
    return text in {"IR", "13", "IR_SLOT"} or text == "13.0"


def is_ir_eligible(
    row: pd.Series,
    ir_eligible_statuses: Optional[set[str]] = None,
) -> bool:
    statuses = ir_eligible_statuses or DEFAULT_IR_ELIGIBLE_STATUSES
    return clean_status(row.get("injury_status")) in statuses


# =============================================================================
# REPLACEMENT LEVEL / ROS VALUE
# =============================================================================

def estimate_replacement_level(
    player_pool: pd.DataFrame,
    *,
    quantile: float = 0.80,
) -> float:
    """
    Estimate generic waiver replacement level from FREE_AGENT players.

    This is intentionally season-value oriented. Schedule flags Mon-Sun are not
    used.

    V1 uses a generic league-wide replacement level. A later version can make
    this position-aware (PG/SG/SF/PF/C scarcity).
    """
    if "roster_status" not in player_pool.columns:
        raise ValueError("player_pool needs roster_status")

    fa = player_pool[
        player_pool["roster_status"].astype(str).str.upper().eq("FREE_AGENT")
    ].copy()

    if fa.empty:
        return 0.0

    vals = fa.apply(projected_fpg, axis=1)
    vals = vals[np.isfinite(vals)]

    if vals.empty:
        return 0.0

    return float(np.quantile(vals, quantile))


def build_roster_values(
    roster: pd.DataFrame,
    replacement_fpg: float,
) -> pd.DataFrame:
    """
    Build the long-term hold-value table.

    Current V1:
        hold_value = projected fantasy points per game - replacement FPG

    Importantly, NO current-week schedule variable is used.

    We keep hard/soft protection as separate flags rather than inflating the
    numeric hold score. That makes the manager's behaviour auditable.
    """
    rows = []

    for _, row in roster.iterrows():
        fpg = projected_fpg(row)
        hold = fpg - replacement_fpg

        rows.append({
            "player_id": norm_id(row["player_id"]),
            "player_name": str(row["player_name"]),
            "fantasy_team": str(row.get("fantasy_team", "")),
            "projected_fpg": fpg,
            "replacement_fpg": replacement_fpg,
            "hold_value": hold,
            "hard_protected": fpg >= HARD_PROTECT_FPG,
            "soft_protected": (
                fpg >= replacement_fpg + SOFT_PROTECT_ABOVE_REPLACEMENT
            ),
            "injury_status": clean_status(row.get("injury_status")),
            "on_ir": is_ir_slot(row),
        })

    return pd.DataFrame(rows)


# =============================================================================
# DROP SELECTION
# =============================================================================

def choose_activation_drop(
    normal_roster: pd.DataFrame,
    replacement_fpg: float,
) -> Tuple[pd.Series, Optional[str]]:
    """
    Choose the player to drop when an IR player must be activated.

    Priority:
        1. Never use games remaining this week.
        2. Prefer players below the hard-protection threshold.
        3. Among those, prefer players without soft protection.
        4. Drop the lowest ROS hold value.
        5. If EVERY player is hard-protected, return the lowest-value player
           with an explicit warning rather than silently pretending the choice
           is normal.
    """
    values = build_roster_values(normal_roster, replacement_fpg)

    if values.empty:
        raise ValueError("No normal-roster players available for drop decision.")

    expendable = values[~values["hard_protected"]].copy()

    warning = None

    if expendable.empty:
        expendable = values.copy()
        warning = (
            "All normal-roster players are at or above the hard-protection "
            f"floor ({HARD_PROTECT_FPG:.1f} FP/game). Forced lowest-value "
            "choice requires manager review."
        )

    # First prefer non-soft-protected assets if any exist.
    non_soft = expendable[~expendable["soft_protected"]]
    if not non_soft.empty:
        expendable = non_soft

    choice = expendable.sort_values(
        ["hold_value", "projected_fpg", "player_name"],
        ascending=[True, True, True],
    ).iloc[0]

    return choice, warning


# =============================================================================
# IR ACTIVATION
# =============================================================================

def manage_healthy_ir_player(
    team_roster: pd.DataFrame,
    healthy_ir_player_id: str,
    replacement_fpg: float,
    *,
    ir_eligible_statuses: Optional[set[str]] = None,
) -> IRDecision:
    """
    Resolve a healthy player who is currently occupying the fantasy IR slot.

    Rules:
        A. If another normal-roster player is IR-eligible:
             activate healthy IR player
             move the BEST injured asset into IR
             drop nobody

        B. Otherwise:
             activate healthy IR player
             drop the lowest long-term-value expendable player

    Why "best injured asset" for the IR swap?
        The IR slot is scarce storage. If several players are injured, preserving
        the most valuable injured asset is the most sensible long-term action.

    This function returns the decision; it does not mutate the input dataframe.
    """
    pid = norm_id(healthy_ir_player_id)
    roster = team_roster.copy()
    roster["_pid"] = roster["player_id"].map(norm_id)

    matches = roster[roster["_pid"] == pid]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one IR player with id {pid}; found {len(matches)}."
        )

    healthy_ir = matches.iloc[0]

    if not is_ir_slot(healthy_ir):
        raise ValueError(
            f"{healthy_ir['player_name']} is not currently in the IR slot."
        )

    normal = roster[~roster.apply(is_ir_slot, axis=1)].copy()

    injured_candidates = normal[
        normal.apply(
            lambda r: is_ir_eligible(r, ir_eligible_statuses),
            axis=1,
        )
    ].copy()

    moves = [
        RosterMove(
            action="ACTIVATE_FROM_IR",
            player_id=pid,
            player_name=str(healthy_ir["player_name"]),
            reason="Player is healthy and should not remain in fantasy IR.",
        )
    ]

    if not injured_candidates.empty:
        injured_values = build_roster_values(
            injured_candidates,
            replacement_fpg,
        )

        # Preserve the highest-value injured asset in the scarce IR slot.
        stash = injured_values.sort_values(
            ["hold_value", "projected_fpg"],
            ascending=[False, False],
        ).iloc[0]

        moves.append(
            RosterMove(
                action="MOVE_TO_IR",
                player_id=str(stash["player_id"]),
                player_name=str(stash["player_name"]),
                reason=(
                    "Another rostered player is IR-eligible; use the IR slot "
                    "instead of dropping a healthy asset."
                ),
            )
        )

        return IRDecision(
            fantasy_team=str(healthy_ir.get("fantasy_team", "")),
            activated_player_id=pid,
            activated_player_name=str(healthy_ir["player_name"]),
            moves=moves,
            dropped_player_id=None,
            dropped_player_name=None,
            used_ir_swap=True,
        )

    drop, warning = choose_activation_drop(
        normal_roster=normal,
        replacement_fpg=replacement_fpg,
    )

    moves.append(
        RosterMove(
            action="DROP",
            player_id=str(drop["player_id"]),
            player_name=str(drop["player_name"]),
            reason=(
                "No other player is IR-eligible. Lowest expendable "
                "rest-of-season hold value on the normal roster."
            ),
        )
    )

    return IRDecision(
        fantasy_team=str(healthy_ir.get("fantasy_team", "")),
        activated_player_id=pid,
        activated_player_name=str(healthy_ir["player_name"]),
        moves=moves,
        dropped_player_id=str(drop["player_id"]),
        dropped_player_name=str(drop["player_name"]),
        used_ir_swap=False,
        warning=warning,
    )


# =============================================================================
# AUDIT / DISPLAY
# =============================================================================

def explain_drop_board(
    team_roster: pd.DataFrame,
    replacement_fpg: float,
) -> pd.DataFrame:
    """
    Human-readable drop ranking for debugging.

    Lowest rows are most expendable. No weekly game counts are included by
    design.
    """
    normal = team_roster[~team_roster.apply(is_ir_slot, axis=1)].copy()
    values = build_roster_values(normal, replacement_fpg)

    return values.sort_values(
        ["hard_protected", "soft_protected", "hold_value"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


if __name__ == "__main__":
    print("fantasy_roster_manager_v1.py")
    print()
    print("Roster decisions: rest-of-season value")
    print("Lineup decisions: daily expected fantasy value")
    print("Weekly games remaining are NOT part of the drop score.")
