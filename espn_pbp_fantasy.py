"""
espn_pbp_fantasy.py

Fantasy points per player per quarter from ESPN's game summary play-by-play
("plays" in site.api.espn.com/.../nba/summary?event=ID), the same feed the box-score
rebuild already downloads. Used for this season's NBA records.

Participants: [0] is the shooter / rebounder / player who turned it over,
[1] is the assister on a make, the blocker on a blocked miss, or the stealer.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "tpm", "ftm", "fta"]


def _pid(play, i):
    parts = play.get("participants") or []
    if len(parts) > i:
        return str(((parts[i] or {}).get("athlete") or {}).get("id") or "")
    return ""


def period_stats_espn(summary: dict) -> pd.DataFrame:
    """One row per (period, ESPN athlete id) with counts and fantasy points (today's scoring)."""
    acc = defaultdict(lambda: dict.fromkeys(STATS, 0))
    for p in summary.get("plays") or []:
        per = int(((p.get("period") or {}).get("number")) or 0)
        typ = str(((p.get("type") or {}).get("text")) or "").lower()
        text = str(p.get("text") or "").lower()
        a, b = _pid(p, 0), _pid(p, 1)
        made = bool(p.get("scoringPlay"))
        if p.get("shootingPlay"):
            pa = int(p.get("pointsAttempted") or 0)
            if pa == 1 or "free throw" in typ:
                if a:
                    acc[(per, a)]["fta"] += 1
                    if made:
                        acc[(per, a)]["ftm"] += 1
                        acc[(per, a)]["pts"] += 1
            elif a and "heave" in typ and not made:
                pass       # missed end-of-quarter heaves don't count as attempts in the box score
            elif a:
                acc[(per, a)]["fga"] += 1
                if made:
                    acc[(per, a)]["fgm"] += 1
                    acc[(per, a)]["pts"] += int(p.get("scoreValue") or (3 if pa == 3 else 2))
                    acc[(per, a)]["tpm"] += int(pa == 3 or p.get("scoreValue") == 3)
                    if b:
                        acc[(per, b)]["ast"] += 1
                elif b and "block" in text:
                    acc[(per, b)]["blk"] += 1
        elif "rebound" in typ and a:
            acc[(per, a)]["reb"] += 1
        elif "turnover" in typ or str(p.get("shortDescription") or "").lower() == "turnover":
            if a:
                acc[(per, a)]["tov"] += 1
            if b and "steal" in text:
                acc[(per, b)]["stl"] += 1
    rows = [{"period": per, "espn_id": pid, **v} for (per, pid), v in acc.items() if per > 0]
    df = pd.DataFrame(rows, columns=["period", "espn_id"] + STATS)
    df["fp"] = (df["pts"] + df["reb"] + 2 * df["ast"] + 4 * df["stl"] + 4 * df["blk"] - 2 * df["tov"]
                + 2 * df["fgm"] - df["fga"] + df["tpm"])
    return df


def players_in_summary(summary: dict) -> dict:
    """ESPN athlete id -> (name, team abbreviation) from the summary's box score."""
    out = {}
    for team in (summary.get("boxscore") or {}).get("players") or []:
        abbr = ((team.get("team") or {}).get("abbreviation") or "").upper()
        for grp in team.get("statistics") or []:
            for ath in grp.get("athletes") or []:
                a = ath.get("athlete") or {}
                if a.get("id"):
                    out[str(a["id"])] = (a.get("displayName") or "", abbr)
    return out
