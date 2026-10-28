"""
rebuild_current_season_history_2027.py

FORCE-REBUILD the 2026-27 current NBA player-game history.

This replaces the earlier current-season CSV because ESPN NBA player boxscores
represent shooting as combined made-attempt strings such as:
    FG  = "12-23"
    3PT = "3-7"
    FT  = "8-10"

Output:
    player_game_history_2027_current.csv
    player_stats_2027_current.csv

Fantasy scoring:
    FP = PTS + REB + 2*AST + 4*STL + 4*BLK - 2*TO
         + 2*FGM - FGA + 3PM
"""

import os
import time
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
START_DATE = date(2026, 10, 20)

HISTORY_FILE = os.path.join(BASE_DIR, "player_game_history_2027_current.csv")
STATS_FILE = os.path.join(BASE_DIR, "player_stats_2027_current.csv")

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/scoreboard"
)
SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/basketball/nba/summary"
)

TEAM_FIXES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}

SESSION = requests.Session()


def normalise_team(abbr):
    if abbr is None:
        return None
    s = str(abbr)
    return TEAM_FIXES.get(s, s)


def num(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_minutes(value):
    if value is None:
        return 0.0

    s = str(value).strip()

    if not s:
        return 0.0

    if ":" in s:
        try:
            m, sec = s.split(":", 1)
            return float(m) + float(sec) / 60.0
        except (TypeError, ValueError):
            return 0.0

    return num(s, 0.0)


def parse_made_attempted(value):
    """
    Parse ESPN shooting strings:
        "12-23" -> (12, 23)
        "3-7"   -> (3, 7)
        "0-0"   -> (0, 0)

    Defensive support for unicode dash variants is included.
    """
    if value is None:
        return 0.0, 0.0

    s = str(value).strip()

    for dash in ["–", "—", "−"]:
        s = s.replace(dash, "-")

    if "-" not in s:
        return 0.0, 0.0

    left, right = s.split("-", 1)

    try:
        return float(left), float(right)
    except ValueError:
        return 0.0, 0.0


def fetch_json(url, params, attempts=4):
    error = None

    for attempt in range(1, attempts + 1):
        try:
            r = SESSION.get(url, params=params, timeout=30)

            if r.status_code == 200:
                return r.json()

            error = RuntimeError(
                f"HTTP {r.status_code}: {r.text[:300]}"
            )

        except requests.RequestException as exc:
            error = exc

        if attempt < attempts:
            time.sleep(0.8 * attempt)

    raise RuntimeError(
        f"Request failed after {attempts} attempts: {error}"
    )


def scoreboard_events(start, end):
    events = {}
    d = start

    while d <= end:
        data = fetch_json(
            SCOREBOARD_URL,
            {"dates": d.strftime("%Y%m%d"), "limit": 100},
        )

        for event in data.get("events", []):
            event_id = str(event.get("id", "")).strip()
            comps = event.get("competitions", [])

            if not event_id or not comps:
                continue

            competitors = comps[0].get("competitors", [])
            if len(competitors) != 2:
                continue

            # Skip explicitly cancelled/postponed events.
            status = (
                comps[0]
                .get("status", {})
                .get("type", {})
                .get("name", "")
            )

            if status in {"STATUS_CANCELED", "STATUS_POSTPONED"}:
                continue

            events[event_id] = d.isoformat()

        print(
            f"{d.isoformat()}: "
            f"{len(data.get('events', []))} scoreboard events"
        )

        d += timedelta(days=1)

    return events


def label_lookup(labels):
    return {
        str(label).strip().upper(): i
        for i, label in enumerate(labels)
    }


def raw_stat(stats, lookup, names):
    for name in names:
        key = name.upper()
        if key in lookup:
            i = lookup[key]
            if i < len(stats):
                return stats[i]
    return None


def scalar_stat(stats, lookup, names, default=0.0):
    return num(
        raw_stat(stats, lookup, names),
        default,
    )


def fantasy_points(row):
    return (
        row["points"]
        + row["rebounds"]
        + 2.0 * row["assists"]
        + 4.0 * row["steals"]
        + 4.0 * row["blocks"]
        - 2.0 * row["turnovers"]
        + 2.0 * row["fgm"]
        - row["fga"]
        + row["three_pm"]
    )


def parse_game(event_id, fallback_date):
    data = fetch_json(
        SUMMARY_URL,
        {"event": event_id},
    )

    game_date = fallback_date

    header_comps = (
        data.get("header", {})
        .get("competitions", [])
    )

    if header_comps:
        raw_date = header_comps[0].get("date")
        parsed = pd.to_datetime(
            raw_date, errors="coerce", utc=True
        )

        if pd.notna(parsed):
            game_date = parsed.date().isoformat()

    rows = []

    for team_block in (
        data.get("boxscore", {}).get("players", [])
    ):
        team = team_block.get("team", {})
        team_abbr = normalise_team(
            team.get("abbreviation")
        )

        for group in team_block.get("statistics", []):
            labels = group.get("labels", []) or []
            lookup = label_lookup(labels)

            for athlete_row in group.get("athletes", []):
                if athlete_row.get("didNotPlay") is True:
                    continue

                athlete = athlete_row.get("athlete", {})
                player_id = str(
                    athlete.get("id", "")
                ).strip()

                if not player_id:
                    continue

                stats = athlete_row.get("stats", []) or []

                minutes_raw = raw_stat(
                    stats, lookup, ["MIN"]
                )
                minutes = parse_minutes(minutes_raw)

                if minutes <= 0:
                    minutes = parse_minutes(
                        athlete_row.get("minutes")
                    )

                if minutes <= 0:
                    continue

                fg_raw = raw_stat(
                    stats, lookup, ["FG", "FGM-A"]
                )
                three_raw = raw_stat(
                    stats, lookup, ["3PT", "3P", "3PM-A"]
                )
                ft_raw = raw_stat(
                    stats, lookup, ["FT", "FTM-A"]
                )

                fgm, fga = parse_made_attempted(fg_raw)
                three_pm, three_pa = parse_made_attempted(
                    three_raw
                )
                ftm, fta = parse_made_attempted(ft_raw)

                points = scalar_stat(
                    stats, lookup, ["PTS"]
                )
                oreb = scalar_stat(
                    stats, lookup, ["OREB", "OR"]
                )
                dreb = scalar_stat(
                    stats, lookup, ["DREB", "DR"]
                )
                rebounds = scalar_stat(
                    stats, lookup, ["REB"]
                )

                if rebounds == 0 and (oreb > 0 or dreb > 0):
                    rebounds = oreb + dreb

                row = {
                    "game_id": str(event_id),
                    "date": game_date,
                    "player_id": player_id,
                    "player_name": athlete.get(
                        "displayName",
                        athlete.get("fullName", "UNKNOWN"),
                    ),
                    "nba_team": team_abbr,
                    "minutes": float(minutes),
                    "points": points,
                    "fgm": fgm,
                    "fga": fga,
                    "three_pm": three_pm,
                    "three_pa": three_pa,
                    "ftm": ftm,
                    "fta": fta,
                    "offensive_rebounds": oreb,
                    "defensive_rebounds": dreb,
                    "rebounds": rebounds,
                    "assists": scalar_stat(
                        stats, lookup, ["AST"]
                    ),
                    "steals": scalar_stat(
                        stats, lookup, ["STL"]
                    ),
                    "blocks": scalar_stat(
                        stats, lookup, ["BLK"]
                    ),
                    "turnovers": scalar_stat(
                        stats, lookup, ["TO", "TOV"]
                    ),
                    "_fg_raw": fg_raw,
                    "_three_raw": three_raw,
                    "_ft_raw": ft_raw,
                }

                row["fantasy_points"] = fantasy_points(row)
                rows.append(row)

    return rows


def validate_rows(df):
    problems = []

    if df.empty:
        problems.append("No player rows were parsed.")
        return problems

    scoring = df[df["points"] > 0]

    if len(scoring):
        zero_fga_rate = float(
            (scoring["fga"] == 0).mean()
        )

        if zero_fga_rate > 0.02:
            problems.append(
                f"{zero_fga_rate:.1%} of players with points "
                "have FGA=0. Shooting parsing is probably broken."
            )

    impossible = df[
        (df["fgm"] > df["fga"])
        | (df["three_pm"] > df["three_pa"])
        | (df["ftm"] > df["fta"])
        | (df["three_pm"] > df["fgm"])
        | (df["three_pa"] > df["fga"])
    ]

    if len(impossible):
        problems.append(
            f"{len(impossible)} rows have impossible shooting totals."
        )

    point_check = (
        2.0 * (df["fgm"] - df["three_pm"])
        + 3.0 * df["three_pm"]
        + df["ftm"]
    )

    mismatch = (point_check - df["points"]).abs() > 0.01

    if mismatch.any():
        problems.append(
            f"{int(mismatch.sum())} rows fail the points-from-shooting check."
        )

    return problems


def build_stats(history):
    return (
        history
        .groupby(["player_id", "player_name"], as_index=False)
        .agg(
            games=("game_id", "nunique"),
            fp_avg=("fantasy_points", "mean"),
            fp_sd=("fantasy_points", "std"),
            fp_min=("fantasy_points", "min"),
            fp_max=("fantasy_points", "max"),
        )
        .sort_values(
            ["fp_avg", "games"],
            ascending=[False, False],
        )
    )


print()
print("=" * 100)
print("FORCE-REBUILDING 2026-27 CURRENT-SEASON HISTORY")
print("=" * 100)
print(
    "This intentionally ignores/replaces the earlier current-season CSV."
)

today = datetime.now().date()
events = scoreboard_events(START_DATE, today)

print()
print("Scoreboard games found:", len(events))
print("Fetching every game again with corrected shooting parser...")
print()

all_rows = []
errors = []

for i, (event_id, fallback_date) in enumerate(
    events.items(),
    start=1,
):
    try:
        rows = parse_game(
            event_id,
            fallback_date,
        )

        if rows:
            all_rows.extend(rows)
            print(
                f"[{i}/{len(events)}] "
                f"{event_id}: {len(rows)} player rows"
            )
        else:
            print(
                f"[{i}/{len(events)}] "
                f"{event_id}: no completed player rows"
            )

    except Exception as exc:
        errors.append((event_id, str(exc)))
        print(
            f"[{i}/{len(events)}] "
            f"{event_id}: ERROR {exc}"
        )

    time.sleep(0.05)

history = pd.DataFrame(all_rows)

if len(history):
    history = (
        history
        .drop_duplicates(
            subset=["game_id", "player_id"],
            keep="last",
        )
        .sort_values(
            ["date", "game_id", "nba_team", "player_name"]
        )
        .reset_index(drop=True)
    )

problems = validate_rows(history)

print()
print("=" * 100)
print("PARSER VALIDATION")
print("=" * 100)

if problems:
    print("FAILED")
    for problem in problems:
        print(" -", problem)

    print()
    print(
        "The existing CSV has NOT been overwritten because "
        "the corrected parser failed validation."
    )
    raise SystemExit(1)

print("PASSED")
print(" - shooting makes/attempts parsed")
print(" - no material points-with-zero-FGA problem")
print(" - makes do not exceed attempts")
print(" - 3PM/3PA do not exceed FGM/FGA")
print(" - PTS reconciles to 2PM + 3PM + FTM")

# Remove raw diagnostic fields before permanent save.
history = history.drop(
    columns=["_fg_raw", "_three_raw", "_ft_raw"],
    errors="ignore",
)

history.to_csv(HISTORY_FILE, index=False)

stats = build_stats(history)
stats.to_csv(STATS_FILE, index=False)

print()
print("=" * 100)
print("CORRECTED CURRENT-SEASON HISTORY SUMMARY")
print("=" * 100)
print("Player-game rows:", len(history))
print(
    "Unique completed games:",
    history["game_id"].nunique()
    if len(history) else 0
)
print(
    "Unique players:",
    history["player_id"].nunique()
    if len(history) else 0
)
print("Fetch errors:", len(errors))

print()
print("TOP CURRENT-SEASON FANTASY AVERAGES")
print("-" * 100)

if len(stats):
    print(
        stats[
            ["player_name", "games", "fp_avg", "fp_sd"]
        ]
        .head(25)
        .round(3)
        .to_string(index=False)
    )

print()
print("NIKOLA JOKIC GAME-LEVEL AUDIT")
print("-" * 100)

jokic = history[
    history["player_name"] == "Nikola Jokic"
]

audit_cols = [
    "date", "game_id", "minutes", "points",
    "fgm", "fga", "three_pm", "three_pa",
    "ftm", "fta", "rebounds", "assists",
    "steals", "blocks", "turnovers",
    "fantasy_points",
]

if len(jokic):
    print(
        jokic[audit_cols]
        .round(3)
        .to_string(index=False)
    )
else:
    print("Nikola Jokic not found.")

print()
print("=" * 100)
print("FILES REPLACED")
print("=" * 100)
print(os.path.basename(HISTORY_FILE))
print(os.path.basename(STATS_FILE))

if errors:
    print()
    print("FETCH ERRORS")
    for event_id, msg in errors:
        print(event_id, msg)

print()
print(
    "Do NOT run build_current_player_projections_v2.py until "
    "the Jokic audit and leaderboard have been checked."
)
