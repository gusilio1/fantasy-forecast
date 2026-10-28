import os
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

LEAGUE_ID = 1421453942
SEASON = 2027

# Historical ESPN seasons
HISTORY_RECENT_SEASON = 2026   # 2025-26 NBA season
HISTORY_OLDER_SEASON = 2025    # 2024-25 NBA season

# Minimum number of historical games required before we trust
# an individual player's calculated standard deviation.
MIN_HISTORY_GAMES = 10

# When estimating an SD, start by looking for historical players
# whose scoring average was within this many fantasy points/game.
COMPARABLE_FP_RANGE = 3.0

# We want at least this many comparable player-seasons.
MIN_COMPARABLES = 20


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HISTORY_RECENT_FILE = os.path.join(
    BASE_DIR,
    "player_stats_2026.csv"
)

HISTORY_OLDER_FILE = os.path.join(
    BASE_DIR,
    "player_stats_2025.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "player_pool.csv"
)


# ============================================================
# LOAD ESPN LOGIN
# ============================================================

load_dotenv(os.path.join(BASE_DIR, ".env"))

SWID = os.getenv("ESPN_SWID")
ESPN_S2 = os.getenv("ESPN_S2")

if not SWID or not ESPN_S2:
    raise ValueError(
        "ESPN_SWID or ESPN_S2 is missing from .env"
    )


# ============================================================
# NBA TEAM MAPPING
# ============================================================

NBA_TEAMS = {
    1: "ATL",
    2: "BOS",
    3: "NOP",
    4: "CHI",
    5: "CLE",
    6: "DAL",
    7: "DEN",
    8: "DET",
    9: "GSW",
    10: "HOU",
    11: "IND",
    12: "LAC",
    13: "LAL",
    14: "MIA",
    15: "MIL",
    16: "MIN",
    17: "BKN",
    18: "NYK",
    19: "ORL",
    20: "PHI",
    21: "PHX",
    22: "POR",
    23: "SAC",
    24: "SAS",
    25: "OKC",
    26: "UTA",
    27: "WAS",
    28: "TOR",
    29: "MEM",
    30: "CHA"
}


TEAM_ABBREVIATION_FIXES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS"
}


# ============================================================
# LINEUP SLOT NAMES
# ============================================================

LINEUP_SLOTS = {
    0: "PG",
    1: "SG",
    2: "SF",
    3: "PF",
    4: "C",
    5: "G",
    6: "F",
    7: "SG/SF",
    8: "G/F",
    9: "PF/C",
    10: "F/C",
    11: "UTIL",
    12: "BENCH",
    13: "IR"
}


# The 10 active lineup positions in this league.
# There are three separate UTIL spots.
ACTIVE_LINEUP_SLOTS = [
    "PG", "SG", "SF", "PF", "C", "G", "F",
    "UTIL", "UTIL", "UTIL"
]


def get_eligibility(player):
    """Return ESPN slot IDs plus a clean readable active-position list."""

    eligible_slot_ids = player.get("eligibleSlots", []) or []

    # ESPN can also return combination slot IDs such as SG/SF, G/F, PF/C
    # and F/C. For simulation legality we keep the original numeric IDs.
    # For display, show only this league's actual active slot types.
    display_slot_ids = [0, 1, 2, 3, 4, 5, 6, 11]

    eligible_positions = [
        LINEUP_SLOTS[slot_id]
        for slot_id in display_slot_ids
        if slot_id in eligible_slot_ids
    ]

    default_position_id = player.get("defaultPositionId")
    default_position = LINEUP_SLOTS.get(default_position_id, str(default_position_id))

    return default_position, eligible_slot_ids, eligible_positions


# ============================================================
# ESPN CONNECTION DETAILS
# ============================================================

LEAGUE_URL = (
    f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/"
    f"seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"
)

COOKIES = {
    "SWID": SWID,
    "espn_s2": ESPN_S2
}


# ============================================================
# HELPER: CURRENT ESPN FANTASY AVERAGES
# ============================================================

def get_current_fantasy_stats(player):

    actual_fp_avg = None
    projected_fp_avg = None

    for stat in player.get("stats", []):

        if stat.get("seasonId") != SEASON:
            continue

        if stat.get("statSplitTypeId") != 0:
            continue

        if stat.get("statSourceId") == 0:
            actual_fp_avg = stat.get("appliedAverage")

        elif stat.get("statSourceId") == 1:
            projected_fp_avg = stat.get("appliedAverage")

    return actual_fp_avg, projected_fp_avg


# ============================================================
# DOWNLOAD CURRENT ROSTERS
# ============================================================

print()
print("Downloading current ESPN fantasy league...")

roster_params = [
    ("view", "mTeam"),
    ("view", "mRoster"),
    ("view", "mSettings")
]

response = requests.get(
    LEAGUE_URL,
    params=roster_params,
    cookies=COOKIES,
    timeout=30
)

print("Fantasy API HTTP status:", response.status_code)

if response.status_code != 200:
    print(response.text[:2000])
    raise SystemExit


league_data = response.json()

teams = league_data.get("teams", [])

team_names = {}

for team in teams:

    team_id = team.get("id")

    team_name = team.get("name")

    if not team_name:
        location = team.get("location", "")
        nickname = team.get("nickname", "")
        team_name = f"{location} {nickname}".strip()

    team_names[team_id] = team_name


rostered_players = []

for team in teams:

    fantasy_team_id = team.get("id")

    fantasy_team = team_names.get(
        fantasy_team_id,
        f"Team {fantasy_team_id}"
    )

    roster = team.get("roster", {})

    for entry in roster.get("entries", []):

        player_pool_entry = entry.get(
            "playerPoolEntry",
            {}
        )

        player = player_pool_entry.get(
            "player",
            {}
        )

        if not player:
            continue

        player_id = player.get("id")

        player_name = player.get(
            "fullName",
            "UNKNOWN"
        )

        pro_team_id = player.get("proTeamId")

        nba_team = NBA_TEAMS.get(
            pro_team_id,
            f"UNKNOWN({pro_team_id})"
        )

        injury_status = player.get(
            "injuryStatus",
            "UNKNOWN"
        )

        default_position, eligible_slot_ids, eligible_positions = (
            get_eligibility(player)
        )

        lineup_slot_id = entry.get(
            "lineupSlotId"
        )

        lineup_slot = LINEUP_SLOTS.get(
            lineup_slot_id,
            str(lineup_slot_id)
        )

        actual_fp_avg, projected_fp_avg = (
            get_current_fantasy_stats(player)
        )

        rostered_players.append({
            "player_id": player_id,
            "player_name": player_name,
            "nba_team": nba_team,
            "fantasy_team_id": fantasy_team_id,
            "fantasy_team": fantasy_team,
            "roster_status": "ROSTERED",
            "lineup_slot": lineup_slot,
            "injury_status": injury_status,
            "default_position": default_position,
            "eligible_slot_ids": json.dumps(eligible_slot_ids),
            "eligible_positions": "/".join(eligible_positions),
            "actual_fp_avg": actual_fp_avg,
            "projected_fp_avg": projected_fp_avg,
            "percent_owned": None,
            "percent_started": None
        })


rostered_df = pd.DataFrame(rostered_players)

print(
    "Current roster entries:",
    len(rostered_df)
)


# ============================================================
# DOWNLOAD FREE AGENTS + WAIVER PLAYERS
# ============================================================

print()
print("Downloading available ESPN players...")

fantasy_filter = {
    "players": {
        "filterStatus": {
            "value": [
                "FREEAGENT",
                "WAIVERS"
            ]
        },
        "limit": 1000,
        "sortPercOwned": {
            "sortPriority": 1,
            "sortAsc": False
        }
    }
}

headers = {
    "X-Fantasy-Filter":
        json.dumps(fantasy_filter)
}

waiver_response = requests.get(
    LEAGUE_URL,
    params={
        "view": "kona_player_info"
    },
    headers=headers,
    cookies=COOKIES,
    timeout=30
)

print(
    "Available-player API HTTP status:",
    waiver_response.status_code
)

if waiver_response.status_code != 200:
    print(waiver_response.text[:2000])
    raise SystemExit


waiver_data = waiver_response.json()

available_entries = waiver_data.get(
    "players",
    []
)

print(
    "Available ESPN entries returned:",
    len(available_entries)
)


available_players = []

for entry in available_entries:

    player = entry.get("player", {})

    if not player:
        continue

    player_id = player.get("id")

    player_name = player.get(
        "fullName",
        "UNKNOWN"
    )

    pro_team_id = player.get("proTeamId")

    nba_team = NBA_TEAMS.get(
        pro_team_id,
        f"UNKNOWN({pro_team_id})"
    )

    injury_status = player.get(
        "injuryStatus",
        "UNKNOWN"
    )

    default_position, eligible_slot_ids, eligible_positions = (
        get_eligibility(player)
    )

    actual_fp_avg, projected_fp_avg = (
        get_current_fantasy_stats(player)
    )

    ownership = player.get(
        "ownership",
        {}
    )

    percent_owned = ownership.get(
        "percentOwned"
    )

    percent_started = ownership.get(
        "percentStarted"
    )

    if projected_fp_avg is None:
        continue

    available_players.append({
        "player_id": player_id,
        "player_name": player_name,
        "nba_team": nba_team,
        "fantasy_team_id": None,
        "fantasy_team": None,
        "roster_status": "FREE_AGENT",
        "lineup_slot": None,
        "injury_status": injury_status,
        "default_position": default_position,
        "eligible_slot_ids": json.dumps(eligible_slot_ids),
        "eligible_positions": "/".join(eligible_positions),
        "actual_fp_avg": actual_fp_avg,
        "projected_fp_avg": projected_fp_avg,
        "percent_owned": percent_owned,
        "percent_started": percent_started
    })


available_df = pd.DataFrame(
    available_players
)

print(
    "Available players with projections:",
    len(available_df)
)


# ============================================================
# COMBINE ROSTERED + AVAILABLE PLAYERS
# ============================================================

player_pool = pd.concat(
    [
        rostered_df,
        available_df
    ],
    ignore_index=True
)


duplicate_ids = player_pool[
    player_pool.duplicated(
        subset=["player_id"],
        keep=False
    )
]

if len(duplicate_ids) > 0:

    print()
    print("WARNING: DUPLICATE PLAYER IDs FOUND")

    print(
        duplicate_ids[
            [
                "player_id",
                "player_name",
                "roster_status",
                "fantasy_team"
            ]
        ].to_string(index=False)
    )


print()
print(
    "Combined player pool:",
    len(player_pool)
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print()
print("Loading historical player data...")


if not os.path.exists(HISTORY_RECENT_FILE):

    print(
        "\nERROR: player_stats_2026.csv "
        "was not found."
    )

    raise SystemExit


if not os.path.exists(HISTORY_OLDER_FILE):

    print(
        "\nERROR: player_stats_2025.csv "
        "was not found."
    )

    raise SystemExit


history_recent = pd.read_csv(
    HISTORY_RECENT_FILE
)

history_older = pd.read_csv(
    HISTORY_OLDER_FILE
)


# ============================================================
# PREPARE HISTORICAL COLUMNS
# ============================================================

history_recent = history_recent.rename(
    columns={
        "games": "games_2026",
        "fp_avg": "fp_avg_2026",
        "fp_sd": "fp_sd_2026",
        "fp_min": "fp_min_2026",
        "fp_max": "fp_max_2026"
    }
)

history_older = history_older.rename(
    columns={
        "games": "games_2025",
        "fp_avg": "fp_avg_2025",
        "fp_sd": "fp_sd_2025",
        "fp_min": "fp_min_2025",
        "fp_max": "fp_max_2025"
    }
)


recent_columns = [
    "player_id",
    "games_2026",
    "fp_avg_2026",
    "fp_sd_2026",
    "fp_min_2026",
    "fp_max_2026"
]

older_columns = [
    "player_id",
    "games_2025",
    "fp_avg_2025",
    "fp_sd_2025",
    "fp_min_2025",
    "fp_max_2025"
]


history_recent = history_recent[
    recent_columns
]

history_older = history_older[
    older_columns
]


# ============================================================
# BUILD COMPARABLE-PLAYER SD DATABASE
# ============================================================

# We build a combined table of historical player-seasons.
#
# Only player-seasons with at least 10 captured games are
# allowed into this database. This prevents tiny samples
# such as a 2-game SD from influencing rookie estimates.

recent_comparables = history_recent[
    (history_recent["games_2026"] >= MIN_HISTORY_GAMES)
    & history_recent["fp_avg_2026"].notna()
    & history_recent["fp_sd_2026"].notna()
][
    [
        "player_id",
        "games_2026",
        "fp_avg_2026",
        "fp_sd_2026"
    ]
].copy()

recent_comparables = recent_comparables.rename(
    columns={
        "games_2026": "hist_games",
        "fp_avg_2026": "hist_fp_avg",
        "fp_sd_2026": "hist_fp_sd"
    }
)

recent_comparables["hist_season"] = "2025-26"


older_comparables = history_older[
    (history_older["games_2025"] >= MIN_HISTORY_GAMES)
    & history_older["fp_avg_2025"].notna()
    & history_older["fp_sd_2025"].notna()
][
    [
        "player_id",
        "games_2025",
        "fp_avg_2025",
        "fp_sd_2025"
    ]
].copy()

older_comparables = older_comparables.rename(
    columns={
        "games_2025": "hist_games",
        "fp_avg_2025": "hist_fp_avg",
        "fp_sd_2025": "hist_fp_sd"
    }
)

older_comparables["hist_season"] = "2024-25"


comparable_database = pd.concat(
    [
        recent_comparables,
        older_comparables
    ],
    ignore_index=True
)


print(
    "Historical player-seasons available "
    "for SD estimation:",
    len(comparable_database)
)


# ============================================================
# MERGE HISTORY INTO MASTER PLAYER POOL
# ============================================================

player_pool = player_pool.merge(
    history_recent,
    on="player_id",
    how="left"
)

player_pool = player_pool.merge(
    history_older,
    on="player_id",
    how="left"
)


# ============================================================
# SIMULATION MEAN
# ============================================================

player_pool["sim_mean"] = (
    player_pool["projected_fp_avg"]
)


# ============================================================
# ESTIMATE SD FROM COMPARABLE PLAYERS
# ============================================================

def estimate_comparable_sd(projected_fp_avg):

    if pd.isna(projected_fp_avg):
        return None, 0, None

    # Start with players within +/- 3 fantasy points/game.
    search_range = COMPARABLE_FP_RANGE

    comparable_players = comparable_database[
        (
            comparable_database["hist_fp_avg"]
            - projected_fp_avg
        ).abs() <= search_range
    ]

    # If we don't have enough comparable player-seasons,
    # gradually widen the scoring range.
    while (
        len(comparable_players) < MIN_COMPARABLES
        and search_range < 10
    ):

        search_range += 1

        comparable_players = comparable_database[
            (
                comparable_database["hist_fp_avg"]
                - projected_fp_avg
            ).abs() <= search_range
        ]

    if len(comparable_players) == 0:
        return None, 0, search_range

    estimated_sd = (
        comparable_players["hist_fp_sd"]
        .median()
    )

    return (
        estimated_sd,
        len(comparable_players),
        search_range
    )


# ============================================================
# CHOOSE SIMULATION SD
# ============================================================

def choose_sd(row):

    # First preference:
    # most recent historical season with >= 10 games.
    if (
        pd.notna(row["fp_sd_2026"])
        and pd.notna(row["games_2026"])
        and row["games_2026"] >= MIN_HISTORY_GAMES
    ):

        return pd.Series({
            "sim_sd": row["fp_sd_2026"],
            "sd_source": "2025-26",
            "sd_comparables": None,
            "sd_range": None
        })

    # Second preference:
    # older historical season with >= 10 games.
    if (
        pd.notna(row["fp_sd_2025"])
        and pd.notna(row["games_2025"])
        and row["games_2025"] >= MIN_HISTORY_GAMES
    ):

        return pd.Series({
            "sim_sd": row["fp_sd_2025"],
            "sd_source": "2024-25",
            "sd_comparables": None,
            "sd_range": None
        })

    # Otherwise estimate variability from players who
    # historically scored at a similar fantasy level.
    estimated_sd, comparable_count, search_range = (
        estimate_comparable_sd(
            row["projected_fp_avg"]
        )
    )

    return pd.Series({
        "sim_sd": estimated_sd,
        "sd_source": "ESTIMATED",
        "sd_comparables": comparable_count,
        "sd_range": search_range
    })


sd_results = player_pool.apply(
    choose_sd,
    axis=1
)

player_pool[
    [
        "sim_sd",
        "sd_source",
        "sd_comparables",
        "sd_range"
    ]
] = sd_results


# ============================================================
# CURRENT WEEK
# ============================================================

# Monday of the current US Eastern week (override with WEEK_START=YYYY-MM-DD).
_week_override = os.environ.get("WEEK_START", "").strip()
if _week_override:
    week_start = datetime.strptime(_week_override, "%Y-%m-%d")
else:
    from zoneinfo import ZoneInfo
    _today_et = datetime.now(ZoneInfo("America/New_York")).date()
    _monday = _today_et - timedelta(days=_today_et.weekday())
    week_start = datetime(_monday.year, _monday.month, _monday.day)

day_names = [
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun"
]


# ============================================================
# DOWNLOAD NBA SCHEDULE
# ============================================================

print()
print(
    "Downloading NBA schedule for",
    week_start.strftime("%Y-%m-%d"),
    "to",
    (
        week_start + timedelta(days=6)
    ).strftime("%Y-%m-%d")
)


team_schedule = {}


for day_number in range(7):

    date = (
        week_start
        + timedelta(days=day_number)
    )

    date_string = date.strftime(
        "%Y%m%d"
    )

    scoreboard_url = (
        "https://site.api.espn.com/apis/site/v2/"
        "sports/basketball/nba/scoreboard"
    )

    schedule_response = requests.get(
        scoreboard_url,
        params={
            "dates": date_string
        },
        timeout=30
    )

    if schedule_response.status_code != 200:

        print(
            "Schedule error:",
            date_string,
            schedule_response.status_code
        )

        continue


    schedule_data = (
        schedule_response.json()
    )

    day_name = day_names[
        day_number
    ]


    for event in schedule_data.get(
        "events",
        []
    ):

        competitions = event.get(
            "competitions",
            []
        )

        if not competitions:
            continue

        competitors = competitions[
            0
        ].get(
            "competitors",
            []
        )

        for competitor in competitors:

            team = competitor.get(
                "team",
                {}
            )

            abbreviation = team.get(
                "abbreviation"
            )

            if not abbreviation:
                continue

            abbreviation = (
                TEAM_ABBREVIATION_FIXES.get(
                    abbreviation,
                    abbreviation
                )
            )

            if abbreviation not in team_schedule:

                team_schedule[
                    abbreviation
                ] = []

            team_schedule[
                abbreviation
            ].append(
                day_name
            )


# ============================================================
# ADD SCHEDULE TO EACH PLAYER
# ============================================================

for day in day_names:

    player_pool[day] = (
        player_pool["nba_team"]
        .apply(
            lambda team:
            1
            if day in team_schedule.get(
                team,
                []
            )
            else 0
        )
    )


player_pool["games"] = (
    player_pool[day_names]
    .sum(axis=1)
)


# ============================================================
# TEAM MAPPING SAFETY CHECK
# ============================================================

known_player_teams = set(
    player_pool.loc[
        ~player_pool[
            "nba_team"
        ].str.startswith(
            "UNKNOWN",
            na=False
        ),
        "nba_team"
    ].dropna()
)

schedule_teams = set(
    team_schedule.keys()
)

missing_schedule_teams = (
    known_player_teams
    - schedule_teams
)


print()

if missing_schedule_teams:

    print(
        "WARNING - NBA TEAMS NOT FOUND "
        "IN SCHEDULE:"
    )

    print(
        sorted(
            missing_schedule_teams
        )
    )

else:

    print(
        "Team mapping check: OK"
    )


# ============================================================
# SORT MASTER TABLE
# ============================================================

player_pool["roster_sort"] = (
    player_pool["roster_status"]
    .map({
        "ROSTERED": 0,
        "FREE_AGENT": 1
    })
)

player_pool = player_pool.sort_values(
    by=[
        "roster_sort",
        "fantasy_team",
        "projected_fp_avg"
    ],
    ascending=[
        True,
        True,
        False
    ],
    na_position="last"
)

player_pool = player_pool.drop(
    columns=["roster_sort"]
)

player_pool = player_pool.reset_index(
    drop=True
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 75)
print("PLAYER POOL SUMMARY")
print("=" * 75)

print(
    player_pool[
        "roster_status"
    ].value_counts()
)


print()
print("=" * 75)
print("SD SOURCE SUMMARY")
print("=" * 75)

print(
    player_pool[
        "sd_source"
    ].value_counts(
        dropna=False
    )
)


# ============================================================
# CHECK FOR ANY SDs STILL MISSING
# ============================================================

missing_sd = player_pool[
    player_pool[
        "sim_sd"
    ].isna()
]


print()
print("=" * 75)
print("PLAYERS STILL MISSING SD")
print("=" * 75)

if len(missing_sd) == 0:

    print("None - every player now has a simulation SD.")

else:

    print(
        missing_sd[
            [
                "player_name",
                "projected_fp_avg",
                "sd_source"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SHOW ESTIMATED SD PLAYERS
# ============================================================

estimated_players = player_pool[
    player_pool[
        "sd_source"
    ] == "ESTIMATED"
].copy()

estimated_players = estimated_players.sort_values(
    "projected_fp_avg",
    ascending=False
)


print()
print("=" * 75)
print("PLAYERS USING ESTIMATED SD")
print("=" * 75)

print(
    estimated_players[
        [
            "roster_status",
            "fantasy_team",
            "player_name",
            "projected_fp_avg",
            "games_2026",
            "games_2025",
            "sim_sd",
            "sd_comparables",
            "sd_range"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# TOP AVAILABLE PLAYERS
# ============================================================

top_available = player_pool[
    player_pool[
        "roster_status"
    ] == "FREE_AGENT"
].sort_values(
    "projected_fp_avg",
    ascending=False
)


print()
print("=" * 75)
print("TOP 30 AVAILABLE PLAYERS BY ESPN PROJECTION")
print("=" * 75)

print(
    top_available[
        [
            "player_name",
            "nba_team",
            "injury_status",
            "projected_fp_avg",
            "sim_sd",
            "sd_source",
            "games"
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# DETERMINISTIC DAILY LINEUP SELECTOR
# ============================================================

# ESPN slot IDs for this league's 10 active lineup places.
# The three UTIL positions are separate lineup places but share slot ID 11.
ACTIVE_LINEUP_SLOT_IDS = [0, 1, 2, 3, 4, 5, 6, 11, 11, 11]


def parse_eligible_slot_ids(value):
    """Convert the saved ESPN eligibility field back into a set of integers."""

    if isinstance(value, list):
        return set(value)

    if pd.isna(value):
        return set()

    text = str(value).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {int(x) for x in parsed}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Handles Python-list-looking strings such as "[0, 5, 11]".
    text = text.strip("[]")
    result = set()
    for piece in text.split(","):
        piece = piece.strip()
        if piece:
            try:
                result.add(int(piece))
            except ValueError:
                pass
    return result


def best_projected_lineup(team_df, day_name):
    """Find the maximum-projection legal lineup for one fantasy team/day."""

    candidates = team_df[
        (team_df[day_name] == 1)
        & team_df["projected_fp_avg"].notna()
    ].copy()

    candidates = candidates.sort_values(
        "projected_fp_avg",
        ascending=False
    ).reset_index(drop=True)

    # DP state: mask of filled physical lineup places -> (score, assignments).
    dp = {0: (0.0, [])}

    for player_index, player in candidates.iterrows():
        eligible_ids = parse_eligible_slot_ids(player["eligible_slot_ids"])
        current_states = list(dp.items())

        for mask, (score, assignments) in current_states:
            for slot_index, slot_id in enumerate(ACTIVE_LINEUP_SLOT_IDS):
                slot_bit = 1 << slot_index

                if mask & slot_bit:
                    continue

                if slot_id not in eligible_ids:
                    continue

                new_mask = mask | slot_bit
                new_score = score + float(player["projected_fp_avg"])
                new_assignments = assignments + [(slot_index, player_index)]

                if new_mask not in dp or new_score > dp[new_mask][0]:
                    dp[new_mask] = (new_score, new_assignments)

    best_mask, (best_score, best_assignments) = max(
        dp.items(),
        key=lambda item: (item[0].bit_count(), item[1][0])
    )

    lineup_rows = []
    used_player_indices = set()

    for slot_index, player_index in sorted(best_assignments):
        player = candidates.loc[player_index]
        used_player_indices.add(player_index)

        lineup_rows.append({
            "slot": ACTIVE_LINEUP_SLOTS[slot_index],
            "player_name": player["player_name"],
            "nba_team": player["nba_team"],
            "eligible_positions": player["eligible_positions"],
            "projected_fp_avg": player["projected_fp_avg"]
        })

    lineup_df = pd.DataFrame(lineup_rows)

    bench_df = candidates.loc[
        ~candidates.index.isin(used_player_indices),
        ["player_name", "nba_team", "eligible_positions", "projected_fp_avg"]
    ].copy()

    return lineup_df, bench_df, best_score, len(candidates)


# ============================================================
# ALL-TEAM LINEUP DIAGNOSTICS
# ============================================================

print()
print("=" * 75)
print("ALL-TEAM LINEUP DIAGNOSTICS")
print("=" * 75)

congested_days = []
position_squeeze_days = []

for team in teams:
    fantasy_team_id = team.get("id")
    fantasy_team_name = team_names.get(
        fantasy_team_id,
        f"Team {fantasy_team_id}"
    )

    team_df = player_pool[
        player_pool["fantasy_team_id"] == fantasy_team_id
    ].copy()

    for day in day_names:
        lineup_df, bench_df, projected_total, players_with_games = (
            best_projected_lineup(team_df, day)
        )

        starters = len(lineup_df)
        benched = len(bench_df)
        bench_points = (
            float(bench_df["projected_fp_avg"].sum())
            if benched > 0 else 0.0
        )

        # More than 10 players have games: unavoidable lineup congestion.
        if players_with_games > len(ACTIVE_LINEUP_SLOTS):
            congested_days.append({
                "fantasy_team": fantasy_team_name,
                "day": day,
                "players_with_games": players_with_games,
                "starters": starters,
                "benched": benched,
                "bench_projected_fp": bench_points,
                "active_projected_fp": projected_total
            })

        # <=10 players had games but somebody still could not be placed.
        # This flags a genuine position-eligibility squeeze for inspection.
        elif benched > 0:
            position_squeeze_days.append({
                "fantasy_team": fantasy_team_name,
                "day": day,
                "players_with_games": players_with_games,
                "starters": starters,
                "benched": benched,
                "bench_projected_fp": bench_points,
                "active_projected_fp": projected_total
            })


print()
print("CONGESTED DAYS (>10 PLAYERS WITH NBA GAMES)")
print("-" * 75)

if not congested_days:
    print("None this week.")
else:
    congested_df = pd.DataFrame(congested_days)
    print(
        congested_df.to_string(
            index=False,
            formatters={
                "bench_projected_fp": "{:.2f}".format,
                "active_projected_fp": "{:.2f}".format
            }
        )
    )

print()
print("POSITION-ELIGIBILITY SQUEEZES (<=10 PLAYERS, BUT SOMEONE BENCHED)")
print("-" * 75)

if not position_squeeze_days:
    print("None - good sign. Every <=10-player day fits legally.")
else:
    squeeze_df = pd.DataFrame(position_squeeze_days)
    print(
        squeeze_df.to_string(
            index=False,
            formatters={
                "bench_projected_fp": "{:.2f}".format,
                "active_projected_fp": "{:.2f}".format
            }
        )
    )


# ============================================================
# DETAILED CONGESTED-DAY CHECK
# ============================================================

print()
print("=" * 75)
print("DETAILED CONGESTED-DAY CHECK")
print("=" * 75)

if not congested_days:
    print("No congested days to inspect this week.")
else:
    for item in congested_days:
        fantasy_team_name = item["fantasy_team"]
        day = item["day"]

        team_id_matches = [
            team_id for team_id, name in team_names.items()
            if name == fantasy_team_name
        ]

        if not team_id_matches:
            continue

        fantasy_team_id = team_id_matches[0]
        team_df = player_pool[
            player_pool["fantasy_team_id"] == fantasy_team_id
        ].copy()

        lineup_df, bench_df, projected_total, players_with_games = (
            best_projected_lineup(team_df, day)
        )

        print()
        print("-" * 75)
        print(f"{fantasy_team_name} - {day}")
        print(
            f"Players with games: {players_with_games} | "
            f"Starters: {len(lineup_df)} | "
            f"Benched: {len(bench_df)}"
        )
        print("-" * 75)

        print("STARTERS")
        print(
            lineup_df.to_string(
                index=False,
                formatters={"projected_fp_avg": "{:.2f}".format}
            )
        )

        print()
        print("BENCHED PLAYERS WHO HAVE A GAME")
        print(
            bench_df.to_string(
                index=False,
                formatters={"projected_fp_avg": "{:.2f}".format}
            )
        )

        print()
        print("Projected active-lineup total:", round(projected_total, 2))
        print(
            "Projected FP left on bench:",
            round(float(bench_df["projected_fp_avg"].sum()), 2)
        )


# ============================================================
# SAVE MASTER PLAYER POOL
# ============================================================

player_pool.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 75)
print("FILE SAVED")
print("=" * 75)

print("player_pool.csv")

print(
    "Total players:",
    len(player_pool)
)