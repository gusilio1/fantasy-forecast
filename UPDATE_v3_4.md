# Updating to v3.4: History tab

v3.4 adds a **History** tab with two parts:

- **NBA fantasy records since 2023-24:** every NBA player's best games, halves and
  quarters, and the worst games, all under your league's scoring, regular season and
  playoffs. You can pick all seasons or a single one. This season's games are added
  automatically after each run.
- **League history since 2024-25** (the first 12-team season), from your spreadsheets
  plus this season from ESPN:
  - week records: highest and lowest week, best and worst PPP week, biggest and
    closest win, highest-scoring matchup
  - all-time standings and each season's standings
  - the top 15 weeks
  - head-to-head records for any manager

## 1. Upload these files

**Add file**, then **Upload files**. Drag in every file from this folder except this
guide and `workflow_hourly-forecast.yml`, then click **Commit changes**. Check that each
name is exact: no " (1)" on the end.

New:

- `nba_history.py`
- `pbp_fantasy.py`
- `nba_fantasy_history_2023_2026.csv.gz` (about 2 MB: the 2023-24 to 2025-26 records)
- `league_history_2024_2026.csv` (your spreadsheet weeks)

Changed: `build_dashboard.py`, `run_pipeline.py`. The other `.py` files are the latest
versions of everything so far. Uploading them again does no harm and makes sure
nothing is out of date.

If you have already installed v3.3, the workflow file is unchanged and you can skip it.

## 2. Run it

**Actions**, then **Hourly forecast**, then **Run workflow**.

## Joining managers across seasons

A manager's seasons are joined by name: the spreadsheet name, and this season's
ESPN first name. Two settings near the top of `build_dashboard.py` fix names that
differ:

- `NAME_ALIASES = {"Hammer": "Hamish"}`: spreadsheet names that changed between
  seasons.
- `TEAM_HISTORY_NAMES = {}`: this season's ESPN team name mapped to the
  spreadsheet name. For example, `{"Gustavo Giants": "Gus"}` if ESPN shows a
  different first name for that team.

If the History tab lists someone twice, once with a team and once as "Former
manager", add them to `TEAM_HISTORY_NAMES`.
