# Updating your GitHub forecast to v3.2

v3.2 adds:

- **Real-life matchups.** Each NBA game gets its own pace, and players' rates are
  adjusted for the opponent's defense. Tested on held-out 2026 games: team-game
  fantasy point error fell from 31.7 to 29.3.
- **Pickups.** Every team streams free agents with its usual number of adds, and
  every team gets its own recommended moves.
- **A redesigned page.** Standings sit in a side rail. New tabs show the next day's
  top players and top teams, top free agents, pickups, and **PPP and records**
  (PPP this week, last 3 weeks and season, plus league records).

## 1. Upload these files

In your repository, click **Add file**, then **Upload files**, drag in these 8
files, and click **Commit changes**. Files with the same name are replaced.

New:

- `weekly_matchup_simulator_v3_2.py`
- `matchup_context.py`
- `matchup_context_settings.json`
- `backtest_matchup_context.py` (only needed to re-test the matchup settings)

Replaced:

- `run_pipeline.py` (now runs v3.2)
- `fetch_espn_matchups.py` (also reads standings, acquisitions, manager names and
  the season's daily lineups)
- `build_dashboard.py` (the new page)
- `requirements.txt` (adds `scipy`)

You can delete `weekly_matchup_simulator_v3_1.py`, but leaving it does no harm.

## 2. Update the hourly schedule file

The schedule now keeps past days' lineups between runs, so each hourly run only
downloads the newest days.

1. In your repository, open `.github/workflows/hourly-forecast.yml`.
2. Click the pencil icon (**Edit this file**).
3. Select everything in the editor and delete it.
4. Open `workflow_hourly-forecast.yml` from this folder, copy everything, and
   paste it in.
5. Click **Commit changes**.

## 3. Run it

**Actions**, then **Hourly forecast**, then **Run workflow**. Expect about 25 to 30
minutes per run. The first run also downloads every finished day of the season
once (about one second per day).

## Settings

All of these are near the top of `weekly_matchup_simulator_v3_2.py`:

| Setting | Default | What it does |
|---|---|---|
| `PUBLISH_PICKUP_ADVICE` | `True` | `False` hides every team's recommended moves on the public page |
| `MIN_STREAM_GAIN` | `5.0` | An add must gain at least this many projected points this week |
| `STREAM_DROP_MARGIN` | `2.0` | Streaming only drops players within this many FP/game of waiver replacement level |
| `USE_MATCHUP_CONTEXT` | `True` | `False` switches off pace and defense adjustments |
| `MY_TEAM_NAME` | `Gustavo Giants` | Whose recommended moves are printed in the run log |

`MIN_STARTS_PPP_WEEK` near the top of `build_dashboard.py` (default 15) sets how
many starts a week needs to count for the best and worst PPP week records.
