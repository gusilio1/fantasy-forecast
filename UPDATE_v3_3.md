# Updating to v3.3: season forecast and self-test

v3.3 adds two page tabs:

- **Season:** each team's projected record, chance of making the playoffs, bye, final and
  title, and its likely finishing position. There is also a week-by-week view of your
  remaining schedule. The standings rail gains a Playoffs column.
- **Model check:** every update saves what it predicted. When a week ends, those
  predictions are scored against ESPN's actual results: win chances, weekly totals,
  80% ranges and player projections, with ESPN's own projections for comparison.
  After two finished weeks, the simulation starts correcting itself if its weekly
  totals have been consistently too high or too low, or its ranges too narrow or
  too wide. The corrections are cautious and capped.

The page also moves on to the next week's forecast as soon as Sunday's last NBA game
is final, instead of waiting hours for ESPN to switch weeks. Last week's results are
counted in the standings straight away.

It is also much faster. The simulation splits its work across all of GitHub's
processor cores and the slowest parts of the code have been rewritten. On the same
inputs, it gives exactly the same results as before. Finished NBA games are no
longer downloaded again every hour.

## 1. Upload these files

**Add file**, then **Upload files**. Drag in all 9 files and click **Commit changes**.
Check that each name is exact: no " (1)" or " (2)" on the end.

New:

- `season_forecast.py`
- `model_check.py`

Replaced:

- `weekly_matchup_simulator_v3_2.py`
- `fetch_espn_matchups.py`
- `run_pipeline.py`
- `build_dashboard.py`
- `fantasy_sim.py` (the retry fix, in case it isn't uploaded yet)
- `injury_rotation_v3_features.py` (faster, same results)
- `rebuild_current_season_history_2027.py` (keeps finished games instead of downloading them again)

## 2. Replace the schedule file

The workflow now saves the self-test's history between runs on a separate branch
called `data`. That needs permission to write to the repository.

1. Open `.github/workflows/hourly-forecast.yml` and click the pencil icon.
2. Select everything, delete it, and paste in everything from
   `workflow_hourly-forecast.yml`.
3. Click **Commit changes**.

## 3. Run it

**Actions**, then **Hourly forecast**, then **Run workflow**.

After the run, a new branch called `data` appears. Leave it alone: it holds the saved
forecasts. If you delete it, the self-test starts again from scratch.

If the run fails at **Save forecast history** with "Permission denied" or a 403 error:

1. Go to **Settings**, then **Actions**, then **General**.
2. Under **Workflow permissions**, choose **Read and write permissions**.
3. Click **Save**, then run the workflow again.

## Settings

| Setting | File | Default | What it does |
|---|---|---|---|
| `USE_LEARNED_CORRECTIONS` | `weekly_matchup_simulator_v3_2.py` | `True` | `False` ignores the self-test's corrections |
| `SEASON_SIMS` | `season_forecast.py` | `10000` | Number of seasons simulated (takes about a second) |
| `PLAYOFF_TEAMS_OVERRIDE` | `season_forecast.py` | `None` | Set this if ESPN's playoff team count is wrong |
| `TIEBREAK_OVERRIDE` | `season_forecast.py` | `None` | `"points_for"` or `"h2h"` |
| `MIN_TEAM_WEEKS` | `model_check.py` | `24` | Number of checked team-weeks before corrections start |

The run log's **STEP 5** shows the league format ESPN reports, for example "16
regular-season matchups, 6 playoff teams". Check that it matches your league.
