# Hourly league forecast: setup

This folder runs the whole forecast every hour on GitHub (free) and publishes a
web page your league can open. Your laptop does not need to be on.

Every hour GitHub runs `run_pipeline.py`, which does:

1. `fantasy_sim.py`: rosters, injury statuses, this week's schedule
2. `rebuild_current_season_history_2027.py`: this season's box scores
3. `build_current_player_projections_v3.py`: player projections
4. `build_full_nba_projection_universe.py`: every NBA roster player
5. `fetch_espn_matchups.py`: matchups and actual points so far
6. `weekly_matchup_simulator_v3_1.py`: the simulation and win probabilities
7. `build_dashboard.py`: the web page

Setup takes about 20 minutes. No command line is needed.

## 1. Make a GitHub account

Go to github.com, sign up (free), and confirm your email.

## 2. Create the repository

1. Click **+** (top right), then **New repository**.
2. Name it `fantasy-forecast`.
3. Choose **Public**. Public repositories get unlimited free hourly runs. Your
   ESPN login stays secret; the code and the forecast page are public.
4. Tick **Add a README file**.
5. Click **Create repository**.

## 3. Upload the files

1. In the new repository, click **Add file**, then **Upload files**.
2. Drag in every file from this folder: all the `.py`, `.csv`, `.npz` files,
   `requirements.txt` and this `SETUP.md`.
3. **Do not upload your `.env` file.** It holds your ESPN login.
4. Click **Commit changes**. The upload can take a minute; the largest file is
   about 19 MB.

## 4. Add the hourly schedule

GitHub only runs workflows stored in a folder called `.github/workflows`.

1. Click **Add file**, then **Create new file**.
2. In the name box type exactly: `.github/workflows/hourly-forecast.yml`
   (typing the `/` creates the folders).
3. Open `workflow_hourly-forecast.yml` from this folder in any text editor, copy
   everything, and paste it into the big box.
4. Click **Commit changes**.

## 5. Add your ESPN login as secrets

Open your `.env` file to get the two values.

1. In the repository, click **Settings**, then **Secrets and variables**, then **Actions**.
2. Click **New repository secret**.
   - Name: `ESPN_SWID`
   - Secret: your SWID value, including the curly braces `{...}`
3. Click **Add secret**, then **New repository secret** again.
   - Name: `ESPN_S2`
   - Secret: your espn_s2 value
4. Click **Add secret**.

Secrets are encrypted. Nobody can read them, including you, once saved.

## 6. Turn on the web page

1. **Settings**, then **Pages** (left menu).
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.

## 7. Run it the first time

1. Click the **Actions** tab. If GitHub asks, click to enable workflows.
2. Click **Hourly forecast** on the left, then **Run workflow**, then the green
   **Run workflow** button.
3. The run takes 15 to 30 minutes. A green tick means it worked.

Your page is now at:

    https://YOUR-GITHUB-USERNAME.github.io/fantasy-forecast/

**Settings**, then **Pages**, also shows the link. Share it with your league.
After this, it updates every hour at about :17 past (GitHub sometimes starts
scheduled runs a few minutes late).

## If a run fails

The page keeps showing the last successful forecast.

1. **Actions**, click the failed run, then click **build**.
2. Open **Run forecast pipeline**. It says which step stopped (for example
   `STEP 5/7: ESPN matchups`).
3. Copy the last 30 or so lines and send them to Claude.

Common causes:

- **ESPN login expired** (step 1 or 5 fails with HTTP 401 or 403): copy fresh
  `SWID` and `espn_s2` cookies from your browser and update both secrets (open a
  secret and click **Update**).
- **Runs stopped after about 60 days**: GitHub pauses schedules on repositories
  with no changes for 60 days and emails you. Click **Enable workflow** in the
  Actions tab, or upload any small change.

## Changing things

- **Number of simulations**: `SIMS = 2000` near the top of
  `weekly_matchup_simulator_v3_1.py`. More is smoother but slower.
- **Page title**: `LEAGUE_TITLE` near the top of `build_dashboard.py`.
- **Updating a file**: upload it again with **Add file**, then **Upload files**;
  the new version replaces the old one.
- **Run now**: **Actions**, **Hourly forecast**, **Run workflow**.
- **Run on your own computer**: `python run_pipeline.py` (needs your `.env`),
  then open `site/index.html`.
