"""
run_pipeline.py

Runs the whole forecast end to end, in order, and stops at the first failure:

    1. fantasy_sim.py                          rosters, injury statuses, schedule flags (ESPN login)
    2. rebuild_current_season_history_2027.py  this season's box scores (public ESPN)
    3. build_current_player_projections_v3.py  player projections
    4. build_full_nba_projection_universe.py   every NBA roster player (public ESPN)
    5. fetch_espn_matchups.py                  matchups + actual points so far (ESPN login)
    6. weekly_matchup_simulator_v3_2.py        Monte Carlo, pickups, win probabilities
    7. season_forecast.py                      rest of season, playoff odds (optional step)
    8. model_check.py                          self-test against real results (optional step)
    9. nba_history.py                          NBA fantasy records since 2023-24 (optional step)
   10. build_dashboard.py                      site/index.html

Optional steps print a warning if they fail and the run carries on.

Used by the hourly GitHub workflow; also works on your own computer:
    python run_pipeline.py
"""

import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))

STEPS = [   # (label, script, required)
    ("Fantasy league + player pool", "fantasy_sim.py", True),
    ("Current-season box scores", "rebuild_current_season_history_2027.py", True),
    ("Player projections", "build_current_player_projections_v3.py", True),
    ("Full NBA roster universe", "build_full_nba_projection_universe.py", True),
    ("ESPN matchups + points so far", "fetch_espn_matchups.py", True),
    ("Weekly simulation", "weekly_matchup_simulator_v3_2.py", True),
    ("Season forecast", "season_forecast.py", False),
    ("Self-test", "model_check.py", False),
    ("NBA records since 2023-24", "nba_history.py", False),
    ("Dashboard page", "build_dashboard.py", True),
]


SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
DONE_STATUSES = {"STATUS_FINAL", "STATUS_FULL_TIME", "STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_FORFEIT"}


def nba_day_finished(day):
    """True / False, or None if ESPN could not be reached."""
    import requests
    for attempt in range(1, 4):
        try:
            r = requests.get(SCOREBOARD, params={"dates": day.strftime("%Y%m%d")},
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if r.status_code == 200:
                events = r.json().get("events") or []
                done = 0
                for ev in events:
                    st = ((ev.get("status") or {}).get("type") or {})
                    done += bool(st.get("completed") or st.get("name") in DONE_STATUSES)
                return done == len(events)
        except Exception:
            pass
        time.sleep(3 * attempt)
    return None


def target_week_start():
    """
    Monday of the fantasy week to forecast (US Eastern). The week moves on as soon as
    Sunday's last NBA game is final, instead of waiting for ESPN's overnight switch;
    and stays on the old week if Sunday's games run past midnight.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/New_York")).date()
    monday = today - timedelta(days=today.weekday())
    if today.weekday() == 6:                       # Sunday: move on once every game is final
        if nba_day_finished(today):
            return monday + timedelta(days=7), "Sunday's games are all final: forecasting next week"
    elif today.weekday() == 0:                     # Monday: unless Sunday's games are still going
        if nba_day_finished(today - timedelta(days=1)) is False:
            return monday - timedelta(days=7), "Sunday's games are still going: staying on last week"
    return monday, "current US Eastern week"


def main():
    start = time.time()
    if not os.environ.get("WEEK_START"):
        week, why = target_week_start()
        os.environ["WEEK_START"] = str(week)
        print(f"Fantasy week starting {week} ({why})", flush=True)
    for i, (label, script, required) in enumerate(STEPS, 1):
        print(f"\n{'=' * 100}\nSTEP {i}/{len(STEPS)}: {label} ({script})\n{'=' * 100}", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, os.path.join(BASE, script)], cwd=BASE)
        if result.returncode != 0 and not required:
            print(f"\nWARNING: step {i} ({script}) failed with exit code {result.returncode}; "
                  "continuing without it.", flush=True)
            continue
        if result.returncode != 0:
            print(f"\nPIPELINE STOPPED: step {i} ({script}) failed with exit code {result.returncode}. "
                  "The previously published dashboard stays online.", flush=True)
            sys.exit(result.returncode)
        print(f"Step {i} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nPipeline finished in {(time.time() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
