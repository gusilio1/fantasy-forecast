"""
run_pipeline.py

Runs the whole forecast end to end, in order, and stops at the first failure:

    1. fantasy_sim.py                          rosters, injury statuses, schedule flags (ESPN login)
    2. rebuild_current_season_history_2027.py  this season's box scores (public ESPN)
    3. build_current_player_projections_v3.py  player projections
    4. build_full_nba_projection_universe.py   every NBA roster player (public ESPN)
    5. fetch_espn_matchups.py                  matchups + actual points so far (ESPN login)
    6. weekly_matchup_simulator_v3_2.py        Monte Carlo, pickups, win probabilities
    7. build_dashboard.py                      site/index.html

Used by the hourly GitHub workflow; also works on your own computer:
    python run_pipeline.py
"""

import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Fantasy league + player pool", "fantasy_sim.py"),
    ("Current-season box scores", "rebuild_current_season_history_2027.py"),
    ("Player projections", "build_current_player_projections_v3.py"),
    ("Full NBA roster universe", "build_full_nba_projection_universe.py"),
    ("ESPN matchups + points so far", "fetch_espn_matchups.py"),
    ("Weekly simulation", "weekly_matchup_simulator_v3_2.py"),
    ("Dashboard page", "build_dashboard.py"),
]


def main():
    start = time.time()
    for i, (label, script) in enumerate(STEPS, 1):
        print(f"\n{'=' * 100}\nSTEP {i}/{len(STEPS)}: {label} ({script})\n{'=' * 100}", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, os.path.join(BASE, script)], cwd=BASE)
        if result.returncode != 0:
            print(f"\nPIPELINE STOPPED: step {i} ({script}) failed with exit code {result.returncode}. "
                  "The previously published dashboard stays online.", flush=True)
            sys.exit(result.returncode)
        print(f"Step {i} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nPipeline finished in {(time.time() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
