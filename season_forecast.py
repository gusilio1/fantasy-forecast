"""
season_forecast.py

Rest-of-season forecast: every remaining fantasy week, final standings, playoff
chances and title chances. Runs after weekly_matchup_simulator_v3_2.py.

How it works (deliberately simpler than the weekly simulator)
-------------------------------------------------------------
1. Each rostered player gets an expected fantasy score per game from his
   projection (rates x expected minutes, league-average pace).
2. Each future NBA day: a fantasy team scores the sum of its best 10 players
   with a game that day (expected score x chance he plays). Injured players
   count from their ESPN return date; everyone else plays 94% of his games
   (the 2024-26 rate for established rotation players).
3. That simple number is scaled so it matches the full weekly simulator on the
   days they both cover (the "anchor"), and given a spread taken from the full
   simulator too.
4. 10,000 seasons are played out. This week uses the full simulator's own
   2,000 results; later weeks use the simple model plus team-strength drift
   (rosters, injuries and form change as the season goes on).
5. Standings are ranked by record, then the league's tiebreak; the top seeds go
   into a playoff bracket that is played out the same way.

Outputs
    season_forecast.csv        one row per team: projected record, playoff/bye/title odds, seed odds
    season_weeks.csv           one row per team per remaining regular-season week
    season_meta.json           league format and model settings used
    history/season_model.json  anchor and spread carried between runs
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE, "history")
CACHE_DIR = os.path.join(BASE, "league_cache")

# ---- settings ---------------------------------------------------------------
SEASON_SIMS = 10_000
SEED = 2027
LINEUP_SLOTS = 10                 # PG SG SF PF C G F UTIL UTIL UTIL
HEALTHY_PLAY_RATE = 0.94          # share of scheduled games established rotation players played, 2024-26
OUT_NO_DATE_DAYS = 14             # OUT with no ESPN return date: assume back after this many days
SHORT_TERM_DAYS = 3               # Questionable / Day-to-day / Doubtful: affects the next few days only
DEFAULT_WEEK_CV = 0.08            # weekly score spread (sd / mean) until the simulator provides one
STRENGTH_SD_START = 0.03          # uncertainty in each team's true level next week
STRENGTH_SD_PER_WEEK = 0.015      # extra drift per week further out (trades, injuries, pickups)
LEAGUE_POSSESSIONS = 101.55       # NBA possessions per game, 2024-26
LEAGUE_TEAM_MISSES = 47.4         # missed field goals per team-game, 2024-26 (rebound chances)

# Overrides if ESPN's settings are missing or wrong. None = use ESPN's.
PLAYOFF_TEAMS_OVERRIDE = None
REGULAR_SEASON_WEEKS_OVERRIDE = None
TIEBREAK_OVERRIDE = None          # "points_for" or "h2h"

STATUS_SHORT_TERM = {"QUESTIONABLE": 0.5, "Q": 0.5, "DAY_TO_DAY": 0.5, "DTD": 0.5,
                     "DOUBTFUL": 0.25, "D": 0.25, "PROBABLE": 0.75, "P": 0.75}
STATUS_OUT = {"OUT", "O", "INJURY_RESERVE", "INJURED_RESERVE", "SUSPENSION", "SSPD"}
TEAM_FIXES = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ET = ZoneInfo("America/New_York")


def nid(v) -> str:
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _get(url, params, attempts=4):
    import time
    for a in range(1, attempts + 1):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * a)
    return None


# =============================================================================
# NBA SCHEDULE (rest of season, cached once a day)
# =============================================================================

def _events_to_rows(events) -> list[dict]:
    rows = []
    for ev in events or []:
        comp = (ev.get("competitions") or [{}])[0]
        teams = []
        for c in comp.get("competitors") or []:
            ab = ((c.get("team") or {}).get("abbreviation") or "").upper()
            teams.append(TEAM_FIXES.get(ab, ab))
        if len(teams) != 2 or not ev.get("date"):
            continue
        d = pd.Timestamp(ev["date"]).tz_convert(ET).date()
        rows.append({"game_id": str(ev.get("id")), "date": str(d), "team_a": teams[0], "team_b": teams[1]})
    return rows


def load_nba_schedule(start: date, end: date) -> pd.DataFrame:
    """Every NBA game from start to end (US Eastern dates). Cached in league_cache/ for a day."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "nba_season_schedule.csv")
    meta_path = os.path.join(CACHE_DIR, "nba_season_schedule.json")
    meta = read_json(meta_path, {})
    today = str(datetime.now(ET).date())
    if os.path.exists(path) and meta.get("fetched") == today and meta.get("end", "") >= str(end) \
            and meta.get("start", "9999") <= str(start):
        return pd.read_csv(path, dtype={"game_id": str})
    rows, d = [], start
    while d <= end:
        w_end = min(d + timedelta(days=6), end)
        data = _get(SCOREBOARD, {"dates": f"{d:%Y%m%d}-{w_end:%Y%m%d}", "limit": 500})
        if data is None:
            if os.path.exists(path):
                print("WARNING: NBA schedule download failed; using the last saved copy.")
                return pd.read_csv(path, dtype={"game_id": str})
            raise SystemExit("NBA schedule download failed.")
        got = _events_to_rows(data.get("events"))
        if not got:   # range query not supported or empty: ask day by day
            for i in range((w_end - d).days + 1):
                day = d + timedelta(days=i)
                got += _events_to_rows((_get(SCOREBOARD, {"dates": f"{day:%Y%m%d}"}) or {}).get("events"))
        rows += got
        d = w_end + timedelta(days=1)
    sched = pd.DataFrame(rows, columns=["game_id", "date", "team_a", "team_b"]).drop_duplicates("game_id")
    sched = sched[(sched["date"] >= str(start)) & (sched["date"] <= str(end))]
    sched.to_csv(path, index=False)
    with open(meta_path, "w") as fh:
        json.dump({"fetched": today, "start": str(start), "end": str(end)}, fh)
    print(f"NBA schedule downloaded: {len(sched)} games from {start} to {end}")
    return sched


# =============================================================================
# PLAYERS
# =============================================================================

def player_values() -> pd.DataFrame:
    """Rostered players: expected fantasy points per game played (healthy role)."""
    pr = pd.read_csv(os.path.join(BASE, "current_player_projections_v3.csv"))
    pr = pr[pr["fantasy_team_id"].notna()].copy()
    pr["fantasy_team_id"] = pr["fantasy_team_id"].map(nid)
    pr["player_id"] = pr["player_id"].map(nid)
    num = ["expected_minutes", "fga_rate", "fta_rate", "ast_rate", "tov_rate", "stl_rate", "blk_rate",
           "three_share", "two_pct", "three_pct", "ft_pct", "oreb_capture", "dreb_capture"]
    for c in num:
        pr[c] = pd.to_numeric(pr[c], errors="coerce").fillna(0.0)
    m = pr["expected_minutes"].clip(0, 48)
    e = LEAGUE_POSSESSIONS * m / 48.0
    fga = pr["fga_rate"] * e
    tpa = fga * pr["three_share"]
    tpm = tpa * pr["three_pct"]
    twm = (fga - tpa) * pr["two_pct"]
    ftm = pr["fta_rate"] * e * pr["ft_pct"]
    pts = 2 * twm + 3 * tpm + ftm
    reb = m / 48.0 * LEAGUE_TEAM_MISSES * (pr["oreb_capture"] + pr["dreb_capture"])
    fp = (pts + reb + 2 * pr["ast_rate"] * e + 4 * pr["stl_rate"] * e + 4 * pr["blk_rate"] * e
          - 2 * pr["tov_rate"] * e + 2 * (twm + tpm) - fga + tpm)
    espn = pd.to_numeric(pr.get("espn_projected_fp_avg"), errors="coerce")
    pr["fp_per_game"] = np.where(fp > 0, fp, espn.fillna(0.0))
    pr["nba_team"] = pr["nba_team"].astype(str).str.upper().map(lambda t: TEAM_FIXES.get(t, t))
    pr["status"] = pr["injury_status"].fillna("ACTIVE").astype(str).str.upper().str.strip()\
        .str.replace(" ", "_").str.replace("-", "_")
    ret_path = os.path.join(BASE, "weekly_matchup_v1_8_expected_returns.csv")
    rets = pd.read_csv(ret_path, dtype=str) if os.path.exists(ret_path) else pd.DataFrame(columns=["player_id", "return_date"])
    ret = {nid(r.player_id): r.return_date for r in rets.itertuples() if isinstance(r.return_date, str) and r.return_date}
    pr["return_date"] = pr["player_id"].map(ret)
    return pr[["player_id", "player_name", "nba_team", "fantasy_team_id", "fp_per_game", "status", "return_date"]]


def play_chance(status: str, return_date, day: date, today: date) -> float:
    if status in STATUS_OUT:
        if isinstance(return_date, str) and return_date:
            back = date.fromisoformat(return_date[:10])
        else:
            back = today + timedelta(days=OUT_NO_DATE_DAYS)
        return 0.0 if day < back else HEALTHY_PLAY_RATE
    if status in STATUS_SHORT_TERM and (day - today).days < SHORT_TERM_DAYS:
        return STATUS_SHORT_TERM[status]
    return HEALTHY_PLAY_RATE


def team_day_points(players: pd.DataFrame, sched: pd.DataFrame, days: list[date], today: date) -> pd.DataFrame:
    """Simple expected points per fantasy team per day: best 10 expected scores among players with a game."""
    playing = {}
    for r in sched.itertuples():
        playing.setdefault(r.date, set()).update([r.team_a, r.team_b])
    rows = []
    groups = {t: g for t, g in players.groupby("fantasy_team_id")}
    for d in days:
        teams_today = playing.get(str(d), set())
        for tid, g in groups.items():
            vals = sorted((float(r.fp_per_game) * play_chance(r.status, r.return_date, d, today)
                           for r in g.itertuples() if r.nba_team in teams_today), reverse=True)
            rows.append({"fantasy_team_id": tid, "date": str(d), "points": float(sum(vals[:LINEUP_SLOTS])),
                         "players_with_games": len(vals)})
    return pd.DataFrame(rows, columns=["fantasy_team_id", "date", "points", "players_with_games"])


# =============================================================================
# LEAGUE CALENDAR
# =============================================================================

def matchup_period_days(settings: dict, current_mp: int, week_start: date, periods: list[int]) -> dict[int, list[date]]:
    """Days of each matchup period. ESPN maps each matchup period to one or more weeks."""
    mp_map = {int(k): [int(x) for x in v] for k, v in (settings.get("matchup_periods") or {}).items()}
    cur_weeks = mp_map.get(current_mp, [current_mp])
    out = {}
    for mp in periods:
        weeks = mp_map.get(mp, [mp - current_mp + cur_weeks[0]])
        days = []
        for w in weeks:
            start = week_start + timedelta(days=7 * (w - cur_weeks[0]))
            days += [start + timedelta(days=i) for i in range(7)]
        out[mp] = days
    return out


def bracket_order(size: int) -> list[int]:
    """Standard fixed bracket, e.g. 8 -> [1, 8, 4, 5, 3, 6, 2, 7] (seeds, 1-based)."""
    order = [1, 2]
    while len(order) < size:
        n = len(order) * 2
        nxt = []
        for s in order:
            nxt += [s, n + 1 - s]
        order = nxt
    return order


# =============================================================================
# MAIN
# =============================================================================

def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    fm = pd.read_csv(os.path.join(BASE, "fantasy_matchups.csv"))
    sched_path = os.path.join(BASE, "league_schedule.csv")
    if not os.path.exists(sched_path):
        raise SystemExit("league_schedule.csv missing: run fetch_espn_matchups.py first.")
    fsched = pd.read_csv(sched_path)
    settings = read_json(os.path.join(BASE, "league_settings.json"), {})
    for c in ("home_id", "away_id"):
        fsched[c] = fsched[c].map(lambda v: "" if pd.isna(v) else nid(v))
    fsched["matchup_period"] = fsched["matchup_period"].astype(int)

    current_mp = int(fm["matchup_period"].iloc[0])
    first_sim_day = int(fm["first_sim_day_index"].iloc[0])
    week_start = date.fromisoformat(str(fm["week_start"].iloc[0]))
    if "current_date" in fm.columns and isinstance(fm["current_date"].iloc[0], str):
        today = date.fromisoformat(fm["current_date"].iloc[0])
    else:
        today = datetime.now(ET).date()

    teams = {nid(k): v.get("name", k) for k, v in (settings.get("teams") or {}).items()}
    for r in fm.itertuples():
        teams.setdefault(nid(r.fantasy_team_id), r.fantasy_team)
    tids = sorted(teams)
    tix = {t: i for i, t in enumerate(tids)}
    T = len(tids)

    reg = fsched[fsched["playoff_tier"].isna() | (fsched["playoff_tier"].astype(str).isin(["NONE", "nan"]))]
    R = int(REGULAR_SEASON_WEEKS_OVERRIDE or settings.get("matchup_period_count") or reg["matchup_period"].max())
    P = int(PLAYOFF_TEAMS_OVERRIDE or settings.get("playoff_team_count") or 4)
    P = max(2, min(P, T))
    L = int(settings.get("playoff_matchup_period_length") or 1)
    rule = str(settings.get("playoff_seeding_rule") or "")
    tiebreak = TIEBREAK_OVERRIDE or ("h2h" if "H2H" in rule.upper() else "points_for")
    divisions = {v.get("division_id") for v in (settings.get("teams") or {}).values()}
    print(f"League: {T} teams, {R} regular-season matchups, top {P} make the playoffs, "
          f"rounds of {L} matchup period(s), tiebreak {tiebreak}"
          + (f" (ESPN rule {rule})" if rule else " (ESPN gave no rule; using points for)"))
    if len(divisions - {None}) > 1:
        print("NOTE: the league has divisions; this forecast seeds by overall record only.")

    if current_mp > R:
        print("The regular season is over; the season forecast covers the regular season only.")
        pd.DataFrame().to_csv(os.path.join(BASE, "season_forecast.csv"), index=False)
        with open(os.path.join(BASE, "season_meta.json"), "w") as fh:
            json.dump({"status": "regular_season_over"}, fh)
        return

    # ---- record so far: decided matchups from the schedule -----------------------
    wins = np.zeros(T); losses = np.zeros(T); ties = np.zeros(T); pf = np.zeros(T)
    decided = reg[(reg["matchup_period"] <= R) & reg["winner"].isin(["HOME", "AWAY", "TIE"])]
    for r in decided.itertuples():
        if r.home_id not in tix or r.away_id not in tix:
            continue
        h, a = tix[r.home_id], tix[r.away_id]
        pf[h] += float(r.home_points or 0); pf[a] += float(r.away_points or 0)
        if r.winner == "HOME":
            wins[h] += 1; losses[a] += 1
        elif r.winner == "AWAY":
            wins[a] += 1; losses[h] += 1
        else:
            ties[h] += 1; ties[a] += 1
    cur_decided = bool((decided["matchup_period"] == current_mp).any())
    print(f"Record so far from {decided['matchup_period'].nunique()} decided matchup periods"
          + (" (this week already decided)" if cur_decided else ""))

    # ---- calendar and NBA schedule ------------------------------------------------
    n_rounds = int(np.ceil(np.log2(P)))
    playoff_mps = list(range(R + 1, R + 1 + n_rounds * L))
    future_mps = list(range(current_mp + 1, R + 1))
    mp_days = matchup_period_days(settings, current_mp, week_start, [current_mp] + future_mps + playoff_mps)
    last_day = max(max(v) for v in mp_days.values())
    rem_days_cur = [d for d in mp_days[current_mp][first_sim_day:]]
    first_needed = min([today] + rem_days_cur)
    nba = load_nba_schedule(first_needed, last_day)
    # This week's games: use exactly the schedule the weekly simulator used.
    wk_path = os.path.join(BASE, "nba_week_event_schedule.csv")
    if os.path.exists(wk_path):
        wk = pd.read_csv(wk_path, dtype={"game_id": str})
        wk["date"] = wk["date"].astype(str).str[:10]
        wk = wk[["game_id", "date", "team_a", "team_b"]]
        nba = pd.concat([nba[~nba["date"].isin(set(wk["date"]))], wk], ignore_index=True)
    players = player_values()
    all_days = sorted({d for mp in [current_mp] + future_mps + playoff_mps for d in mp_days[mp] if d >= today})
    daily = team_day_points(players, nba, all_days, today)
    daily["points"] = daily["points"].astype(float)
    by_day = {(r.fantasy_team_id, r.date): r.points for r in daily.itertuples()}

    def simple_mean(tid, days):
        return float(sum(by_day.get((tid, str(d)), 0.0) for d in days))

    # ---- anchor the simple model to the full weekly simulator ---------------------
    state_path = os.path.join(HISTORY_DIR, "season_model.json")
    state = read_json(state_path, {"anchor": 1.0, "week_cv": DEFAULT_WEEK_CV, "updates": 0})
    raw = os.path.join(BASE, "weekly_matchup_v1_8_team_rest_raw.csv")
    anchor_now = cv_now = None
    if os.path.exists(raw) and len(rem_days_cur) >= 3:
        rr = pd.read_csv(raw)
        rr["fantasy_team_id"] = rr["fantasy_team_id"].map(nid)
        simple = np.array([simple_mean(t, rem_days_cur) for t in rr["fantasy_team_id"]])
        if simple.sum() > 0 and rr["raw_rest_mean"].sum() > 0:
            anchor_now = float(rr["raw_rest_mean"].sum() / simple.sum())
            ok = rr["raw_rest_mean"] > 0
            cv_d = float(np.median(rr.loc[ok, "raw_rest_sd"] / rr.loc[ok, "raw_rest_mean"]))
            cv_now = cv_d * np.sqrt(len(rem_days_cur) / 7.0)       # variance grows with days
            corr = float(np.corrcoef(simple, rr["raw_rest_mean"])[0, 1]) if len(rr) > 2 else float("nan")
            print(f"Simple model vs full simulator ({len(rem_days_cur)} days left this week): "
                  f"ratio {anchor_now:.3f}, team correlation {corr:.2f}, weekly spread {cv_now:.3f}")
            a = 0.3 if state.get("updates", 0) else 1.0
            state["anchor"] = float(np.clip((1 - a) * state["anchor"] + a * anchor_now, 0.7, 1.3))
            state["week_cv"] = float(np.clip((1 - a) * state["week_cv"] + a * cv_now, 0.03, 0.2))
            state["updates"] = int(state.get("updates", 0)) + 1
            state["updated"] = str(today)
            with open(state_path, "w") as fh:
                json.dump(state, fh, indent=1)
    anchor, week_cv = float(state["anchor"]), float(state["week_cv"])
    cal = read_json(os.path.join(HISTORY_DIR, "calibration.json"), {})
    season_spread = float(np.clip(((cal.get("season") or {}).get("spread", 1.0) if (cal.get("season") or {}).get("active") else 1.0), 0.75, 1.5))
    print(f"Using anchor {anchor:.3f}, weekly spread {week_cv:.3f} x learned {season_spread:.2f}")

    # ---- means for every future week ----------------------------------------------
    sim_weeks = future_mps + playoff_mps
    W = len(sim_weeks)
    mean_tw = np.zeros((T, W))
    for j, mp in enumerate(sim_weeks):
        for t in tids:
            mean_tw[tix[t], j] = anchor * simple_mean(t, mp_days[mp])

    N = SEASON_SIMS
    # Strength drift: random walk per team, larger further out.
    steps = rng.normal(0, STRENGTH_SD_PER_WEEK, size=(N, T, W))
    steps[:, :, 0] = rng.normal(0, STRENGTH_SD_START, size=(N, T)) if W else steps[:, :, 0]
    strength = 1.0 + np.cumsum(steps, axis=2) if W else np.ones((N, T, 0))
    noise = rng.standard_normal((N, T, W))
    scores = np.maximum(mean_tw[None] * strength * (1 + week_cv * season_spread * noise), 0.0)

    # This week: the full simulator's own results.
    W_ = wins[None].repeat(N, 0); L_ = losses[None].repeat(N, 0); T_ = ties[None].repeat(N, 0)
    PF_ = pf[None].repeat(N, 0)
    samples_path = os.path.join(BASE, "weekly_matchup_v1_8_team_score_samples.csv")
    cur_week_scores = None
    if not cur_decided and os.path.exists(samples_path):
        smp = pd.read_csv(samples_path)
        smp.columns = [nid(c) for c in smp.columns]
        idx = rng.integers(0, len(smp), N)
        cur_week_scores = np.zeros((N, T))
        for t in tids:
            if t in smp.columns:
                cur_week_scores[:, tix[t]] = smp[t].to_numpy()[idx]

    def play_week(mp, sc):
        nonlocal W_, L_, T_, PF_
        rows = reg[reg["matchup_period"] == mp]
        for r in rows.itertuples():
            if r.home_id not in tix or r.away_id not in tix:
                continue
            h, a = tix[r.home_id], tix[r.away_id]
            sh, sa = sc[:, h], sc[:, a]
            W_[:, h] += sh > sa; L_[:, h] += sh < sa; T_[:, h] += sh == sa
            W_[:, a] += sa > sh; L_[:, a] += sa < sh; T_[:, a] += sh == sa
            PF_[:, h] += sh; PF_[:, a] += sa

    if cur_week_scores is not None:
        play_week(current_mp, cur_week_scores)
    week_rows = []
    for j, mp in enumerate(future_mps):
        sc = scores[:, :, j]
        play_week(mp, sc)
        for r in reg[reg["matchup_period"] == mp].itertuples():
            if r.home_id not in tix or r.away_id not in tix:
                continue
            for me, op in ((r.home_id, r.away_id), (r.away_id, r.home_id)):
                a, b = sc[:, tix[me]], sc[:, tix[op]]
                week_rows.append({
                    "matchup_period": mp, "start": str(mp_days[mp][0]), "end": str(mp_days[mp][-1]),
                    "fantasy_team_id": me, "fantasy_team": teams[me], "opponent_id": op, "opponent": teams[op],
                    "mean": round(float(a.mean()), 1), "sd": round(float(a.std()), 1),
                    "opponent_mean": round(float(b.mean()), 1),
                    "win_prob": round(float(np.mean(a > b) + 0.5 * np.mean(a == b)), 4),
                })
    if not len(reg[reg["matchup_period"].isin(future_mps)]) and future_mps:
        print("WARNING: ESPN returned no fantasy matchups for future weeks; the forecast covers this week only.")

    # ---- standings and seeds ------------------------------------------------------------
    games = W_ + L_ + T_
    wpct = np.where(games > 0, (W_ + 0.5 * T_) / np.maximum(games, 1), 0.0)
    key = np.round(wpct * 1e4) * 1e7 + PF_ + rng.random((N, T)) * 1e-3   # record, then points for
    order = np.argsort(-key, axis=1)                  # order[s, k] = team at rank k
    rank = np.empty_like(order)
    rank[np.arange(N)[:, None], order] = np.arange(T)[None]   # 0-based finish
    if tiebreak == "h2h":
        print("NOTE: head-to-head tiebreaks are approximated by points for.")

    # ---- playoffs ------------------------------------------------------------------------
    B = 1 << n_rounds
    slots = bracket_order(B)                           # seed per slot
    bracket = np.full((N, B), -1)
    for sidx, seed in enumerate(slots):
        if seed <= P:
            bracket[:, sidx] = order[:, seed - 1]
    byes = B - P
    alive = bracket.copy()
    finalists = None
    for rnd in range(n_rounds):
        if alive.shape[1] == 2:
            finalists = alive.copy()
        wks = [playoff_mps.index(m) + len(future_mps) for m in playoff_mps[rnd * L:(rnd + 1) * L]]
        round_scores = scores[:, :, wks].sum(axis=2) if wks else np.zeros((N, T))
        nxt = np.full((N, alive.shape[1] // 2), -1)
        for k in range(alive.shape[1] // 2):
            a, b = alive[:, 2 * k], alive[:, 2 * k + 1]
            sa = np.where(a >= 0, round_scores[np.arange(N), np.maximum(a, 0)], -np.inf)
            sb = np.where(b >= 0, round_scores[np.arange(N), np.maximum(b, 0)], -np.inf)
            coin = rng.random(N) < 0.5
            nxt[:, k] = np.where((sa > sb) | ((sa == sb) & coin), a, b)
        alive = nxt
    champ = alive[:, 0]

    rows = []
    for t in tids:
        i = tix[t]
        seeds = rank[:, i] + 1
        rows.append({
            "fantasy_team_id": t, "fantasy_team": teams[t],
            "wins": int(wins[i]), "losses": int(losses[i]), "ties": int(ties[i]), "points_for": round(float(pf[i]), 1),
            "proj_wins": round(float(W_[:, i].mean()), 2), "proj_losses": round(float(L_[:, i].mean()), 2),
            "proj_ties": round(float(T_[:, i].mean()), 2),
            "wins_p10": float(np.percentile(W_[:, i], 10)), "wins_p90": float(np.percentile(W_[:, i], 90)),
            "proj_points_for": round(float(PF_[:, i].mean()), 1),
            "playoff_prob": float(np.mean(seeds <= P)),
            "bye_prob": float(np.mean(seeds <= byes)) if byes else None,
            "final_prob": float(np.mean((finalists == i).any(axis=1))) if finalists is not None else float(np.mean(champ == i)),
            "title_prob": float(np.mean(champ == i)),
            "first_place_prob": float(np.mean(seeds == 1)),
            "last_place_prob": float(np.mean(seeds == T)),
            "mean_finish": round(float(seeds.mean()), 2),
            **{f"seed_{k}": float(np.mean(seeds == k)) for k in range(1, T + 1)},
            "future_mean_per_week": round(float(mean_tw[i, :len(future_mps)].mean()), 1) if future_mps else None,
        })
    out = pd.DataFrame(rows).sort_values(["mean_finish"]).reset_index(drop=True)
    out.to_csv(os.path.join(BASE, "season_forecast.csv"), index=False)
    pd.DataFrame(week_rows).to_csv(os.path.join(BASE, "season_weeks.csv"), index=False)
    meta = {
        "status": "ok", "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "sims": N,
        "teams": T, "regular_season_weeks": R, "current_matchup_period": current_mp,
        "weeks_left_after_this": len(future_mps), "playoff_teams": P, "byes": byes,
        "playoff_round_length": L, "tiebreak": tiebreak, "espn_seeding_rule": rule or None,
        "anchor": anchor, "anchor_this_run": anchor_now, "week_cv": week_cv, "season_spread": season_spread,
        "healthy_play_rate": HEALTHY_PLAY_RATE,
        "regular_season_end": str(mp_days[R][-1]) if R in mp_days else None,
        "this_week_from_simulator": cur_week_scores is not None,
    }
    with open(os.path.join(BASE, "season_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    print()
    print(f"SEASON FORECAST ({N:,} seasons; {len(future_mps)} regular-season weeks after this one)")
    show = out[["fantasy_team", "wins", "losses", "proj_wins", "proj_losses", "playoff_prob", "title_prob", "mean_finish"]]
    print(show.to_string(index=False, formatters={"playoff_prob": "{:.1%}".format, "title_prob": "{:.1%}".format}))


if __name__ == "__main__":
    main()
