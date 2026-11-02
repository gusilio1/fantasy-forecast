"""
model_check.py

The forecast's self-test. Runs after the simulations each hour and:

1. Records what was predicted (history/ folder, kept between runs):
     team_snapshots.csv    each team's projected weekly total and win chance, saved once per
                           day of the week (the last run before that day's games finished)
     player_snapshots.csv  each player's projected fantasy points for his next game
     season_snapshots.csv  the season model's projection for each future week
2. Once a week (or a day, for players) is over, scores those predictions against
   ESPN's actual results.
3. Learns two small corrections for the weekly simulation, used from the next run:
     level   are weekly totals projected too high or too low overall?
     spread  are the ranges too narrow or too wide (do 80% ranges hold the result 80% of the time)?
   Both are shrunk toward "no change" until several weeks have been checked, and capped.
   The season model's spread is learned the same way.

Output for the page: model_check_report.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(BASE, "history")
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Learning settings
SHRINK_TEAM_WEEKS = 36        # pseudo team-weeks of "no change" (3 weeks of a 12-team league)
MIN_TEAM_WEEKS = 24        # corrections switch on after this many checked team-weeks (2 weeks of a 12-team league)
MAX_LEVEL_CHANGE = 0.08       # level correction capped at +/- 8%
SPREAD_RANGE = (0.8, 1.4)
LEARN_FROM_DAYS = 6           # snapshots taken with at least 1 day left (Mon..Sat)


def nid(v) -> str:
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def read(path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kw)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def upsert(path: str, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    old = read(path, dtype=str)
    if len(new) == 0:
        return old
    new = new.copy()
    if len(old):
        new_keys = new[keys].astype(str).agg("|".join, axis=1)
        old = old[~old[keys].astype(str).agg("|".join, axis=1).isin(set(new_keys))]
        out = pd.concat([old, new.astype(str)], ignore_index=True)
    else:
        out = new.astype(str)
    out.to_csv(path, index=False)
    return out


def f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# =============================================================================
# 1. RECORD THIS RUN'S PREDICTIONS
# =============================================================================

def record_snapshots(fm: pd.DataFrame, run_utc: str, corr_used: dict):
    mp = int(fm["matchup_period"].iloc[0])
    snap_day = int(fm["first_sim_day_index"].iloc[0])
    if snap_day >= 7:
        return
    samples = read(os.path.join(BASE, "weekly_matchup_v1_8_team_score_samples.csv"))
    raw = read(os.path.join(BASE, "weekly_matchup_v1_8_team_rest_raw.csv"))
    if samples.empty or raw.empty:
        print("No simulation results to record.")
        return
    samples.columns = [nid(c) for c in samples.columns]
    raw["fantasy_team_id"] = raw["fantasy_team_id"].map(nid)
    raw = raw.set_index("fantasy_team_id")
    rows = []
    for r in fm.itertuples():
        t, o = nid(r.fantasy_team_id), nid(r.opponent_team_id) if pd.notna(r.opponent_team_id) else ""
        if t not in samples.columns:
            continue
        s = samples[t].to_numpy(float)
        so = samples[o].to_numpy(float) if o in samples.columns else None
        rows.append({
            "matchup_period": mp, "snap_day": snap_day, "run_utc": run_utc,
            "team_id": t, "team": str(r.fantasy_team).strip(), "opponent_id": o,
            "completed_points": float(r.completed_points),
            "raw_rest_mean": float(raw.loc[t, "raw_rest_mean"]) if t in raw.index else np.nan,
            "raw_rest_sd": float(raw.loc[t, "raw_rest_sd"]) if t in raw.index else np.nan,
            "final_mean": float(s.mean()), "final_p10": float(np.percentile(s, 10)),
            "final_p90": float(np.percentile(s, 90)),
            "win_prob": float(np.mean(s > so) + 0.5 * np.mean(s == so)) if so is not None else np.nan,
            "level_used": corr_used.get("bias", 0.0), "spread_used": corr_used.get("spread", 1.0),
        })
    upsert(os.path.join(HIST, "team_snapshots.csv"), pd.DataFrame(rows), ["matchup_period", "snap_day", "team_id"])

    # Players: projection for the next day still to be played (the first simulated day).
    meta = read_json(os.path.join(BASE, "weekly_matchup_v1_8_run_meta.json"), {})
    day = DAY_NAMES[snap_day]
    date = (meta.get("week_dates") or {}).get(day)
    pdays = read(os.path.join(BASE, "weekly_matchup_v1_8_player_day_projections.csv"))
    if date and not pdays.empty:
        p = pdays[(pdays["day"] == day) & (pdays["fp_if_plays"].fillna(0) > 0)].copy()
        pool = read(os.path.join(BASE, "player_pool.csv"))
        espn = {nid(a): b for a, b in zip(pool.get("player_id", []), pool.get("projected_fp_avg", []))}
        avg = {nid(a): b for a, b in zip(pool.get("player_id", []), pool.get("actual_fp_avg", []))}
        p["player_id"] = p["player_id"].map(nid)
        out = pd.DataFrame({
            "date": date, "matchup_period": mp, "player_id": p["player_id"], "player": p["player_name"],
            "fantasy_team": p["fantasy_team"].fillna(""), "fp_if_plays": p["fp_if_plays"].round(2),
            "play_prob": p["play_prob"].round(3), "espn_projection": p["player_id"].map(espn),
            "season_average": p["player_id"].map(avg), "run_utc": run_utc,
        })
        upsert(os.path.join(HIST, "player_snapshots.csv"), out, ["date", "player_id"])

    # Season model: projections for each future week (latest run during this week).
    weeks = read(os.path.join(BASE, "season_weeks.csv"))
    if not weeks.empty:
        out = pd.DataFrame({
            "snap_matchup_period": mp, "target_matchup_period": weeks["matchup_period"],
            "horizon": weeks["matchup_period"] - mp, "team_id": weeks["fantasy_team_id"].map(nid),
            "mean": weeks["mean"], "sd": weeks["sd"], "win_prob": weeks["win_prob"],
            "opponent_id": weeks["opponent_id"].map(nid), "run_utc": run_utc,
        })
        upsert(os.path.join(HIST, "season_snapshots.csv"), out,
               ["snap_matchup_period", "target_matchup_period", "team_id"])
    sf = read(os.path.join(BASE, "season_forecast.csv"))
    if not sf.empty:
        out = sf[["fantasy_team_id", "fantasy_team", "playoff_prob", "title_prob", "proj_wins", "mean_finish"]].copy()
        out.insert(0, "matchup_period", mp)
        out.insert(1, "snap_day", snap_day)
        upsert(os.path.join(HIST, "season_odds_log.csv"), out, ["matchup_period", "snap_day", "fantasy_team_id"])


# =============================================================================
# 2. TRUTH
# =============================================================================

def actual_week_totals(fm: pd.DataFrame) -> pd.DataFrame:
    """ESPN actual weekly totals for every finished week."""
    td = read(os.path.join(BASE, "league_team_days.csv"))
    if td.empty:
        return pd.DataFrame(columns=["matchup_period", "team_id", "actual"])
    cur = int(fm["matchup_period"].iloc[0])
    done = int(fm["first_sim_day_index"].iloc[0]) >= 7
    td["team_id"] = td["fantasy_team_id"].map(nid)
    td["matchup_period"] = pd.to_numeric(td["matchup_period"], errors="coerce")
    td = td[td["matchup_period"].notna()]
    td["matchup_period"] = td["matchup_period"].astype(int)
    g = td.groupby(["matchup_period", "team_id"])["espn_points"].sum().reset_index(name="actual")
    g = g[(g["matchup_period"] < cur) | ((g["matchup_period"] == cur) & done)]
    return g


# =============================================================================
# 3. EVALUATE + LEARN
# =============================================================================

def evaluate(fm: pd.DataFrame) -> dict:
    report = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    truth = actual_week_totals(fm)
    snaps = read(os.path.join(HIST, "team_snapshots.csv"))
    cal = read_json(os.path.join(HIST, "calibration.json"), {})
    weekly_cal = {"active": False, "bias": 0.0, "spread": 1.0, "team_weeks": 0,
                  "reason": "no finished weeks checked yet"}

    if not snaps.empty and not truth.empty:
        for c in ["matchup_period", "snap_day", "completed_points", "raw_rest_mean", "raw_rest_sd", "final_mean",
                  "final_p10", "final_p90", "win_prob"]:
            snaps[c] = pd.to_numeric(snaps[c], errors="coerce")
        truth["matchup_period"] = truth["matchup_period"].astype(int)
        snaps["matchup_period"] = snaps["matchup_period"].astype(int)
        snaps["team_id"] = snaps["team_id"].map(nid)
        snaps["opponent_id"] = snaps["opponent_id"].map(lambda v: "" if pd.isna(v) else nid(v))
        e = snaps.merge(truth, on=["matchup_period", "team_id"], how="inner")
        opp = truth.rename(columns={"team_id": "opponent_id", "actual": "opp_actual"})
        e = e.merge(opp, on=["matchup_period", "opponent_id"], how="left")
        e["error"] = e["actual"] - e["final_mean"]
        e["in80"] = (e["actual"] >= e["final_p10"]) & (e["actual"] <= e["final_p90"])
        e["outcome"] = np.where(e["actual"] > e["opp_actual"], 1.0, np.where(e["actual"] < e["opp_actual"], 0.0, 0.5))
        e["brier"] = (e["win_prob"] - e["outcome"]) ** 2
        e["fav_right"] = np.where(e["win_prob"] == 0.5, np.nan,
                                  ((e["win_prob"] > 0.5) == (e["outcome"] == 1.0)).astype(float))
        e["actual_rest"] = e["actual"] - e["completed_points"]
        report["weeks_checked"] = int(e["matchup_period"].nunique())
        report["team_weeks_checked"] = int(e[["matchup_period", "team_id"]].drop_duplicates().shape[0])

        by_day = []
        for d, g in e.groupby("snap_day"):
            gm = g[g["opp_actual"].notna()]
            by_day.append({
                "snap_day": int(d), "day": DAY_NAMES[int(d)], "team_weeks": int(len(g)),
                "brier": f(gm["brier"].mean()), "favourite_won": f(gm["fav_right"].mean()),
                "mean_abs_error": f(g["error"].abs().mean()), "mean_error": f(g["error"].mean()),
                "in_80_range": f(g["in80"].mean()),
            })
        report["weekly_by_day"] = by_day

        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
        e["bucket"] = pd.cut(e["win_prob"], bins, right=False)
        rel = []
        for b, g in e[e["opp_actual"].notna()].groupby("bucket", observed=True):
            rel.append({"range": f"{int(b.left * 100)}–{min(int(round(b.right * 100)), 100)}%",
                        "n": int(len(g)), "predicted": f(g["win_prob"].mean()), "actual": f(g["outcome"].mean())})
        report["reliability"] = rel

        recent = []
        for (mp, t), g in e.sort_values("snap_day").groupby(["matchup_period", "team_id"]):
            first = g.iloc[0]
            recent.append({"matchup_period": int(mp), "team": first["team"], "monday_projection": f(first["final_mean"]),
                           "monday_win_prob": f(first["win_prob"]), "actual": f(first["actual"]),
                           "won": f(first["outcome"])})
        report["team_weeks"] = sorted(recent, key=lambda r: (-r["matchup_period"], r["team"]))[:48]

        # ---- learning (from raw, uncorrected simulations) -----------------------------
        L = e[(e["snap_day"] < LEARN_FROM_DAYS) & (e["raw_rest_mean"] > 0) & (e["raw_rest_sd"] > 0)]
        n = int(L[["matchup_period", "team_id"]].drop_duplicates().shape[0])
        if n:
            ratio = float(L["actual_rest"].sum() / L["raw_rest_mean"].sum())
            w = n / (n + SHRINK_TEAM_WEEKS)
            if not np.isfinite(ratio) or ratio <= 0:
                ratio, w = 1.0, 0.0
            bias = float(np.clip((ratio - 1) * w, -MAX_LEVEL_CHANGE, MAX_LEVEL_CHANGE))
            z = (L["actual_rest"] - L["raw_rest_mean"] * ratio) / (L["raw_rest_sd"] * ratio)
            z = z[np.isfinite(z)]
            spread_raw = float(np.sqrt(np.mean(z ** 2))) if len(z) else 1.0
            spread = float(np.clip(1 + (spread_raw - 1) * w, *SPREAD_RANGE))
            weekly_cal = {
                "active": n >= MIN_TEAM_WEEKS, "bias": round(bias, 4), "spread": round(spread, 4),
                "team_weeks": n, "raw_level_ratio": round(ratio, 4), "raw_spread": round(spread_raw, 4),
                "weight": round(w, 3),
                "reason": "" if n >= MIN_TEAM_WEEKS else f"needs {MIN_TEAM_WEEKS} checked team-weeks (has {n})",
            }

    # ---- players -----------------------------------------------------------------------
    ps = read(os.path.join(HIST, "player_snapshots.csv"), dtype={"date": str})
    starts = read(os.path.join(BASE, "league_daily_starts.csv"))
    if not ps.empty and not starts.empty:
        starts["player_id"] = starts["player_id"].map(nid)
        played = starts[starts["played"].astype(str).str.lower() == "true"][["date", "player_id", "points"]]
        played = played.drop_duplicates(["date", "player_id"])
        ps["player_id"] = ps["player_id"].map(nid)
        pe = ps.merge(played, on=["date", "player_id"], how="inner")
        for c in ["fp_if_plays", "espn_projection", "season_average", "points"]:
            pe[c] = pd.to_numeric(pe[c], errors="coerce")
        if len(pe):
            def stats(col):
                d = (pe["points"] - pe[col]).dropna()
                return {"mae": f(d.abs().mean()), "bias": f(-d.mean()), "n": int(len(d))}
            report["players"] = {
                "games": int(len(pe)), "days": int(pe["date"].nunique()),
                "ours": stats("fp_if_plays"), "espn": stats("espn_projection"), "season_average": stats("season_average"),
            }
            pe["matchup_period"] = pd.to_numeric(pe["matchup_period"], errors="coerce")
            report["players_by_week"] = [
                {"matchup_period": int(mp), "games": int(len(g)),
                 "ours_mae": f((g["points"] - g["fp_if_plays"]).abs().mean()),
                 "espn_mae": f((g["points"] - g["espn_projection"]).abs().mean())}
                for mp, g in pe.groupby("matchup_period")]
            pe["miss"] = pe["points"] - pe["fp_if_plays"]
            top = pe.reindex(pe["miss"].abs().sort_values(ascending=False).index).head(8)
            report["player_biggest_misses"] = [
                {"date": r.date, "player": r.player, "projected": f(r.fp_if_plays), "actual": f(r.points)}
                for r in top.itertuples()]

    # ---- season model ------------------------------------------------------------------
    ss = read(os.path.join(HIST, "season_snapshots.csv"))
    season_cal = {"active": False, "spread": 1.0, "reason": "no forecast weeks finished yet"}
    if not ss.empty and not truth.empty:
        for c in ["target_matchup_period", "horizon", "mean", "sd", "win_prob"]:
            ss[c] = pd.to_numeric(ss[c], errors="coerce")
        ss["target_matchup_period"] = ss["target_matchup_period"].astype(int)
        ss["team_id"] = ss["team_id"].map(nid)
        ss["opponent_id"] = ss["opponent_id"].map(lambda v: "" if pd.isna(v) else nid(v))
        se = ss.merge(truth.rename(columns={"matchup_period": "target_matchup_period"}),
                      on=["target_matchup_period", "team_id"], how="inner")
        opp = truth.rename(columns={"matchup_period": "target_matchup_period", "team_id": "opponent_id",
                                    "actual": "opp_actual"})
        se = se.merge(opp, on=["target_matchup_period", "opponent_id"], how="left")
        if len(se):
            se["error"] = se["actual"] - se["mean"]
            se["z"] = se["error"] / se["sd"].replace(0, np.nan)
            se["outcome"] = np.where(se["actual"] > se["opp_actual"], 1.0, np.where(se["actual"] < se["opp_actual"], 0.0, 0.5))
            se["brier"] = (se["win_prob"] - se["outcome"]) ** 2
            report["season_by_horizon"] = [
                {"weeks_ahead": int(h), "team_weeks": int(len(g)), "mean_abs_error": f(g["error"].abs().mean()),
                 "mean_error": f(g["error"].mean()), "brier": f(g["brier"].mean())}
                for h, g in se.groupby("horizon") if h <= 6]
            n = len(se[se["horizon"] == 1])
            zz = se.loc[se["horizon"] <= 3, "z"].dropna()
            if len(zz):
                w = len(zz) / (len(zz) + SHRINK_TEAM_WEEKS * 2)
                sp = float(np.clip(1 + (np.sqrt(np.mean(zz ** 2)) - 1) * w, *SPREAD_RANGE))
                season_cal = {"active": n >= MIN_TEAM_WEEKS, "spread": round(sp, 4), "team_weeks": int(len(zz)),
                              "reason": "" if n >= MIN_TEAM_WEEKS else "needs more finished weeks"}

    new_cal = {"weekly": weekly_cal, "season": season_cal, "updated_utc": report["generated_utc"]}
    with open(os.path.join(HIST, "calibration.json"), "w") as fh:
        json.dump(new_cal, fh, indent=1)
    log = pd.DataFrame([{"date_utc": report["generated_utc"][:10], "level": weekly_cal["bias"],
                         "spread": weekly_cal["spread"], "active": weekly_cal["active"],
                         "team_weeks": weekly_cal["team_weeks"], "season_spread": season_cal["spread"]}])
    upsert(os.path.join(HIST, "calibration_log.csv"), log, ["date_utc"])
    report["calibration"] = new_cal
    report["calibration_previous"] = cal
    report["history_since"] = None
    if not snaps.empty:
        report["history_since"] = str(pd.to_datetime(snaps["run_utc"], errors="coerce").min())[:10]
    return report


def main():
    os.makedirs(HIST, exist_ok=True)
    fm = read(os.path.join(BASE, "fantasy_matchups.csv"))
    if fm.empty:
        raise SystemExit("fantasy_matchups.csv missing.")
    run_utc = pd.Timestamp.now(tz="UTC").isoformat()
    cal = read_json(os.path.join(HIST, "calibration.json"), {})
    w = cal.get("weekly") or {}
    used = {"bias": w.get("bias", 0.0) if w.get("active") else 0.0,
            "spread": w.get("spread", 1.0) if w.get("active") else 1.0}
    record_snapshots(fm, run_utc, used)
    report = evaluate(fm)
    with open(os.path.join(BASE, "model_check_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    print("SELF-TEST")
    print(f"Weeks checked: {report.get('weeks_checked', 0)}; team-weeks: {report.get('team_weeks_checked', 0)}")
    for r in report.get("weekly_by_day", []):
        print(f"  from {r['day']}: Brier {r['brier'] if r['brier'] is None else round(r['brier'], 3)}, "
              f"favourite won {'-' if r['favourite_won'] is None else format(r['favourite_won'], '.0%')}, mean abs error {r['mean_abs_error']:.1f}, "
              f"in 80% range {r['in_80_range']:.0%}")
    if report.get("players"):
        p = report["players"]
        print(f"  Player games checked: {p['games']}. Mean abs error: ours {p['ours']['mae']:.2f}, "
              f"ESPN projection {p['espn']['mae'] if p['espn']['mae'] is not None else float('nan'):.2f}")
    wc = report["calibration"]["weekly"]
    print(f"  Corrections for next run: level {wc['bias']:+.1%}, spread x{wc['spread']:.2f} "
          f"({'on' if wc['active'] else 'off: ' + wc.get('reason', '')})")


if __name__ == "__main__":
    main()
