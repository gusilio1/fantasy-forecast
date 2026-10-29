"""
matchup_context.py

Real-life matchup adjustments for the weekly simulator, shared by:
    backtest_matchup_context.py      (validation on held-out games)
    weekly_matchup_simulator_v3_1.py (live use)

Two adjustments, both validated leak-free on held-out 2025-26 games:

PACE     Each game's expected possessions = league average + each team's recent
         pace above/below average (shrunk with k pseudo-games), instead of the
         same 101.55 possessions for every game.

DEFENSE  For each stat, how much more/less opponents produce against a team than
         those opponents' own rates predict (a residual, so facing weak offenses
         does not look like good defense), shrunk toward 1.0. Applied to the
         offensive players' rates in that game.

Every rate and factor uses games on strictly earlier calendar days.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

TEAM_FIX = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
FIXED_POSSESSIONS = 101.55          # simulator_core_v1 DEFAULT_GAME_POSSESSIONS_MEAN
PLAYER_HALF_LIFE = 20               # games of memory for each player's own rates
MIN_PLAYER_GAMES = 5

COUNT_STATS = {
    "FGA": "fga", "FTA": "fta", "AST": "assists", "TOV": "turnovers",
    "STL": "steals", "BLK": "blocks", "OREB": "offensive_rebounds", "DREB": "defensive_rebounds",
}
RATIO_STATS = {
    "3PA_SHARE": ("three_pa", "fga"),
    "2P_PCT": ("two_pm", "two_pa"),
    "3P_PCT": ("three_pm", "three_pa"),
    "FT_PCT": ("ftm", "fta"),
}
STATS = list(COUNT_STATS) + list(RATIO_STATS)
ACTUAL_FOR = {**COUNT_STATS, "3PA_SHARE": "three_pa", "2P_PCT": "two_pm", "3P_PCT": "three_pm", "FT_PCT": "ftm"}

# Simulator PlayerProjection field each factor scales. Rebounds are NOT scaled in
# the simulator: its rebounds come from missed-shot opportunities, which already
# respond to the shooting/volume factors, so scaling capture rates too would
# double count.
SIM_FIELD = {
    "FGA": "fga_rate", "FTA": "fta_rate", "AST": "ast_rate", "TOV": "tov_rate",
    "STL": "stl_rate", "BLK": "blk_rate", "3PA_SHARE": "three_share",
    "2P_PCT": "two_pct", "3P_PCT": "three_pct", "FT_PCT": "ft_pct",
}

NUM_COLS = ["minutes", "fga", "fgm", "three_pa", "three_pm", "fta", "ftm", "assists", "turnovers",
            "steals", "blocks", "offensive_rebounds", "defensive_rebounds", "rebounds", "points",
            "fantasy_points"]


def fantasy_points(pts, reb, ast, stl, blk, tov, fgm, fga, tpm):
    return pts + reb + 2 * ast + 4 * stl + 4 * blk - 2 * tov + 2 * fgm - fga + tpm


# =============================================================================
# DATA PREPARATION
# =============================================================================

def add_box_possessions(df: pd.DataFrame) -> pd.DataFrame:
    """Game possessions from box scores: mean over both teams of FGA - OREB + TOV + 0.44*FTA."""
    t = df.groupby(["game_id", "team"]).agg(
        fga=("fga", "sum"), oreb=("offensive_rebounds", "sum"),
        tov=("turnovers", "sum"), fta=("fta", "sum")).reset_index()
    t["est"] = t["fga"] - t["oreb"] + t["tov"] + 0.44 * t["fta"]
    g = t.groupby("game_id").agg(est=("est", "mean"), n=("team", "nunique"))
    g = g[g["n"] == 2]["est"]
    out = df.copy()
    out["game_possessions"] = out["game_id"].map(g)
    return out[out["game_possessions"].notna()]


def load_history(model_history_path: str, current_path: str | None = None,
                 cutoff_day=None) -> pd.DataFrame:
    """
    Player-game rows from player_game_model_history.csv, plus this season's
    player_game_history_2027_current.csv when given (possessions computed from
    box scores). Games on/after cutoff_day are dropped.
    """
    m = pd.read_csv(model_history_path, dtype={"game_id": str, "player_id": str}, low_memory=False)
    m["team"] = m["team"].replace(TEAM_FIX)
    m["opponent"] = m["opponent"].replace(TEAM_FIX)
    frames = [m]
    if current_path and os.path.exists(current_path):
        c = pd.read_csv(current_path, dtype={"game_id": str, "player_id": str})
        c = c.rename(columns={"nba_team": "team"})
        c["team"] = c["team"].replace(TEAM_FIX)
        for col in NUM_COLS:
            c[col] = pd.to_numeric(c.get(col), errors="coerce").fillna(0.0)
        teams = c.groupby("game_id")["team"].unique()
        opp = {}
        for gid, ts in teams.items():
            if len(ts) == 2:
                opp[(gid, ts[0])], opp[(gid, ts[1])] = ts[1], ts[0]
        c["opponent"] = [opp.get((g, t)) for g, t in zip(c["game_id"], c["team"])]
        c = c[c["opponent"].notna()]
        c = add_box_possessions(c)
        c["history_season"] = 2027
        frames.append(c)
    keep = ["game_id", "date", "player_id", "team", "opponent", "history_season", "game_possessions"] + NUM_COLS
    m = pd.concat([f[[x for x in keep if x in f.columns]] for f in frames], ignore_index=True)
    m["day"] = pd.to_datetime(m["date"]).dt.normalize()
    if cutoff_day is not None:
        m = m[m["day"] < pd.Timestamp(cutoff_day).normalize()]
    for col in NUM_COLS + ["game_possessions"]:
        m[col] = pd.to_numeric(m[col], errors="coerce").fillna(0.0)
    m["two_pa"] = (m["fga"] - m["three_pa"]).clip(lower=0)
    m["two_pm"] = (m["fgm"] - m["three_pm"]).clip(lower=0)
    m["exposure"] = m["game_possessions"] * m["minutes"] / 48.0
    return m.reset_index(drop=True)


def add_player_rates(m: pd.DataFrame) -> pd.DataFrame:
    """Each player's pre-game per-exposure rates (strictly earlier games)."""
    m = m.sort_values(["player_id", "day", "game_id"]).reset_index(drop=True)
    g = m.groupby("player_id", sort=False)
    alpha = 1 - 0.5 ** (1 / PLAYER_HALF_LIFE)

    def ewm_prev(col):
        return g[col].transform(lambda s: s.ewm(alpha=alpha, adjust=True).mean().shift(1))

    m["n_prev"] = g.cumcount()
    ew_exp = ewm_prev("exposure")
    for s, col in COUNT_STATS.items():
        m[f"rate_{s}"] = ewm_prev(col) / ew_exp
    for s, (num, den) in RATIO_STATS.items():
        d = ewm_prev(den)
        m[f"rate_{s}"] = np.where(d > 0, ewm_prev(num) / d, np.nan)
    m["rate_3P_PCT"] = m["rate_3P_PCT"].fillna(0.36)
    m["rate_2P_PCT"] = m["rate_2P_PCT"].fillna(0.52)
    m["rate_FT_PCT"] = m["rate_FT_PCT"].fillna(0.75)
    m["rate_3PA_SHARE"] = m["rate_3PA_SHARE"].fillna(0.0)
    m["has_rate"] = (m["n_prev"] >= MIN_PLAYER_GAMES) & np.isfinite(m["rate_FGA"])
    for s in COUNT_STATS:
        m[f"exp_{s}"] = m[f"rate_{s}"] * m["exposure"]
    m["exp_3PA_SHARE"] = m["rate_3PA_SHARE"] * m["fga"]
    m["exp_2P_PCT"] = m["rate_2P_PCT"] * m["two_pa"]
    m["exp_3P_PCT"] = m["rate_3P_PCT"] * m["three_pa"]
    m["exp_FT_PCT"] = m["rate_FT_PCT"] * m["fta"]
    return m


def team_game_table(m: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: pace, and what the OPPONENT produced vs expected (this team's defense)."""
    ok = m[m["has_rate"]]
    offense = ok.groupby(["game_id", "team"]).agg(
        **{f"A_{s}": (ACTUAL_FOR[s], "sum") for s in STATS},
        **{f"E_{s}": (f"exp_{s}", "sum") for s in STATS},
    ).reset_index()
    tg = m.groupby(["game_id", "team"]).agg(day=("day", "first"), opponent=("opponent", "first"),
                                             poss=("game_possessions", "first"),
                                             team_min=("minutes", "sum")).reset_index()
    tg = tg.merge(offense, on=["game_id", "team"], how="left").fillna(0.0)
    tg["pace48"] = tg["poss"] * 240.0 / tg["team_min"].clip(lower=240.0)
    against = tg[["game_id", "team"] + [f"A_{s}" for s in STATS] + [f"E_{s}" for s in STATS]].copy()
    against = against.rename(columns={"team": "opponent"})
    tg = tg.merge(against, on=["game_id", "opponent"], suffixes=("", "_vs"), how="left").fillna(0.0)
    return tg.sort_values(["day", "game_id", "team"]).reset_index(drop=True)


def _blank_state():
    return {"pace_dev": 0.0, "pace_w": 0.0, "def_w": 0.0,
            **{f"dA_{s}": 0.0 for s in STATS}, **{f"dE_{s}": 0.0 for s in STATS}}


def team_states(tg: pd.DataFrame, half_life: float):
    """
    Walk games day by day. Returns:
        pregame  DataFrame: each (game_id, team)'s state from strictly earlier days
        final    dict team -> state after every game in tg
        league   league mean possessions after every game in tg
    """
    keep = 0.5 ** (1 / half_life)
    state, rows = {}, []
    league_sum, league_n = 0.0, 0.0
    for day, dg in tg.groupby("day", sort=True):
        L = league_sum / league_n if league_n else FIXED_POSSESSIONS
        for r in dg.itertuples(index=False):
            rec = {"game_id": r.game_id, "team": r.team, "L": L}
            rec.update(state.get(r.team, _blank_state()))
            rows.append(rec)
        for r in dg.itertuples(index=False):
            st = state.setdefault(r.team, _blank_state())
            st["pace_dev"] = keep * st["pace_dev"] + (r.pace48 - L)
            st["pace_w"] = keep * st["pace_w"] + 1.0
            st["def_w"] = keep * st["def_w"] + 1.0
            for s in STATS:
                st[f"dA_{s}"] = keep * st[f"dA_{s}"] + getattr(r, f"A_{s}_vs")
                st[f"dE_{s}"] = keep * st[f"dE_{s}"] + getattr(r, f"E_{s}_vs")
            league_sum += r.poss / 2.0
            league_n += 0.5
    league = league_sum / league_n if league_n else FIXED_POSSESSIONS
    return pd.DataFrame(rows), state, league


def shrink_factor(dA, dE, w, k):
    """Defense factor toward 1.0 with k pseudo-games of the team's average expected volume."""
    dA, dE, w = (np.asarray(x, float) for x in (dA, dE, w))
    if k is None or not np.isfinite(k):
        return np.ones_like(dA)
    ebar = np.where(w > 0, dE / np.maximum(w, 1e-9), 0.0)
    den = dE + k * ebar
    return np.where(den > 0, (dA + k * ebar) / np.maximum(den, 1e-9), 1.0)


def pace_term(dev, w, k):
    if k is None or not np.isfinite(k):
        return 0.0
    return np.asarray(dev, float) / (np.asarray(w, float) + k)


# =============================================================================
# EXPECTED BOX SCORE (used by the backtest)
# =============================================================================

def predict(rates: dict, minutes, poss_hat, fac: dict):
    """Expected stats and FP for player-games given rates, minutes, possessions and factors."""
    exp_hat = np.asarray(poss_hat, float) * np.asarray(minutes, float) / 48.0
    out = {s: np.asarray(rates[s], float) * fac[s] * exp_hat for s in COUNT_STATS}
    share = np.clip(np.asarray(rates["3PA_SHARE"], float) * fac["3PA_SHARE"], 0, 1)
    p2 = np.clip(np.asarray(rates["2P_PCT"], float) * fac["2P_PCT"], 0, 1)
    p3 = np.clip(np.asarray(rates["3P_PCT"], float) * fac["3P_PCT"], 0, 1)
    pft = np.clip(np.asarray(rates["FT_PCT"], float) * fac["FT_PCT"], 0, 1)
    tpa = out["FGA"] * share
    twopa = out["FGA"] - tpa
    tpm, twopm, ftm = tpa * p3, twopa * p2, out["FTA"] * pft
    fgm = tpm + twopm
    pts = 2 * twopm + 3 * tpm + ftm
    reb = out["OREB"] + out["DREB"]
    out.update({"3PA": tpa, "3PM": tpm, "FGM": fgm, "FTM": ftm, "PTS": pts, "REB": reb})
    out["FP"] = fantasy_points(pts, reb, out["AST"], out["STL"], out["BLK"], out["TOV"], fgm, out["FGA"], tpm)
    return out


# =============================================================================
# LIVE CONTEXT FOR THE SIMULATOR
# =============================================================================

class LiveMatchupContext:
    """Pace and defense factors for every team as of the first simulated day."""

    def __init__(self, settings: dict, final_states: dict, league_poss: float):
        self.settings = settings
        self.league = float(league_poss)
        self.final = final_states            # half_life -> {team: state}

    @classmethod
    def build(cls, settings_path: str, model_history_path: str, current_path: str, cutoff_day):
        with open(settings_path, encoding="utf-8") as fh:
            settings = json.load(fh)
        m = add_player_rates(load_history(model_history_path, current_path, cutoff_day))
        tg = team_game_table(m)
        hls = {float(settings["pace"]["half_life"])} | {float(v["half_life"]) for v in settings["defense"].values()}
        finals, league = {}, FIXED_POSSESSIONS
        for hl in hls:
            _, final, league = team_states(tg, hl)
            finals[hl] = final
        return cls(settings, finals, league)

    def _state(self, team, half_life):
        return self.final.get(float(half_life), {}).get(str(team).upper(), _blank_state())

    def possessions(self, team_a: str, team_b: str) -> float:
        p = self.settings["pace"]
        k = p["k"] if p["k"] is not None else None
        sa, sb = self._state(team_a, p["half_life"]), self._state(team_b, p["half_life"])
        return float(self.league + pace_term(sa["pace_dev"], sa["pace_w"], k)
                     + pace_term(sb["pace_dev"], sb["pace_w"], k))

    def defense_factors(self, defense_team: str) -> dict:
        out = {}
        for s, c in self.settings["defense"].items():
            st = self._state(defense_team, c["half_life"])
            out[s] = float(shrink_factor([st[f"dA_{s}"]], [st[f"dE_{s}"]], [st["def_w"]], c["k"])[0])
        return out

    def team_table(self, teams) -> pd.DataFrame:
        rows = []
        for t in sorted(teams):
            f = self.defense_factors(t)
            p = self.settings["pace"]
            st = self._state(t, p["half_life"])
            rows.append({"team": t, "pace_vs_league": float(pace_term(st["pace_dev"], st["pace_w"], p["k"])),
                         **{f"def_{s}": v for s, v in f.items()}})
        return pd.DataFrame(rows)
