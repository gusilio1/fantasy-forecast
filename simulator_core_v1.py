"""
simulator_core_v1.py

Core Monte Carlo engine for the ESPN NBA fantasy simulator.

This module deliberately separates:
    1. availability / injury-adjusted expected minutes
    2. NBA game possessions
    3. player possession exposure
    4. coherent box-score simulation
    5. ESPN fantasy points

Frozen V1 statistical decisions
--------------------------------
Minutes:
    expected minutes are supplied by the projection/rotation layer.
    Simulation residual scale = 0.87.

3PA share:
    game-varying Beta share, variance scale = 0.55.

FTA:
    game-varying Gamma rate -> Poisson attempts, variance scale = 0.70.

Rebounds:
    generated from team/opponent missed-FG opportunities.

AST / TO / STL / BLK:
    Poisson per player possession exposure.

Fantasy scoring:
    FP = PTS + REB + 2*AST + 4*STL + 4*BLK - 2*TO
         + 2*FGM - FGA + 3PM

Important:
    Injury-driven per-possession role changes are NOT used in V1.
    The validated injury effect is applied through expected minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import numpy as np


# =============================================================================
# CALIBRATED CONSTANTS
# =============================================================================

MINUTES_RESIDUAL_SCALE = 0.87
THREE_SHARE_VARIANCE_SCALE = 0.55
FTA_RATE_VARIANCE_SCALE = 0.70

MINUTES_CAP = 48.0
DEFAULT_GAME_POSSESSIONS_MEAN = 101.55
DEFAULT_GAME_POSSESSIONS_SD = 5.39

EPS = 1e-9


# =============================================================================
# DATA OBJECTS
# =============================================================================

@dataclass
class PlayerProjection:
    player_id: str
    name: str
    nba_team: str

    # Expected minutes AFTER availability/rotation logic for this simulation.
    expected_minutes: float

    # Per-player-possession rates.
    fga_rate: float
    fta_rate: float
    ast_rate: float
    tov_rate: float
    stl_rate: float
    blk_rate: float

    # Shooting composition / efficiency.
    three_share: float       # 3PA / FGA
    two_pct: float           # 2PM / 2PA
    three_pct: float         # 3PM / 3PA
    ft_pct: float            # FTM / FTA

    # Rebound capture rates from available missed-FG opportunities.
    oreb_capture: float
    dreb_capture: float

    # Historical distributions used for uncertainty.
    minutes_residuals: Sequence[float] = field(default_factory=list)

    # Optional estimated variance for game-varying 3PA share.
    three_share_variance: Optional[float] = None

    # Optional variance of the latent FTA rate.
    fta_rate_variance: Optional[float] = None

    available: bool = True


@dataclass
class SimulatedPlayerGame:
    player_id: str
    name: str
    nba_team: str
    minutes: float
    exposure: float

    fga: int
    fgm: int
    three_pa: int
    three_pm: int
    two_pa: int
    two_pm: int

    fta: int
    ftm: int

    oreb: int
    dreb: int
    reb: int

    ast: int
    stl: int
    blk: int
    tov: int
    pts: int

    fantasy_points: float


@dataclass
class SimulatedNBAGame:
    game_id: str
    team_a: str
    team_b: str
    possessions: float
    players: List[SimulatedPlayerGame]


# =============================================================================
# HELPERS
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def fantasy_points(
    pts: int,
    reb: int,
    ast: int,
    stl: int,
    blk: int,
    tov: int,
    fgm: int,
    fga: int,
    three_pm: int,
) -> float:
    """Validated league scoring formula."""
    return float(
        pts
        + reb
        + 2 * ast
        + 4 * stl
        + 4 * blk
        - 2 * tov
        + 2 * fgm
        - fga
        + three_pm
    )


def sample_minutes(
    expected_minutes: float,
    residuals: Sequence[float],
    rng: np.random.Generator,
) -> float:
    """
    Validated V1 minutes uncertainty:
        MIN = expected MIN + 0.87 * empirical residual
    Residual pool should already be approximately zero-centred.
    """
    mu = clamp(expected_minutes, 0.0, MINUTES_CAP)

    if mu <= 0:
        return 0.0

    arr = np.asarray(residuals, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        # Safe fallback for players without a residual pool.
        residual = rng.normal(0.0, 6.397)
    else:
        residual = float(rng.choice(arr))
        residual -= float(np.mean(arr))

    return clamp(mu + MINUTES_RESIDUAL_SCALE * residual, 0.0, MINUTES_CAP)


def sample_beta_from_mean_variance(
    mean: float,
    variance: Optional[float],
    variance_scale: float,
    rng: np.random.Generator,
) -> float:
    """
    Draw a probability from a Beta distribution while preserving the supplied
    mean. Falls back to the mean if variance is unavailable or invalid.
    """
    m = clamp(mean, EPS, 1.0 - EPS)

    if variance is None or not np.isfinite(variance) or variance <= 0:
        return m

    v = float(variance) * variance_scale
    max_v = m * (1.0 - m)

    if v <= 0 or v >= max_v:
        return m

    concentration = max_v / v - 1.0
    alpha = m * concentration
    beta = (1.0 - m) * concentration

    if alpha <= 0 or beta <= 0:
        return m

    return float(rng.beta(alpha, beta))


def sample_gamma_rate(
    mean_rate: float,
    rate_variance: Optional[float],
    variance_scale: float,
    rng: np.random.Generator,
) -> float:
    """Gamma mixing for an overdispersed Poisson rate."""
    m = max(float(mean_rate), 0.0)

    if m <= 0:
        return 0.0

    if (
        rate_variance is None
        or not np.isfinite(rate_variance)
        or rate_variance <= 0
    ):
        return m

    v = float(rate_variance) * variance_scale

    if v <= 0:
        return m

    shape = (m * m) / v
    scale = v / m

    if shape <= 0 or scale <= 0:
        return m

    return float(rng.gamma(shape, scale))


def poisson(rate: float, rng: np.random.Generator) -> int:
    return int(rng.poisson(max(float(rate), 0.0)))


def binomial(n: int, p: float, rng: np.random.Generator) -> int:
    if n <= 0:
        return 0
    return int(rng.binomial(int(n), clamp(float(p), 0.0, 1.0)))


# =============================================================================
# TEAM MINUTES
# =============================================================================

def normalize_team_minutes(
    players: Sequence[PlayerProjection],
    target: float = 240.0,
) -> List[PlayerProjection]:
    """
    Scale available players' expected minutes to the requested team total.

    This should be called AFTER the injury rotation model has produced its
    expected-minute predictions.

    It does not alter unavailable players: they remain at zero.
    """
    copied = [
        PlayerProjection(**{**p.__dict__})
        for p in players
    ]

    available = [p for p in copied if p.available and p.expected_minutes > 0]

    if not available:
        return copied

    total = sum(p.expected_minutes for p in available)

    if total <= 0:
        return copied

    scale = target / total

    # First pass.
    for p in copied:
        if not p.available:
            p.expected_minutes = 0.0
        else:
            p.expected_minutes = clamp(
                p.expected_minutes * scale, 0.0, MINUTES_CAP
            )

    # Iteratively distribute any remainder without exceeding 48.
    for _ in range(20):
        current = sum(p.expected_minutes for p in copied if p.available)
        diff = target - current

        if abs(diff) < 1e-6:
            break

        if diff > 0:
            eligible = [
                p for p in copied
                if p.available and p.expected_minutes < MINUTES_CAP - 1e-9
            ]
            if not eligible:
                break
            room = np.array(
                [MINUTES_CAP - p.expected_minutes for p in eligible],
                dtype=float,
            )
            room_sum = room.sum()
            if room_sum <= 0:
                break
            for p, r in zip(eligible, room):
                p.expected_minutes += min(diff * r / room_sum, r)
        else:
            eligible = [
                p for p in copied
                if p.available and p.expected_minutes > 0
            ]
            if not eligible:
                break
            vals = np.array([p.expected_minutes for p in eligible], dtype=float)
            vals_sum = vals.sum()
            if vals_sum <= 0:
                break
            need = -diff
            for p, v in zip(eligible, vals):
                p.expected_minutes -= min(need * v / vals_sum, p.expected_minutes)

    return copied


def sample_team_minutes(
    players: Sequence[PlayerProjection],
    rng: np.random.Generator,
    target: float = 240.0,
) -> Dict[str, float]:
    """
    Sample player minutes then renormalize the available rotation to 240.

    This preserves game-to-game player minute uncertainty while enforcing the
    NBA team-minute identity.
    """
    raw = {}

    for p in players:
        if not p.available or p.expected_minutes <= 0:
            raw[p.player_id] = 0.0
        else:
            raw[p.player_id] = sample_minutes(
                p.expected_minutes,
                p.minutes_residuals,
                rng,
            )

    active_ids = [
        p.player_id for p in players
        if p.available and raw[p.player_id] > 0
    ]

    if not active_ids:
        return raw

    # Iterative capped normalization.
    vals = {pid: raw[pid] for pid in active_ids}

    for _ in range(30):
        total = sum(vals.values())
        diff = target - total

        if abs(diff) < 1e-6:
            break

        if diff > 0:
            eligible = [pid for pid in active_ids if vals[pid] < MINUTES_CAP]
            if not eligible:
                break
            room = np.array([MINUTES_CAP - vals[pid] for pid in eligible])
            denom = room.sum()
            if denom <= 0:
                break
            for pid, r in zip(eligible, room):
                vals[pid] += min(diff * r / denom, r)
        else:
            eligible = [pid for pid in active_ids if vals[pid] > 0]
            if not eligible:
                break
            weights = np.array([vals[pid] for pid in eligible])
            denom = weights.sum()
            if denom <= 0:
                break
            remove = -diff
            for pid, w in zip(eligible, weights):
                vals[pid] -= min(remove * w / denom, vals[pid])

    raw.update(vals)
    return raw


# =============================================================================
# GAME SIMULATION
# =============================================================================

def simulate_game_possessions(
    rng: np.random.Generator,
    mean: float = DEFAULT_GAME_POSSESSIONS_MEAN,
    sd: float = DEFAULT_GAME_POSSESSIONS_SD,
) -> float:
    # Conservative bounds prevent absurd tails.
    return clamp(rng.normal(mean, sd), 80.0, 130.0)


def simulate_pre_rebound_box(
    p: PlayerProjection,
    minutes: float,
    game_possessions: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Simulate all player stats except OREB/DREB.

    Player exposure:
        game possessions * minutes / 48
    """
    exposure = max(game_possessions * minutes / 48.0, 0.0)

    if exposure <= 0:
        return {
            "minutes": 0.0,
            "exposure": 0.0,
            "fga": 0, "fgm": 0,
            "three_pa": 0, "three_pm": 0,
            "two_pa": 0, "two_pm": 0,
            "fta": 0, "ftm": 0,
            "ast": 0, "stl": 0, "blk": 0, "tov": 0,
            "pts": 0,
        }

    # FGA volume.
    fga = poisson(p.fga_rate * exposure, rng)

    # 3PA is a subset of FGA.
    three_share = sample_beta_from_mean_variance(
        p.three_share,
        p.three_share_variance,
        THREE_SHARE_VARIANCE_SCALE,
        rng,
    )
    three_pa = binomial(fga, three_share, rng)
    two_pa = fga - three_pa

    three_pm = binomial(three_pa, p.three_pct, rng)
    two_pm = binomial(two_pa, p.two_pct, rng)
    fgm = two_pm + three_pm

    # FTA: gamma-mixed rate -> Poisson attempts.
    fta_rate = sample_gamma_rate(
        p.fta_rate,
        p.fta_rate_variance,
        FTA_RATE_VARIANCE_SCALE,
        rng,
    )
    fta = poisson(fta_rate * exposure, rng)
    ftm = binomial(fta, p.ft_pct, rng)

    # Other exposure-based stats.
    ast = poisson(p.ast_rate * exposure, rng)
    tov = poisson(p.tov_rate * exposure, rng)
    stl = poisson(p.stl_rate * exposure, rng)
    blk = poisson(p.blk_rate * exposure, rng)

    pts = 2 * two_pm + 3 * three_pm + ftm

    return {
        "minutes": float(minutes),
        "exposure": float(exposure),
        "fga": fga,
        "fgm": fgm,
        "three_pa": three_pa,
        "three_pm": three_pm,
        "two_pa": two_pa,
        "two_pm": two_pm,
        "fta": fta,
        "ftm": ftm,
        "ast": ast,
        "stl": stl,
        "blk": blk,
        "tov": tov,
        "pts": pts,
    }


def simulate_nba_game(
    game_id: str,
    team_a: str,
    team_b: str,
    team_a_players: Sequence[PlayerProjection],
    team_b_players: Sequence[PlayerProjection],
    rng: np.random.Generator,
    possession_mean: float = DEFAULT_GAME_POSSESSIONS_MEAN,
    possession_sd: float = DEFAULT_GAME_POSSESSIONS_SD,
) -> SimulatedNBAGame:
    """
    Simulate one NBA game.

    Shared game possessions induce correlation among all players in the game.
    Rebounds are generated from the simulated missed-FG environment.
    """
    possessions = simulate_game_possessions(
        rng,
        mean=possession_mean,
        sd=possession_sd,
    )

    a_players = normalize_team_minutes(team_a_players)
    b_players = normalize_team_minutes(team_b_players)

    a_minutes = sample_team_minutes(a_players, rng)
    b_minutes = sample_team_minutes(b_players, rng)

    a_box = {
        p.player_id: simulate_pre_rebound_box(
            p, a_minutes.get(p.player_id, 0.0), possessions, rng
        )
        for p in a_players
    }
    b_box = {
        p.player_id: simulate_pre_rebound_box(
            p, b_minutes.get(p.player_id, 0.0), possessions, rng
        )
        for p in b_players
    }

    # Team missed FGs create rebound opportunities.
    a_missed_fg = sum(x["fga"] - x["fgm"] for x in a_box.values())
    b_missed_fg = sum(x["fga"] - x["fgm"] for x in b_box.values())

    results: List[SimulatedPlayerGame] = []

    def finish_team(
        players: Sequence[PlayerProjection],
        boxes: Mapping[str, Dict[str, float]],
        own_missed_fg: int,
        opp_missed_fg: int,
    ) -> None:
        for p in players:
            x = boxes[p.player_id]
            minute_fraction = x["minutes"] / 48.0

            # Validated rebound-opportunity model.
            oreb_opp = own_missed_fg * minute_fraction
            dreb_opp = opp_missed_fg * minute_fraction

            oreb = binomial(
                max(int(round(oreb_opp)), 0),
                p.oreb_capture,
                rng,
            )
            dreb = binomial(
                max(int(round(dreb_opp)), 0),
                p.dreb_capture,
                rng,
            )
            reb = oreb + dreb

            fp = fantasy_points(
                pts=int(x["pts"]),
                reb=reb,
                ast=int(x["ast"]),
                stl=int(x["stl"]),
                blk=int(x["blk"]),
                tov=int(x["tov"]),
                fgm=int(x["fgm"]),
                fga=int(x["fga"]),
                three_pm=int(x["three_pm"]),
            )

            results.append(
                SimulatedPlayerGame(
                    player_id=p.player_id,
                    name=p.name,
                    nba_team=p.nba_team,
                    minutes=float(x["minutes"]),
                    exposure=float(x["exposure"]),
                    fga=int(x["fga"]),
                    fgm=int(x["fgm"]),
                    three_pa=int(x["three_pa"]),
                    three_pm=int(x["three_pm"]),
                    two_pa=int(x["two_pa"]),
                    two_pm=int(x["two_pm"]),
                    fta=int(x["fta"]),
                    ftm=int(x["ftm"]),
                    oreb=oreb,
                    dreb=dreb,
                    reb=reb,
                    ast=int(x["ast"]),
                    stl=int(x["stl"]),
                    blk=int(x["blk"]),
                    tov=int(x["tov"]),
                    pts=int(x["pts"]),
                    fantasy_points=fp,
                )
            )

    finish_team(a_players, a_box, a_missed_fg, b_missed_fg)
    finish_team(b_players, b_box, b_missed_fg, a_missed_fg)

    return SimulatedNBAGame(
        game_id=str(game_id),
        team_a=team_a,
        team_b=team_b,
        possessions=possessions,
        players=results,
    )


# =============================================================================
# MONTE CARLO WRAPPER
# =============================================================================

def simulate_game_many(
    game_id: str,
    team_a: str,
    team_b: str,
    team_a_players: Sequence[PlayerProjection],
    team_b_players: Sequence[PlayerProjection],
    sims: int = 10_000,
    seed: Optional[int] = None,
    possession_mean: float = DEFAULT_GAME_POSSESSIONS_MEAN,
    possession_sd: float = DEFAULT_GAME_POSSESSIONS_SD,
):
    """
    Convenience wrapper for validating one NBA game/player distribution.

    Returns:
        list[SimulatedNBAGame]
    """
    if sims <= 0:
        raise ValueError("sims must be > 0")

    rng = np.random.default_rng(seed)

    return [
        simulate_nba_game(
            game_id=game_id,
            team_a=team_a,
            team_b=team_b,
            team_a_players=team_a_players,
            team_b_players=team_b_players,
            rng=rng,
            possession_mean=possession_mean,
            possession_sd=possession_sd,
        )
        for _ in range(sims)
    ]


def summarize_player(
    simulations: Sequence[SimulatedNBAGame],
    player_id: str,
) -> Dict[str, float]:
    """Summarize a player's Monte Carlo game distribution."""
    rows = [
        p
        for sim in simulations
        for p in sim.players
        if p.player_id == str(player_id)
    ]

    if not rows:
        raise ValueError(f"Player {player_id} not found in simulations.")

    fp = np.array([r.fantasy_points for r in rows], dtype=float)
    mins = np.array([r.minutes for r in rows], dtype=float)

    return {
        "simulations": float(len(rows)),
        "mean_minutes": float(np.mean(mins)),
        "mean_fp": float(np.mean(fp)),
        "sd_fp": float(np.std(fp, ddof=1)) if len(fp) > 1 else 0.0,
        "p10_fp": float(np.percentile(fp, 10)),
        "median_fp": float(np.percentile(fp, 50)),
        "p90_fp": float(np.percentile(fp, 90)),
    }


# =============================================================================
# INJURY / AVAILABILITY INTERFACE
# =============================================================================

def draw_availability(
    play_probability: float,
    rng: np.random.Generator,
) -> bool:
    """
    Monte Carlo availability draw.

    Example:
        play_probability = 0.35
        -> player plays in roughly 35% of simulations.
    """
    return bool(rng.random() < clamp(play_probability, 0.0, 1.0))


def apply_availability(
    players: Sequence[PlayerProjection],
    availability: Mapping[str, bool],
) -> List[PlayerProjection]:
    """
    Apply a PLAYS/OUT draw.

    This function only marks availability and zeros OUT players.
    The learned V2 injury rotation model should then supply the replacement
    expected minutes before simulate_nba_game() is called.
    """
    out = []

    for p in players:
        q = PlayerProjection(**{**p.__dict__})
        q.available = bool(availability.get(p.player_id, p.available))

        if not q.available:
            q.expected_minutes = 0.0

        out.append(q)

    return out


# =============================================================================
# FANTASY-WEEK HELPERS
# =============================================================================

def accumulate_fantasy_points(
    games: Iterable[SimulatedNBAGame],
) -> Dict[str, float]:
    """Sum simulated fantasy points by player across multiple NBA games."""
    totals: Dict[str, float] = {}

    for game in games:
        for p in game.players:
            totals[p.player_id] = totals.get(p.player_id, 0.0) + p.fantasy_points

    return totals


if __name__ == "__main__":
    print("simulator_core_v1.py")
    print("Core module loaded successfully.")
    print()
    print("Next integration step:")
    print("  current ESPN projections")
    print("    -> PlayerProjection objects")
    print("    -> availability draw")
    print("    -> V2 injury-adjusted minutes")
    print("    -> simulate_nba_game()")
    print("    -> fantasy lineup optimiser")
    print("    -> weekly matchup win probability")
