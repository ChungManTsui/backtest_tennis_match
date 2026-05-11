# markov.py — Barnett-Clarke Markov chain: point -> game -> set -> match

import numpy as np
from collections import defaultdict


def prob_hold_game(p: float) -> float:
    """P(server holds a game) given p = P(server wins a point on serve)."""
    q = 1 - p
    p_deuce = p ** 2 / (p ** 2 + q ** 2) if (p ** 2 + q ** 2) > 0 else 0.5
    hold = (
        p**4
        + 4 * p**4 * q
        + 10 * p**4 * q**2
        + 20 * p**3 * q**3 * p_deuce
    )
    return min(max(hold, 0.0), 1.0)


def _set_distribution(p_a_hold: float, p_b_hold: float):
    """
    Returns (P(A wins set), {games_in_set: probability}).
    A serves first. Serve alternates each game.
    Handles tiebreak at 6-6 (7-6 = 13 games total).
    """
    states = defaultdict(float)
    states[(0, 0, 0)] = 1.0  # (a_games, b_games, server: 0=A, 1=B)

    finished_a = 0.0
    game_dist = defaultdict(float)

    while states:
        new_states = defaultdict(float)
        for (ag, bg, server), prob in states.items():
            if prob < 1e-14:
                continue
            hold_p = p_a_hold if server == 0 else p_b_hold
            ns = 1 - server

            for (dga, dgb), p_ev in [
                ((1, 0) if server == 0 else (0, 1), hold_p),
                ((0, 1) if server == 0 else (1, 0), 1 - hold_p),
            ]:
                ga, gb = ag + dga, bg + dgb
                total = ga + gb

                set_over = False
                a_won = False

                if ga == 7:
                    set_over, a_won = True, True
                elif gb == 7:
                    set_over = True
                elif ga >= 6 and ga - gb >= 2:
                    set_over, a_won = True, True
                elif gb >= 6 and gb - ga >= 2:
                    set_over = True
                elif ga == 6 and gb == 6:
                    p_tb = p_a_hold / (p_a_hold + p_b_hold) if (p_a_hold + p_b_hold) > 0 else 0.5
                    game_dist[13] += prob * p_ev * p_tb
                    finished_a += prob * p_ev * p_tb
                    game_dist[13] += prob * p_ev * (1 - p_tb)
                    continue

                if set_over:
                    game_dist[total] += prob * p_ev
                    if a_won:
                        finished_a += prob * p_ev
                else:
                    new_states[(ga, gb, ns)] += prob * p_ev

        states = new_states

    total_prob = sum(game_dist.values())
    if total_prob > 0:
        game_dist = {k: v / total_prob for k, v in game_dist.items()}

    return finished_a, dict(game_dist)


def games_distribution_fast(p_hold_a: float, p_hold_b: float, best_of: int = 3) -> dict:
    """Returns {total_games_in_match: probability} distribution."""
    sets_to_win = (best_of + 1) // 2

    p_a_set, set_game_dist = _set_distribution(p_hold_a, p_hold_b)
    p_b_set = 1 - p_a_set

    match_states = defaultdict(float)
    match_states[(0, 0, 0)] = 1.0

    finished = defaultdict(float)

    while match_states:
        new_states = defaultdict(float)
        for (sa, sb, tg), prob in match_states.items():
            if prob < 1e-14:
                continue
            for n_games, p_set in set_game_dist.items():
                for (dsa, dsb), p_sw in [((1, 0), p_a_set), ((0, 1), p_b_set)]:
                    nsa, nsb = sa + dsa, sb + dsb
                    ntg = tg + n_games
                    combined = prob * p_set * p_sw
                    if nsa == sets_to_win or nsb == sets_to_win:
                        finished[ntg] += combined
                    else:
                        new_states[(nsa, nsb, ntg)] += combined

        match_states = new_states

    return dict(finished)


def prob_over_under(p_hold_a: float, p_hold_b: float, line: float, best_of: int = 3) -> tuple:
    """Returns (P(over line), P(under line)) for total games in a match."""
    dist = games_distribution_fast(p_hold_a, p_hold_b, best_of)
    p_over = sum(p for g, p in dist.items() if g > line)
    p_under = sum(p for g, p in dist.items() if g < line)
    return p_over, p_under
