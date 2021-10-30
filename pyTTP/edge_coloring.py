
import numpy as np


def _canonical_home_away_pattern_coloring(num_nodes: int):
    n = num_nodes
    E = np.zeros((n - 1, n // 2, 2), dtype=np.int32)
    for i in range(1, n):
        E[i - 1][0][:] = [i, n] if i % 2 == 0 else [n, i]
        for j in range(1, n // 2):
            a = n - 1 if ((i - j) % (n - 1) == 0) else ((i - j) % (n - 1))
            b = n - 1 if ((i + j) % (n - 1) == 0) else ((i + j) % (n - 1))
            E[i - 1][j][:] = [a, b] if j % 2 != 0 else [b, a]
    return E


def create_initial_schedule(num_teams: int):
    E = _canonical_home_away_pattern_coloring(num_teams)
    num_teams = 2 * E.shape[1]
    num_rounds = E.shape[0]
    schedule = np.zeros((num_teams, num_rounds), dtype=np.int32)
    for s in range(0, num_rounds):
        for r in range(0, (num_rounds // 2) + 1):
            # match at home
            schedule[E[s, r, 0] - 1, s] = E[s, r, 1]
            # match not at home
            schedule[E[s, r, 1] - 1, s] = -1 * E[s, r, 0]
    return np.hstack((schedule, -schedule))
