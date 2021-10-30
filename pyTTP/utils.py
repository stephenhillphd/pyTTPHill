#!/usr/bin/env python3

from .localsearch import ttp_objective as objective


def print_solution(schedule, teams: list[str]):
    num_rounds = 2*len(teams) - 2
    print(f"Slot\t" + "\t".join(teams), end="\n\n")
    for s in range(num_rounds):
        print(f"{s:2d}\t", end="")
        for i, t1 in enumerate(teams):
            for j, t2 in enumerate(teams):
                if schedule[i, s] == -(j + 1):
                    print(f"@{t2:4s}\t", end="")
                if schedule[i, s] == j + 1:
                    print(f"{t2:>4s}\t", end="")
        print("")
    print("")
