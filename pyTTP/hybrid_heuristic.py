#!/usr/bin/env python3

import numpy as np
from .localsearch import ttp_local_search, ttp_objective
from .ttp_mip import TTPMip
from .edge_coloring import create_initial_schedule


def _phase1(start_obj_val, start_sol, D, max_runs: int = 20, max_k: int = 3):
    start_sols = []
    obj_vals = []
    for i in range(max_runs):
        sols, objvals = ttp_local_search(start_sol, D, max_k)
        if len(sols) > 0:
            # found solutions
            for sol, objval in zip(sols, objvals):
                # only add if it is a new solution
                if objval not in obj_vals:
                    start_sols.append(sol)
                    obj_vals.append(objval)
        # take the first solution and do a local search
        # if len(obj_vals) > 0:
            start_sol = start_sols.pop(0)
            obj_val = obj_vals.pop(0)
        else:
            return False, None, None
    # finish, take the best solution
    idx = np.argmin(obj_vals)
    # print(f"Best solution: {obj_vals[idx]}")
    # is it better than the given solution before phase I?
    if obj_vals[idx] < start_obj_val:
        indices = np.argsort(obj_vals)
        obj_vals = [obj_vals[i] for i in indices[:3]]
        start_sols = [start_sols[i] for i in indices[:3]]
        return True, obj_vals, start_sols
    else:
        return False, obj_vals[idx], start_sols[idx]


def algo(D, max_k: int = 3, max_phase1_runs: int = 10, max_gap_phase2: float = 0.05):
    """ solves the TTP by a hybrid local search and MIP heuristic.

    Args:
        D (array_like): the distance matrix 
        max_k (int, optional): The maximum number of allowed home stands and road trips. Defaults to 3.
        max_phase1_runs (int, optional): number of complete local search runs phase I. Defaults to 10.
        max_gap_phase2 (float, optional): the maximal MIP gap used in phase II. Defaults to 0.05.

    Returns:
        tuple: the objective value and the best found solution
    """
    # create feasible schedule
    N = D.shape[0]
    start_sol = create_initial_schedule(N)
    start_obj_val = ttp_objective(start_sol, D)

    # Create MIP model for phase II
    prob = TTPMip(D, max_gap_phase2, max_k)
    print(f"Canonical start solution            : {start_obj_val:6d}")
    while True:
        # Phase I: Try to improve the solution by a local search
        improved, start_obj_vals, start_sols = _phase1(
            start_obj_val, start_sol, D, max_runs=max_phase1_runs, max_k=max_k)
        if improved:
            print(f"Phase  I found new solution      : {start_obj_val:6.0f}")
        else:
            print(f"Phase  I couldn't further improve the solution. Finished.")
            return start_obj_val, start_sol

        # Phase II: Try to improve the solution by solving the MIPs
        while len(start_sols) > 0:
            start_obj_val, start_sol = start_obj_vals.pop(0), start_sols.pop(0)
            ret = prob.solve(start_obj_val, start_sol)
            if ret is not None and ret[0] is False:
                print(("Phase II couldn't improve the best solution of Phase I. "
                       "Try it with the next one."))
            else:
                # update the start solution and continue again with phase I
                start_obj_val, start_sol = ret[1:]
                break
