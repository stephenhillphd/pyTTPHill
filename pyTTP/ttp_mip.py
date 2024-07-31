#!/usr/bin/env python3

from mip import Model, xsum, MINIMIZE
import itertools
import numpy as np
import gurobipy as gp


class TTPMip:
    def __init__(self, distance_matrix, max_mip_gap: float = 0.0,
                 max_home_away_stand: int = 3):
        self.distance_matrix = distance_matrix
        self.num_teams = distance_matrix.shape[0]
        self.num_rounds = 2*self.num_teams - 2
        self.teams = list(range(self.num_teams))
        self.max_home_away_stand = max_home_away_stand
        self._fix_scheduling_constrs = []
        self._fix_home_away_pattern_constrs = []
        self._create_mip_model()
        self.mdl.solver.set_verbose(0)
        self.mdl.max_mip_gap = max_mip_gap

    def solve(self, start_obj_val: float, start_sol):
        #sol = self._transform_start_sol(start_sol)
        # Phase II
        proceed = True
        improved = False
        while proceed is True:
            proceed = False
            # optimize home_away_pattern
            self._fix_home_away_pattern(start_sol)
            self.mdl.optimize()
            if self.mdl.status != self.mdl.status.OPTIMAL:
                return (improved,)
            if self.mdl.objective_value < start_obj_val:
                proceed = True
                improved = True
                # get the solution of the previous optimization
                start_sol = self.get_solution()
                start_obj_val = self.mdl.objective_value
                print(
                    f"Phase II found new solution ( HA): {start_obj_val:6.0f}")
            # remove the previous constraints
            self.mdl.remove(self._fix_home_away_pattern_constrs)
            # optimize schedule pattern and use previous solution as start
            self._fix_scheduling(start_sol)
            self.mdl.optimize()
            if self.mdl.status != self.mdl.status.OPTIMAL:
                return (improved,)
            if self.mdl.objective_value < start_obj_val:
                proceed = True
                improved = True
                # get the solution of the previous optimization
                start_sol = self.get_solution()
                start_obj_val = self.mdl.objective_value
                print(
                    f"Phase II found new solution (nHA): {start_obj_val:6.0f}")
            # else, get solution and continue
            start_sol = self.get_solution()
            # remove the previous constraints
            self.mdl.remove(self._fix_scheduling_constrs)
        return improved, start_obj_val, start_sol

    def get_solution(self):
        S = np.zeros((self.num_teams, self.num_rounds), dtype=np.int32)
        for s in range(self.num_rounds):
            for i, t1 in enumerate(self.teams):
                for j, t2 in enumerate(self.teams):
                    if self.x[t1, t2, s].x > 0.0:
                        S[[i, j], s] = [-(j+1), i+1]
                    if self.x[t2, t1, s].x > 0.0:
                        S[[i, j], s] = [j+1, -(i+1)]
        return S

    def _transform_start_sol(self, start_sol):
        sol = {
            (t1, t2, s): 0 for t1 in self.teams for t2 in self.teams for s in range(self.num_rounds) if t1 != t2}
        for i, team in enumerate(self.teams):
            for s in range(self.num_rounds):
                if (idx := start_sol[i, s]) > 0:
                    opponent = self.teams[idx-1]
                    sol[team, opponent, s] = 1
        # print(sol)
        return sol

    def _create_mip_model(self):

        self.rounds = list(range(self.num_rounds))
        D = {(t1, t2): self.distance_matrix[i, j] for i, t1 in enumerate(self.teams)
             for j, t2 in enumerate(self.teams)}
        K = self.max_home_away_stand  # max length of home stand and road trips

        # -- sets --
        self.PairTeams = list((i, j) for i in self.teams for j in self.teams)
        self.TripleTeams = list((i, j, t)
                                for (i, j) in self.PairTeams for t in self.teams)

        # mip model object
        self.mdl = Model("bla", sense=MINIMIZE, solver_name="Gurobi")

        # -- variables --

        # x[i,j,s] = 1, if team i plays team j at home on round s
        x = {(i, j, s): self.mdl.add_var(var_type="B", name=f"x[{i},{j},{s}]")
             for (i, j) in self.PairTeams for s in self.rounds}
        self.x = x

        # y[t,i,j] = 1, if team t travels from team i to team j
        y = {(i, j, t): self.mdl.add_var(var_type="B", name=f"y[{i},{j},{t}]")
             for (i, j, t) in self.TripleTeams}

        # z[i,j,k] = 1, if
        z = {(i, j, s): self.mdl.add_var(var_type="B", name=f"z[{i},{j},{s}]")
             for (i, j) in self.PairTeams for s in self.rounds}

        # objective
        expr1 = xsum(D[i, j]*x[i, j, 0] for (i, j) in self.PairTeams)
        expr2 = xsum(D[i, j]*y[t, i, j] for (i, j, t) in self.TripleTeams)
        expr3 = xsum(D[i, j]*x[i, j, self.num_rounds-1]
                     for (i, j) in self.PairTeams)
        self.mdl.objective = expr1 + expr2 + expr3

        for i in self.teams:
            for s in self.rounds:
                self.mdl.add_constr(x[i, i, s] == 0)

        # each team has exact one match per round s
        for s in self.rounds:
            for i in self.teams:
                self.mdl.add_constr(
                    xsum(x[i, j, s] + x[j, i, s] for j in self.teams) == 1)

        # each team plays the other teams exactly once at home and exactly once away
        for (i, j) in self.PairTeams:
            if i != j:
                self.mdl.add_constr(xsum(x[i, j, s] for s in self.rounds) == 1)

        # do not exceed the number of allowed home stands or road trips
        for s in self.rounds[:-K]:
            for i in self.teams:
                self.mdl.add_constr(xsum(x[i, j, s+l]
                                         for l in range(K+1) for j in self.teams) <= K)
            for j in self.teams:
                self.mdl.add_constr(xsum(x[i, j, s+l]
                                         for l in range(K+1) for i in self.teams) <= K)

        # no repeaters, i.e.
        # x[j,i,s] + x[i,j,s] == 2 --> x[j,i,s+1] + x[i, j,s+1] == 0
        for (i, j) in self.PairTeams:
            for s in self.rounds[:-1]:
                self.mdl.add_constr(x[i, j, s]+x[j, i, s] +
                                    x[i, j, s+1]+x[j, i, s+1] <= 1)

        for i in self.teams:
            for s in self.rounds:
                self.mdl.add_constr(z[i, i, s] == xsum(
                    x[j, i, s] for j in self.teams))

        for (i, j) in self.PairTeams:
            if i != j:
                for s in self.rounds:
                    self.mdl.add_constr(z[i, j, s] == x[i, j, s])

        for (i, j, t) in self.TripleTeams:
            for s in self.rounds[:-1]:
                self.mdl.add_constr(
                    y[t, i, j] >= z[t, i, s] + z[t, j, s+1] - 1)

    def _fix_home_away_pattern(self, start_sol):
        self._fix_home_away_pattern_constrs = []
        sol = self._transform_start_sol(start_sol)
        for (i, j) in self.PairTeams:
            if i != j:
                for s in range(self.num_rounds):
                    con = self.mdl.add_constr(
                        self.x[i, j, s] + self.x[j, i, s] == sol[i, j, s] + sol[j, i, s])
                    # save the constraints such that we can delete them later
                    self._fix_home_away_pattern_constrs.append(con)

    def _fix_scheduling(self, start_sol):
        self._fix_scheduling_constrs = []
        sol = self._transform_start_sol(start_sol)
        for j in self.teams:
            for s in self.rounds:
                lhs1 = xsum(self.x[i, j, s] for i in self.teams if i != j)
                lhs2 = xsum(self.x[j, i, s] for i in self.teams if i != j)
                rhs1 = xsum(sol[i, j, s] for i in self.teams if i != j)
                rhs2 = xsum(sol[j, i, s] for i in self.teams if i != j)
                con1 = self.mdl.add_constr(lhs1 == rhs1)
                con2 = self.mdl.add_constr(lhs2 == rhs2)
                # save the constraints such that we can delete them later
                self._fix_scheduling_constrs.append(con1)
                self._fix_scheduling_constrs.append(con2)
