#!/usr/bin/env python3

from mip import Model, xsum, MINIMIZE
import itertools
import numpy as np

mdl = Model("bla", sense=MINIMIZE, solver_name="Gurobi")

fix_home_away_pattern = False
fix_scheduling = False

# Konstanten
num_teams = 4
num_spieltage = 2*num_teams - 2
Teams = ["ATL", "NYM", "PHI", "MON", "FLA", "PIT", "CIN", "CHI",
         "STL", "MIL", "HOU", "COL", "SF", "SD", "LA", "ARI"][:num_teams]
Spieltage = list(range(num_spieltage))
D = np.loadtxt(f"tests/data{num_teams}.txt", dtype=np.int32)
D = {(t1, t2): D[i, j] for i, t1 in enumerate(Teams)
     for j, t2 in enumerate(Teams)}
K = 3  # max length of home stand and road trips

# Sets
PairTeams = list((i, j) for i in Teams for j in Teams)
TripleTeams = list((i, j, t) for (i, j) in PairTeams for t in Teams)


# x[i,j,s] = 1, if team i plays team j at home on round s
x = {(i, j, s): mdl.add_var(var_type="B", name=f"x[{i},{j},{s}]")
     for (i, j) in PairTeams for s in Spieltage}

# y[t,i,j] = 1, if team t travels from team i to team j
y = {(i, j, t): mdl.add_var(var_type="B", name=f"y[{i},{j},{t}]")
     for (i, j, t) in TripleTeams}

# z[i,j,k] = 1, if
z = {(i, j, s): mdl.add_var(var_type="B", name=f"z[{i},{j},{s}]")
     for (i, j) in PairTeams for s in Spieltage}

# Zielfunktion
expr1 = xsum(D[i, j]*x[i, j, 0] for (i, j) in PairTeams)
expr2 = xsum(D[i, j]*y[t, i, j] for (i, j, t) in TripleTeams)
expr3 = xsum(D[i, j]*x[i, j, num_spieltage-1] for (i, j) in PairTeams)
mdl.objective = expr1 + expr2 + expr3

for i in Teams:
    for s in Spieltage:
        mdl.add_constr(x[i, i, s] == 0)

# Jedes Team bestreitet pro Spieltag s genau ein Spiel
for s in Spieltage:
    for i in Teams:
        mdl.add_constr(
            xsum(x[i, j, s] + x[j, i, s] for j in Teams) == 1)

# Jedes Team spielt gegen jedes andere Team genau einmal zuhause und einmal
# auswärts
for (i, j) in PairTeams:
    if i != j:
        mdl.add_constr(xsum(x[i, j, s] for s in Spieltage) == 1)

# home/away stand
for s in Spieltage[:-K]:
    for i in Teams:
        mdl.add_constr(xsum(x[i, j, s+l]
                            for l in range(K+1) for j in Teams) <= K)
    for j in Teams:
        mdl.add_constr(xsum(x[i, j, s+l]
                            for l in range(K+1) for i in Teams) <= K)

# Die Partie j vs i darf nicht am darauffolgenden Spieltag sein wie die Partie
# i vs j, d.h. x[j,i,s] + x[i,j,s] == 2 --> x[j,i,s+1] + x[i, j,s+1] == 0
for (i, j) in PairTeams:
    for s in Spieltage[:-1]:
        mdl.add_constr(x[i, j, s]+x[j, i, s]+x[i, j, s+1]+x[j, i, s+1] <= 1)

for i in Teams:
    for s in Spieltage:
        mdl.add_constr(z[i, i, s] == xsum(x[j, i, s] for j in Teams))

for (i, j) in PairTeams:
    if i != j:
        for s in Spieltage:
            mdl.add_constr(z[i, j, s] == x[i, j, s])

for (i, j, t) in TripleTeams:
    for s in Spieltage[:-1]:
        mdl.add_constr(y[t, i, j] >= z[t, i, s] + z[t, j, s+1] - 1)

# Falls home away pattern vorgegeben
# if fix_home_away_pattern:
#     for (i, j) in PairTeams:
#         if i != j:
#             for s in Spieltage:
#                 mdl.add_constr(x[i, j, s] + x[j, i, s] ==
#                                sol[i, j, s] + sol[j, i, s])


mdl.write("bla.lp")

mdl.max_mip_gap = 0.8
mdl.optimize()

mdl.status == mdl.status.OPTIMAL


#   Print the solution
S = np.zeros((num_teams, num_spieltage), dtype=np.int32)
# Print Header
print(f"Slot\t" + "\t".join(Teams), end="\n\n")
for s in Spieltage:
    print(f"{s:2d}\t", end="")
    for i, t1 in enumerate(Teams):
        for j, t2 in enumerate(Teams):
            if x[t1, t2, s].x > 0.0:
                print(f"@{t2:4s}\t", end="")
                S[[i, j], s] = [-(j+1), i+1]
            if x[t2, t1, s].x > 0.0:
                print(f"{t2:>4s}\t", end="")
                S[[i, j], s] = [j+1, -(i+1)]

    print("")

if mdl.status == mdl.status.OPTIMAL:
    print(mdl.objective_value)
    print("hi")

print(S)

np.savetxt("S_test.txt", S, fmt="%3d")
