# pyTTP
🏈 A hybrid local search and MIP-heuristic for the **t**raveling **t**ournament **p**roblem.

This python package contains a simple hybrid local search and MIP-heuristic for 
the traveling tournament problem.
In detail, it uses a modified algorithm variant of the original approach described 
[here](https://link.springer.com/article/10.1007%2Fs10479-014-1586-6), which is 
due to M. Goerigk and S. Westphal.

The definition of the TTP is as follows: Let a set of `n` teams and a distance 
matrix `D` be given. Here, `D[i,j]` is the distance from team `i` to team `j`. 
One wants to find a feasible 
[double round robin tournament](https://en.wikipedia.org/wiki/Round-robin_tournament) 
of the teams satisfying the following conditions:

- the length of any home stand is at most `max_k`.
- the length of any road trip is at most `max_k`.
- the two games between any two teams do not follow immediately.
- the sum of the distances traveled by the teams is minimized.

### The basic algorithm

Starting from a canonical feasible solution, the algorithm uses a greedy local 
search to improve the solution (Phase I). As soon as we are stuck at a local 
minimum, the algorithm then tries to escape the local minimum by a MIP-heuristic 
(Phase II). Here, it optimizes the home-away patterns and the team matchups in 
alternation as long as it is not possible to improve the previous solution. 
Then, it switches back to Phase I and repeats. It terminates if neither Phase I 
nor Phase II can further improve the current best solution.

### Details

The local search is written in Cython and thus guarantees a high performance. 
Since most of the algorithm runtime is spent in Phase II, it's possible to set a 
custom MIP Gap. This can reduce the runtime dramatically but will lead to worse 
solutions in most cases. However, there are some cases where a higher MIP Gap 
gives better solutions, which in turn indicates that the neighborhoods used by 
the local search are not fully connected.

## Install

After installing Cython and a C++ compiler with C++17 support, this package
can be installed via

``` bash
pip3 install git+https://github.com/jhelgert/pyTTP
```

## Example

``` python
import numpy as np
from pyTTP import solve, print_schedule

team_names = ["ATL", "NYM", "PHI", "MON"]
distances = np.array([[0,   745,  665,  929],
                     [745,   0,   80,  337],
                     [665,  80,    0,  380],
                     [929, 337,  380,    0]], dtype=np.int32)

objval, schedule = solve(distances, max_k=3)
print_schedule(schedule, team_names)
```

gives the optimal schedule

``` none
Slot    ATL     NYM     PHI     MON

 0      @MON     PHI    @NYM     ATL
 1      @NYM     ATL    @MON     PHI
 2      @PHI    @MON     ATL     NYM
 3       NYM    @ATL     MON    @PHI 
 4       MON    @PHI     NYM    @ATL 
 5       PHI     MON    @ATL    @NYM 
```

with objective value `8276`.
