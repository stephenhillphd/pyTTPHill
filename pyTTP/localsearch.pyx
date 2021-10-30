# distutils: language = c++

from cython cimport wraparound, boundscheck, cdivision, initializedcheck
from libcpp cimport bool
from libcpp.vector cimport vector
import numpy as np
cimport numpy as cnp
cnp.import_array()

from .neighborhoods cimport *

# needed for Cython 0.29.x
cdef extern from "<algorithm>" namespace "std" nogil:
    bool all_of[Iter, Pred](Iter first, Iter last, Pred pred) except +

@initializedcheck(False)
@wraparound(False)
@boundscheck(False)
cpdef int ttp_objective(int[:, ::1] S, int[:, ::1] D):
    cdef int ss, g1, g2, res = 0
    cdef int ii

    for ii in range(S.shape[0]):
        # first round, each time starts at home
        g1 = S[ii, 0]
        if g1 < 0:
            res += D[ii, -1*g1-1]
        for ss in range(1, S.shape[1]):
            g1 = S[ii, ss-1]
            g2 = S[ii, ss]
            if g1 < 0 and g2 < 0:
                # previous match and current match are away matches
                res += D[-1*g1-1, -1*g2-1]
            if g1 < 0 and g2 > 0:
                # previous match was away and the current match is at home
                res += D[ii, -1*g1-1]
            if g1 > 0 and g2 < 0:
                # previous match was at home and current match is away
                res += D[ii, -1*g2-1]
        # all teams travel back home after the last round
        g2 = S[ii, S.shape[1]-1]
        if g2 < 0:
            res += D[ii, -1*g2-1]
    return res

cdef bool is_negative(int val):
    return val < 0

cdef bool is_positive(int val):
    return val > 0

@initializedcheck(False)
@cdivision(True)
@wraparound(False)
@boundscheck(False)
cpdef bool satisfies_home_away_stands(int[:, ::1] S, int max_k):
    cdef int t, sk, s = 0
    cdef int steps = (S.shape[1] - S.shape[1] % (max_k+1)) - 1
    cdef bint helper1, helper2
    for t in range(S.shape[0]):
        for s in range(steps):
            #if all(S[t, s] < 0, S[t, s+1] < 0, ..., S[t, s+max_k] < 0):
            if all_of(&S[t, s], &S[t, s] + max_k + 1, is_negative):
                return False
            #if all(S[t, s] > 0, S[t, s+1] > 0, ..., S[t, s+max_k] > 0):
            if all_of(&S[t, s], &S[t, s] + max_k + 1, is_positive):
                return False
    return True


@wraparound(False)
@boundscheck(False)
cpdef ttp_local_search(int[:, ::1] S, int[:, ::1] D, int max_k):
    cdef int[:, ::1] S_tmp
    cdef int curr_obj_val = ttp_objective(S, D)
    cdef int num_teams = S.shape[0]
    cdef int num_rounds = S.shape[1]
    cdef int ii, jj, ss, ss1, ss2
    cdef int obj_val
    cdef vector[int] best_obj_vals = vector[int](5, curr_obj_val)
    cdef int[:, :, ::1] best_S = np.zeros((5, num_teams, num_rounds), dtype=np.int32)
    
    # search swap_homes
    for ii in range(num_teams):
        for jj in range(ii+1, num_teams):
            S_tmp = swap_homes(S.copy(), ii+1, jj+1)
            obj_val = ttp_objective(S_tmp, D)
            if obj_val < curr_obj_val and obj_val < best_obj_vals[0]:
                if satisfies_home_away_stands(S_tmp, max_k):
                    best_obj_vals[0] = obj_val
                    best_S[0, :, :] = S_tmp.copy()
    
    # search swap_rounds
    for ss1 in range(num_rounds):
        for ss2 in range(ss1+1, num_rounds):
            S_tmp = swap_rounds(S.copy(), ss1+1, ss2+1)
            obj_val = ttp_objective(S_tmp, D)
            if obj_val < curr_obj_val and obj_val < best_obj_vals[1]:
                if satisfies_home_away_stands(S_tmp, max_k):
                    best_obj_vals[1] = obj_val
                    best_S[1, :, :] = S_tmp.copy()
    
    # search swap_teams
    for ii in range(num_teams):
        for jj in range(ii+1, num_teams):
            S_tmp = swap_teams(S.copy(), ii+1, jj+1)
            obj_val = ttp_objective(S_tmp, D)
            if obj_val < curr_obj_val and obj_val < best_obj_vals[2]:
                if satisfies_home_away_stands(S_tmp, max_k):
                    best_obj_vals[2] = obj_val
                    best_S[2, :, :] = S_tmp.copy()
    
    # search partial swap rounds
    for ii in range(num_teams):
        for ss1 in range(num_rounds):
            for ss2 in range(ss1+1, num_rounds):
                S_tmp = partial_swap_rounds(S.copy(), ii+1, ss1+1, ss2+1)
                obj_val = ttp_objective(S_tmp, D)
                if obj_val < curr_obj_val and obj_val < best_obj_vals[3]:
                    if satisfies_home_away_stands(S_tmp, max_k):
                        best_obj_vals[3] = obj_val
                        best_S[3, :, :] = S_tmp.copy()
    
    # search partial_swap_teams
    for ii in range(num_teams):
        for jj in range(ii+1, num_teams):
            for ss in range(num_rounds):
                S_tmp = partial_swap_teams(S.copy(), ii+1, jj+1, ss)
                obj_val = ttp_objective(S_tmp, D)
                if obj_val < curr_obj_val and obj_val < best_obj_vals[4]:
                    if satisfies_home_away_stands(S_tmp, max_k):
                        best_obj_vals[4] = obj_val
                        best_S[4, :, :] = S_tmp.copy()

    # Finished. remove the zeros from best_S (we didn't found new solutions in
    # the corresponding neighborhood)
    bestS = np.array(best_S)
    bestS = bestS[np.all(bestS != 0, axis=(1,2)), :, :]
    return bestS, best_obj_vals
