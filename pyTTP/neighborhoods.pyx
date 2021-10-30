# distutils: language = c++
# distutils: extra_compile_args=-std=c++17

from cython cimport wraparound, boundscheck, initializedcheck
import numpy as np
cimport numpy as cnp

cnp.import_array()
from libcpp.vector cimport vector

# Necessary for Cython 0.29.x
cdef extern from "<algorithm>" namespace "std":
    Iter find[Iter, T](Iter first, Iter last, const T& value)


@initializedcheck(False)
@wraparound(False)
@boundscheck(False)
cpdef swap_homes(int[:, ::1] S_in, int i, int j):
    cdef int[:, ::1] S = S_in.copy()
    cdef int s, g1, g2, N = S.shape[1]
    for s in range(N):
        g1 = S[i-1, s]
        g2 = S[j-1, s]
        if abs(g1) == j:
            S[i-1, s] = -1*g1
        if abs(g2) == i:
            S[j-1, s] = -1*g2
    return np.array(S, copy=False)


@initializedcheck(False)
@wraparound(False)
@boundscheck(False)
cpdef swap_rounds(int[:, ::1] S_in, int s1, int s2):
    cdef int[:, ::1] S = S_in.copy()
    cdef int t, tmp, num_teams = S.shape[0]
    for t in range(num_teams):
        tmp = S[t, s1-1]
        S[t, s1-1] = S[t, s2-1]
        S[t, s2-1] = tmp
    return np.array(S, copy=False)


@initializedcheck(False)
@wraparound(False)
@boundscheck(False)
cpdef swap_teams(int[:, ::1] S_in, int i, int j):
    cdef int[:, ::1] S = S_in.copy()
    cdef int s, g1, g2, tmp, N = S.shape[1]
    for s in range(N):
        if abs(S[i-1, s]) != abs(j):
            g1 = abs(S[i-1, s]) - 1
            g2 = abs(S[j-1, s]) - 1
            # Swap values
            tmp = S[j-1, s]
            S[j-1, s] = S[i-1, s]
            S[i-1, s] = tmp 
            # fix home/away pattern
            if S[g1, s] < 0:
                S[g1, s] = -1*abs(S_in[g2, s])
            else:
                S[g1, s] = abs(S_in[g2, s])
            if S[g2, s] < 0:
                S[g2, s] = -1*abs(S_in[g1, s])
            else:
                S[g2, s] = abs(S_in[g1, s])
    return np.array(S, copy=False)


@initializedcheck(False)
@wraparound(False)
@boundscheck(False)
cpdef partial_swap_rounds(int[:, ::1] S_in, int i, int s1, int s2):
    cdef int[:, ::1] S = S_in.copy()
    cdef int t, kandidat, links, rechts, tmp
    cdef vector[int] L = vector[int](1, i)
    cdef vector[int] K = vector[int](2, 0)
    cdef size_t ii = 0
    
    K[0] = abs(S[i - 1, s1 - 1])
    K[1] = abs(S[i - 1, s2 - 1])

    while ii < K.size():
        kandidat = K[ii]
        # if kandidat not in L
        if find(L.begin(), L.end(), kandidat) == L.end():
            L.push_back(kandidat)
        links = abs(S[kandidat-1, s1-1])
        rechts = abs(S[kandidat-1, s2-1])
        # is links not in L and not in K?
        if find(L.begin(), L.end(), links) == L.end() and find(K.begin(), K.end(), links) == K.end():
            K.push_back(links)
        # is rechts not in L and not in K?
        if find(L.begin(), L.end(), rechts) == L.end() and find(K.begin(), K.end(), rechts) == K.end():
            K.push_back(rechts)
        ii += 1
    # swap the rows
    for t in L:
        tmp = S[t-1, s1-1]
        S[t-1, s1-1] = S[t-1, s2-1]
        S[t-1, s2-1] = tmp
    return np.array(S, copy=False)


cdef int sign(int x):
    return (x > 0) - (x < 0)


@initializedcheck(False)
@wraparound(False)
@boundscheck(False)
cpdef partial_swap_teams(int[:, ::1] S_in, int i, int j, int s):
    cdef int[:, ::1] S = S_in.copy()
    cdef vector[int] L = vector[int](S.shape[1], 0)
    cdef vector[int] K = vector[int](S.shape[1], 0)
    cdef int a, b, q, r, g1, g2, tmp
    cdef size_t ii, ss

    L[0] = s-1
    K[0] = s-1

    for ii in range(K.size()):
        q = K[ii]
        a = S[i-1, q]
        b = S[j-1, q]
        # find r such that S[i-1, r] = abs(b) or S[j-1, r] = abs(a)
        for r in range(S.shape[1]):
            if abs(S[i-1, r]) == abs(b) or abs(S[j-1, r]) == abs(a):
                # if r not in L and not in K
                if find(L.begin(), L.end(), r) == L.end() and find(K.begin(), K.end(), r) == K.end():
                    L[ii] = r
                    K[ii] = r
    # swap the columns
    for ss in range(L.size()):
        g1 = abs(S[i-1, ss])
        g2 = abs(S[j-1, ss])
        S[i-1, ss] = sign(S[i-1, ss]) * g2
        S[j-1, ss] = sign(S[j-1, ss]) * g1
        tmp = S[g1-1, ss]
        S[g1-1, ss] = S[g2-1, ss]
        S[g2-1, ss] = tmp
    return np.array(S, copy=False) 





