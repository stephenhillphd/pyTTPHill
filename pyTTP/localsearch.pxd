
from libcpp cimport bool

cpdef int ttp_objective(int[:, ::1] S, int[:, ::1] D)
cdef bool is_negative(int val)
cdef bool is_positive(int val)
cpdef bool satisfies_home_away_stands(int[:, ::1] S, int max_k)
cpdef ttp_local_search(int[:, ::1] S, int[:, ::1] D, int max_k)