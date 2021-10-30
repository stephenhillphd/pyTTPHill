# distutils: language = c++
# distutils: extra_compile_args=-std=c++17

cdef int sign(int x)

cpdef swap_homes(int[:, ::1] S_in, int i, int j)
cpdef swap_rounds(int[:, ::1] S_in, int s1, int s2)
cpdef swap_teams(int[:, ::1] S_in, int i, int j)
cpdef partial_swap_rounds(int[:, ::1] S_in, int i, int s1, int s2)
cpdef partial_swap_teams(int[:, ::1] S_in, int i, int j, int s)





