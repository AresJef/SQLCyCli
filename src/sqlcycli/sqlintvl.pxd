# cython: language_level=3

# Constant
cdef set INTERVAL_UNITS

# Base class
cdef class SQLInterval:
    cdef:
        str _name
        object _expr
    cpdef str generate(self)
