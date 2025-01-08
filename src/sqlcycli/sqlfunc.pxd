# cython: language_level=3

# Constant
cdef:
    set TRIM_MODES
    set JSON_VALUE_RETURNING_TYPES

# Base class
cdef class SQLFunction:
    cdef:
        str _name
        int _arg_count
        tuple _args
        str _kwargs
        str _sep
    cdef inline str _validate_kwargs(self, dict kwargs)
    cpdef str generate(self)

cdef class Sentinel:
    pass

cdef Sentinel IGNORED

cdef class RawText:
    cdef:
        str _value
    cpdef str generate(self)
