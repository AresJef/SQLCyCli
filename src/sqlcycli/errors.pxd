# cython: language_level=3

# Mysql Exceptions
cdef:
    dict MYSQL_ERROR_MAP

cpdef bint raise_mysql_exception(char* data, unsigned long long size) except -1