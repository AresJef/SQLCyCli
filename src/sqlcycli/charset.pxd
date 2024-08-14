# cython: language_level=3

# Charset(s)
cdef class Charset:
    cdef:
        int _id
        str _name
        str _collation
        bytes _encoding
        char* _encoding_c
        bint _is_default
    # Methods
    cpdef bint is_binary(self)

cdef class Charsets:
    cdef: 
        dict _by_id
        dict _by_name
        dict _by_collation
        dict _by_name_n_collation
    # Add Charset
    cpdef bint add(self, Charset charset) except -1
    cdef inline bint _add_by_id(self, Charset charset) except -1
    cdef inline bint _add_by_name(self, Charset charset) except -1
    cdef inline bint _add_by_collation(self, Charset charset) except -1
    cdef inline bint _add_by_name_n_collation(self, Charset charset) except -1
    cdef inline str _gen_namecoll_key(self, str name, str collation)
    # Access Charset
    cpdef Charset by_id(self, object id)
    cpdef Charset by_name(self, str name)
    cpdef Charset by_collation(self, str collation)
    cpdef Charset by_name_n_collation(self, str name, str collation)

# Init
cdef Charsets _charsets
cpdef Charsets all_charsets()
cpdef Charset by_id(object id)
cpdef Charset by_name(str name)
cpdef Charset by_collation(str collation)
cpdef Charset by_name_n_collation(str name, str collation)
