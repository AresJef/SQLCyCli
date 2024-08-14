# cython: language_level=3

# Constant
cdef:
    bint SSL_ENABLED_C

# Utils
cpdef bint is_ssl(object obj) except -1
cpdef bint is_ssl_ctx(object obj) except -1

# SSL
cdef class SSL:
    cdef:
        bint _has_ca
        object _ca_file
        object _ca_path
        object _cert_file
        object _cert_key
        object _cert_key_password
        bint _verify_identity
        object _verify_mode
        object _cipher
        object _context
    # Methods
    cdef inline bint _create_ssl_context(self) except -1
    cdef inline object _validate_path(self, object path, str arg_name)
