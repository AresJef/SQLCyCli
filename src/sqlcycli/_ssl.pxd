# cython: language_level=3

# Constant
cdef:
    object CERT_NONE, CERT_REQUIRED, CERT_OPTIONAL
    bint SSL_ENABLED_C

# Utils
cpdef bint is_ssl(object obj) except -1
cpdef bint is_ssl_ctx(object obj) except -1

# SSL
cdef class SSL:
    cdef:
        bint _has_ca, _verify_identity
        object _ca_file, _ca_path
        object _cert_file, _cert_key, _cert_key_password
        object _verify_mode, _cipher, _context
    # Methods
    cdef inline bint _create_ssl_context(self) except -1
    cdef inline object _validate_path(self, object path, str arg_name)
