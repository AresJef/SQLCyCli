# cython: language_level=3

# Constant
cdef:
    bint CRYPTOGRAPHY_AVAILABLE_C
    bint NACL_AVAILABLE_C
    int SCRAMBLE_LENGTH

# Auth Plugin
cdef class AuthPlugin:
    cdef:
        # . plugins registry
        dict _plugins
        # . common plugins
        object _mysql_native_password
        object _caching_sha2_password
        object _sha256_password
        object _client_ed25519
        object _mysql_old_password
        object _mysql_clear_password
        object _dialog
    # Handler
    cpdef object get(self, object plugin_name)
    cpdef bint set(self, object plugin_name, object handler) except -1
    # Validate
    cdef inline object _validete_plugin_name(self, object plugin_name)
    cdef inline object _validate_plugin_handler(self, object plugin_name, object handler)

# Password
cpdef bytes scramble_native_password(bytes password, bytes salt)
cpdef bytes scramble_caching_sha2(bytes password, bytes salt)
cpdef bytes sha2_rsa_encrypt(bytes password, bytes salt, bytes public_key)
cpdef bytes ed25519_password(bytes password, bytes scramble)
