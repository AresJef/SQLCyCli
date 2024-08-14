# cython: language_level=3
from sqlcycli._ssl cimport SSL # type: ignore

# MysqlOption
cdef class OptionFile:
    cdef:
        # . file
        object _opt_file
        str _opt_group
        # . basic
        str _host
        int _port
        str _user
        str _password
        str _database 
        # . charset
        str _charset
        # . client
        str _bind_address
        str _unix_socket
        str _max_allowed_packet
        # . ssl
        SSL _ssl
    # Methods
    cdef inline bint _load_options(self) except -1
    cdef inline object _access_option(self, object cfg, str value, object default)
    cdef inline object _validate_path(self, object path, str arg_name)
