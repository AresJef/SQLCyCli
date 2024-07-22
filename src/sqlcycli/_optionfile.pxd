# cython: language_level=3
from sqlcycli._ssl cimport SSL # type: ignore

# MysqlOption
cdef class OptionFile:
    cdef:
        object _opt_file
        int _port
        str _opt_group, _host, _user, _password, _database 
        str _charset, _bind_address, _unix_socket, _max_allowed_packet
        SSL _ssl
    # Methods
    cdef inline bint _load_options(self) except -1
    cdef inline object _access_option(self, object cfg, str value, object default)
    cdef inline object _validate_path(self, object path, str arg_name)
