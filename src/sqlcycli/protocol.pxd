# cython: language_level=3

# MySQL Packet
cdef class MysqlPacket:
    cdef:
        # . raw data
        bytes _data
        char* _data_c
        char* _encoding
        unsigned long long _size
        unsigned long long _pos
        # . packet data
        unsigned long long _affected_rows
        unsigned long long _insert_id
        int _server_status
        unsigned int _warning_count
        bint _has_next
        bytes _message
        bytes _filename
        bytes _plugin_name
        bytes _salt
    # Read Data
    cdef inline bytes read_all_data(self)
    cdef inline bytes read(self, unsigned long long size)
    cdef inline bytes read_remains(self)
    cdef inline unsigned long long read_length_encoded_integer(self)
    cdef inline bytes read_length_encoded_string(self)
    cdef inline unsigned char _read_uint8(self)
    cdef inline unsigned short _read_uint16(self)
    cdef inline unsigned int _read_uint24(self)
    cdef inline unsigned int _read_uint32(self)
    cdef inline unsigned long long _read_uint64(self)
    # Read Packet
    cdef inline bint is_ok_packet(self) except -1
    cpdef bint read_ok_packet(self) except -1
    cdef inline bint is_load_local_packet(self) except -1
    cpdef bint read_load_local_packet(self) except -1
    cdef inline bint is_eof_packet(self) except -1
    cpdef bint read_eof_packet(self) except -1
    cdef inline bint is_auth_switch_request(self) except -1
    cpdef bint read_auth_switch_request(self) except -1
    cdef inline bint is_extra_auth_data(self) except -1
    cdef inline bint is_resultset_packet(self) except -1
    cdef inline bint is_error_packet(self) except -1
    # Curosr
    cdef inline bint advance(self, unsigned long long length) except -1
    cdef inline bint rewind(self, unsigned long long position) except -1
    # Error
    cpdef bint check_error(self) except -1
    cdef inline bint raise_error(self) except -1

cdef class FieldDescriptorPacket(MysqlPacket):
    cdef:
        # . packet data
        bytes _catalog
        str _db
        str _table
        str _table_org
        str _column
        str _column_org
        unsigned int _charsetnr
        unsigned long long _length
        unsigned int _type_code
        unsigned int _flags
        unsigned int _scale
        bint _is_binary
    # Read Packet
    cpdef tuple description(self)
    cdef inline unsigned long long _get_column_length(self)
