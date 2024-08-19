# cython: language_level=3

# Constants
cdef:
    # . python types
    object NONE
    object DICT_KEYS
    object DICT_VALUES
    object DICT_ITEMS
    object DECIMAL
    object STRUCT_TIME
    # . numpy types
    object FLOAT16
    object FLOAT32
    object FLOAT64
    object INT8
    object INT16
    object INT32
    object INT64
    object UINT8
    object UINT16
    object UINT32
    object UINT64
    object DATETIME64
    object TIMEDELTA64
    object STR_
    object BOOL_
    object BYTES_
    object RECORD
    # . pandas types
    object SERIES
    object DATAFRAME
    object PD_TIMESTAMP
    object PD_TIMEDELTA
    object DATETIMEINDEX
    object TIMEDELTAINDEX
    # . cytimes types
    bint CYTIMES_AVAILABLE
    object PYDT
    object PDDT
