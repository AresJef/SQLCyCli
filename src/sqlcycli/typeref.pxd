# cython: language_level=3

# Constants
cdef:
    # . python types
    object NONE, DICT_KEYS, DICT_VALUES
    object DECIMAL, STRUCT_TIME
    # . numpy types
    object FLOAT16, FLOAT32, FLOAT64
    object INT8, INT16, INT32, INT64
    object UINT8, UINT16, UINT32, UINT64
    object DATETIME64, TIMEDELTA64
    object STR_, BOOL_, BYTES_, RECORD
    # . pandas types
    object DATAFRAME, SERIES
    object TIMESTAMP, DATETIMEINDEX
    object TIMEDELTA, TIMEDELTAINDEX
    # . cytimes
    object PDDT, PYDT
    