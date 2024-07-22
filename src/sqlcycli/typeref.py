# cython: language_level=3

# Python imports
import numpy as np
from decimal import Decimal
from time import struct_time
from pandas import Series, DataFrame
from pandas import Timestamp, DatetimeIndex
from pandas import Timedelta, TimedeltaIndex
from cytimes import pddt, pydt

# Constants -------------------------------------------------------------------------
# . python types
NONE: type = type(None)
DICT_KEYS: type = type(dict().keys())
DICT_VALUES: type = type(dict().values())
DECIMAL: type[Decimal] = Decimal
STRUCT_TIME: type[struct_time] = struct_time
# . numpy types
FLOAT16: type[np.float16] = np.float16
FLOAT32: type[np.float32] = np.float32
FLOAT64: type[np.float64] = np.float64
INT8: type[np.int8] = np.int8
INT16: type[np.int16] = np.int16
INT32: type[np.int32] = np.int32
INT64: type[np.int64] = np.int64
UINT8: type[np.uint8] = np.uint8
UINT16: type[np.uint16] = np.uint16
UINT32: type[np.uint32] = np.uint32
UINT64: type[np.uint64] = np.uint64
DATETIME64: type[np.datetime64] = np.datetime64
TIMEDELTA64: type[np.timedelta64] = np.timedelta64
STR_: type[np.str_] = np.str_
BOOL_: type[np.bool_] = np.bool_
BYTES_: type[np.bytes_] = np.bytes_
RECORD: type[np.record] = np.record
# . pandas types
DATAFRAME: type[DataFrame] = DataFrame
SERIES: type[Series] = Series
TIMESTAMP: type[Timestamp] = Timestamp
DATETIMEINDEX: type[DatetimeIndex] = DatetimeIndex
TIMEDELTA: type[Timedelta] = Timedelta
TIMEDELTAINDEX: type[TimedeltaIndex] = TimedeltaIndex
# . cytimes
PDDT: type[pddt] = pddt
PYDT: type[pydt] = pydt
