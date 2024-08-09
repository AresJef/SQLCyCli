# cython: language_level=3

cdef:
    unsigned int SERVER_STATUS_IN_TRANS
    unsigned int SERVER_STATUS_AUTOCOMMIT
    unsigned int SERVER_MORE_RESULTS_EXISTS
    unsigned int SERVER_QUERY_NO_GOOD_INDEX_USED
    unsigned int SERVER_QUERY_NO_INDEX_USED
    unsigned int SERVER_STATUS_CURSOR_EXISTS
    unsigned int SERVER_STATUS_LAST_ROW_SENT
    unsigned int SERVER_STATUS_DB_DROPPED
    unsigned int SERVER_STATUS_NO_BACKSLASH_ESCAPES
    unsigned int SERVER_STATUS_METADATA_CHANGED
