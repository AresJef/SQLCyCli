# cython: language_level=3

cdef:
    unsigned int LONG_PASSWORD
    unsigned int FOUND_ROWS
    unsigned int LONG_FLAG
    unsigned int CONNECT_WITH_DB
    unsigned int NO_SCHEMA
    unsigned int COMPRESS
    unsigned int ODBC
    unsigned int LOCAL_FILES
    unsigned int IGNORE_SPACE
    unsigned int PROTOCOL_41
    unsigned int INTERACTIVE
    unsigned int SSL
    unsigned int IGNORE_SIGPIPE
    unsigned int TRANSACTIONS
    unsigned int SECURE_CONNECTION
    unsigned int MULTI_STATEMENTS
    unsigned int MULTI_RESULTS
    unsigned int PS_MULTI_RESULTS
    unsigned int PLUGIN_AUTH
    unsigned int CONNECT_ATTRS
    unsigned int PLUGIN_AUTH_LENENC_CLIENT_DATA
    unsigned int CAPABILITIES
    
    # Not done yet
    unsigned int HANDLE_EXPIRED_PASSWORDS
    unsigned int SESSION_TRACK
    unsigned int DEPRECATE_EOF
