# cython: language_level=3
import cython
from sqlcycli.constants import COMMAND

COM_SLEEP: cython.uint = COMMAND.COM_SLEEP
COM_QUIT: cython.uint = COMMAND.COM_QUIT
COM_INIT_DB: cython.uint = COMMAND.COM_INIT_DB
COM_QUERY: cython.uint = COMMAND.COM_QUERY
COM_FIELD_LIST: cython.uint = COMMAND.COM_FIELD_LIST
COM_CREATE_DB: cython.uint = COMMAND.COM_CREATE_DB
COM_DROP_DB: cython.uint = COMMAND.COM_DROP_DB
COM_REFRESH: cython.uint = COMMAND.COM_REFRESH
COM_SHUTDOWN: cython.uint = COMMAND.COM_SHUTDOWN
COM_STATISTICS: cython.uint = COMMAND.COM_STATISTICS
COM_PROCESS_INFO: cython.uint = COMMAND.COM_PROCESS_INFO
COM_CONNECT: cython.uint = COMMAND.COM_CONNECT
COM_PROCESS_KILL: cython.uint = COMMAND.COM_PROCESS_KILL
COM_DEBUG: cython.uint = COMMAND.COM_DEBUG
COM_PING: cython.uint = COMMAND.COM_PING
COM_TIME: cython.uint = COMMAND.COM_TIME
COM_DELAYED_INSERT: cython.uint = COMMAND.COM_DELAYED_INSERT
COM_CHANGE_USER: cython.uint = COMMAND.COM_CHANGE_USER
COM_BINLOG_DUMP: cython.uint = COMMAND.COM_BINLOG_DUMP
COM_TABLE_DUMP: cython.uint = COMMAND.COM_TABLE_DUMP
COM_CONNECT_OUT: cython.uint = COMMAND.COM_CONNECT_OUT
COM_REGISTER_SLAVE: cython.uint = COMMAND.COM_REGISTER_SLAVE
COM_STMT_PREPARE: cython.uint = COMMAND.COM_STMT_PREPARE
COM_STMT_EXECUTE: cython.uint = COMMAND.COM_STMT_EXECUTE
COM_STMT_SEND_LONG_DATA: cython.uint = COMMAND.COM_STMT_SEND_LONG_DATA
COM_STMT_CLOSE: cython.uint = COMMAND.COM_STMT_CLOSE
COM_STMT_RESET: cython.uint = COMMAND.COM_STMT_RESET
COM_SET_OPTION: cython.uint = COMMAND.COM_SET_OPTION
COM_STMT_FETCH: cython.uint = COMMAND.COM_STMT_FETCH
COM_DAEMON: cython.uint = COMMAND.COM_DAEMON
COM_BINLOG_DUMP_GTID: cython.uint = COMMAND.COM_BINLOG_DUMP_GTID
COM_END: cython.uint = COMMAND.COM_END
