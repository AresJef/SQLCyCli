# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.sqlcycli._ssl import SSL_ENABLED_C, SSL  # type: ignore


# Python imports
import os
from os import PathLike
from configparser import RawConfigParser
from sqlcycli import errors
from sqlcycli._ssl import SSL

__all__ = ["OptionFile"]


# Parser --------------------------------------------------------------------------------------
class _Parser(RawConfigParser):
    """Custom parser for MySQL option file."""

    def __init__(self, **kwargs):
        kwargs["allow_no_value"] = True
        RawConfigParser.__init__(self, **kwargs)

    @cython.wraparound(True)
    def __remove_quotes(self, value):
        quotes = ["'", '"']
        for quote in quotes:
            if len(value) >= 2 and value[0] == value[-1] == quote:
                return value[1:-1]
        return value

    def optionxform(self, key: str):
        return key.lower().replace("_", "-")

    def get(self, section, option):
        value = RawConfigParser.get(self, section, option)
        return self.__remove_quotes(value)


# MysqlOption ---------------------------------------------------------------------------------
@cython.cclass
class OptionFile:
    """Represents the configuration form MySQL option file (my.cnf or my.ini).

    It takes the responsibility of reading local MySQL option file
    from the 'PyMySQL' package's <'Connection'> class.
    """

    # File
    _opt_file: object
    _opt_group: str
    # Basic
    _host: str
    _port: cython.int
    _user: str
    _password: str
    _database: str
    # Charset
    _charset: str
    # Client
    _bind_address: str
    _unix_socket: str
    _max_allowed_packet: str
    # SSL
    _ssl: SSL

    def __init__(
        self,
        opt_file: str | bytes | PathLike,
        opt_group: str = "client",
    ) -> None:
        """the configuration form MySQL option file (my.cnf or my.ini).

        It takes the responsibility of reading local MySQL option file
        from the 'PyMySQL' package's <'Connection'> class.

        :param opt_file `<'str/bytes/Path'>`: The path to the MySQL option file (my.cnf or my.ini).
        :param opt_group `<'str'>`: The 'group' to read from the MySQL option file. Defaults to `'client'`.
        """
        self._opt_file = self._validate_path(opt_file, "opt_file")
        self._opt_group = opt_group
        # Load options
        try:
            self._load_options()
        except Exception as err:
            raise errors.InvalidOptionFileError(
                "<'%s'>\nFailed to load MySQL option file '%s'.\n"
                "Error: %s" % (self.__class__.__name__, self._opt_file, err)
            ) from err

    # Property --------------------------------------------------------------------------------
    @property
    def opt_file(self) -> str | bytes | PathLike:
        """The path to the MySQL option file `<'str/bytes/Path'>`."""
        return self._opt_file

    @property
    def opt_group(self) -> str:
        """The 'group' to read from the MySQL option file `<'str'>`."""
        return self._opt_group

    @property
    def host(self) -> str:
        """The 'host' from the MySQL option `<'str'>`."""
        return self._host

    @property
    def port(self) -> int:
        """The 'port' from the MySQL option `<'int'>`."""
        return self._port if self._port != -1 else None

    @property
    def user(self) -> str:
        """The 'user' from the MySQL option `<'str'>`."""
        return self._user

    @property
    def password(self) -> str:
        """The 'password' from the MySQL option `<'str'>`."""
        return self._password

    @property
    def database(self) -> str:
        """The 'database' from the MySQL option `<'str'>`."""
        return self._database

    @property
    def charset(self) -> str:
        """The 'default-character-set' from the MySQL option `<'str'>`."""
        return self._charset

    @property
    def bind_address(self) -> str:
        """The 'bind-address' from the MySQL option `<'str'>`."""
        return self._bind_address

    @property
    def unix_socket(self) -> str:
        """The 'socket' from the MySQL option `<'str'>`."""
        return self._unix_socket

    @property
    def max_allowed_packet(self) -> str:
        """The 'max-allowed-packet' from the MySQL option `<'str'>`."""
        return self._max_allowed_packet

    @property
    def ssl(self) -> SSL:
        """The 'SSL' from the MySQL option `<'SSL'>`.

        ## Notice
        If Python `ssl` module is not available, returns `None`.
        """
        return self._ssl

    # Methods ---------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _load_options(self) -> cython.bint:
        """(cfunc) Load settings from MySQL option file."""
        # Parse options
        cfg = _Parser()
        cfg.read(self._opt_file)
        if not cfg.has_section(self._opt_group):
            raise ValueError(
                "MySQL option file does not contain '%s' group." % self._opt_group
            )
        # . basic
        self._host = self._access_option(cfg, "host", None)
        port: cython.int = int(self._access_option(cfg, "port", -1))
        self._port = port if port >= 0 else -1
        self._user = self._access_option(cfg, "user", None)
        self._password = self._access_option(cfg, "password", None)
        self._database = self._access_option(cfg, "database", None)
        # . charset
        self._charset = self._access_option(cfg, "default-character-set", None)
        # . client
        self._bind_address = self._access_option(cfg, "bind-address", None)
        self._unix_socket = self._access_option(cfg, "socket", None)
        self._max_allowed_packet = self._access_option(cfg, "max-allowed-packet", None)
        # . ssl
        if SSL_ENABLED_C:
            ssl = SSL(
                self._access_option(cfg, "ssl-ca", None),
                self._access_option(cfg, "ssl-capath", None),
                self._access_option(cfg, "ssl-cert", None),
                self._access_option(cfg, "ssl-key", None),
                self._access_option(cfg, "ssl-password", None),
                True,
                self._access_option(cfg, "ssl-verify-cert", None),
                self._access_option(cfg, "ssl-cipher", None),
            )
            self._ssl = ssl if ssl else None
        else:
            self._ssl = None
        # Success
        return True

    @cython.cfunc
    @cython.inline(True)
    def _access_option(self, cfg: _Parser, value: str, default: object) -> object:
        """(cfunc) Access the settings in the MySQL options `<'object'>`"""
        try:
            return cfg.get(self._opt_group, value)
        except Exception:
            return default

    @cython.cfunc
    @cython.inline(True)
    def _validate_path(self, path: object, arg_name: str) -> object:
        """(cfunc) Expand '~' and '~user' constructions and validate 
        path existence. If user or $HOME is unknown, do nothing. 
        Only applies to <'str'> or <'Path'> objects.
        """
        if path is None:
            return None
        try:
            path = os.path.expanduser(path)
        except Exception as err:
            raise errors.InvalidOptionFileError(
                "<'%s'>\nPath for '%s' is invalid: '%s'.\n"
                "Error: %s" % (self.__class__.__name__, arg_name, path, err)
            ) from err
        if not os.path.exists(path):
            raise errors.OptionFileNotFoundError(
                "<'%s'>\nPath for '%s' does not exist: '%s'."
                % (self.__class__.__name__, arg_name, path)
            )
        return path

    def __repr__(self) -> str:
        reprs = {
            "opt_file": self._opt_file,
            "opt_group": self._opt_group,
            "host": self._host,
            "port": self._port if self._port != -1 else None,
            "user": self._user,
            "password": self._password,
            "database": self._database,
            "charset": self._charset,
            "bind_address": self._bind_address,
            "unix_socket": self._unix_socket,
            "max_allowed_packet": self._max_allowed_packet,
            "ssl": self._ssl,
        }
        # fmt: off
        return "<%s(\n  %s)>" % (
            self.__class__.__name__,
            ",\n  ".join("%s=%r" % (k, None if v is None else v) for k, v in reprs.items())
        )
        # fmt: on
