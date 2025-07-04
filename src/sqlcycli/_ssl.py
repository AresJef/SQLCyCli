# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython

# Pyhton imports
import os, warnings
from os import PathLike
from typing import Literal
from sqlcycli import errors

__all__ = ["SSL_ENABLED", "SSL", "is_ssl", "is_ssl_ctx"]


# Constants -----------------------------------------------------------------------------------
try:
    import ssl as _py_ssl

    SSL_ENABLED: bool = True
except ImportError:
    SSL_ENABLED: bool = False
SSL_ENABLED_C: cython.bint = SSL_ENABLED


# Utils ---------------------------------------------------------------------------------------
@cython.ccall
@cython.exceptval(-1, check=False)
def is_ssl(obj: object) -> cython.bint:
    """Determine if the pass-in object is a `sqlcycli.SSL` configuration. `<'bool'>`."""
    return isinstance(obj, SSL) if SSL_ENABLED_C else False


@cython.ccall
@cython.exceptval(-1, check=False)
def is_ssl_ctx(obj: object) -> cython.bint:
    """Determine if the pass-in object is a `ssl.SSLContext` configuration. `<'bool'>`."""
    return isinstance(obj, _py_ssl.SSLContext) if SSL_ENABLED_C else False


# SSL -----------------------------------------------------------------------------------------
@cython.cclass
class SSL:
    """Configure and produce an SSLContext for MySQL connections.

    Handles CA certificates, client cert/key, verification modes, and cipher
    suites. Access the resulting `ssl.SSLContext` via the `context` property.
    If Python's `ssl` module is unavailable, a warning is issued and `context`
    property will always be `None`.
    """

    _has_ca: cython.bint
    _ca_file: object
    _ca_path: object
    _cert_file: object
    _cert_key: object
    _cert_key_password: object
    _verify_identity: cython.bint
    _verify_mode: object
    _cipher: object
    _context: object

    def __init__(
        self,
        ca_file: str | bytes | PathLike | None = None,
        ca_path: str | bytes | PathLike | None = None,
        cert_file: str | bytes | PathLike | None = None,
        cert_key: str | bytes | PathLike | None = None,
        cert_key_password: str | bytes | bytearray | None = None,
        verify_identity: bool = True,
        verify_mode: bool | Literal["Required", "Optional", "None"] | None = None,
        cipher: str | None = None,
    ) -> None:
        """Configure and produce an SSLContext for MySQL connections.

        Handles CA certificates, client cert/key, verification modes, and cipher
        suites. Access the resulting `ssl.SSLContext` via the `context` property.
        If Python's `ssl` module is unavailable, a warning is issued and `context`
        property will always be `None`.

        :param ca_file `<'str/bytes/Path'>`: Path to a PEM-formatted CA certificate file.. Defaults to `None`.
        :param ca_path `<'str/bytes/Path'>`: Path to a directory of CA certificate files. Defaults to `None`.
        :param cert_file: `<'str/bytes/Path'>`: Path to a PEM-formatted client certificate file. Defaults to `None`.
        :param cert_key: `<'str/bytes/Path'>`: Path to a PEM-formatted private key for the client certificate. Defaults to `None`.
        :param cert_key_password: `<'str/bytes/bytearray'>`: Password for the private key, if encrypted.. Defaults to `None`.
        :param verify_server_identity: `<'bool'>`: Whether to verify the server's hostname against its certificate. Defaults to `False`.
        :param verify_server_mode: `<'bool/str'>`: Certificate verification mode. Defaults to `None`.
        :param cipher: `<'str'>`: OpenSSL cipher string to restrict the cipher suites. Defaults to `None`.
        """
        self._has_ca = ca_file is not None or ca_path is not None
        self._ca_file = self._validate_path(ca_file, "ca_file")
        self._ca_path = self._validate_path(ca_path, "ca_path")
        self._cert_file = self._validate_path(cert_file, "cert_file")
        self._cert_key = self._validate_path(cert_key, "cert_key")
        self._cert_key_password = cert_key_password
        self._verify_identity = self._has_ca and bool(verify_identity)
        self._verify_mode = verify_mode
        self._cipher = cipher
        if SSL_ENABLED_C:
            try:
                self._create_ssl_context()
            except Exception as err:
                raise errors.InvalidSSLConfigError(
                    "<'%s'>\nSSL settings is invalid.\nError: %s"
                    % (self.__class__.__name__, err)
                ) from err
        else:
            self._context = None
            warnings.warn(
                "<'%s'> Python 'ssl' module is not available."
                % self.__class__.__name__,
                RuntimeWarning,
            )

    # Property --------------------------------------------------------------------------------
    @property
    def context(self) -> object | None:
        """Retrieve the configured SSLContext `<'SSLContext/None'>`."""
        return self._context

    # Internal --------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _create_ssl_context(self) -> cython.bint:
        """(Internal) Build the SSLContext according to current parameters.

        Loads CA certificates, sets verification policies,
        client cert/key, and cipher suites.
        """
        # . ca certificate
        context = _py_ssl.create_default_context(
            cafile=self._ca_file, capath=self._ca_path
        )
        # . check hostname
        context.check_hostname = self._verify_identity
        # . verify mode
        if self._verify_mode is None:
            if self._has_ca:
                context.verify_mode = _py_ssl.CERT_REQUIRED
                self._verify_mode = "Required"
            else:
                context.verify_mode = _py_ssl.CERT_NONE
                self._verify_mode = "None"
        elif isinstance(self._verify_mode, bool):
            if self._verify_mode:
                context.verify_mode = _py_ssl.CERT_REQUIRED
                self._verify_mode = "Required"
            else:
                context.verify_mode = _py_ssl.CERT_NONE
                self._verify_mode = "None"
        else:
            if isinstance(self._verify_mode, str):
                self._verify_mode = self._verify_mode.lower()
            if self._verify_mode in ("none", "0", "false", "no"):
                context.verify_mode = _py_ssl.CERT_NONE
                self._verify_mode = "None"
            elif self._verify_mode == "optional":
                context.verify_mode = _py_ssl.CERT_OPTIONAL
                self._verify_mode = "Optional"
            elif self._has_ca or self._verify_mode in ("required", "1", "true", "yes"):
                context.verify_mode = _py_ssl.CERT_REQUIRED
                self._verify_mode = "Required"
            else:
                context.verify_mode = _py_ssl.CERT_NONE
                self._verify_mode = "None"
        # . client certificate
        if self._cert_file is not None:
            context.load_cert_chain(
                certfile=self._cert_file,
                keyfile=self._cert_key,
                password=self._cert_key_password,
            )
        # . cipher
        if self._cipher is not None:
            context.set_ciphers(self._cipher)
        # . set context
        self._context = context
        return True

    @cython.cfunc
    @cython.inline(True)
    def _validate_path(self, path: object, arg_name: str) -> object:
        """(Internal) Expand and verify a filesystem path `<'str/PathLike'>`.

        Expands `~` and `~user`, then checks that the path exists.

        :param path `<'str/bytes/PathLike'>`: The path to validate.
        :param arg_name `<'str'>`: Name of the parameter for error messages.
        """
        if path is None:
            return None
        try:
            path = os.path.expanduser(path)
        except Exception as err:
            raise errors.InvalidSSLConfigError(
                "<'%s'>\nPath for '%s' is invalid: '%s'.\n"
                "Error: %s" % (self.__class__.__name__, arg_name, path, err)
            ) from err
        if not os.path.exists(path):
            raise errors.SSLConfigFileNotFoundError(
                "<'%s'>\nPath for '%s' does not exist: '%s'."
                % (self.__class__.__name__, arg_name, path)
            )
        return path

    # Special Methods -------------------------------------------------------------------------
    def __repr__(self) -> str:
        if self._context is None:
            return "<%s(SSL Disabled)>." % self.__class__.__name__
        reprs = {
            "has_ca": self._has_ca,
            "ca_file": self._ca_file,
            "ca_path": self._ca_path,
            "cert_file": self._cert_file,
            "cert_key": self._cert_key,
            "cert_key_password": self._cert_key_password,
            "verify_identity": self._verify_identity,
            "verify_mode": self._verify_mode,
            "cipher": self._cipher,
        }
        # fmt: off
        return "<%s(\n  %s)>" % (
            self.__class__.__name__,
            ",\n  ".join("%s=%r" % (k, None if v is None else v) for k, v in reprs.items()),
        )
        # fmt: on

    def __bool__(self) -> bool:
        return self._context is not None and (
            self._has_ca or self._cert_file is not None
        )
