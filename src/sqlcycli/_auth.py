# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython.bytes import PyBytes_Size as bytes_len  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_AsString as bytes_to_chars  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_FromStringAndSize as bytes_fr_chars_wlen  # type: ignore
from cython.cimports.cpython.bytearray import PyByteArray_GET_SIZE as bytearray_len  # type: ignore
from cython.cimports.cpython.bytearray import PyByteArray_AS_STRING as bytearray_to_chars  # type: ignore
from cython.cimports.sqlcycli import utils  # type: ignore

# Python imports
try:
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend as _default_backend

    CRYPTOGRAPHY_AVAILABLE: cython.bint = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE: cython.bint = False
# MariaDB's client_ed25519-plugin
# https://mariadb.com/kb/en/library/connection/#client_ed25519-plugin
try:
    from nacl import bindings

    NACL_AVAILABLE: cython.bint = True
except ImportError:
    NACL_AVAILABLE: cython.bint = False

from hashlib import (
    sha1 as _hashlib_sha1,
    sha256 as _hashlib_sha256,
    sha512 as _hashlib_sha512,
)
from sqlcycli import utils, errors

__all__ = [
    "AuthPlugin",
    "scramble_native_password",
    "scramble_caching_sha2",
    "sha2_rsa_encrypt",
    "ed25519_password",
]

# Constants -----------------------------------------------------------------------------------
SCRAMBLE_LENGTH: cython.int = 20


# Auth Plugin ---------------------------------------------------------------------------------
@cython.cclass
class AuthPlugin:
    """Represents the authentication pluging handlers for MySQL."""

    _mysql_native_password: type
    _caching_sha2_password: type
    _sha256_password: type
    _client_ed25519: type
    _mysql_old_password: type
    _mysql_clear_password: type
    _dialog: type
    _plugins: dict[bytes, type]

    def __init__(self, plugins: dict[str | bytes, type] | None = None) -> None:
        """The authentication pluging handlers for MySQL.

        :param plugins `<'dict'>`: The plugin handlers for MySQL authentication,
            where key is the plugin name <'str/bytes'> and value is the handler
            Class <'type'>. Defaults to `None`.
        """
        if plugins is None:
            self._mysql_native_password = None
            self._caching_sha2_password = None
            self._sha256_password = None
            self._client_ed25519 = None
            self._mysql_old_password = None
            self._mysql_clear_password = None
            self._dialog = None
            self._plugins = {}
        elif isinstance(plugins, dict):
            _plugins: dict[bytes, type] = {}
            for name, plugin in plugins.items():
                # . validate plugin name
                if isinstance(name, str):
                    name = utils.encode_str(name, "ascii")
                elif not isinstance(name, bytes):
                    raise errors.InvalidAuthPluginError(
                        "<'%s'>\nAuth plugin name '%s' must be type of <'str/bytes'>, "
                        "instead of: %s" % (self.__class__.__name__, name, type(name))
                    )
                # . validate plugin handler
                if type(plugin) is not type:
                    raise errors.InvalidAuthPluginError(
                        "<'%s'>\nAuth plugin handler '%s' "
                        "must be <class 'type'>, instead of: %s"
                        % (
                            self.__class__.__name__,
                            utils.decode_bytes_ascii(name),
                            type(plugin),
                        )
                    )
                _plugins[name] = plugin
            self._mysql_native_password = _plugins.get(b"mysql_native_password")
            self._caching_sha2_password = _plugins.get(b"caching_sha2_password")
            self._sha256_password = _plugins.get(b"sha256_password")
            self._client_ed25519 = _plugins.get(b"client_ed25519")
            self._mysql_old_password = _plugins.get(b"mysql_old_password")
            self._mysql_clear_password = _plugins.get(b"mysql_clear_password")
            self._dialog = _plugins.get(b"dialog")
            self._plugins = _plugins
        else:
            raise errors.InvalidAuthPluginError(
                "<'%s'>\nAuth 'plugins' must be type of <'dict'>, instead of: %s"
                % (self.__class__.__name__, type(plugins))
            )

    # Property --------------------------------------------------------------------------------
    @property
    def mysql_native_password(self) -> type | None:
        """The handler for 'mysql_native_password' auth plugin `<'type'>`."""
        return self._mysql_native_password

    @mysql_native_password.setter
    def mysql_native_password(self, plugin: type) -> None:
        self._mysql_native_password = plugin
        self._plugins[b"mysql_native_password"] = plugin

    @property
    def caching_sha2_password(self) -> type | None:
        """The handler for 'caching_sha2_password' auth plugin `<'type'>`."""
        return self._caching_sha2_password

    @caching_sha2_password.setter
    def caching_sha2_password(self, plugin: type) -> None:
        self._caching_sha2_password = plugin
        self._plugins[b"caching_sha2_password"] = plugin

    @property
    def sha256_password(self) -> type | None:
        """The handler for 'sha256_password' auth plugin `<'type'>`."""
        return self._sha256_password

    @sha256_password.setter
    def sha256_password(self, plugin: type) -> None:
        self._sha256_password = plugin
        self._plugins[b"sha256_password"] = plugin

    @property
    def client_ed25519(self) -> type | None:
        """The handler for 'client_ed25519' auth plugin `<'type'>`."""
        return self._client_ed25519

    @client_ed25519.setter
    def client_ed25519(self, plugin: type) -> None:
        self._client_ed25519 = plugin
        self._plugins[b"client_ed25519"] = plugin

    @property
    def mysql_old_password(self) -> type | None:
        """The handler for 'mysql_old_password' auth plugin `<'type'>`."""
        return self._mysql_old_password

    @mysql_old_password.setter
    def mysql_old_password(self, plugin: type) -> None:
        self._mysql_old_password = plugin
        self._plugins[b"mysql_old_password"] = plugin

    @property
    def mysql_clear_password(self) -> type | None:
        """The handler for 'mysql_clear_password' auth plugin `<'type'>`."""
        return self._mysql_clear_password

    @mysql_clear_password.setter
    def mysql_clear_password(self, plugin: type) -> None:
        self._mysql_clear_password = plugin
        self._plugins[b"mysql_clear_password"] = plugin

    @property
    def dialog(self) -> type | None:
        """The handler for 'dialog' auth plugin `<'type'>`."""
        return self._dialog

    @dialog.setter
    def dialog(self, plugin: type) -> None:
        self._dialog = plugin
        self._plugins[b"dialog"] = plugin

    # Methods ---------------------------------------------------------------------------------
    @cython.ccall
    def get(self, plugin_name: bytes) -> object:
        """Get the handler for the plugin by name `<'type/None'>`."""
        if plugin_name == b"mysql_native_password":
            return self._mysql_native_password
        if plugin_name == b"caching_sha2_password":
            return self._caching_sha2_password
        if plugin_name == b"sha256_password":
            return self._sha256_password
        if plugin_name == b"client_ed25519":
            return self._client_ed25519
        if plugin_name == b"mysql_old_password":
            return self._mysql_old_password
        if plugin_name == b"mysql_clear_password":
            return self._mysql_clear_password
        if plugin_name == b"dialog":
            return self._dialog
        return self._plugins.get(plugin_name)

    def __repr__(self) -> str:
        if not self._plugins:
            return "<%s(No Handlers)>" % self.__class__.__name__
        else:
            return "<%s(\n  %s)>" % (
                self.__class__.__name__,
                ",\n  ".join("%s=%s" % (k, v) for k, v in self._plugins.items()),
            )

    def __bool__(self) -> bool:
        return bool(self._plugins)


# Password ------------------------------------------------------------------------------------
@cython.ccall
@cython.boundscheck(True)
def scramble_native_password(password: bytes, salt: bytes) -> bytes:
    """Scramble used for mysql_native_password `<'bytes'>"""
    if not password:
        return b""

    stage1: bytes = _hashlib_sha1(password).digest()
    stage2: bytes = _hashlib_sha1(stage1).digest()
    s = _hashlib_sha1()
    s.update(salt[0:SCRAMBLE_LENGTH])
    s.update(stage2)
    res: bytes = s.digest()

    msg1: cython.pchar = bytes_to_chars(res)
    msg2: cython.pchar = bytes_to_chars(stage1)
    length: cython.Py_ssize_t = bytes_len(res)
    i: cython.Py_ssize_t
    for i in range(length):
        msg1[i] ^= msg2[i]
    return bytes_fr_chars_wlen(msg1, length)


@cython.ccall
def scramble_caching_sha2(password: bytes, salt: bytes) -> bytes:
    """Scramble algorithm used in cached_sha2_password fast path `<'bytes'>`.

    XOR(SHA256(password), SHA256(SHA256(SHA256(password)), nonce))
    """
    if not password:
        return b""

    p1: bytes = _hashlib_sha256(password).digest()
    p2: bytes = _hashlib_sha256(p1).digest()
    p3: bytes = _hashlib_sha256(p2 + salt).digest()

    msg1: cython.pchar = bytes_to_chars(p1)
    msg2: cython.pchar = bytes_to_chars(p3)
    length: cython.Py_ssize_t = bytes_len(p3)
    i: cython.Py_ssize_t
    for i in range(length):
        msg1[i] ^= msg2[i]
    return bytes_fr_chars_wlen(msg1, length)


@cython.ccall
@cython.boundscheck(True)
def sha2_rsa_encrypt(password: bytes, salt: bytes, public_key: bytes) -> bytes:
    """Encrypt password with salt and public_key `<'bytes'>`.

    Used for sha256_password and caching_sha2_password.
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        raise RuntimeError(
            "The 'cryptography' package is required for 'sha256_password' "
            "or 'caching_sha2_password' authentication."
        )

    # xor_password
    # Trailing NUL character will be added in Auth Switch Request.
    # See https://github.com/mysql/mysql-server/blob/7d10c82196c8e45554f27c00681474a9fb86d137/sql/auth/sha2_password.cc#L939-L945
    pswd: bytearray = bytearray(password) + b"\0"
    pswd_len: cython.Py_ssize_t = bytearray_len(pswd)
    msg1: cython.pchar = bytearray_to_chars(pswd)
    salt: bytes = salt[0:SCRAMBLE_LENGTH]
    msg2: cython.pchar = bytes_to_chars(salt)
    salt_len: cython.Py_ssize_t = bytes_len(salt)
    i: cython.Py_ssize_t
    for i in range(pswd_len):
        msg1[i] ^= msg2[i % salt_len]
    message: bytes = bytes_fr_chars_wlen(msg1, pswd_len)

    # rsa encryption
    rsa_key = serialization.load_pem_public_key(public_key, _default_backend())
    return rsa_key.encrypt(
        message,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()),
            algorithm=hashes.SHA1(),
            label=None,
        ),
    )


@cython.ccall
@cython.boundscheck(True)
def ed25519_password(password: bytes, scramble: bytes) -> bytes:
    """Sign a random scramble with elliptic curve Ed25519 `<'bytes'>`.

    Secret and public key are derived from password.
    """
    if not NACL_AVAILABLE:
        raise RuntimeError(
            "The 'nacl (pynacl)' package is required "
            "for 'client_ed25519' authentication."
        )
    # h = SHA512(password)
    h: bytes = _hashlib_sha512(password).digest()

    # s = prune(first_half(h))
    s32: bytearray = bytearray(h[0:32])
    ba: cython.pchar = bytearray_to_chars(s32)
    ba0: bytes = utils.pack_uint8(ba[0] & 248)
    ba31: bytes = utils.pack_uint8((ba[31] & 127) | 64)
    ba_m: bytes = ba[1:31]
    s: bytes = ba0 + ba_m + ba31

    # r = SHA512(second_half(h) || M)
    length = bytes_len(h)
    r = _hashlib_sha512(h[32:length] + scramble).digest()

    # R = encoded point [r]B
    r = bindings.crypto_core_ed25519_scalar_reduce(r)
    R = bindings.crypto_scalarmult_ed25519_base_noclamp(r)

    # A = encoded point [s]B
    A = bindings.crypto_scalarmult_ed25519_base_noclamp(s)

    # k = SHA512(R || A || M)
    k = _hashlib_sha512(R + A + scramble).digest()

    # S = (k * s + r) mod L
    k = bindings.crypto_core_ed25519_scalar_reduce(k)
    ks = bindings.crypto_core_ed25519_scalar_mul(k, s)
    S = bindings.crypto_core_ed25519_scalar_add(ks, r)

    # signature = R || S
    return R + S
