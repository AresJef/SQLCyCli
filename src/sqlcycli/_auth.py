# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython.dict import PyDict_Size as dict_len  # type: ignore
from cython.cimports.cpython.dict import PyDict_SetItem as dict_setitem  # type: ignore
from cython.cimports.cpython.dict import PyDict_Contains as dict_contains  # type: ignore
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

from typing import Iterator
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
    """Registry and manager for MySQL authentication plugin handlers.

    This class holds and validates handler classes for various MySQL
    authentication methods (native, sha2, sha256, clear-text, dialog, etc.).
    You can provide a custom mapping of plugin names to handler classes,
    or rely on the built-in attributes and setters to register handlers
    individually.
    """

    # . plugins registry
    _plugins: dict[bytes, type]
    # . common plugins
    _mysql_native_password: object
    _caching_sha2_password: object
    _sha256_password: object
    _client_ed25519: object
    _mysql_old_password: object
    _mysql_clear_password: object
    _dialog: object

    def __init__(self, plugins: dict[str | bytes, type] | None = None) -> None:
        """Registry and manager for MySQL authentication plugin handlers.

        This class holds and validates handler classes for various MySQL
        authentication methods (native, sha2, sha256, clear-text, dialog, etc.).
        You can provide a custom mapping of plugin names to handler classes,
        or rely on the built-in attributes and setters to register handlers
        individually.

        :param plugins `<'dict'>`: Optional initial mapping of plugin names
            (str or bytes) to the handler classes. Defaults to `None`.

        Common plugin names:
        ```python
        - "mysql_native_password"
        - "caching_sha2_password"
        - "sha256_password"
        - "client_ed25519"
        - "mysql_old_password"
        - "mysql_clear_password"
        - "dialog"
        ```
        """
        self._plugins = {}
        self._mysql_native_password = None
        self._caching_sha2_password = None
        self._sha256_password = None
        self._client_ed25519 = None
        self._mysql_old_password = None
        self._mysql_clear_password = None
        self._dialog = None

        # Validate and register plugins
        if plugins is None:
            pass
        elif isinstance(plugins, dict):
            for name, handler in plugins.items():
                self.set(name, handler)
        else:
            raise errors.InvalidAuthPluginError(
                "<'%s'>\nAuth 'plugins' must be type of <'dict'>, instead of: %s"
                % (self.__class__.__name__, type(plugins))
            )

    # Property --------------------------------------------------------------------------------
    @property
    def mysql_native_password(self) -> type | None:
        """Handler class for the `mysql_native_password` authentication plugin `<'type'>`."""
        return self._mysql_native_password

    @mysql_native_password.setter
    def mysql_native_password(self, handler: object) -> None:
        self.set(b"mysql_native_password", handler)

    @property
    def caching_sha2_password(self) -> type | None:
        """Handler class for the `caching_sha2_password` authentication plugin `<'type'>`."""
        return self._caching_sha2_password

    @caching_sha2_password.setter
    def caching_sha2_password(self, handler: object) -> None:
        self.set(b"caching_sha2_password", handler)

    @property
    def sha256_password(self) -> type | None:
        """Handler class for the `sha256_password` authentication plugin `<'type'>`."""
        return self._sha256_password

    @sha256_password.setter
    def sha256_password(self, handler: object) -> None:
        self.set(b"sha256_password", handler)

    @property
    def client_ed25519(self) -> type | None:
        """Handler class for the `client_ed25519` authentication plugin `<'type'>`."""
        return self._client_ed25519

    @client_ed25519.setter
    def client_ed25519(self, handler: object) -> None:
        self.set(b"client_ed25519", handler)

    @property
    def mysql_old_password(self) -> type | None:
        """Handler class for the `mysql_old_password` authentication plugin `<'type'>`."""
        return self._mysql_old_password

    @mysql_old_password.setter
    def mysql_old_password(self, handler: object) -> None:
        self.set(b"mysql_old_password", handler)

    @property
    def mysql_clear_password(self) -> type | None:
        """Handler class for the `mysql_clear_password` authentication plugin `<'type'>`."""
        return self._mysql_clear_password

    @mysql_clear_password.setter
    def mysql_clear_password(self, handler: object) -> None:
        self.set(b"mysql_clear_password", handler)

    @property
    def dialog(self) -> type | None:
        """Handler class for the `dialog` authentication plugin `<'type'>`."""
        return self._dialog

    @dialog.setter
    def dialog(self, handler: object) -> None:
        self.set(b"dialog", handler)

    # Handler ---------------------------------------------------------------------------------
    @cython.ccall
    def get(self, plugin_name: str | bytes) -> object:
        """Retrieve the registered handler class for a given plugin name `<'type/None'>`.

        :param plugin_name `<'str/bytes'>`: The name of the authentication plugin.
        :returns `<'type/None'>`: The handler class for the specified plugin, or
            `None` if the plugin is not registered.
        """
        return self._plugins.get(self._validete_plugin_name(plugin_name))

    @cython.ccall
    @cython.exceptval(-1, check=False)
    def set(
        self,
        plugin_name: str | bytes,
        handler: object,
    ) -> cython.bint:
        """Register a handler class under a given plugin name.

        :param plugin_name `<'str/bytes'>`: The name of the authentication plugin.
        :param handler `<'type'>`: The handler class to register.
        :raises `<'InvalidAuthPluginError'>`: If the plugin name or handler is invalid.
        """
        # Register plugins
        name: bytes = self._validete_plugin_name(plugin_name)
        handler = self._validate_plugin_handler(name, handler)
        dict_setitem(self._plugins, name, handler)

        # Common plugins
        if name == b"mysql_native_password":
            self._mysql_native_password = handler
        elif name == b"caching_sha2_password":
            self._caching_sha2_password = handler
        elif name == b"sha256_password":
            self._sha256_password = handler
        elif name == b"client_ed25519":
            self._client_ed25519 = handler
        elif name == b"mysql_old_password":
            self._mysql_old_password = handler
        elif name == b"mysql_clear_password":
            self._mysql_clear_password = handler
        elif name == b"dialog":
            self._dialog = handler
        return True

    # Validate --------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _validete_plugin_name(self, plugin_name: str | bytes) -> object:
        """(internal) Validate and convert plugin name to bytes `<'bytes'>`.

        :param plugin_name `<'str/bytes'>`: The name of the authentication plugin.
        :returns `<'bytes'>`: The validated plugin name.
        :raises `<'InvalidAuthPluginError'>`: If the plugin name is not a valid string or bytes.
        """
        if isinstance(plugin_name, bytes):
            return plugin_name
        elif isinstance(plugin_name, str):
            return utils.encode_str(plugin_name, "ascii")
        else:
            raise errors.InvalidAuthPluginError(
                "<'%s'>\nAuth plugin name (%s %r) is invalid, must be <'str'> or <'bytes'>."
                % (self.__class__.__name__, type(plugin_name), plugin_name)
            )

    @cython.cfunc
    @cython.inline(True)
    def _validate_plugin_handler(
        self,
        pluging_name: str | bytes,
        handler: object,
    ) -> object:
        """(internal) Validate the plugin handler class `<'type'>`.

        :param pluging_name `<'str/bytes'>`: The name of the plugin.
        :param handler `<'type'>`: The handler class to validate.
        :returns `<'type'>`: The validated handler class.
        :raises `<'InvalidAuthPluginError'>`: If the handler is not a valid class.
        """
        if type(handler) is not type:
            if isinstance(pluging_name, bytes):
                pluging_name = utils.decode_bytes_ascii(pluging_name)
            raise errors.InvalidAuthPluginError(
                "<'%s'>\nAuth plugin handler '%s' (%s %r) is invalid, must be <class 'type'>."
                % (self.__class__.__name__, pluging_name, type(handler), handler)
            )
        return handler

    # Special Methods -------------------------------------------------------------------------
    def __repr__(self) -> str:
        if dict_len(self._plugins) == 0:
            return "<%s ()>" % self.__class__.__name__
        else:
            return "<%s (\n\t%s\n)>" % (
                self.__class__.__name__,
                ",\n\t".join("%s=%s" % (k, v) for k, v in self._plugins.items()),
            )

    def __getitem__(self, key: str | bytes) -> type:
        handler = self.get(key)
        if handler is None:
            raise KeyError(
                "<'%s'>\nAuth plugin '%s' is not registered."
                % (self.__class__.__name__, key)
            )
        return handler

    def __contains__(self, plugin_name: str | bytes) -> bool:
        return dict_contains(self._plugins, self._validete_plugin_name(plugin_name))

    def __iter__(self) -> Iterator[type]:
        return iter(self._plugins.values())

    def __bool__(self) -> bool:
        return dict_len(self._plugins) != 0

    def __len__(self) -> int:
        return dict_len(self._plugins)


# Password ------------------------------------------------------------------------------------
@cython.ccall
@cython.boundscheck(True)
def scramble_native_password(password: bytes, salt: bytes) -> bytes:
    """Compute the MySQL `native_password` scramble `<'bytes'>`.

    Performs:
    - 1. SHA1(password) → stage1
    - 2. SHA1(stage1) → stage2
    - 3. SHA1(salt[:20] + stage2) → scramble
    - 4. XOR(stage1, scramble) → result

    :param password `<'bytes'>`: User's plaintext password as UTF-8 encoded bytes.
    :param salt `<'bytes'>`: Server-provided challenge (salt), at least SCRAMBLE_LENGTH bytes.
    :returns `<'bytes'>`: The scrambled password bytes to send to the server.
        Returns empty `bytes` if the `password` is empty.
    """
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
    """Compute the fast-path scramble for `caching_sha2_password` authentication `<'bytes'>`.

    Performs:
    - 1. p1 = SHA256(password)
    - 2. p2 = SHA256(p1)
    - 3. p3 = SHA256(p2 + salt)
    - 4. result = XOR(p1, p3)

    :param password `<'bytes'>`: User's plaintext password as bytes.
    :param salt `<'bytes'>`: Server-provided 20-byte challenge (salt).
    :returns `<'bytes'>`: The scrambled password for fast authentication.
        Returns empty `bytes` if the `password` is empty.
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
    """Encrypt the password for full `sha256_password` or
    full-path `caching_sha2_password` `<'bytes'>`.

    Performs:
    - 1. Append a NUL byte to `password`.
    - 2. XOR the password bytes cyclically with `salt[:20]`.
    - 3. Load the server's RSA public key (PEM) and perform OAEP(SHA1)
         encryption on the XOR'd message.

    :param password `<'bytes'>`: User's password as bytes.
    :param salt `<'bytes'>`: Server-provided challenge (salt).
    :param public_key `<'bytes'>`: PEM-encoded RSA public key sent by the server.
    :returns `<'bytes'>`: The RSA-encrypted password blob to send to the server.

    :raises `RuntimeError`: If the `cryptography` library is not available.
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        raise RuntimeError(
            "The 'cryptography' libarary is required for 'sha256_password' "
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
    """Generate an Ed25519 signature for the `client_ed25519` authentication plugin `<'bytes'>`.

    Performs:
    - 1. Compute SHA512(password) → h (64 bytes)
    - 2. Derive secret scalar s by pruning the first 32 bytes of h.
    - 3. Compute nonce r = SHA512(h[32:] + scramble), reduce to 32-byte scalar.
    - 4. Compute R = r·B, A = s·B (public key).
    - 5. Compute challenge k = SHA512(R‖A‖scramble), reduce to scalar.
    - 6. Compute S = (k·s + r) mod L.
    - 7. Return concatenation R‖S.

    :param password `<'bytes'>`: User's password as bytes.
    :param scramble `<'bytes'>`: Server-provided random scramble bytes.
    :returns `<'bytes'>`: A 64-byte Ed25519 signature (R‖S).
    :raises `RuntimeError`: If the PyNaCl (`nacl`) library is not available.
    """
    if not NACL_AVAILABLE:
        raise RuntimeError(
            "The 'nacl (pynacl)' library is required "
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
