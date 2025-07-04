from os import PathLike
from typing import Any, Literal

# Utils ---------------------------------------------------------------------------------------
def is_ssl(obj: Any) -> bool: ...
def is_ssl_ctx(obj: Any) -> bool: ...

# SSL -----------------------------------------------------------------------------------------
class SSL:
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
    ) -> None: ...
    # Property --------------------------------------------------------------------------------
    def context(self) -> object | None: ...
    # Special Methods -------------------------------------------------------------------------
    def __repr__(self) -> str: ...
    def __bool__(self) -> bool: ...
