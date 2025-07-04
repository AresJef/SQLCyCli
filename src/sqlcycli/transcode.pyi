# Escape --------------------------------------------------------------------------------------
def escape(
    data: object,
    many: bool = False,
    itemize: bool = True,
) -> str | tuple[str] | list[str | tuple[str]]: ...

# Decode --------------------------------------------------------------------------------------
def decode(
    value: bytes,
    field_type: int,
    encoding: bytes,
    is_binary: bool,
    use_decimal: bool,
    decode_bit: bool,
    decode_json: bool,
) -> object: ...
