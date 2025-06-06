# Base class
class SQLInterval:
    def __init__(self, interval_name: str, expr: object): ...
    @property
    def name(self) -> str: ...
    @property
    def expr(self) -> object: ...
    def syntax(self) -> str: ...
    def __repr__(self) -> str: ...

# Intervals
class MICROSECOND(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class SECOND(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class MINUTE(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class HOUR(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class DAY(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class WEEK(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class MONTH(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class QUARTER(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class YEAR(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class SECOND_MICROSECOND(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class MINUTE_MICROSECOND(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class MINUTE_SECOND(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class HOUR_MICROSECOND(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class HOUR_SECOND(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class HOUR_MINUTE(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class DAY_MICROSECOND(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class DAY_SECOND(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class DAY_MINUTE(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class DAY_HOUR(SQLInterval):
    def __init__(self, expr: int | float | str): ...

class YEAR_MONTH(SQLInterval):
    def __init__(self, expr: int | float | str): ...
