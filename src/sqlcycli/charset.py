# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython.dict import PyDict_SetItem as dict_setitem  # type: ignore
from cython.cimports.cpython.dict import PyDict_GetItem as dict_getitem  # type: ignore
from cython.cimports.cpython.dict import PyDict_Contains as dict_contains  # type: ignore

# Python imports
from typing import Iterator, Any
from sqlcycli import errors

__all__ = [
    "Charset",
    "Charsets",
    "all_charsets",
    "by_id",
    "by_name",
    "by_collation",
    "by_name_n_collation",
]


# Charset(s) ----------------------------------------------------------------------------------
@cython.cclass
class Charset:
    """Represents a MySQL character set."""

    _id: cython.int
    _name: str
    _collation: str
    _encoding: bytes
    _encoding_c: cython.pchar
    _is_default: cython.bint
    _hashcode: cython.Py_ssize_t

    def __init__(
        self,
        id: cython.int,
        name: str,
        collation: str,
        is_default: cython.bint = False,
    ) -> None:
        """The MySQL character set

        :param id `<'int'>`: Numeric MySQL charset identifier.
        :param name `<'str'>`: The charset name (e.g., `"utf8mb4"`).
        :param collation `<'str'>`: The collation name (e.g., `"utf8mb4_general_ci"`).
        :param is_default `<'bool'>`: Whether the charset is one of MySQL's defaults. Defaults to `False`.
        """
        self._id = id
        if name is None:
            raise AssertionError("Charset name cannot be 'None'.")
        self._name = name.lower().strip()
        if collation is None:
            raise AssertionError("Charset collation cannot be 'None'.")
        self._collation = collation.lower().strip()
        if self._name in ("utf8mb4", "utf8mb3"):
            self._encoding = b"utf8"
        elif self._name == "latin1":
            self._encoding = b"cp1252"
        elif self._name == "koi8r":
            self._encoding = b"koi8_r"
        elif self._name == "koi8u":
            self._encoding = b"koi8_u"
        else:
            self._encoding = self._name.encode("ascii", "strict")
        self._encoding_c = self._encoding
        self._is_default = is_default
        self._hashcode = -1

    # Property ---------------------------------------------------------------------
    @property
    def id(self) -> int:
        """Numeric MySQL charset identifier `<'int'>`."""
        return self._id

    @property
    def name(self) -> str:
        """The charset name (e.g., `"utf8mb4"`) `<'str'>`."""
        return self._name

    @property
    def collation(self) -> str:
        """The collation name (e.g., `"utf8mb4_general_ci"`) `<'str'>`."""
        return self._collation

    @property
    def is_default(self) -> bool:
        """Whether the charset is one of MySQL's defaults. `<'bool'>`."""
        return self._is_default

    @property
    def encoding(self) -> bytes:
        """The Python encoding of the charset `<'bytes'>`."""
        return self._encoding

    # Methods ----------------------------------------------------------------------
    @cython.ccall
    def is_binary(self) -> cython.bint:
        """Check if this charset is the MySQL binary charset (ID == 63). `<'bool'>`."""
        return self._id == 63

    def __repr__(self) -> str:
        return "<Charset(id=%d, name='%s', collation='%s', encoding=%s)>" % (
            self._id,
            self._name,
            self._collation,
            self._encoding,
        )

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Charset):
            _o: Charset = o
            return (
                self._id == _o._id
                and self._name == _o._name
                and self._collation == _o._collation
                and self._is_default == _o._is_default
            )
        return NotImplemented

    def __hash__(self) -> int:
        if self._hashcode == -1:
            self._hashcode = id(self)
        return self._hashcode


@cython.cclass
class Charsets:
    """Collection for managing all the MySQL charsets."""

    _by_id: dict[int, Charset]
    _by_name: dict[str, Charset]
    _by_collation: dict[str, Charset]
    _by_name_n_collation: dict[str, Charset]

    def __init__(self) -> None:
        """Collection for managing all the MySQL charsets."""
        self._by_id = {}
        self._by_name = {}
        self._by_collation = {}
        self._by_name_n_collation = {}

    # Add Charset ------------------------------------------------------------------
    @cython.ccall
    @cython.exceptval(-1, check=False)
    def add(self, charset: Charset) -> cython.bint:
        """Add MySQL charset to the collection.

        :param charset `<'Charset'>`: The charset instance to add.
        """
        if charset is None:
            raise AssertionError("charset cannot be 'None'.")
        self._index_by_id(charset)
        self._index_by_name(charset)
        self._index_by_collation(charset)
        self._index_by_name_n_collation(charset)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _index_by_id(self, charset: Charset) -> cython.bint:
        """(internal) Index MySQL charset by its id.

        :param charset `<'Charset'>`: The charset instance to index.
        """
        dict_setitem(self._by_id, charset._id, charset)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _index_by_name(self, charset: Charset) -> cython.bint:
        """(internal) Index MySQL charset by its name.

        :param charset `<'Charset'>`: The charset instance to index.
        """
        if charset._is_default:
            dict_setitem(self._by_name, charset._name, charset)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _index_by_collation(self, charset: Charset) -> cython.bint:
        """(internal) Index MySQL charset by its collation.

        :param charset `<'Charset'>`: The charset instance to index.
        """
        dict_setitem(self._by_collation, charset._collation, charset)
        return True

    @cython.cfunc
    @cython.inline(True)
    @cython.exceptval(-1, check=False)
    def _index_by_name_n_collation(self, charset: Charset) -> cython.bint:
        """(internal) Index MySQL charset by its name and collation.

        :param charset `<'Charset'>`: The charset instance to index.
        """
        dict_setitem(
            self._by_name_n_collation,
            self._gen_charset_n_collate_key(charset._name, charset._collation),
            charset,
        )
        return True

    @cython.cfunc
    @cython.inline(True)
    def _gen_charset_n_collate_key(self, name: object, collation: object) -> str:
        """(internal) Generate a unique lookup key from name and collation `<'str'>."""
        return "%s:%s" % (name, collation)

    # Access Charset ---------------------------------------------------------------
    @cython.ccall
    def by_id(self, id: object) -> Charset:
        """Retrieve a Charset by its MySQL ID `<'Charset'>`.

        :param id `<'int'>`: The ID of the charset.
        """
        val = dict_getitem(self._by_id, id)
        if val == cython.NULL:
            raise errors.CharsetNotFoundError(
                "<'%s'>\nMySQL charset ID '%d' does not exist."
                % (self.__class__.__name__, id)
            )
        return cython.cast(Charset, val)

    @cython.ccall
    def by_name(self, name: object) -> Charset:
        """Retrieve a default Charset by its name `<'Charset'>`.

        :param name `<'str'>`: The name of the charset.
        """
        if name in ("utf8mb4", "utf8", "utf-8"):
            return _default_utf8mb4

        val = dict_getitem(self._by_name, name)
        if val == cython.NULL:
            raise errors.CharsetNotFoundError(
                "<'%s'>\nMySQL charactor set '%s' does not exist."
                % (self.__class__.__name__, name)
            )
        return cython.cast(Charset, val)

    @cython.ccall
    def by_collation(self, collation: object) -> Charset:
        """Retrieve a Charset by its collation `<'Charset'>`.

        :param collation `<'str'>`: The collation of the charset.
        """
        if collation == "utf8mb4_general_ci":
            return _default_utf8mb4

        val = dict_getitem(self._by_collation, collation)
        if val == cython.NULL:
            raise errors.CharsetNotFoundError(
                "<'%s'>\nMySQL charactor collation '%s' does not exist."
                % (self.__class__.__name__, collation)
            )
        return cython.cast(Charset, val)

    @cython.ccall
    def by_name_n_collation(self, name: object, collation: object) -> Charset:
        """Retrieve a Charset by both its name and collation combination `<'Charset'>`.

        :param name `<'str'>`: The name of the charset.
        :param collation `<'str'>`: The collation of the charset.
        """
        if name in ("utf8mb4", "utf8", "utf-8"):
            if collation == "utf8mb4_general_ci":
                return _default_utf8mb4
            _key: str = self._gen_charset_n_collate_key("utf8mb4", collation)
        else:
            _key: str = self._gen_charset_n_collate_key(name, collation)

        val = dict_getitem(self._by_name_n_collation, _key)
        if val == cython.NULL:
            raise errors.CharsetNotFoundError(
                "<'%s'>\nMySQL charactor set & collation '%s & %s' does not exist."
                % (self.__class__.__name__, name, collation)
            )
        return cython.cast(Charset, val)

    # Special Methods -------------------------------------------------------------
    def __iter__(self) -> Iterator[Charset]:
        return self._by_id.values().__iter__()

    def __repr__(self) -> str:
        return "<Charsets(\n%s\n)>" % (
            "\n".join(str(charset) for charset in self._by_id.values()),
        )


_charsets: Charsets = Charsets()


# Functions -----------------------------------------------------------------------------------
@cython.ccall
def all_charsets() -> Charsets:
    """Retrieve the collection of all the MySQL charsets `<'Charsets'>`."""
    return _charsets


@cython.ccall
def by_id(id: object) -> Charset:
    """Retrieve a Charset by its MySQL ID `<'Charset'>`.

    :param id `<'int'>`: The ID of the charset.
    """
    return _charsets.by_id(id)


@cython.ccall
def by_name(name: object) -> Charset:
    """Retrieve a default Charset by its name `<'Charset'>`.

    :param name `<'str'>`: The name of the charset.
    """
    return _charsets.by_name(name)


@cython.ccall
def by_collation(collation: object) -> Charset:
    """Retrieve a Charset by its collation `<'Charset'>`.

    :param collation `<'str'>`: The collation of the charset.
    """
    return _charsets.by_collation(collation)


@cython.ccall
def by_name_n_collation(name: str | Any, collation: str | Any) -> Charset:
    """Retrieve a Charset by both its name and collation combination `<'Charset'>`.

    :param name `<'str'>`: The name of the charset.
    :param collation `<'str'>`: The collation of the charset.
    """
    return _charsets.by_name_n_collation(name, collation)


_charsets.add(Charset(1, "big5", "big5_chinese_ci", True))
_charsets.add(Charset(2, "latin2", "latin2_czech_cs"))
_charsets.add(Charset(3, "dec8", "dec8_swedish_ci", True))
_charsets.add(Charset(4, "cp850", "cp850_general_ci", True))
_charsets.add(Charset(5, "latin1", "latin1_german1_ci"))
_charsets.add(Charset(6, "hp8", "hp8_english_ci", True))
_charsets.add(Charset(7, "koi8r", "koi8r_general_ci", True))
_charsets.add(Charset(8, "latin1", "latin1_swedish_ci", True))
_charsets.add(Charset(9, "latin2", "latin2_general_ci", True))
_charsets.add(Charset(10, "swe7", "swe7_swedish_ci", True))
_charsets.add(Charset(11, "ascii", "ascii_general_ci", True))
_charsets.add(Charset(12, "ujis", "ujis_japanese_ci", True))
_charsets.add(Charset(13, "sjis", "sjis_japanese_ci", True))
_charsets.add(Charset(14, "cp1251", "cp1251_bulgarian_ci"))
_charsets.add(Charset(15, "latin1", "latin1_danish_ci"))
_charsets.add(Charset(16, "hebrew", "hebrew_general_ci", True))
_charsets.add(Charset(18, "tis620", "tis620_thai_ci", True))
_charsets.add(Charset(19, "euckr", "euckr_korean_ci", True))
_charsets.add(Charset(20, "latin7", "latin7_estonian_cs"))
_charsets.add(Charset(21, "latin2", "latin2_hungarian_ci"))
_charsets.add(Charset(22, "koi8u", "koi8u_general_ci", True))
_charsets.add(Charset(23, "cp1251", "cp1251_ukrainian_ci"))
_charsets.add(Charset(24, "gb2312", "gb2312_chinese_ci", True))
_charsets.add(Charset(25, "greek", "greek_general_ci", True))
_charsets.add(Charset(26, "cp1250", "cp1250_general_ci", True))
_charsets.add(Charset(27, "latin2", "latin2_croatian_ci"))
_charsets.add(Charset(28, "gbk", "gbk_chinese_ci", True))
_charsets.add(Charset(29, "cp1257", "cp1257_lithuanian_ci"))
_charsets.add(Charset(30, "latin5", "latin5_turkish_ci", True))
_charsets.add(Charset(31, "latin1", "latin1_german2_ci"))
_charsets.add(Charset(32, "armscii8", "armscii8_general_ci", True))
_charsets.add(Charset(33, "utf8mb3", "utf8mb3_general_ci", True))
_charsets.add(Charset(34, "cp1250", "cp1250_czech_cs"))
_charsets.add(Charset(36, "cp866", "cp866_general_ci", True))
_charsets.add(Charset(37, "keybcs2", "keybcs2_general_ci", True))
_charsets.add(Charset(38, "macce", "macce_general_ci", True))
_charsets.add(Charset(39, "macroman", "macroman_general_ci", True))
_charsets.add(Charset(40, "cp852", "cp852_general_ci", True))
_charsets.add(Charset(41, "latin7", "latin7_general_ci", True))
_charsets.add(Charset(42, "latin7", "latin7_general_cs"))
_charsets.add(Charset(43, "macce", "macce_bin"))
_charsets.add(Charset(44, "cp1250", "cp1250_croatian_ci"))
_charsets.add(Charset(45, "utf8mb4", "utf8mb4_general_ci", True))
_charsets.add(Charset(46, "utf8mb4", "utf8mb4_bin"))
_charsets.add(Charset(47, "latin1", "latin1_bin"))
_charsets.add(Charset(48, "latin1", "latin1_general_ci"))
_charsets.add(Charset(49, "latin1", "latin1_general_cs"))
_charsets.add(Charset(50, "cp1251", "cp1251_bin"))
_charsets.add(Charset(51, "cp1251", "cp1251_general_ci", True))
_charsets.add(Charset(52, "cp1251", "cp1251_general_cs"))
_charsets.add(Charset(53, "macroman", "macroman_bin"))
_charsets.add(Charset(57, "cp1256", "cp1256_general_ci", True))
_charsets.add(Charset(58, "cp1257", "cp1257_bin"))
_charsets.add(Charset(59, "cp1257", "cp1257_general_ci", True))
_charsets.add(Charset(63, "binary", "binary", True))
_charsets.add(Charset(64, "armscii8", "armscii8_bin"))
_charsets.add(Charset(65, "ascii", "ascii_bin"))
_charsets.add(Charset(66, "cp1250", "cp1250_bin"))
_charsets.add(Charset(67, "cp1256", "cp1256_bin"))
_charsets.add(Charset(68, "cp866", "cp866_bin"))
_charsets.add(Charset(69, "dec8", "dec8_bin"))
_charsets.add(Charset(70, "greek", "greek_bin"))
_charsets.add(Charset(71, "hebrew", "hebrew_bin"))
_charsets.add(Charset(72, "hp8", "hp8_bin"))
_charsets.add(Charset(73, "keybcs2", "keybcs2_bin"))
_charsets.add(Charset(74, "koi8r", "koi8r_bin"))
_charsets.add(Charset(75, "koi8u", "koi8u_bin"))
_charsets.add(Charset(76, "utf8mb3", "utf8mb3_tolower_ci"))
_charsets.add(Charset(77, "latin2", "latin2_bin"))
_charsets.add(Charset(78, "latin5", "latin5_bin"))
_charsets.add(Charset(79, "latin7", "latin7_bin"))
_charsets.add(Charset(80, "cp850", "cp850_bin"))
_charsets.add(Charset(81, "cp852", "cp852_bin"))
_charsets.add(Charset(82, "swe7", "swe7_bin"))
_charsets.add(Charset(83, "utf8mb3", "utf8mb3_bin"))
_charsets.add(Charset(84, "big5", "big5_bin"))
_charsets.add(Charset(85, "euckr", "euckr_bin"))
_charsets.add(Charset(86, "gb2312", "gb2312_bin"))
_charsets.add(Charset(87, "gbk", "gbk_bin"))
_charsets.add(Charset(88, "sjis", "sjis_bin"))
_charsets.add(Charset(89, "tis620", "tis620_bin"))
_charsets.add(Charset(91, "ujis", "ujis_bin"))
_charsets.add(Charset(92, "geostd8", "geostd8_general_ci", True))
_charsets.add(Charset(93, "geostd8", "geostd8_bin"))
_charsets.add(Charset(94, "latin1", "latin1_spanish_ci"))
_charsets.add(Charset(95, "cp932", "cp932_japanese_ci", True))
_charsets.add(Charset(96, "cp932", "cp932_bin"))
_charsets.add(Charset(97, "eucjpms", "eucjpms_japanese_ci", True))
_charsets.add(Charset(98, "eucjpms", "eucjpms_bin"))
_charsets.add(Charset(99, "cp1250", "cp1250_polish_ci"))
_charsets.add(Charset(192, "utf8mb3", "utf8mb3_unicode_ci"))
_charsets.add(Charset(193, "utf8mb3", "utf8mb3_icelandic_ci"))
_charsets.add(Charset(194, "utf8mb3", "utf8mb3_latvian_ci"))
_charsets.add(Charset(195, "utf8mb3", "utf8mb3_romanian_ci"))
_charsets.add(Charset(196, "utf8mb3", "utf8mb3_slovenian_ci"))
_charsets.add(Charset(197, "utf8mb3", "utf8mb3_polish_ci"))
_charsets.add(Charset(198, "utf8mb3", "utf8mb3_estonian_ci"))
_charsets.add(Charset(199, "utf8mb3", "utf8mb3_spanish_ci"))
_charsets.add(Charset(200, "utf8mb3", "utf8mb3_swedish_ci"))
_charsets.add(Charset(201, "utf8mb3", "utf8mb3_turkish_ci"))
_charsets.add(Charset(202, "utf8mb3", "utf8mb3_czech_ci"))
_charsets.add(Charset(203, "utf8mb3", "utf8mb3_danish_ci"))
_charsets.add(Charset(204, "utf8mb3", "utf8mb3_lithuanian_ci"))
_charsets.add(Charset(205, "utf8mb3", "utf8mb3_slovak_ci"))
_charsets.add(Charset(206, "utf8mb3", "utf8mb3_spanish2_ci"))
_charsets.add(Charset(207, "utf8mb3", "utf8mb3_roman_ci"))
_charsets.add(Charset(208, "utf8mb3", "utf8mb3_persian_ci"))
_charsets.add(Charset(209, "utf8mb3", "utf8mb3_esperanto_ci"))
_charsets.add(Charset(210, "utf8mb3", "utf8mb3_hungarian_ci"))
_charsets.add(Charset(211, "utf8mb3", "utf8mb3_sinhala_ci"))
_charsets.add(Charset(212, "utf8mb3", "utf8mb3_german2_ci"))
_charsets.add(Charset(213, "utf8mb3", "utf8mb3_croatian_ci"))
_charsets.add(Charset(214, "utf8mb3", "utf8mb3_unicode_520_ci"))
_charsets.add(Charset(215, "utf8mb3", "utf8mb3_vietnamese_ci"))
_charsets.add(Charset(223, "utf8mb3", "utf8mb3_general_mysql500_ci"))
_charsets.add(Charset(224, "utf8mb4", "utf8mb4_unicode_ci"))
_charsets.add(Charset(225, "utf8mb4", "utf8mb4_icelandic_ci"))
_charsets.add(Charset(226, "utf8mb4", "utf8mb4_latvian_ci"))
_charsets.add(Charset(227, "utf8mb4", "utf8mb4_romanian_ci"))
_charsets.add(Charset(228, "utf8mb4", "utf8mb4_slovenian_ci"))
_charsets.add(Charset(229, "utf8mb4", "utf8mb4_polish_ci"))
_charsets.add(Charset(230, "utf8mb4", "utf8mb4_estonian_ci"))
_charsets.add(Charset(231, "utf8mb4", "utf8mb4_spanish_ci"))
_charsets.add(Charset(232, "utf8mb4", "utf8mb4_swedish_ci"))
_charsets.add(Charset(233, "utf8mb4", "utf8mb4_turkish_ci"))
_charsets.add(Charset(234, "utf8mb4", "utf8mb4_czech_ci"))
_charsets.add(Charset(235, "utf8mb4", "utf8mb4_danish_ci"))
_charsets.add(Charset(236, "utf8mb4", "utf8mb4_lithuanian_ci"))
_charsets.add(Charset(237, "utf8mb4", "utf8mb4_slovak_ci"))
_charsets.add(Charset(238, "utf8mb4", "utf8mb4_spanish2_ci"))
_charsets.add(Charset(239, "utf8mb4", "utf8mb4_roman_ci"))
_charsets.add(Charset(240, "utf8mb4", "utf8mb4_persian_ci"))
_charsets.add(Charset(241, "utf8mb4", "utf8mb4_esperanto_ci"))
_charsets.add(Charset(242, "utf8mb4", "utf8mb4_hungarian_ci"))
_charsets.add(Charset(243, "utf8mb4", "utf8mb4_sinhala_ci"))
_charsets.add(Charset(244, "utf8mb4", "utf8mb4_german2_ci"))
_charsets.add(Charset(245, "utf8mb4", "utf8mb4_croatian_ci"))
_charsets.add(Charset(246, "utf8mb4", "utf8mb4_unicode_520_ci"))
_charsets.add(Charset(247, "utf8mb4", "utf8mb4_vietnamese_ci"))
_charsets.add(Charset(248, "gb18030", "gb18030_chinese_ci", True))
_charsets.add(Charset(249, "gb18030", "gb18030_bin"))
_charsets.add(Charset(250, "gb18030", "gb18030_unicode_520_ci"))
_charsets.add(Charset(255, "utf8mb4", "utf8mb4_0900_ai_ci"))

# default utf8mb4: utf8mb4_general_ci
_default_utf8mb4: Charset = _charsets.by_id(45)
