from typing import Any

# Base class
class SQLFunction:
    def __init__(
        self,
        function_name: str,
        arg_count: int,
        *args,
        sep: str = ",",
        **kwargs,
    ): ...
    @property
    def name(self) -> str: ...
    @property
    def args(self) -> tuple[object]: ...
    def syntax(self) -> str: ...
    def __repr__(self) -> str: ...

# Custom class
class Sentinel:
    def __repr__(self): ...

IGNORED: Sentinel

class RawText:
    def __init__(self, value: str): ...
    @property
    def value(self) -> str: ...
    def __repr__(self) -> str: ...

class ObjStr:
    def __str__(self) -> str: ...

# Functions: Custom
class RANDINT(SQLFunction):
    def __init__(self, i: int, j: int): ...

# Functions: A
class ABS(SQLFunction):
    def __init__(self, X): ...

class ACOS(SQLFunction):
    def __init__(self, X): ...

class ADDDATE(SQLFunction):
    def __init__(self, date, expr): ...

class ADDTIME(SQLFunction):
    def __init__(self, expr1, expr2): ...

class ASCII(SQLFunction):
    def __init__(self, string): ...

class ASIN(SQLFunction):
    def __init__(self, X): ...

class ATAN(SQLFunction):
    def __init__(self, X, Y: Any | Sentinel = IGNORED): ...

# Functions: B
class BIN(SQLFunction):
    def __init__(self, N): ...

class BIN_TO_UUID(SQLFunction):
    def __init__(self, binary_uuid, swap_flag: int | Sentinel = IGNORED): ...

class BIT_COUNT(SQLFunction):
    def __init__(self, N): ...

class BIT_LENGTH(SQLFunction):
    def __init__(self, string): ...

# Functions: C
class CEIL(SQLFunction):
    def __init__(self, X): ...

class CHAR(SQLFunction):
    def __init__(self, *N, using: str | Sentinel = IGNORED): ...

class CHAR_LENGTH(SQLFunction):
    def __init__(self, string): ...

class CHARSET(SQLFunction):
    def __init__(self, string): ...

class COALESCE(SQLFunction):
    def __init__(self, *values): ...

class COERCIBILITY(SQLFunction):
    def __init__(self, string): ...

class COLLATION(SQLFunction):
    def __init__(self, string): ...

class COMPRESS(SQLFunction):
    def __init__(self, string): ...

class CONCAT(SQLFunction):
    def __init__(self, *strings): ...

class CONCAT_WS(SQLFunction):
    def __init__(self, sep, *strings): ...

class CONNECTION_ID(SQLFunction):
    def __init__(self): ...

class CONV(SQLFunction):
    def __init__(self, N, from_base, to_base): ...

class CONVERT(SQLFunction):
    def __init__(self, expr, using): ...

class CONVERT_TZ(SQLFunction):
    def __init__(self, dt, from_tz, to_tz): ...

class COS(SQLFunction):
    def __init__(self, X): ...

class COT(SQLFunction):
    def __init__(self, X): ...

class CRC32(SQLFunction):
    def __init__(self, expr): ...

class CUME_DIST(SQLFunction):
    def __init__(self): ...

class CURRENT_DATE(SQLFunction):
    def __init__(self): ...

class CURRENT_ROLE(SQLFunction):
    def __init__(self): ...

class CURRENT_TIME(SQLFunction):
    def __init__(self, fsp: Any | Sentinel = IGNORED): ...

class CURRENT_TIMESTAMP(SQLFunction):
    def __init__(self, fsp: Any | Sentinel = IGNORED): ...

class CURRENT_USER(SQLFunction):
    def __init__(self): ...

# Functions: D
class DATABASE(SQLFunction):
    def __init__(self): ...

class DATE(SQLFunction):
    def __init__(self, expr): ...

class DATE_ADD(SQLFunction):
    def __init__(self, date, expr): ...

class DATE_FORMAT(SQLFunction):
    def __init__(self, date, format): ...

class DATE_SUB(SQLFunction):
    def __init__(self, date, expr): ...

class DATEDIFF(SQLFunction):
    def __init__(self, expr1, expr2): ...

class DAYNAME(SQLFunction):
    def __init__(self, date): ...

class DAYOFMONTH(SQLFunction):
    def __init__(self, date): ...

class DAYOFWEEK(SQLFunction):
    def __init__(self, date): ...

class DAYOFYEAR(SQLFunction):
    def __init__(self, date): ...

class DEGREES(SQLFunction):
    def __init__(self, X): ...

class DENSE_RANK(SQLFunction):
    def __init__(self): ...

# Functions: E
class ELT(SQLFunction):
    def __init__(self, N, *strings): ...

class EXP(SQLFunction):
    def __init__(self, X): ...

class EXPORT_SET(SQLFunction):
    def __init__(
        self,
        bits,
        on,
        off,
        separator: str | Sentinel = IGNORED,
        number_of_bits: int | Sentinel = IGNORED,
    ): ...

class EXTRACT(SQLFunction):
    def __init__(self, unit, date): ...

# Functions: F
class FIELD(SQLFunction):
    def __init__(self, *strings): ...

class FIND_IN_SET(SQLFunction):
    def __init__(self, string, string_list): ...

class FLOOR(SQLFunction):
    def __init__(self, X): ...

class FORMAT(SQLFunction):
    def __init__(self, X, D, locale: str | Sentinel = IGNORED): ...

class FORMAT_BYTES(SQLFunction):
    def __init__(self, count): ...

class FORMAT_PICO_TIME(SQLFunction):
    def __init__(self, time_val): ...

class FROM_DAYS(SQLFunction):
    def __init__(self, N): ...

class FROM_UNIXTIME(SQLFunction):
    def __init__(self, ts, format: str | Sentinel = IGNORED): ...

# Functions: G
class GeomCollection(SQLFunction):
    def __init__(self, *g): ...

class GET_FORMAT(SQLFunction):
    def __init__(self, date: str, format: str): ...

class GET_LOCK(SQLFunction):
    def __init__(self, string, timeout): ...

class GREATEST(SQLFunction):
    def __init__(self, *values): ...

# Functions: H
class HEX(SQLFunction):
    def __init__(self, X): ...

class HOUR(SQLFunction):
    def __init__(self, time): ...

# Functions: I
class IFNULL(SQLFunction):
    def __init__(self, expr1, expr2): ...

class IN(SQLFunction):
    def __init__(self, *values): ...

class INET_ATON(SQLFunction):
    def __init__(self, expr): ...

class INET6_ATON(SQLFunction):
    def __init__(self, expr): ...

class INET_NTOA(SQLFunction):
    def __init__(self, expr): ...

class INET6_NTOA(SQLFunction):
    def __init__(self, expr): ...

class INSERT(SQLFunction):
    def __init__(self, string, pos, length, newstr): ...

class INSTR(SQLFunction):
    def __init__(self, string, substr): ...

class INTERVAL(SQLFunction):
    def __init__(self, N, N1, *Nn): ...

class IS_FREE_LOCK(SQLFunction):
    def __init__(self, string): ...

class IS_USER_LOCK(SQLFunction):
    def __init__(self, string): ...

class IS_UUID(SQLFunction):
    def __init__(self, string_uuid): ...

class ISNULL(SQLFunction):
    def __init__(self, expr): ...

# Functions: J -------------------------------------------------------------------------------------------------------
class JSON_ARRAY(SQLFunction):
    def __init__(self, *values): ...

class JSON_ARRAY_APPEND(SQLFunction):
    def __init__(self, json_doc, *path_val_pairs): ...

class JSON_ARRAY_INSERT(SQLFunction):
    def __init__(self, json_doc, *path_val_pairs): ...

class JSON_CONTAINS(SQLFunction):
    def __init__(self, target, candidate, path: str | Sentinel = IGNORED): ...

class JSON_CONTAINS_PATH(SQLFunction):
    def __init__(self, json_doc, one_or_all, *paths): ...

class JSON_DEPTH(SQLFunction):
    def __init__(self, json_doc): ...

class JSON_EXTRACT(SQLFunction):
    def __init__(self, json_doc, *paths): ...

class JSON_INSERT(SQLFunction):
    def __init__(self, json_doc, *path_val_pairs): ...

class JSON_KEYS(SQLFunction):
    def __init__(self, json_doc, path: str | Sentinel = IGNORED): ...

class JSON_LENGTH(SQLFunction):
    def __init__(self, json_doc, path: str | Sentinel = IGNORED): ...

class JSON_MERGE_PATCH(SQLFunction):
    def __init__(self, *json_docs): ...

class JSON_MERGE_PRESERVE(SQLFunction):
    def __init__(self, *json_docs): ...

class JSON_OBJECT(SQLFunction):
    def __init__(self, *key_val_pairs): ...

class JSON_OVERLAPS(SQLFunction):
    def __init__(self, json_doc1, json_doc2): ...

class JSON_PRETTY(SQLFunction):
    def __init__(self, json_doc): ...

class JSON_QUOTE(SQLFunction):
    def __init__(self, string): ...

class JSON_REMOVE(SQLFunction):
    def __init__(self, json_doc, *paths): ...

class JSON_REPLACE(SQLFunction):
    def __init__(self, json_doc, *path_val_pairs): ...

class JSON_SEARCH(SQLFunction):
    def __init__(
        self,
        json_doc,
        one_or_all,
        search_str,
        escape_char: str | Sentinel = IGNORED,
        path: str | Sentinel = IGNORED,
    ): ...

class JSON_SET(SQLFunction):
    def __init__(self, json_doc, *path_val_pairs): ...

class JSON_TYPE(SQLFunction):
    def __init__(self, json_val): ...

class JSON_UNQUOTE(SQLFunction):
    def __init__(self, json_val): ...

class JSON_VALID(SQLFunction):
    def __init__(self, val): ...

# Functions: L
class LAST_DAY(SQLFunction):
    def __init__(self, date): ...

class LEAST(SQLFunction):
    def __init__(self, *values): ...

class LEFT(SQLFunction):
    def __init__(self, string, length): ...

class LENGTH(SQLFunction):
    def __init__(self, string): ...

class LineString(SQLFunction):
    def __init__(self, *pt): ...

class LN(SQLFunction):
    def __init__(self, X): ...

class LOAD_FILE(SQLFunction):
    def __init__(self, file_name): ...

class LOCALTIME(SQLFunction):
    def __init__(self, fsp: Any | Sentinel = IGNORED): ...

class LOCATE(SQLFunction):
    def __init__(self, substr, string, pos: Any | Sentinel = IGNORED): ...

class LOG(SQLFunction):
    def __init__(self, X, B: Any | Sentinel = IGNORED): ...

class LOG2(SQLFunction):
    def __init__(self, X): ...

class LOG10(SQLFunction):
    def __init__(self, X): ...

class LOWER(SQLFunction):
    def __init__(self, string): ...

class LPAD(SQLFunction):
    def __init__(self, string, length, pad_string): ...

class LTRIM(SQLFunction):
    def __init__(self, string): ...

# Functions: M
class MAKE_SET(SQLFunction):
    def __init__(self, bits, *strings): ...

class MAKEDATE(SQLFunction):
    def __init__(self, year, dayofyear): ...

class MAKETIME(SQLFunction):
    def __init__(self, hour, minute, second): ...

class MBRContains(SQLFunction):
    def __init__(self, g1, g2): ...

class MBRCoveredBy(SQLFunction):
    def __init__(self, g1, g2): ...

class MBRCovers(SQLFunction):
    def __init__(self, g1, g2): ...

class MBRDisjoint(SQLFunction):
    def __init__(self, g1, g2): ...

class MBREquals(SQLFunction):
    def __init__(self, g1, g2): ...

class MBRIntersects(SQLFunction):
    def __init__(self, g1, g2): ...

class MBROverlaps(SQLFunction):
    def __init__(self, g1, g2): ...

class MBRTouches(SQLFunction):
    def __init__(self, g1, g2): ...

class MBRWithin(SQLFunction):
    def __init__(self, g1, g2): ...

class MD5(SQLFunction):
    def __init__(self, string): ...

class MICROSECOND(SQLFunction):
    def __init__(self, expr): ...

class MINUTE(SQLFunction):
    def __init__(self, expr): ...

class MOD(SQLFunction):
    def __init__(self, N, M): ...

class MONTH(SQLFunction):
    def __init__(self, date): ...

class MONTHNAME(SQLFunction):
    def __init__(self, date): ...

class MultiLineString(SQLFunction):
    def __init__(self, *ls): ...

class MultiPoint(SQLFunction):
    def __init__(self, *pt): ...

class MultiPolygon(SQLFunction):
    def __init__(self, *poly): ...

# Functions: N
class NOT_IN(SQLFunction):
    def __init__(self, *values): ...

class NOW(SQLFunction):
    def __init__(self, fsp: Any | Sentinel = IGNORED): ...

class NULLIF(SQLFunction):
    def __init__(self, expr1, expr2): ...

# Functions: O
class OCT(SQLFunction):
    def __init__(self, N): ...

class ORD(SQLFunction):
    def __init__(self, string): ...

# Functions: P
class PERIOD_ADD(SQLFunction):
    def __init__(self, P, N): ...

class PERIOD_DIFF(SQLFunction):
    def __init__(self, P1, P2): ...

class PI(SQLFunction):
    def __init__(self): ...

class Point(SQLFunction):
    def __init__(self, x, y): ...

class Polygon(SQLFunction):
    def __init__(self, *ls): ...

class POW(SQLFunction):
    def __init__(self, X, Y): ...

class PS_CURRENT_THREAD_ID(SQLFunction):
    def __init__(self): ...

class PS_THREAD_ID(SQLFunction):
    def __init__(self, connection_id): ...

# Functions: Q
class QUARTER(SQLFunction):
    def __init__(self, date): ...

class QUOTE(SQLFunction):
    def __init__(self, string): ...

# Functions: R
class RADIANS(SQLFunction):
    def __init__(self, X): ...

class RAND(SQLFunction):
    def __init__(self, N: Any | Sentinel = IGNORED): ...

class RANDOM_BYTES(SQLFunction):
    def __init__(self, length): ...

class RANK(SQLFunction):
    def __init__(self): ...

class REGEXP_INSTR(SQLFunction):
    def __init__(
        self,
        expr,
        pat,
        pos: int | Sentinel = IGNORED,
        occurrence: int | Sentinel = IGNORED,
        return_option: int | Sentinel = IGNORED,
        match_type: str | Sentinel = IGNORED,
    ): ...

class REGEXP_LIKE(SQLFunction):
    def __init__(self, expr, pat, match_type: str | Sentinel = IGNORED): ...

class REGEXP_REPLACE(SQLFunction):
    def __init__(
        self,
        expr,
        pat,
        repl,
        pos: int | Sentinel = IGNORED,
        occurrence: int | Sentinel = IGNORED,
        match_type: str | Sentinel = IGNORED,
    ): ...

class REGEXP_SUBSTR(SQLFunction):
    def __init__(
        self,
        expr,
        pat,
        pos: int | Sentinel = IGNORED,
        occurrence: int | Sentinel = IGNORED,
        match_type: str | Sentinel = IGNORED,
    ): ...

class RELEASE_ALL_LOCKS(SQLFunction):
    def __init__(self): ...

class RELEASE_LOCK(SQLFunction):
    def __init__(self, string): ...

class REPEAT(SQLFunction):
    def __init__(self, string, count): ...

class REPLACE(SQLFunction):
    def __init__(self, string, from_str, to_str): ...

class REVERSE(SQLFunction):
    def __init__(self, string): ...

class RIGHT(SQLFunction):
    def __init__(self, string, length): ...

class ROLES_GRAPHML(SQLFunction):
    def __init__(self): ...

class ROUND(SQLFunction):
    def __init__(self, X, D: Any | Sentinel = IGNORED): ...

class ROW_COUNT(SQLFunction):
    def __init__(self): ...

class ROW_NUMBER(SQLFunction):
    def __init__(self): ...

class RPAD(SQLFunction):
    def __init__(self, string, length, padstr): ...

class RTRIM(SQLFunction):
    def __init__(self, string): ...

# Functions: S
class SEC_TO_TIME(SQLFunction):
    def __init__(self, seconds): ...

class SECOND(SQLFunction):
    def __init__(self, time): ...

class SHA1(SQLFunction):
    def __init__(self, string): ...

class SHA2(SQLFunction):
    def __init__(self, string, hash_length: int): ...

class SIGN(SQLFunction):
    def __init__(self, X): ...

class SIN(SQLFunction):
    def __init__(self, X): ...

class SLEEP(SQLFunction):
    def __init__(self, duration): ...

class SOUNDEX(SQLFunction):
    def __init__(self, string): ...

class SPACE(SQLFunction):
    def __init__(self, N): ...

class SQRT(SQLFunction):
    def __init__(self, X): ...

class ST_Area(SQLFunction):
    def __init__(self, poly): ...

class ST_AsBinary(SQLFunction):
    def __init__(self, g, options: Any | Sentinel = IGNORED): ...

class ST_AsWKB(SQLFunction):
    def __init__(self, g, options: Any | Sentinel = IGNORED): ...

class ST_AsGeoJSON(SQLFunction):
    def __init__(
        self,
        g,
        max_dec_digits: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_AsText(SQLFunction):
    def __init__(self, g, options: Any | Sentinel = IGNORED): ...

class ST_AsWKT(SQLFunction):
    def __init__(self, g, options: Any | Sentinel = IGNORED): ...

class ST_Buffer(SQLFunction):
    def __init__(
        self,
        g,
        d,
        strategy1: Any | Sentinel = IGNORED,
        strategy2: Any | Sentinel = IGNORED,
        strategy3: Any | Sentinel = IGNORED,
    ): ...

class ST_Buffer_Strategy(SQLFunction):
    def __init__(self, strategy: str, points_per_circle: Any | Sentinel = IGNORED): ...

class ST_Centroid(SQLFunction):
    def __init__(self, poly): ...

class ST_Contains(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_ConvexHull(SQLFunction):
    def __init__(self, g): ...

class ST_Crosses(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_Difference(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_Dimension(SQLFunction):
    def __init__(self, g): ...

class ST_Disjoint(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_Distance(SQLFunction):
    def __init__(self, g1, g2, unit: Any | Sentinel = IGNORED): ...

class ST_Distance_Sphere(SQLFunction):
    def __init__(self, g1, g2, radius: Any | Sentinel = IGNORED): ...

class ST_EndPoint(SQLFunction):
    def __init__(self, ls): ...

class ST_Envelope(SQLFunction):
    def __init__(self, g): ...

class ST_Equals(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_ExteriorRing(SQLFunction):
    def __init__(self, poly): ...

class ST_FrechetDistance(SQLFunction):
    def __init__(self, g1, g2, unit: Any | Sentinel = IGNORED): ...

class ST_GeoHash(SQLFunction):
    def __init__(self, arg1, arg2, arg3: Any | Sentinel = IGNORED): ...

class ST_GeomCollFromText(SQLFunction):
    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_GeomCollFromWKB(SQLFunction):
    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_GeometryN(SQLFunction):
    def __init__(self, gc, N): ...

class ST_GeometryType(SQLFunction):
    def __init__(self, g): ...

class ST_GeomFromGeoJSON(SQLFunction):
    def __init__(
        self,
        string,
        options: Any | Sentinel = IGNORED,
        srid: Any | Sentinel = IGNORED,
    ): ...

class ST_GeomFromText(SQLFunction):
    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_GeomFromWKB(SQLFunction):
    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_HausdorffDistance(SQLFunction):
    def __init__(self, g1, g2, unit: Any | Sentinel = IGNORED): ...

class ST_InteriorRingN(SQLFunction):
    def __init__(self, poly, N): ...

class ST_Intersection(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_Intersects(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_IsClosed(SQLFunction):
    def __init__(self, ls): ...

class ST_IsEmpty(SQLFunction):
    def __init__(self, g): ...

class ST_IsSimple(SQLFunction):
    def __init__(self, g): ...

class ST_IsValid(SQLFunction):
    def __init__(self, g): ...

class ST_LatFromGeoHash(SQLFunction):
    def __init__(self, geohash_str): ...

class ST_Latitude(SQLFunction):
    def __init__(self, p, new_latitude_val: Any | Sentinel = IGNORED): ...

class ST_Length(SQLFunction):
    def __init__(self, ls, unit: Any | Sentinel = IGNORED): ...

class ST_LineFromText(SQLFunction):
    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_LineFromWKB(SQLFunction):
    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_LineInterpolatePoint(SQLFunction):
    def __init__(self, ls, fractional_distance): ...

class ST_LineInterpolatePoints(SQLFunction):
    def __init__(self, ls, fractional_distance): ...

class ST_LongFromGeoHash(SQLFunction):
    def __init__(self, geohash_str): ...

class ST_Longitude(SQLFunction):
    def __init__(self, p, new_longitude_val: Any | Sentinel = IGNORED): ...

class ST_MakeEnvelope(SQLFunction):
    def __init__(self, pt1, pt2): ...

class ST_MLineFromText(SQLFunction):
    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_MLineFromWKB(SQLFunction):
    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_MPointFromText(SQLFunction):
    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_MPointFromWKB(SQLFunction):
    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_MPolyFromText(SQLFunction):
    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_MPolyFromWKB(SQLFunction):
    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_NumGeometries(SQLFunction):
    def __init__(self, gc): ...

class ST_NumInteriorRing(SQLFunction):
    def __init__(self, poly): ...

class ST_NumPoints(SQLFunction):
    def __init__(self, ls): ...

class ST_Overlaps(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_PointAtDistance(SQLFunction):
    def __init__(self, ls, distance): ...

class ST_PointFromGeoHash(SQLFunction):
    def __init__(self, geohash_str, srid): ...

class ST_PointFromText(SQLFunction):
    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_PointFromWKB(SQLFunction):
    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_PointN(SQLFunction):
    def __init__(self, ls, N): ...

class ST_PolyFromText(SQLFunction):
    def __init__(
        self,
        wkt,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_PolyFromWKB(SQLFunction):
    def __init__(
        self,
        wkb,
        srid: Any | Sentinel = IGNORED,
        options: Any | Sentinel = IGNORED,
    ): ...

class ST_Simplify(SQLFunction):
    def __init__(self, g, max_distance): ...

class ST_SRID(SQLFunction):
    def __init__(self, g, srid: Any | Sentinel = IGNORED): ...

class ST_StartPoint(SQLFunction):
    def __init__(self, ls): ...

class ST_SwapXY(SQLFunction):
    def __init__(self, g): ...

class ST_SymDifference(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_Touches(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_Transform(SQLFunction):
    def __init__(self, g, target_srid): ...

class ST_Union(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_Validate(SQLFunction):
    def __init__(self, g): ...

class ST_Within(SQLFunction):
    def __init__(self, g1, g2): ...

class ST_X(SQLFunction):
    def __init__(self, p, new_x_val: Any | Sentinel = IGNORED): ...

class ST_Y(SQLFunction):
    def __init__(self, p, new_y_val: Any | Sentinel = IGNORED): ...

class STR_TO_DATE(SQLFunction):
    def __init__(self, string, format): ...

class STRCMP(SQLFunction):
    def __init__(self, expr1, expr2): ...

class SUBDATE(SQLFunction):
    def __init__(self, date, expr): ...

class SUBSTRING(SQLFunction):
    def __init__(self, string, pos, length: Any | Sentinel = IGNORED): ...

class SUBSTRING_INDEX(SQLFunction):
    def __init__(self, string, delim, count): ...

class SUBTIME(SQLFunction):
    def __init__(self, expr1, expr2): ...

class SYSDATE(SQLFunction):
    def __init__(self, fsp: Any | Sentinel = IGNORED): ...

# Functions: T
class TAN(SQLFunction):
    def __init__(self, X): ...

class TIME(SQLFunction):
    def __init__(self, expr): ...

class TIME_FORMAT(SQLFunction):
    def __init__(self, time, format): ...

class TIME_TO_SEC(SQLFunction):
    def __init__(self, time): ...

class TIMEDIFF(SQLFunction):
    def __init__(self, expr1, expr2): ...

class TIMESTAMP(SQLFunction):
    def __init__(self, expr1, expr2: Any | Sentinel = IGNORED): ...

class TIMESTAMPADD(SQLFunction):
    def __init__(self, unit, interval, datetime_expr): ...

class TIMESTAMPDIFF(SQLFunction):
    def __init__(self, unit, datetime_expr1, datetime_expr2): ...

class TO_DAYS(SQLFunction):
    def __init__(self, date): ...

class TO_SECONDS(SQLFunction):
    def __init__(self, expr): ...

class TRIM(SQLFunction):
    def __init__(
        self,
        string,
        remstr: Any | Sentinel = IGNORED,
        mode: str | Sentinel = IGNORED,
    ): ...

class TRUNCATE(SQLFunction):
    def __init__(self, X, D): ...

# Functions: U
class UNCOMPRESS(SQLFunction):
    def __init__(self, string): ...

class UNCOMPRESSED_LENGTH(SQLFunction):
    def __init__(self, compressed_string): ...

class UNHEX(SQLFunction):
    def __init__(self, string): ...

class UNIX_TIMESTAMP(SQLFunction):
    def __init__(self, date: Any | Sentinel = IGNORED): ...

class UPPER(SQLFunction):
    def __init__(self, string): ...

class USER(SQLFunction):
    def __init__(self): ...

class UTC_DATE(SQLFunction):
    def __init__(self): ...

class UTC_TIME(SQLFunction):
    def __init__(self, fsp: int | Sentinel = IGNORED): ...

class UTC_TIMESTAMP(SQLFunction):
    def __init__(self, fsp: int | Sentinel = IGNORED): ...

class UUID(SQLFunction):
    def __init__(self): ...

class UUID_SHORT(SQLFunction):
    def __init__(self): ...

class UUID_TO_BIN(SQLFunction):
    def __init__(self, string_uuid, swap_flag: Any | Sentinel = IGNORED): ...

# Functions: V
class VALIDATE_PASSWORD_STRENGTH(SQLFunction):
    def __init__(self, string): ...

class VERSION(SQLFunction):
    def __init__(self): ...

# Functions: W
class WEEK(SQLFunction):
    def __init__(self, date, mode: int | Sentinel = IGNORED): ...

class WEEKDAY(SQLFunction):
    def __init__(self, date): ...

class WEEKOFYEAR(SQLFunction):
    def __init__(self, date): ...

# Functions: Y
class YEAR(SQLFunction):
    def __init__(self, date): ...

class YEARWEEK(SQLFunction):
    def __init__(self, date, mode: int | Sentinel = IGNORED): ...
