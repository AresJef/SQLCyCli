import unittest, time, datetime
from sqlcycli import escape, sqlfunc, sqlintvl


class TestCase(unittest.TestCase):
    name: str = "Case"

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self._start_time = None
        self._ended_time = None
        self._hashs = set()

    def test_all(self) -> None:
        pass

    # utils
    def log_start(self, msg: str) -> None:
        msg = "START TEST '%s': %s" % (self.name, msg)
        print(msg.ljust(60), end="\r")
        self._start_time = time.perf_counter()

    def log_ended(self, msg: str, skip: bool = False) -> None:
        self._ended_time = time.perf_counter()
        msg = "%s TEST '%s': %s" % ("SKIP" if skip else "PASS", self.name, msg)
        if self._start_time is not None:
            msg += " (%.6fs)" % (self._ended_time - self._start_time)
        print(msg.ljust(60))

    def reset_hashs(self) -> None:
        self._hashs.clear()

    def validate_hash(self, obj: object) -> None:
        h = hash(obj)
        self.assertTrue(h not in self._hashs)
        self._hashs.add(h)

    def compare(self, obj: object, expected: str) -> None:
        self.validate_hash(obj)
        self.assertEqual(escape(obj), expected)


class TestCustomClass(TestCase):
    name: str = "Custom Class"

    def test_all(self) -> None:
        self.test_rawtext()

    def test_rawtext(self) -> None:
        self.log_start("RawText")

        arg = sqlfunc.RawText("apple")
        self.assertEqual(escape(arg), "apple")

        self.log_ended("RawText")

    def test_objstr(self) -> None:
        self.log_start("ObjStr")

        class CustomClass(sqlfunc.ObjStr):
            def __str__(self) -> str:
                return "apple"

        arg = CustomClass()
        self.assertEqual(escape(arg), "apple")

        self.log_ended("ObjStr")


class TestSQLInterval(TestCase):
    name: str = "SQLInterval"

    def test_all(self) -> None:
        self.reset_hashs()
        self.log_start("escape")
        # fmt: off

        self.compare(sqlintvl.MICROSECOND(5), "INTERVAL 5 MICROSECOND")
        self.compare(sqlintvl.SECOND(5), "INTERVAL 5 SECOND")
        self.compare(sqlintvl.MINUTE(5), "INTERVAL 5 MINUTE")
        self.compare(sqlintvl.HOUR(5), "INTERVAL 5 HOUR")
        self.compare(sqlintvl.DAY(5), "INTERVAL 5 DAY")
        self.compare(sqlintvl.WEEK(5), "INTERVAL 5 WEEK")
        self.compare(sqlintvl.MONTH(5), "INTERVAL 5 MONTH")
        self.compare(sqlintvl.QUARTER(5), "INTERVAL 5 QUARTER")
        self.compare(sqlintvl.YEAR(5), "INTERVAL 5 YEAR")
        self.compare(sqlintvl.SECOND_MICROSECOND("01.000001"), "INTERVAL '01.000001' SECOND_MICROSECOND")
        self.compare(sqlintvl.MINUTE_MICROSECOND("01:01.000001"), "INTERVAL '01:01.000001' MINUTE_MICROSECOND")
        self.compare(sqlintvl.MINUTE_SECOND("01:01"), "INTERVAL '01:01' MINUTE_SECOND")
        self.compare(sqlintvl.HOUR_MICROSECOND("01:01:01.000001"), "INTERVAL '01:01:01.000001' HOUR_MICROSECOND")
        self.compare(sqlintvl.HOUR_SECOND("01:01:01"), "INTERVAL '01:01:01' HOUR_SECOND")
        self.compare(sqlintvl.HOUR_MINUTE("01:01"), "INTERVAL '01:01' HOUR_MINUTE")
        self.compare(sqlintvl.DAY_MICROSECOND("01 01:01:01.000001"), "INTERVAL '01 01:01:01.000001' DAY_MICROSECOND")
        self.compare(sqlintvl.DAY_SECOND("01 01:01:01"), "INTERVAL '01 01:01:01' DAY_SECOND")
        self.compare(sqlintvl.DAY_MINUTE("01 01:01"), "INTERVAL '01 01:01' DAY_MINUTE")
        self.compare(sqlintvl.DAY_HOUR("01 01"), "INTERVAL '01 01' DAY_HOUR")
        self.compare(sqlintvl.YEAR_MONTH("2-5"), "INTERVAL '2-5' YEAR_MONTH")

        # fmt: on
        self.log_ended("escape")


class TestSQLFunction(TestCase):
    name: str = "SQLFunction"

    def test_all(self) -> None:
        self.reset_hashs()
        self.log_start("escape")
        # fmt: off

        # Custom: RANDINT
        self.compare(sqlfunc.RANDINT(1, 10), "FLOOR(1 + RAND() * 9)")
        # ABS
        self.compare(sqlfunc.ABS(-1), "ABS(-1)")
        self.compare(sqlfunc.ABS("-2"), "ABS('-2')")
        # ACOS
        self.compare(sqlfunc.ACOS(1), "ACOS(1)")
        self.compare(sqlfunc.ACOS(1.0001), "ACOS(1.0001)")
        self.compare(sqlfunc.ACOS("0"), "ACOS('0')")
        # ADDDATE
        self.compare(sqlfunc.ADDDATE("2008-01-02", 31), "ADDDATE('2008-01-02',31)",)
        self.compare(sqlfunc.ADDDATE(datetime.date(2008, 1, 2), sqlintvl.DAY(31)), "ADDDATE('2008-01-02',INTERVAL 31 DAY)")
        self.compare(sqlfunc.ADDDATE(datetime.datetime(2008, 1, 2), sqlintvl.DAY(31)), "ADDDATE('2008-01-02 00:00:00',INTERVAL 31 DAY)")
        # ADDTIME
        self.compare(sqlfunc.ADDTIME("2007-12-31 23:59:59.999999", "1 1:1:1.000002"), "ADDTIME('2007-12-31 23:59:59.999999','1 1:1:1.000002')")
        self.compare(sqlfunc.ADDTIME(datetime.datetime(2007, 12, 31, 23,59, 59, 999999), "1 1:1:1.000002"), "ADDTIME('2007-12-31 23:59:59.999999','1 1:1:1.000002')")
        self.compare(sqlfunc.ADDTIME(datetime.time(1, 0, 0, 999999), "02:00:00.999998"), "ADDTIME('01:00:00.999999','02:00:00.999998')")
        # ASCII
        self.compare(sqlfunc.ASCII("2"), "ASCII('2')")
        self.compare(sqlfunc.ASCII(2), "ASCII(2)")
        self.compare(sqlfunc.ASCII("dx"), "ASCII('dx')")
        # ASIN
        self.compare(sqlfunc.ASIN(0.2), "ASIN(0.2)")
        self.compare(sqlfunc.ASIN("0.2"), "ASIN('0.2')")
        # ATAN
        self.compare(sqlfunc.ATAN(2), "ATAN(2)")
        self.compare(sqlfunc.ATAN("-2"), "ATAN('-2')")
        self.compare(sqlfunc.ATAN(-2, 2), "ATAN(-2,2)")
        # BIN
        self.compare(sqlfunc.BIN(12), "BIN(12)")
        self.compare(sqlfunc.BIN("12"), "BIN('12')")
        # BIN_TO_UUID
        uuid = "6ccd780c-baba-1026-9564-5b8c656024db"
        self.compare(sqlfunc.BIN_TO_UUID(sqlfunc.UUID_TO_BIN(uuid)), f"BIN_TO_UUID(UUID_TO_BIN('{uuid}'))")
        self.compare(sqlfunc.BIN_TO_UUID(sqlfunc.UUID_TO_BIN(uuid), 1), f"BIN_TO_UUID(UUID_TO_BIN('{uuid}'),1)")
        # BIT_COUNT
        self.compare(sqlfunc.BIT_COUNT(64), "BIT_COUNT(64)")
        self.compare(sqlfunc.BIT_COUNT(b"64"), "BIT_COUNT(_binary'64')")
        # BIT_LENGHT
        self.compare(sqlfunc.BIT_LENGTH('text'), "BIT_LENGTH('text')")
        # CEIL
        self.compare(sqlfunc.CEIL(1.23), "CEIL(1.23)")
        self.compare(sqlfunc.CEIL("-1.23"), "CEIL('-1.23')")
        # CHAR
        self.compare(sqlfunc.CHAR(77, 121, 83, 81, '76'), "CHAR(77,121,83,81,'76')")
        self.compare(sqlfunc.CHAR(77, 121, 83, 81, "76", using="utf8mb4"), "CHAR(77,121,83,81,'76' USING utf8mb4)")
        # CHAR_LENGTH
        self.compare(sqlfunc.CHAR_LENGTH('text'), "CHAR_LENGTH('text')")
        self.compare(sqlfunc.CHAR_LENGTH('海豚'), "CHAR_LENGTH('海豚')")
        # CHARSET
        self.compare(sqlfunc.CHARSET('abs'), "CHARSET('abs')")
        self.compare(sqlfunc.CHARSET(sqlfunc.CONVERT('abc', 'latin1')), "CHARSET(CONVERT('abc' USING latin1))")
        # COALESCE
        self.compare(sqlfunc.COALESCE(None, 1), "COALESCE(NULL,1)")
        self.compare(sqlfunc.COALESCE(None, None, None), "COALESCE(NULL,NULL,NULL)")
        # COERCIBILITY
        self.compare(sqlfunc.COERCIBILITY("abc"), "COERCIBILITY('abc')")
        # COLLATION
        self.compare(sqlfunc.COLLATION("abc"), "COLLATION('abc')")
        # COMPRESS
        self.compare(sqlfunc.LENGTH(sqlfunc.COMPRESS("")), "LENGTH(COMPRESS(''))")
        self.compare(sqlfunc.LENGTH(sqlfunc.COMPRESS("a")), "LENGTH(COMPRESS('a'))")
        # CONCAT
        self.compare(sqlfunc.CONCAT("My", "S", "QL"), "CONCAT('My','S','QL')")
        self.compare(sqlfunc.CONCAT("My", None, "QL"), "CONCAT('My',NULL,'QL')")
        self.compare(sqlfunc.CONCAT(14.3), "CONCAT(14.3)")
        # CONCAT_WS
        self.compare(sqlfunc.CONCAT_WS(",", "First name", "Second name", "Last Name"), "CONCAT_WS(',','First name','Second name','Last Name')")
        self.compare(sqlfunc.CONCAT_WS(",", "First name", None, "Last Name"), "CONCAT_WS(',','First name',NULL,'Last Name')")
        # CONNECTION_ID
        self.compare(sqlfunc.CONNECTION_ID(), "CONNECTION_ID()")
        # CONV
        self.compare(sqlfunc.CONV("a",16,2), "CONV('a',16,2)")
        self.compare(sqlfunc.CONV("6E",18,8), "CONV('6E',18,8)")
        # CONVERT
        self.compare(sqlfunc.CONVERT('MySQL', 'utf8mb4'), "CONVERT('MySQL' USING utf8mb4)")
        self.compare(sqlfunc.CONVERT('MySQL', 'latin1'), "CONVERT('MySQL' USING latin1)")
        # CONVERT_TZ
        self.compare(sqlfunc.CONVERT_TZ("2004-01-01 12:00:00", "+00:00", "+10:00"), "CONVERT_TZ('2004-01-01 12:00:00','+00:00','+10:00')")
        self.compare(sqlfunc.CONVERT_TZ(datetime.datetime(2004, 1, 1, 12), "+00:00", "+10:00"), "CONVERT_TZ('2004-01-01 12:00:00','+00:00','+10:00')")
        # COS
        self.compare(sqlfunc.COS(0), "COS(0)")
        self.compare(sqlfunc.COS(sqlfunc.PI()), "COS(PI())")
        # COT
        self.compare(sqlfunc.COT(12), "COT(12)")
        self.compare(sqlfunc.COT("12"), "COT('12')")
        # CRC32
        self.compare(sqlfunc.CRC32('MySQL'), "CRC32('MySQL')")
        self.compare(sqlfunc.CRC32("mysql"), "CRC32('mysql')")
        # CUME_DIST
        self.compare(sqlfunc.CUME_DIST(), "CUME_DIST()")
        # CURRENT_DATE
        self.compare(sqlfunc.CURRENT_DATE(), "CURRENT_DATE()")
        # CURRENT_ROLE
        self.compare(sqlfunc.CURRENT_ROLE(), "CURRENT_ROLE()")
        # CURRENT_TIME
        self.compare(sqlfunc.CURRENT_TIME(), "CURRENT_TIME()")
        self.compare(sqlfunc.CURRENT_TIME(3), "CURRENT_TIME(3)")
        # CURRENT_TIMESTAMP
        self.compare(sqlfunc.CURRENT_TIMESTAMP(), "CURRENT_TIMESTAMP()")
        self.compare(sqlfunc.CURRENT_TIMESTAMP(3), "CURRENT_TIMESTAMP(3)")
        # CURRENT_USER
        self.compare(sqlfunc.CURRENT_USER(), "CURRENT_USER()")
        # DATABASE
        self.compare(sqlfunc.DATABASE(), "DATABASE()")
        # DATE
        self.compare(sqlfunc.DATE("2003-12-31 01:02:03"), "DATE('2003-12-31 01:02:03')")
        self.compare(sqlfunc.DATE(datetime.datetime(2003, 12, 31, 1, 2, 3)), "DATE('2003-12-31 01:02:03')")
        # DATE_ADD
        self.compare(sqlfunc.DATE_ADD("2021-01-01", sqlintvl.WEEK(1)), "DATE_ADD('2021-01-01',INTERVAL 1 WEEK)")
        self.compare(sqlfunc.DATE_ADD(datetime.date(2021, 1, 1), sqlintvl.DAY(1)), "DATE_ADD('2021-01-01',INTERVAL 1 DAY)")
        # DATE_FORMAT
        self.compare(sqlfunc.DATE_FORMAT("2009-10-04 22:23:00", "%W %M %Y"), "DATE_FORMAT('2009-10-04 22:23:00','%W %M %Y')")
        self.compare(sqlfunc.DATE_FORMAT(datetime.datetime(2007, 10, 4, 22, 23), "%H:%i:%s"), "DATE_FORMAT('2007-10-04 22:23:00','%H:%i:%s')")
        # DATE_SUB
        self.compare(sqlfunc.DATE_SUB("2021-01-08", sqlintvl.WEEK(1)), "DATE_SUB('2021-01-08',INTERVAL 1 WEEK)")
        self.compare(sqlfunc.DATE_SUB(datetime.date(2021, 1, 2), sqlintvl.DAY(1)), "DATE_SUB('2021-01-02',INTERVAL 1 DAY)")
        # DATEDIFF
        self.compare(sqlfunc.DATEDIFF("2007-12-31 23:59:59", "2007-12-30"), "DATEDIFF('2007-12-31 23:59:59','2007-12-30')")
        self.compare(sqlfunc.DATEDIFF(datetime.datetime(2010, 11, 30, 23, 59, 59), "2010-12-31"), "DATEDIFF('2010-11-30 23:59:59','2010-12-31')")
        # DATENAME
        self.compare(sqlfunc.DAYNAME("2007-02-03"), "DAYNAME('2007-02-03')")
        self.compare(sqlfunc.DAYNAME(datetime.date(2007, 2, 3)), "DAYNAME('2007-02-03')")
        # DAYOFMONTH
        self.compare(sqlfunc.DAYOFMONTH("2007-02-03"), "DAYOFMONTH('2007-02-03')")
        self.compare(sqlfunc.DAYOFMONTH(datetime.date(2007, 2, 3)), "DAYOFMONTH('2007-02-03')")
        # DAYOFWEEK
        self.compare(sqlfunc.DAYOFWEEK("2007-02-03"), "DAYOFWEEK('2007-02-03')")
        self.compare(sqlfunc.DAYOFWEEK(datetime.date(2007, 2, 3)), "DAYOFWEEK('2007-02-03')")
        # DAYOFYEAR
        self.compare(sqlfunc.DAYOFYEAR("2007-02-03"), "DAYOFYEAR('2007-02-03')")
        self.compare(sqlfunc.DAYOFYEAR(datetime.date(2007, 2, 3)), "DAYOFYEAR('2007-02-03')")
        # DEGREES
        self.compare(sqlfunc.DEGREES(sqlfunc.PI()), "DEGREES(PI())")
        # DENSE_RANK
        self.compare(sqlfunc.DENSE_RANK(), "DENSE_RANK()")
        # ELT
        self.compare(sqlfunc.ELT(1, "Aa", "Bb", "Cc", "Dd"), "ELT(1,'Aa','Bb','Cc','Dd')")
        self.compare(sqlfunc.ELT(4, "Aa", "Bb", "Cc", "Dd"), "ELT(4,'Aa','Bb','Cc','Dd')")
        # EXP
        self.compare(sqlfunc.EXP(2), "EXP(2)")
        self.compare(sqlfunc.EXP("-2"), "EXP('-2')")
        self.compare(sqlfunc.EXP(0), "EXP(0)")
        # EXPORT_SET
        self.compare(sqlfunc.EXPORT_SET(5,"Y","N",",",4), "EXPORT_SET(5,'Y','N',',',4)")
        self.compare(sqlfunc.EXPORT_SET(6,"1","0",",",10), "EXPORT_SET(6,'1','0',',',10)")
        self.compare(sqlfunc.EXPORT_SET(6,"1","0",""), "EXPORT_SET(6,'1','0','')")
        # EXTRACT
        self.compare(sqlfunc.EXTRACT("YEAR", datetime.date(2007, 12, 31)), "EXTRACT(YEAR FROM '2007-12-31')")
        self.compare(sqlfunc.EXTRACT(sqlintvl.MONTH, "2007-12-31"), "EXTRACT(MONTH FROM '2007-12-31')")
        self.compare(sqlfunc.EXTRACT(sqlintvl.YEAR_MONTH, datetime.date(2007, 12, 31)), "EXTRACT(YEAR_MONTH FROM '2007-12-31')")
        # FIELD
        self.compare(sqlfunc.FIELD("Bb", "Aa", "Bb", "Cc", "Dd", "Ff"), "FIELD('Bb','Aa','Bb','Cc','Dd','Ff')")
        self.compare(sqlfunc.FIELD("Gg", "Aa", "Bb", "Cc", "Dd", "Ff"), "FIELD('Gg','Aa','Bb','Cc','Dd','Ff')")
        # FIND_IN_SET
        self.compare(sqlfunc.FIND_IN_SET("b", "a,b,c,d"), "FIND_IN_SET('b','a,b,c,d')")
        self.compare(sqlfunc.FIND_IN_SET("b", ("a", "b", "c", "d", "e")), "FIND_IN_SET('b','a,b,c,d,e')")
        # FLOOR
        self.compare(sqlfunc.FLOOR(1.23), "FLOOR(1.23)")
        self.compare(sqlfunc.FLOOR("-1.23"), "FLOOR('-1.23')")
        # FORMAT
        self.compare(sqlfunc.FORMAT(12332.123456, 4), "FORMAT(12332.123456,4)")
        self.compare(sqlfunc.FORMAT("12332.2", 0), "FORMAT('12332.2',0)")
        self.compare(sqlfunc.FORMAT(12332.2, 2, "de_DE"), "FORMAT(12332.2,2,'de_DE')")
        # FORMAT_BYTES
        self.compare(sqlfunc.FORMAT_BYTES(512), "FORMAT_BYTES(512)")
        self.compare(sqlfunc.FORMAT_BYTES(18446644073709551615), "FORMAT_BYTES(18446644073709551615)")
        # FORMAT_PICO_TIME
        self.compare(sqlfunc.FORMAT_PICO_TIME(3501), "FORMAT_PICO_TIME(3501)")
        self.compare(sqlfunc.FORMAT_PICO_TIME(188732396662000), "FORMAT_PICO_TIME(188732396662000)")
        # FROM_DAY
        self.compare(sqlfunc.FROM_DAYS(730669), "FROM_DAYS(730669)")
        self.compare(sqlfunc.FROM_DAYS("736695"), "FROM_DAYS('736695')")
        # FROM_UNIXTIME
        self.compare(sqlfunc.FROM_UNIXTIME(1447430881), "FROM_UNIXTIME(1447430881)")
        self.compare(sqlfunc.FROM_UNIXTIME(1447430881, "%Y %D %M %h:%i:%s %x"), "FROM_UNIXTIME(1447430881,'%Y %D %M %h:%i:%s %x')")
        # GeomCollection
        self.compare(sqlfunc.GeomCollection(sqlfunc.Point(1, 1), sqlfunc.Point(2, 2)), "GeomCollection(Point(1,1),Point(2,2))")
        # GET_FORMAT
        self.compare(sqlfunc.GET_FORMAT("DATE", "EUR"), "GET_FORMAT(DATE,'EUR')")
        self.compare(sqlfunc.GET_FORMAT("DATETIME", "USA"), "GET_FORMAT(DATETIME,'USA')")
        # GET_LOCK
        self.compare(sqlfunc.GET_LOCK("lock1", 10), "GET_LOCK('lock1',10)")
        # GREATEST
        self.compare(sqlfunc.GREATEST(2, 0), "GREATEST(2,0)")
        self.compare(sqlfunc.GREATEST("B", "A", "C"), "GREATEST('B','A','C')")
        # HEX
        self.compare(sqlfunc.HEX("abc"), "HEX('abc')")
        self.compare(sqlfunc.HEX(255), "HEX(255)")
        # HOUR
        self.compare(sqlfunc.HOUR("10:05:03"), "HOUR('10:05:03')")
        self.compare(sqlfunc.HOUR(datetime.time(11, 59, 59)), "HOUR('11:59:59')")
        # IFNULL
        self.compare(sqlfunc.IFNULL(1, 0), "IFNULL(1,0)")
        self.compare(sqlfunc.IFNULL(None, "a"), "IFNULL(NULL,'a')")
        # IN
        self.compare(sqlfunc.IN(0, 3, 5, 7), "IN(0,3,5,7)")
        self.compare(sqlfunc.IN(*["a", "b", "c"]), "IN('a','b','c')")
        # INET_ATON
        self.compare(sqlfunc.INET_ATON("10.0.5.9"), "INET_ATON('10.0.5.9')")
        # INET6_ATON
        self.compare(sqlfunc.HEX(sqlfunc.INET6_ATON("fdfe::5a55:caff:fefa:9089")), "HEX(INET6_ATON('fdfe::5a55:caff:fefa:9089'))")
        self.compare(sqlfunc.HEX(sqlfunc.INET6_ATON("10.0.5.9")), "HEX(INET6_ATON('10.0.5.9'))")
        # INET_NTOA
        self.compare(sqlfunc.INET_NTOA(167773449), "INET_NTOA(167773449)")
        # INET6_NTOA
        self.compare(sqlfunc.INET6_NTOA(sqlfunc.INET6_ATON("fdfe::5a55:caff:fefa:9089")), "INET6_NTOA(INET6_ATON('fdfe::5a55:caff:fefa:9089'))")
        self.compare(sqlfunc.INET6_NTOA(sqlfunc.INET6_ATON("10.0.5.9")), "INET6_NTOA(INET6_ATON('10.0.5.9'))")
        # INSERT
        self.compare(sqlfunc.INSERT("Quadratic", 3, 4, "What"), "INSERT('Quadratic',3,4,'What')")
        self.compare(sqlfunc.INSERT("Quadratic", -1, 4, "What"), "INSERT('Quadratic',-1,4,'What')")
        self.compare(sqlfunc.INSERT("Quadratic", 3, 100, "What"), "INSERT('Quadratic',3,100,'What')")
        # INSTR
        self.compare(sqlfunc.INSTR("foobarbar", "bar"), "INSTR('foobarbar','bar')")
        self.compare(sqlfunc.INSTR("xbar", "foobar"), "INSTR('xbar','foobar')")
        # INTERVAL
        self.compare(sqlfunc.INTERVAL(23, 1, 15, 17, 30, 44, 200), "INTERVAL(23,1,15,17,30,44,200)")
        self.compare(sqlfunc.INTERVAL(22, 23, 30, 44, 200), "INTERVAL(22,23,30,44,200)")
        # IS_FREE_LOCK
        self.compare(sqlfunc.IS_FREE_LOCK("lock1"), "IS_FREE_LOCK('lock1')")
        # IS_USED_LOCK
        self.compare(sqlfunc.IS_USER_LOCK("user1"), "IS_USER_LOCK('user1')")
        # IS_UUID
        self.compare(sqlfunc.IS_UUID("6ccd780c-baba-1026-9564-5b8c656024db"), "IS_UUID('6ccd780c-baba-1026-9564-5b8c656024db')")
        self.compare(sqlfunc.IS_UUID("6ccd780c-baba-1026-9564-5b8c6560"), "IS_UUID('6ccd780c-baba-1026-9564-5b8c6560')")
        # ISNULL
        self.compare(sqlfunc.ISNULL(None), "ISNULL(NULL)")
        self.compare(sqlfunc.ISNULL(0), "ISNULL(0)")
        # JSON_ARRAY
        self.compare(sqlfunc.JSON_ARRAY(1, "abc", None, sqlfunc.CURRENT_TIME()), "JSON_ARRAY(1,'abc',NULL,CURRENT_TIME())")
        # JSON_ARRAY_APPEND
        js = '["a", ["b", "c"], "d"]'
        self.compare(sqlfunc.JSON_ARRAY_APPEND(js, "$[1]", 1), 
            'JSON_ARRAY_APPEND(\'[\\"a\\", [\\"b\\", \\"c\\"], \\"d\\"]\',\'$[1]\',1)')
        self.compare(sqlfunc.JSON_ARRAY_APPEND(js, "$[1]", 2, "$[1]", 3),
            'JSON_ARRAY_APPEND(\'[\\"a\\", [\\"b\\", \\"c\\"], \\"d\\"]\',\'$[1]\',2,\'$[1]\',3)')
        # JSON_ARRAY_INSERT
        js = '["a", {"b": [1, 2]}, [3, 4]]'
        self.compare(sqlfunc.JSON_ARRAY_INSERT(js, "$[1]", "x"), 
            'JSON_ARRAY_INSERT(\'[\\"a\\", {\\"b\\": [1, 2]}, [3, 4]]\',\'$[1]\',\'x\')')
        self.compare(sqlfunc.JSON_ARRAY_INSERT(js, "$[0]", "x", "$[2][1]", "y"),
            'JSON_ARRAY_INSERT(\'[\\"a\\", {\\"b\\": [1, 2]}, [3, 4]]\',\'$[0]\',\'x\',\'$[2][1]\',\'y\')')
        # JSON_CONTAINS
        js1, js2 = '{"a": 1, "b": 2, "c": {"d": 4}}', "1"
        self.compare(sqlfunc.JSON_CONTAINS(js1, js2), 
            'JSON_CONTAINS(\'{\\"a\\": 1, \\"b\\": 2, \\"c\\": {\\"d\\": 4}}\',\'1\')')
        self.compare(sqlfunc.JSON_CONTAINS(js1, js2, '$.a'),
            'JSON_CONTAINS(\'{\\"a\\": 1, \\"b\\": 2, \\"c\\": {\\"d\\": 4}}\',\'1\',\'$.a\')')
        # JSON_CONTAINS_PATH
        js = '{"a": 1, "b": 2, "c": {"d": 4}}'
        self.compare(sqlfunc.JSON_CONTAINS_PATH(js, "one", "$.a", "$.e"),
            'JSON_CONTAINS_PATH(\'{\\"a\\": 1, \\"b\\": 2, \\"c\\": {\\"d\\": 4}}\',\'one\',\'$.a\',\'$.e\')')
        self.compare(sqlfunc.JSON_CONTAINS_PATH(js, "all", "$.a", "$.e"),
            'JSON_CONTAINS_PATH(\'{\\"a\\": 1, \\"b\\": 2, \\"c\\": {\\"d\\": 4}}\',\'all\',\'$.a\',\'$.e\')')
        # JSON_DEPTH
        self.compare(sqlfunc.JSON_DEPTH("{}"), "JSON_DEPTH('{}')")
        self.compare(sqlfunc.JSON_DEPTH("[10, 20]"), "JSON_DEPTH('[10, 20]')")
        self.compare(sqlfunc.JSON_DEPTH('[10, {"a": 20}]'), 'JSON_DEPTH(\'[10, {\\"a\\": 20}]\')')
        # JSON_EXTRACT
        self.compare(sqlfunc.JSON_EXTRACT("[10, 20, [30, 40]]", "$[1]"),
            "JSON_EXTRACT('[10, 20, [30, 40]]','$[1]')")
        self.compare(sqlfunc.JSON_EXTRACT("[10, 20, [30, 40]]", "$[1]", "$[0]"),
            "JSON_EXTRACT('[10, 20, [30, 40]]','$[1]','$[0]')")
        # JSON_INSERT
        js = '{ "a": 1, "b": [2, 3]}'
        self.compare(sqlfunc.JSON_INSERT(js, "$.a", 10, "$.c", "[true, false]"),
            'JSON_INSERT(\'{ \\"a\\": 1, \\"b\\": [2, 3]}\',\'$.a\',10,\'$.c\',\'[true, false]\')')
        # JSON_KEYS
        self.compare(sqlfunc.JSON_KEYS('{"a": 1, "b": {"c": 30}}'),
            'JSON_KEYS(\'{\\"a\\": 1, \\"b\\": {\\"c\\": 30}}\')')
        self.compare(sqlfunc.JSON_KEYS('{"a": 1, "b": {"c": 30}}', '$.b'),
            'JSON_KEYS(\'{\\"a\\": 1, \\"b\\": {\\"c\\": 30}}\',\'$.b\')')
        # JSON_LENGTH
        self.compare(sqlfunc.JSON_LENGTH('[1, 2, {"a": 3}]'), 'JSON_LENGTH(\'[1, 2, {\\"a\\": 3}]\')')
        self.compare(sqlfunc.JSON_LENGTH('{"a": 1, "b": {"c": 30}}', '$.b'),
            'JSON_LENGTH(\'{\\"a\\": 1, \\"b\\": {\\"c\\": 30}}\',\'$.b\')')
        # JSON_MERGE_PATCH
        self.compare(sqlfunc.JSON_MERGE_PATCH("[1, 2]", "[true, false]"),
            "JSON_MERGE_PATCH('[1, 2]','[true, false]')")
        self.compare(sqlfunc.JSON_MERGE_PATCH('{"name": "x"}', '{"id": 47}'),
            'JSON_MERGE_PATCH(\'{\\"name\\": \\"x\\"}\',\'{\\"id\\": 47}\')')
        self.compare(sqlfunc.JSON_MERGE_PATCH('{"a": 1, "b":2}','{"a": 3, "c":4}','{"a": 5, "d":6}'),
            'JSON_MERGE_PATCH(\'{\\"a\\": 1, \\"b\\":2}\',\'{\\"a\\": 3, \\"c\\":4}\',\'{\\"a\\": 5, \\"d\\":6}\')')
        # JSON_MERGE_PRESERVE
        self.compare(sqlfunc.JSON_MERGE_PRESERVE("[1, 2]", "[true, false]"),
            "JSON_MERGE_PRESERVE('[1, 2]','[true, false]')")
        self.compare(sqlfunc.JSON_MERGE_PRESERVE('{"name": "x"}', '{"id": 47}'),
            'JSON_MERGE_PRESERVE(\'{\\"name\\": \\"x\\"}\',\'{\\"id\\": 47}\')')
        self.compare(sqlfunc.JSON_MERGE_PRESERVE('{ "a": 1, "b": 2 }','{ "a": 3, "c": 4 }','{ "a": 5, "d": 6 }'),
            'JSON_MERGE_PRESERVE(\'{ \\"a\\": 1, \\"b\\": 2 }\',\'{ \\"a\\": 3, \\"c\\": 4 }\',\'{ \\"a\\": 5, \\"d\\": 6 }\')')
        # JSON_OBJECT
        self.compare(sqlfunc.JSON_OBJECT("id", 87, "name", "carrot"), "JSON_OBJECT('id',87,'name','carrot')")
        # JSON_OVERLAPS
        self.compare(sqlfunc.JSON_OVERLAPS("[1,3,5,7]", "[2,5,7]"), "JSON_OVERLAPS('[1,3,5,7]','[2,5,7]')")
        self.compare(sqlfunc.JSON_OVERLAPS("[1,3,5,7]", "[2,6,8]"), "JSON_OVERLAPS('[1,3,5,7]','[2,6,8]')")
        # JSON_PRETTY
        self.compare(sqlfunc.JSON_PRETTY("[1,3,5]"), "JSON_PRETTY('[1,3,5]')")
        self.compare(sqlfunc.JSON_PRETTY('{"a":"10","b":"15","x":"25"}'),
            'JSON_PRETTY(\'{\\"a\\":\\"10\\",\\"b\\":\\"15\\",\\"x\\":\\"25\\"}\')')
        # JSON_QUOTE
        self.compare(sqlfunc.JSON_QUOTE("null"), "JSON_QUOTE('null')")
        self.compare(sqlfunc.JSON_QUOTE('"null"'), 'JSON_QUOTE(\'\\"null\\"\')')
        self.compare(sqlfunc.JSON_QUOTE("[1, 2, 3]"), "JSON_QUOTE('[1, 2, 3]')")
        # JSON_REMOVE
        js = '["a", ["b", "c"], "d"]'
        self.compare(sqlfunc.JSON_REMOVE(js, "$[1]"), 
            'JSON_REMOVE(\'[\\"a\\", [\\"b\\", \\"c\\"], \\"d\\"]\',\'$[1]\')')
        self.compare(sqlfunc.JSON_REMOVE(js, "$[1]", "$[1]"),
            'JSON_REMOVE(\'[\\"a\\", [\\"b\\", \\"c\\"], \\"d\\"]\',\'$[1]\',\'$[1]\')')
        # JSON_REPLACE
        self.compare(sqlfunc.JSON_REPLACE(js, "$.a", 10, "$.c", "[true, false]"),
            'JSON_REPLACE(\'[\\"a\\", [\\"b\\", \\"c\\"], \\"d\\"]\',\'$.a\',10,\'$.c\',\'[true, false]\')')
        self.compare(sqlfunc.JSON_REPLACE(js, "$.a", None, "$.c", "[true, false]"),
            'JSON_REPLACE(\'[\\"a\\", [\\"b\\", \\"c\\"], \\"d\\"]\',\'$.a\',NULL,\'$.c\',\'[true, false]\')')
        self.compare(sqlfunc.JSON_REPLACE(None, "$.a", 10, "$.c", "[true, false]"),
            "JSON_REPLACE(NULL,'$.a',10,'$.c','[true, false]')")
        self.compare(sqlfunc.JSON_REPLACE(js, None, 10, "$.c", "[true, false]"),
            'JSON_REPLACE(\'[\\"a\\", [\\"b\\", \\"c\\"], \\"d\\"]\',NULL,10,\'$.c\',\'[true, false]\')')
        # JSON_SEARCH
        js = '["abc", [{"k": "10"}, "def"], {"x":"abc"}, {"y":"bcd"}]'
        self.compare(sqlfunc.JSON_SEARCH(js, "one", "abc"),
            'JSON_SEARCH(\'[\\"abc\\", [{\\"k\\": \\"10\\"}, \\"def\\"], {\\"x\\":\\"abc\\"}, {\\"y\\":\\"bcd\\"}]\',\'one\',\'abc\')')
        self.compare(sqlfunc.JSON_SEARCH(js, "all", "abc"),
            'JSON_SEARCH(\'[\\"abc\\", [{\\"k\\": \\"10\\"}, \\"def\\"], {\\"x\\":\\"abc\\"}, {\\"y\\":\\"bcd\\"}]\',\'all\',\'abc\')')
        self.compare(sqlfunc.JSON_SEARCH(js, "all", "10", None, "$"),
            'JSON_SEARCH(\'[\\"abc\\", [{\\"k\\": \\"10\\"}, \\"def\\"], {\\"x\\":\\"abc\\"}, {\\"y\\":\\"bcd\\"}]\',\'all\',\'10\',NULL,\'$\')')
        self.compare(sqlfunc.JSON_SEARCH(js, "all", "%b%", "", "$[3]"),
            'JSON_SEARCH(\'[\\"abc\\", [{\\"k\\": \\"10\\"}, \\"def\\"], {\\"x\\":\\"abc\\"}, {\\"y\\":\\"bcd\\"}]\',\'all\',\'%b%\',\'\',\'$[3]\')')
        # JSON_SET
        js = '{ "a": 1, "b": [2, 3]}'
        self.compare(sqlfunc.JSON_SET(js, "$.a", 10, "$.c", "[true, false]"),
            'JSON_SET(\'{ \\"a\\": 1, \\"b\\": [2, 3]}\',\'$.a\',10,\'$.c\',\'[true, false]\')')
        # JSON_TYPE
        js = '{"a": [10, true]}'
        self.compare(sqlfunc.JSON_TYPE(sqlfunc.JSON_EXTRACT(js, "$.a")),
            'JSON_TYPE(JSON_EXTRACT(\'{\\"a\\": [10, true]}\',\'$.a\'))')
        self.compare(sqlfunc.JSON_TYPE(sqlfunc.JSON_EXTRACT(js, "$.a[0]")),
            'JSON_TYPE(JSON_EXTRACT(\'{\\"a\\": [10, true]}\',\'$.a[0]\'))')
        self.compare(sqlfunc.JSON_TYPE(sqlfunc.JSON_EXTRACT(js, "$.a[1]")),
            'JSON_TYPE(JSON_EXTRACT(\'{\\"a\\": [10, true]}\',\'$.a[1]\'))')
        # JSON_UNQUOTE
        self.compare(sqlfunc.JSON_UNQUOTE('"abc"'), 'JSON_UNQUOTE(\'\\"abc\\"\')')
        self.compare(sqlfunc.JSON_UNQUOTE(sqlfunc.JSON_QUOTE("abc")), "JSON_UNQUOTE(JSON_QUOTE('abc'))")
        # JSON_VALID
        self.compare(sqlfunc.JSON_VALID('{"a": 1}'), 'JSON_VALID(\'{\\"a\\": 1}\')')
        self.compare(sqlfunc.JSON_VALID('hello'), "JSON_VALID('hello')")
        # LAST_DAY
        self.compare(sqlfunc.LAST_DAY("2003-02-05"), "LAST_DAY('2003-02-05')")
        self.compare(sqlfunc.LAST_DAY(datetime.date(2004, 2, 5)), "LAST_DAY('2004-02-05')")
        self.compare(sqlfunc.LAST_DAY("2004-01-01 01:01:01"), "LAST_DAY('2004-01-01 01:01:01')")
        # LEAST
        self.compare(sqlfunc.LEAST(2, 0), "LEAST(2,0)")
        self.compare(sqlfunc.LEAST("B", "A", "C"), "LEAST('B','A','C')")
        # LEFT
        self.compare(sqlfunc.LEFT("foobarbar", 5), "LEFT('foobarbar',5)")
        # LENGTH
        self.compare(sqlfunc.LENGTH("text"), "LENGTH('text')")
        # LingString
        self.compare(sqlfunc.LineString(sqlfunc.Point(1, 1), sqlfunc.Point(2, 2)), "LineString(Point(1,1),Point(2,2))")
        # LN
        self.compare(sqlfunc.LN(2), "LN(2)")
        self.compare(sqlfunc.LN("-2"), "LN('-2')")
        # LOAD_FILE
        self.compare(sqlfunc.LOAD_FILE("/tmp/picture"), "LOAD_FILE('/tmp/picture')")
        # LOCALTIME
        self.compare(sqlfunc.LOCALTIME(), "LOCALTIME()")
        self.compare(sqlfunc.LOCALTIME(3), "LOCALTIME(3)")
        # LOCATE
        self.compare(sqlfunc.LOCATE("bar", "foobarbar"), "LOCATE('bar','foobarbar')")
        self.compare(sqlfunc.LOCATE("bar", "foobarbar", 5), "LOCATE('bar','foobarbar',5)")
        # LOG
        self.compare(sqlfunc.LOG(2), "LOG(2)")
        self.compare(sqlfunc.LOG(2, 10), "LOG(2,10)")
        # LOG2
        self.compare(sqlfunc.LOG2(65536), "LOG2(65536)")
        self.compare(sqlfunc.LOG2("-100"), "LOG2('-100')")
        # LOG10
        self.compare(sqlfunc.LOG10(2), "LOG10(2)")
        self.compare(sqlfunc.LOG10("100"), "LOG10('100')")
        # LOWER
        self.compare(sqlfunc.LOWER("QUADRATICALLY"), "LOWER('QUADRATICALLY')")
        # LPAD
        self.compare(sqlfunc.LPAD("hi", 4, "?"), "LPAD('hi',4,'?')")
        self.compare(sqlfunc.LPAD("hi", 1, "?"), "LPAD('hi',1,'?')")
        # LTRIM
        self.compare(sqlfunc.LTRIM("  barbar"), "LTRIM('  barbar')")
        # MAKE_SET
        self.compare(sqlfunc.MAKE_SET(1, "a", "b", "c"), "MAKE_SET(1,'a','b','c')")
        self.compare(sqlfunc.MAKE_SET(1 | 4, "hello", "nice", "world"), "MAKE_SET(5,'hello','nice','world')")
        self.compare(sqlfunc.MAKE_SET(1 | 4, "hello", "nice", None, "world"), "MAKE_SET(5,'hello','nice',NULL,'world')")
        self.compare(sqlfunc.MAKE_SET(0, "a", "b", "c"), "MAKE_SET(0,'a','b','c')")
        # MAKEDATE
        self.compare(sqlfunc.MAKEDATE(2011, 31), "MAKEDATE(2011,31)")
        self.compare(sqlfunc.MAKEDATE(2011, 0), "MAKEDATE(2011,0)")
        # MAKETIME
        self.compare(sqlfunc.MAKETIME(12, 15, 30), "MAKETIME(12,15,30)")
        self.compare(sqlfunc.MAKETIME(12, 15, 30.123), "MAKETIME(12,15,30.123)")
        # MBRContains
        g1 = "Polygon((0 0,0 3,3 3,3 0,0 0))"
        g2 = "Polygon((1 1,1 2,2 2,2 1,1 1))"
        g3 = "Polygon((5 5,5 10,10 10,10 5,5 5))"
        p1, p2 = "Point(1 1)", "Point(3 3)"
        self.compare(sqlfunc.MBRContains(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2)),
            "MBRContains(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))")
        self.compare(sqlfunc.MBRContains(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g3)),
            "MBRContains(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))")
        self.compare(sqlfunc.MBRContains(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(p1)),
            "MBRContains(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Point(1 1)'))")
        self.compare(sqlfunc.MBRContains(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(p2)),
            "MBRContains(ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'),ST_GeomFromText('Point(3 3)'))")
        # MBRCoveredBy
        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Point(1 1)"
        self.compare(sqlfunc.MBRCoveredBy(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2)),
            "MBRCoveredBy(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Point(1 1)'))")
        self.compare(sqlfunc.MBRCoveredBy(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(g1)),
            "MBRCoveredBy(ST_GeomFromText('Point(1 1)'),ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'))")
        # MBRCovers
        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Point(1 1)"
        self.compare(sqlfunc.MBRCovers(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2)),
            "MBRCovers(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Point(1 1)'))")
        self.compare(sqlfunc.MBRCovers(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(g1)),
            "MBRCovers(ST_GeomFromText('Point(1 1)'),ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'))")
        # MBRDisjoint
        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        g3, g4 = "Polygon((0 0,0 5,5 5,5 0,0 0))", "Polygon((5 5,5 10,10 10,10 5,5 5))"
        self.compare(sqlfunc.MBRDisjoint(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g4)),
            "MBRDisjoint(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))")
        self.compare(sqlfunc.MBRDisjoint(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(g4)),
            "MBRDisjoint(ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))")
        self.compare(sqlfunc.MBRDisjoint(sqlfunc.ST_GeomFromText(g3), sqlfunc.ST_GeomFromText(g4)),
            "MBRDisjoint(ST_GeomFromText('Polygon((0 0,0 5,5 5,5 0,0 0))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))")
        self.compare(sqlfunc.MBRDisjoint(sqlfunc.ST_GeomFromText(g4), sqlfunc.ST_GeomFromText(g4)),
            "MBRDisjoint(ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))")
        # MBREquals
        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        self.compare(sqlfunc.MBREquals(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g1)),
            "MBREquals(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'))")
        self.compare(sqlfunc.MBREquals(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2)),
            "MBREquals(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))")
        # MBRIntersects
        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        g3, g4 = "Polygon((0 0,0 5,5 5,5 0,0 0))", "Polygon((5 5,5 10,10 10,10 5,5 5))"
        self.compare(sqlfunc.MBRIntersects(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2)),
            "MBRIntersects(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))")
        self.compare(sqlfunc.MBRIntersects(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g3)),
            "MBRIntersects(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((0 0,0 5,5 5,5 0,0 0))'))")
        self.compare(sqlfunc.MBRIntersects(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g4)),
            "MBRIntersects(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((5 5,5 10,10 10,10 5,5 5))'))")
        # MBROverlaps
        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        self.compare(sqlfunc.MBROverlaps(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2)),
            "MBROverlaps(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))")
        # MBRTouches
        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        self.compare(sqlfunc.MBRTouches(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2)),
            "MBRTouches(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))")
        # MBRWithin
        g1, g2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        self.compare(sqlfunc.MBRWithin(sqlfunc.ST_GeomFromText(g1), sqlfunc.ST_GeomFromText(g2)),
            "MBRWithin(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))")
        self.compare(sqlfunc.MBRWithin(sqlfunc.ST_GeomFromText(g2), sqlfunc.ST_GeomFromText(g1)),
            "MBRWithin(ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'),ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'))")
        # MD5
        self.compare(sqlfunc.MD5("testing"), "MD5('testing')")
        # MICROSECOND
        self.compare(sqlfunc.MICROSECOND("12:00:00.123456"), "MICROSECOND('12:00:00.123456')")
        self.compare(sqlfunc.MICROSECOND(datetime.time(23, 59, 59, 10)), "MICROSECOND('23:59:59.000010')")
        # MINUTE
        self.compare(sqlfunc.MINUTE("2008-02-03 10:05:03"), "MINUTE('2008-02-03 10:05:03')")
        self.compare(sqlfunc.MINUTE(datetime.time(23, 59, 59)), "MINUTE('23:59:59')")
        # MOD
        self.compare(sqlfunc.MOD(234, 10), "MOD(234,10)")
        self.compare(sqlfunc.MOD("29", "9"), "MOD('29','9')")
        # MONTH
        self.compare(sqlfunc.MONTH("2008-02-03"), "MONTH('2008-02-03')")
        self.compare(sqlfunc.MONTH(datetime.datetime(2010, 12, 31)), "MONTH('2010-12-31 00:00:00')")
        # MONTHNAME
        self.compare(sqlfunc.MONTHNAME("2008-02-03"), "MONTHNAME('2008-02-03')")
        self.compare(sqlfunc.MONTHNAME(datetime.datetime(2010, 12, 31)), "MONTHNAME('2010-12-31 00:00:00')")
        # MultiLineString
        self.compare(sqlfunc.MultiLineString(sqlfunc.LineString(sqlfunc.Point(1, 1), sqlfunc.Point(2, 2))), 
            "MultiLineString(LineString(Point(1,1),Point(2,2)))")
        # MultiPoint
        self.compare(sqlfunc.MultiPoint(sqlfunc.Point(1, 1), sqlfunc.Point(2, 2)), "MultiPoint(Point(1,1),Point(2,2))")
        # MultiPolygon
        self.compare(sqlfunc.MultiPolygon(
            sqlfunc.Polygon(sqlfunc.LineString(sqlfunc.Point(0, 0),sqlfunc.Point(5, 0),sqlfunc.Point(5, 5),sqlfunc.Point(0, 5),sqlfunc.Point(0, 0)))),
            "MultiPolygon(Polygon(LineString(Point(0,0),Point(5,0),Point(5,5),Point(0,5),Point(0,0))))"
        )
        # NOT_IN
        self.compare(sqlfunc.NOT_IN(0, 3, 5, 7), "NOT IN(0,3,5,7)")
        self.compare(sqlfunc.NOT_IN(*["a", "b", "c"]), "NOT IN('a','b','c')")
        # NOW
        self.compare(sqlfunc.NOW(), "NOW()")
        self.compare(sqlfunc.NOW(3), "NOW(3)")
        # NULLIF
        self.compare(sqlfunc.NULLIF(1, 1), "NULLIF(1,1)")
        self.compare(sqlfunc.NULLIF(1, 2), "NULLIF(1,2)")
        # OCT
        self.compare(sqlfunc.OCT(12), "OCT(12)")
        self.compare(sqlfunc.OCT("10"), "OCT('10')")
        # ORD
        self.compare(sqlfunc.ORD('2'), "ORD('2')")
        # PERIOD_ADD
        self.compare(sqlfunc.PERIOD_ADD(200801, 2), "PERIOD_ADD(200801,2)")
        self.compare(sqlfunc.PERIOD_ADD("200801", "2"), "PERIOD_ADD('200801','2')")
        # PERIOD_DIFF
        self.compare(sqlfunc.PERIOD_DIFF(200802, 200703), "PERIOD_DIFF(200802,200703)")
        self.compare(sqlfunc.PERIOD_DIFF("200802", "200703"), "PERIOD_DIFF('200802','200703')")
        # PI
        self.compare(sqlfunc.PI(), "PI()")
        # Point
        self.compare(sqlfunc.Point(1, 1), "Point(1,1)")
        # Polygon
        self.compare(sqlfunc.Polygon(sqlfunc.LineString(
            sqlfunc.Point(0, 0),sqlfunc.Point(5, 0),sqlfunc.Point(5, 5),sqlfunc.Point(0, 5),sqlfunc.Point(0, 0))),
            "Polygon(LineString(Point(0,0),Point(5,0),Point(5,5),Point(0,5),Point(0,0)))"
        )
        # POW
        self.compare(sqlfunc.POW(2, 2), "POW(2,2)")
        self.compare(sqlfunc.POW("2", "-2"), "POW('2','-2')")
        # PS_CURRENT_THREAD_ID
        self.compare(sqlfunc.PS_CURRENT_THREAD_ID(), "PS_CURRENT_THREAD_ID()")
        # PS_THREAD_ID
        self.compare(sqlfunc.PS_THREAD_ID(sqlfunc.CONNECTION_ID()), "PS_THREAD_ID(CONNECTION_ID())")
        # QUARTER
        self.compare(sqlfunc.QUARTER("2008-04-01"), "QUARTER('2008-04-01')")
        self.compare(sqlfunc.QUARTER(datetime.datetime(2010, 12, 31)), "QUARTER('2010-12-31 00:00:00')")
        # QUOTE
        self.compare(sqlfunc.QUOTE("It's a beautiful day"), "QUOTE('It\\'s a beautiful day')")
        self.compare(sqlfunc.QUOTE(None), "QUOTE(NULL)")
        # RADIANS
        self.compare(sqlfunc.RADIANS(90), "RADIANS(90)")
        self.compare(sqlfunc.RADIANS("180"), "RADIANS('180')")
        # RAND
        self.compare(sqlfunc.RAND(), "RAND()")
        self.compare(sqlfunc.RAND(3), "RAND(3)")
        # RANDOM_BYTES
        self.compare(sqlfunc.RANDOM_BYTES(16), "RANDOM_BYTES(16)")
        # REGEXP_INSTR
        self.compare(sqlfunc.REGEXP_INSTR("dog cat dog", "dog"), "REGEXP_INSTR('dog cat dog','dog')")
        self.compare(sqlfunc.REGEXP_INSTR("dog cat dog", "dog", 2), "REGEXP_INSTR('dog cat dog','dog',2)")
        self.compare(sqlfunc.REGEXP_INSTR("dog cat dog", "dog", 1, 2), "REGEXP_INSTR('dog cat dog','dog',1,2)")
        self.compare(sqlfunc.REGEXP_INSTR("dog cat dog", "dog", 1, 1, 1), "REGEXP_INSTR('dog cat dog','dog',1,1,1)")
        self.compare(sqlfunc.REGEXP_INSTR("dog cat dog", "Dog", 1, 1, 0, "i"), "REGEXP_INSTR('dog cat dog','Dog',1,1,0,'i')")
        # REGEXP_LIKE
        self.compare(sqlfunc.REGEXP_LIKE("CamelCase", "CAMELCASE"), "REGEXP_LIKE('CamelCase','CAMELCASE')")
        self.compare(sqlfunc.REGEXP_LIKE("CamelCase", "CAMELCASE", "c"), "REGEXP_LIKE('CamelCase','CAMELCASE','c')")
        self.compare(sqlfunc.REGEXP_LIKE("CamelCase", "CAMELCASE", "imnu"), "REGEXP_LIKE('CamelCase','CAMELCASE','imnu')")
        # REGEXP_REPLACE
        self.compare(sqlfunc.REGEXP_REPLACE("a b c", "b", "X"), "REGEXP_REPLACE('a b c','b','X')")
        self.compare(sqlfunc.REGEXP_REPLACE("a b b c", "b", "X", 4), "REGEXP_REPLACE('a b b c','b','X',4)")
        self.compare(sqlfunc.REGEXP_REPLACE("a b b b c", "b", "X", 1, 2), "REGEXP_REPLACE('a b b b c','b','X',1,2)")
        self.compare(sqlfunc.REGEXP_REPLACE("a b b b c", "B", "X", 1, 2, "i"), "REGEXP_REPLACE('a b b b c','B','X',1,2,'i')")
        # REGEXP_SUBSTR
        self.compare(sqlfunc.REGEXP_SUBSTR("abc def ghi", "[a-z]+"), "REGEXP_SUBSTR('abc def ghi','[a-z]+')")
        self.compare(sqlfunc.REGEXP_SUBSTR("abc def ghi", "[a-z]+", 4), "REGEXP_SUBSTR('abc def ghi','[a-z]+',4)")
        self.compare(sqlfunc.REGEXP_SUBSTR("abc def ghi", "[a-z]+", 1, 2), "REGEXP_SUBSTR('abc def ghi','[a-z]+',1,2)")
        self.compare(sqlfunc.REGEXP_SUBSTR("abc def ghi", "[A-Z]+", 1, 2, "i"), "REGEXP_SUBSTR('abc def ghi','[A-Z]+',1,2,'i')")
        # RANK
        self.compare(sqlfunc.RANK(), "RANK()")
        # RELEASE_ALL_LOCKS
        self.compare(sqlfunc.RELEASE_ALL_LOCKS(), "RELEASE_ALL_LOCKS()")
        # RELEASE_LOCK
        self.compare(sqlfunc.RELEASE_LOCK("lock1"), "RELEASE_LOCK('lock1')")
        # REPEAT
        self.compare(sqlfunc.REPEAT("MySQL", 3), "REPEAT('MySQL',3)")
        self.compare(sqlfunc.REPEAT("MySQL", 0), "REPEAT('MySQL',0)")
        # REPLACE
        self.compare(sqlfunc.REPLACE("www.mysql.com", "w", "Ww"), "REPLACE('www.mysql.com','w','Ww')")
        # REVERSE
        self.compare(sqlfunc.REVERSE("abc"), "REVERSE('abc')")
        # RIGHT
        self.compare(sqlfunc.RIGHT("foobarbar", 4), "RIGHT('foobarbar',4)")
        # ROUND
        self.compare(sqlfunc.ROUND(-1.23), "ROUND(-1.23)")
        self.compare(sqlfunc.ROUND(1.298, 1), "ROUND(1.298,1)")
        # ROW_COUNT
        self.compare(sqlfunc.ROW_COUNT(), "ROW_COUNT()")
        # ROW_NUMBER
        self.compare(sqlfunc.ROW_NUMBER(), "ROW_NUMBER()")
        # RPAD
        self.compare(sqlfunc.RPAD("hi", 5, "?"), "RPAD('hi',5,'?')")
        self.compare(sqlfunc.RPAD("hi", 1, "?"), "RPAD('hi',1,'?')")
        # RTRIM
        self.compare(sqlfunc.RTRIM("barbar   "), "RTRIM('barbar   ')")
        # SEC_TO_TIME
        self.compare(sqlfunc.SEC_TO_TIME(2378), "SEC_TO_TIME(2378)")
        self.compare(sqlfunc.SEC_TO_TIME("2378"), "SEC_TO_TIME('2378')")
        # SECOND
        self.compare(sqlfunc.SECOND("10:05:03"), "SECOND('10:05:03')")
        self.compare(sqlfunc.SECOND(datetime.datetime(2021, 12, 25, 19, 25, 37)), "SECOND('2021-12-25 19:25:37')")
        # SHA1
        self.compare(sqlfunc.SHA1("abc"), "SHA1('abc')")
        # SHA2
        self.compare(sqlfunc.SHA2("abc", 224), "SHA2('abc',224)")
        self.compare(sqlfunc.SHA2("abc", 256), "SHA2('abc',256)")
        # SIGN
        self.compare(sqlfunc.SIGN(-32), "SIGN(-32)")
        self.compare(sqlfunc.SIGN("0"), "SIGN('0')")
        self.compare(sqlfunc.SIGN(234), "SIGN(234)")
        # SIN
        self.compare(sqlfunc.SIN(1), "SIN(1)")
        self.compare(sqlfunc.SIN(sqlfunc.PI()), "SIN(PI())")
        # SLEEP
        self.compare(sqlfunc.SLEEP(5), "SLEEP(5)")
        # SOUNDEX
        self.compare(sqlfunc.SOUNDEX("Hello"), "SOUNDEX('Hello')")
        self.compare(sqlfunc.SOUNDEX("Quadratically"), "SOUNDEX('Quadratically')")
        # SPACE
        self.compare(sqlfunc.SPACE(6), "SPACE(6)")
        # SQRT
        self.compare(sqlfunc.SQRT(4), "SQRT(4)")
        self.compare(sqlfunc.SQRT("20"), "SQRT('20')")
        # ST_Area
        poly = "Polygon((0 0,0 3,3 0,0 0),(1 1,1 2,2 1,1 1))"
        self.compare(sqlfunc.ST_Area(sqlfunc.ST_GeomFromText(poly)), 
            "ST_Area(ST_GeomFromText('Polygon((0 0,0 3,3 0,0 0),(1 1,1 2,2 1,1 1))'))")
        multipoly = "MultiPolygon(((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1)))"
        self.compare(sqlfunc.ST_Area(sqlfunc.ST_GeomFromText(multipoly)),
            "ST_Area(ST_GeomFromText('MultiPolygon(((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1)))'))")
        # ST_AsBinary/ST_AsWKB
        ls = "LineString(0 5,5 10,10 15)"
        self.compare(sqlfunc.ST_AsBinary(sqlfunc.ST_GeomFromText(ls)),
            "ST_AsBinary(ST_GeomFromText('LineString(0 5,5 10,10 15)'))")
        self.compare(sqlfunc.ST_AsBinary(sqlfunc.ST_GeomFromText(ls), "axis-order=long-lat"),
            "ST_AsBinary(ST_GeomFromText('LineString(0 5,5 10,10 15)'),'axis-order=long-lat')")
        self.compare(sqlfunc.ST_AsWKB(sqlfunc.ST_GeomFromText(ls)),
            "ST_AsWKB(ST_GeomFromText('LineString(0 5,5 10,10 15)'))")
        self.compare(sqlfunc.ST_AsWKB(sqlfunc.ST_GeomFromText(ls), "axis-order=long-lat"),
            "ST_AsWKB(ST_GeomFromText('LineString(0 5,5 10,10 15)'),'axis-order=long-lat')")
        # ST_AsGeoJSON
        p = "POINT(11.11111 12.22222)"
        self.compare(sqlfunc.ST_AsGeoJSON(sqlfunc.ST_GeomFromText(p)),
            "ST_AsGeoJSON(ST_GeomFromText('POINT(11.11111 12.22222)'))")
        self.compare(sqlfunc.ST_AsGeoJSON(sqlfunc.ST_GeomFromText(p), 2),
            "ST_AsGeoJSON(ST_GeomFromText('POINT(11.11111 12.22222)'),2)")
        # ST_AsText/ST_AsWKT
        ls, multi_p = "LineString(1 1,2 2,3 3)", "MultiPoint(0 0,1 2,2 3)"
        self.compare(sqlfunc.ST_AsText(sqlfunc.ST_GeomFromText(ls)),
            "ST_AsText(ST_GeomFromText('LineString(1 1,2 2,3 3)'))")
        self.compare(sqlfunc.ST_AsText(sqlfunc.ST_GeomFromText(multi_p)),
            "ST_AsText(ST_GeomFromText('MultiPoint(0 0,1 2,2 3)'))")
        self.compare(sqlfunc.ST_AsWKT(sqlfunc.ST_GeomFromText(ls)),
            "ST_AsWKT(ST_GeomFromText('LineString(1 1,2 2,3 3)'))")
        self.compare(sqlfunc.ST_AsWKT(sqlfunc.ST_GeomFromText(multi_p)),
            "ST_AsWKT(ST_GeomFromText('MultiPoint(0 0,1 2,2 3)'))")
        # ST_Buffer
        self.compare(sqlfunc.ST_AsText(sqlfunc.ST_Buffer(sqlfunc.ST_GeomFromText("POINT(0 0)"), 0)),
            "ST_AsText(ST_Buffer(ST_GeomFromText('POINT(0 0)'),0))")
        self.compare(sqlfunc.ST_AsText(sqlfunc.ST_Buffer(sqlfunc.ST_GeomFromText("POINT(0 0)"), 2, sqlfunc.ST_Buffer_Strategy("point_square"))),
            "ST_AsText(ST_Buffer(ST_GeomFromText('POINT(0 0)'),2,ST_Buffer_Strategy('point_square')))")
        # ST_Buffer_Strategy
        self.compare(sqlfunc.ST_Buffer_Strategy("end_flat"), "ST_Buffer_Strategy('end_flat')")
        self.compare(sqlfunc.ST_Buffer_Strategy("join_round", 10), "ST_Buffer_Strategy('join_round',10)")
        # ST_Centroid
        poly = "Polygon((0 0,10 0,10 10,0 10,0 0),(5 5,7 5,7 7,5 7,5 5))"
        self.compare(sqlfunc.ST_Centroid(sqlfunc.ST_GeomFromText(poly)),
            "ST_Centroid(ST_GeomFromText('Polygon((0 0,10 0,10 10,0 10,0 0),(5 5,7 5,7 7,5 7,5 5))'))")
        # ST_Contains
        poly = "POLYGON((0 0,0 3,3 3,3 0,0 0))"
        p1, p2, p3 = "POINT(1 1)", "POINT(3 3)", "POINT(5 5)"
        self.compare(sqlfunc.ST_Contains(sqlfunc.ST_GeomFromText(poly), sqlfunc.ST_GeomFromText(p1)),
            "ST_Contains(ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('POINT(1 1)'))")
        self.compare(sqlfunc.ST_Contains(sqlfunc.ST_GeomFromText(poly), sqlfunc.ST_GeomFromText(p2)),
            "ST_Contains(ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('POINT(3 3)'))")
        self.compare(sqlfunc.ST_Contains(sqlfunc.ST_GeomFromText(poly), sqlfunc.ST_GeomFromText(p3)),
            "ST_Contains(ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('POINT(5 5)'))")
        # ST_ConvexHull
        multi_p = "MULTIPOINT(5 0,25 0,15 10,15 25)"
        self.compare(sqlfunc.ST_ConvexHull(sqlfunc.ST_GeomFromText(multi_p)),
            "ST_ConvexHull(ST_GeomFromText('MULTIPOINT(5 0,25 0,15 10,15 25)'))")
        # ST_Crosses
        ls1, ls2 = "LINESTRING(0 0, 10 10)", "LINESTRING(0 10, 10 0)"
        self.compare(sqlfunc.ST_Crosses(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls2)),
            "ST_Crosses(ST_GeomFromText('LINESTRING(0 0, 10 10)'),ST_GeomFromText('LINESTRING(0 10, 10 0)'))")
        # ST_Difference
        p1, p2 = "Point(1 1)", "Point(2 2)"
        self.compare(sqlfunc.ST_Difference(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2)),
            "ST_Difference(ST_GeomFromText('Point(1 1)'),ST_GeomFromText('Point(2 2)'))")
        # ST_Dimension
        ls = "LineString(1 1,2 2)"
        self.compare(sqlfunc.ST_Dimension(sqlfunc.ST_GeomFromText(ls)),
            "ST_Dimension(ST_GeomFromText('LineString(1 1,2 2)'))")
        # ST_Disjoint
        poly1, poly2 = "POLYGON((0 0,0 3,3 3,3 0,0 0))", "POLYGON((4 4,4 6,6 6,6 4,4 4))"
        self.compare(sqlfunc.ST_Disjoint(sqlfunc.ST_GeomFromText(poly1), sqlfunc.ST_GeomFromText(poly2)),
            "ST_Disjoint(ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('POLYGON((4 4,4 6,6 6,6 4,4 4))'))")
        # ST_Distance
        p1, p2 = "POINT(1 1)", "POINT(2 2)"
        self.compare(sqlfunc.ST_Distance(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2)),
            "ST_Distance(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(2 2)'))")
        self.compare(sqlfunc.ST_Distance(sqlfunc.ST_GeomFromText(p1, 4326), sqlfunc.ST_GeomFromText(p2, 4326), "foot"),
            "ST_Distance(ST_GeomFromText('POINT(1 1)',4326),ST_GeomFromText('POINT(2 2)',4326),'foot')")
        # ST_Distance_Sphere
        p1, p2 = "POINT(0 0)", "POINT(180 0)"
        self.compare(sqlfunc.ST_Distance_Sphere(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2)), 
            "ST_Distance_Sphere(ST_GeomFromText('POINT(0 0)'),ST_GeomFromText('POINT(180 0)'))")
        # ST_EndPoint
        ls = "LineString(1 1,2 2,3 3)"
        self.compare(sqlfunc.ST_EndPoint(sqlfunc.ST_GeomFromText(ls)), 
            "ST_EndPoint(ST_GeomFromText('LineString(1 1,2 2,3 3)'))")
        # ST_Envelope
        ls1, ls2 = "LineString(1 1,2 2)", "LineString(1 1,1 2)"
        self.compare(sqlfunc.ST_Envelope(sqlfunc.ST_GeomFromText(ls1)),
            "ST_Envelope(ST_GeomFromText('LineString(1 1,2 2)'))")
        self.compare(sqlfunc.ST_Envelope(sqlfunc.ST_GeomFromText(ls2)),
            "ST_Envelope(ST_GeomFromText('LineString(1 1,1 2)'))")
        # ST_Equals
        p1, p2 = "POINT(1 1)", "POINT(2 2)"
        self.compare(sqlfunc.ST_Equals(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p1)),
            "ST_Equals(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(1 1)'))")
        self.compare(sqlfunc.ST_Equals(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2)),
            "ST_Equals(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(2 2)'))")
        # ST_ExteriorRing
        poly = "Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"
        self.compare(sqlfunc.ST_ExteriorRing(sqlfunc.ST_GeomFromText(poly)),
            "ST_ExteriorRing(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))'))")
        # ST_FrechetDistance
        ls1, ls2 = "LINESTRING(0 0,0 5,5 5)", "LINESTRING(0 1,0 6,3 3,5 6)"
        self.compare(sqlfunc.ST_FrechetDistance(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls2)),
            "ST_FrechetDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)'))")
        self.compare(sqlfunc.ST_FrechetDistance(sqlfunc.ST_GeomFromText(ls1, 4326), sqlfunc.ST_GeomFromText(ls2, 4326), "foot"),
            "ST_FrechetDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)',4326),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)',4326),'foot')")
        # ST_GeoHash
        self.compare(sqlfunc.ST_GeoHash(180, 0, 10), "ST_GeoHash(180,0,10)")
        self.compare(sqlfunc.ST_GeoHash(sqlfunc.Point(180, 0), 10), "ST_GeoHash(Point(180,0),10)")
        # ST_GeomCollFromText
        multi_ls = "MULTILINESTRING((10 10, 11 11), (9 9, 10 10))"
        self.compare(sqlfunc.ST_GeomCollFromText(multi_ls), 
            "ST_GeomCollFromText('MULTILINESTRING((10 10, 11 11), (9 9, 10 10))')")
        self.compare(sqlfunc.ST_GeomCollFromText(multi_ls, 4326),
            "ST_GeomCollFromText('MULTILINESTRING((10 10, 11 11), (9 9, 10 10))',4326)")
        # ST_GeomCollFromWKB
        multi_ls = "MULTILINESTRING((10 10, 11 11), (9 9, 10 10))"
        self.compare(sqlfunc.ST_GeomCollFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_GeomCollFromText(multi_ls))),
            "ST_GeomCollFromWKB(ST_AsBinary(ST_GeomCollFromText('MULTILINESTRING((10 10, 11 11), (9 9, 10 10))')))")
        # ST_GeometryN
        gc = "GeometryCollection(Point(1 1),LineString(2 2, 3 3))"
        self.compare(sqlfunc.ST_GeometryN(sqlfunc.ST_GeomFromText(gc), 1),
            "ST_GeometryN(ST_GeomFromText('GeometryCollection(Point(1 1),LineString(2 2, 3 3))'),1)")
        # ST_GeometryType
        self.compare(sqlfunc.ST_GeometryType(sqlfunc.ST_GeomFromText('POINT(1 1)')),
            "ST_GeometryType(ST_GeomFromText('POINT(1 1)'))")
        # ST_GeomFromGeoJSON
        geojson = '{ "type": "Point", "coordinates": [102.0, 0.0]}'
        self.compare(sqlfunc.ST_GeomFromGeoJSON(geojson),
            'ST_GeomFromGeoJSON(\'{ \\"type\\": \\"Point\\", \\"coordinates\\": [102.0, 0.0]}\')')
        self.compare(sqlfunc.ST_GeomFromGeoJSON(geojson, 2),
            'ST_GeomFromGeoJSON(\'{ \\"type\\": \\"Point\\", \\"coordinates\\": [102.0, 0.0]}\',2)')
        # ST_GeomFromText
        p1, p2 = "POINT(10 20)", "POINT(-73.935242 40.730610)"
        self.compare(sqlfunc.ST_GeomFromText(p1), "ST_GeomFromText('POINT(10 20)')")
        self.compare(sqlfunc.ST_GeomFromText(p2, 4326), "ST_GeomFromText('POINT(-73.935242 40.730610)',4326)")
        self.compare(sqlfunc.ST_GeomFromText(p1, 0, "axis-order=lat-long"), "ST_GeomFromText('POINT(10 20)',0,'axis-order=lat-long')")
        # ST_GeomFromWKB
        p = "POINT(10 20)"
        self.compare(sqlfunc.ST_GeomFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_GeomFromText(p))),
            "ST_GeomFromWKB(ST_AsBinary(ST_GeomFromText('POINT(10 20)')))")
        self.compare(sqlfunc.ST_GeomFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_GeomFromText(p)), 4326),
            "ST_GeomFromWKB(ST_AsBinary(ST_GeomFromText('POINT(10 20)')),4326)")
        # ST_HausdorffDistance
        ls1, ls2 = "LINESTRING(0 0,0 5,5 5)", "LINESTRING(0 1,0 6,3 3,5 6)"
        self.compare(sqlfunc.ST_HausdorffDistance(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls2)),
            "ST_HausdorffDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)'))")
        self.compare(sqlfunc.ST_HausdorffDistance(sqlfunc.ST_GeomFromText(ls1, 4326), sqlfunc.ST_GeomFromText(ls2, 4326)),
            "ST_HausdorffDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)',4326),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)',4326))")
        self.compare(sqlfunc.ST_HausdorffDistance(sqlfunc.ST_GeomFromText(ls1, 4326), sqlfunc.ST_GeomFromText(ls2, 4326), "foot"),
            "ST_HausdorffDistance(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)',4326),ST_GeomFromText('LINESTRING(0 1,0 6,3 3,5 6)',4326),'foot')")
        # ST_InteriorRingN
        poly = "Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"
        self.compare(sqlfunc.ST_InteriorRingN(sqlfunc.ST_GeomFromText(poly), 1),
            "ST_InteriorRingN(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))'),1)")
        # ST_Intersection
        ls1, ls2 = "LineString(1 1, 3 3)", "LineString(1 3, 3 1)"
        self.compare(sqlfunc.ST_Intersection(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls1)),
            "ST_Intersection(ST_GeomFromText('LineString(1 1, 3 3)'),ST_GeomFromText('LineString(1 1, 3 3)'))")
        # ST_Intersects
        p1, p2 = "POINT(1 1)", "POINT(2 2)"
        self.compare(sqlfunc.ST_Intersects(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p1)),
            "ST_Intersects(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(1 1)'))")
        self.compare(sqlfunc.ST_Intersects(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(p2)),
            "ST_Intersects(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POINT(2 2)'))")
        # ST_IsClosed
        ls1, ls2 = "LineString(1 1,2 2,3 3,2 2)", "LineString(1 1,2 2,3 3,1 1)"
        self.compare(sqlfunc.ST_IsClosed(sqlfunc.ST_GeomFromText(ls1)),
            "ST_IsClosed(ST_GeomFromText('LineString(1 1,2 2,3 3,2 2)'))")
        self.compare(sqlfunc.ST_IsClosed(sqlfunc.ST_GeomFromText(ls2)),
            "ST_IsClosed(ST_GeomFromText('LineString(1 1,2 2,3 3,1 1)'))")
        # ST_IsEmpty
        self.compare(sqlfunc.ST_IsEmpty(sqlfunc.ST_GeomFromText('POINT(1 1)')),
            "ST_IsEmpty(ST_GeomFromText('POINT(1 1)'))")
        self.compare(sqlfunc.ST_IsEmpty(sqlfunc.ST_GeomFromText('GEOMETRYCOLLECTION EMPTY')),
            "ST_IsEmpty(ST_GeomFromText('GEOMETRYCOLLECTION EMPTY'))")
        # ST_IsSimple
        ls1 = "LINESTRING(0 0,0 5,5 5)"
        self.compare(sqlfunc.ST_IsSimple(sqlfunc.ST_GeomFromText(ls1)),
            "ST_IsSimple(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'))")
        # ST_IsValid
        ls1, ls2 = "LINESTRING(0 0,-0.00 0,0.0 0)", "LINESTRING(0 0, 1 1)"
        self.compare(sqlfunc.ST_IsValid(sqlfunc.ST_GeomFromText(ls1)),
            "ST_IsValid(ST_GeomFromText('LINESTRING(0 0,-0.00 0,0.0 0)'))")
        self.compare(sqlfunc.ST_IsValid(sqlfunc.ST_GeomFromText(ls2)),
            "ST_IsValid(ST_GeomFromText('LINESTRING(0 0, 1 1)'))")
        # ST_LatFromGeoHash
        self.compare(sqlfunc.ST_LatFromGeoHash(sqlfunc.ST_GeoHash(45, -20, 10)),
            "ST_LatFromGeoHash(ST_GeoHash(45,-20,10))")
        # ST_Latitude
        p = "POINT(45 90)"
        self.compare(sqlfunc.ST_Latitude(sqlfunc.ST_GeomFromText(p, 4326)),
            "ST_Latitude(ST_GeomFromText('POINT(45 90)',4326))")
        self.compare(sqlfunc.ST_Latitude(sqlfunc.ST_GeomFromText(p, 4326), 10),
            "ST_Latitude(ST_GeomFromText('POINT(45 90)',4326),10)")
        # ST_Length
        ls, multi_ls = "LineString(1 1,2 2,3 3)", "MultiLineString((1 1,2 2,3 3),(4 4,5 5))"
        self.compare(sqlfunc.ST_Length(sqlfunc.ST_GeomFromText(ls)),
            "ST_Length(ST_GeomFromText('LineString(1 1,2 2,3 3)'))")
        self.compare(sqlfunc.ST_Length(sqlfunc.ST_GeomFromText(ls, 4326), "foot"),
            "ST_Length(ST_GeomFromText('LineString(1 1,2 2,3 3)',4326),'foot')")
        self.compare(sqlfunc.ST_Length(sqlfunc.ST_GeomFromText(multi_ls)),
            "ST_Length(ST_GeomFromText('MultiLineString((1 1,2 2,3 3),(4 4,5 5))'))")
        # ST_LineFromText
        ls = "LINESTRING(0 0,0 5,5 5)"
        self.compare(sqlfunc.ST_LineFromText(ls), "ST_LineFromText('LINESTRING(0 0,0 5,5 5)')")
        self.compare(sqlfunc.ST_LineFromText(ls, 4326), "ST_LineFromText('LINESTRING(0 0,0 5,5 5)',4326)")
        # ST_LineFromWKB
        ls = "LINESTRING(0 0,0 5,5 5)"
        self.compare(sqlfunc.ST_LineFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_LineFromText(ls))),
            "ST_LineFromWKB(ST_AsBinary(ST_LineFromText('LINESTRING(0 0,0 5,5 5)')))")
        # ST_LineInterpolatePoint
        ls = "LINESTRING(0 0,0 5,5 5)"
        self.compare(sqlfunc.ST_LineInterpolatePoint(sqlfunc.ST_GeomFromText(ls), 0.5),
            "ST_LineInterpolatePoint(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),0.5)")
        # ST_LineInterpolatePoints
        ls = "LINESTRING(0 0,0 5,5 5)"
        self.compare(sqlfunc.ST_LineInterpolatePoints(sqlfunc.ST_GeomFromText(ls), 0.5),
            "ST_LineInterpolatePoints(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),0.5)")
        self.compare(sqlfunc.ST_LineInterpolatePoints(sqlfunc.ST_GeomFromText(ls), 0.25),
            "ST_LineInterpolatePoints(ST_GeomFromText('LINESTRING(0 0,0 5,5 5)'),0.25)")
        # ST_LongFromGeoHash
        self.compare(sqlfunc.ST_LongFromGeoHash(sqlfunc.ST_GeoHash(45, -20, 10)),
            "ST_LongFromGeoHash(ST_GeoHash(45,-20,10))")
        # ST_Longitude
        p = "POINT(45 90)"
        self.compare(sqlfunc.ST_Longitude(sqlfunc.ST_GeomFromText(p, 4326)),
            "ST_Longitude(ST_GeomFromText('POINT(45 90)',4326))")
        self.compare(sqlfunc.ST_Longitude(sqlfunc.ST_GeomFromText(p, 4326), 10),
            "ST_Longitude(ST_GeomFromText('POINT(45 90)',4326),10)")
        # ST_MakeEnvelope
        pt1, pt2 = "POINT(0 0)", "POINT(1 1)"
        self.compare(sqlfunc.ST_MakeEnvelope(sqlfunc.ST_GeomFromText(pt1), sqlfunc.ST_GeomFromText(pt2)),
            "ST_MakeEnvelope(ST_GeomFromText('POINT(0 0)'),ST_GeomFromText('POINT(1 1)'))")
        # ST_MLineFromText
        ls = "MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))"
        self.compare(sqlfunc.ST_MLineFromText(ls),
            "ST_MLineFromText('MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))')")
        self.compare(sqlfunc.ST_MLineFromText(ls, 4326),
            "ST_MLineFromText('MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))',4326)")
        # ST_MLineFromWKB
        ls = "MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))"
        self.compare(sqlfunc.ST_MLineFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MLineFromText(ls))),
            "ST_MLineFromWKB(ST_AsBinary(ST_MLineFromText('MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))')))")
        self.compare(sqlfunc.ST_MLineFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MLineFromText(ls)), 4326),
            "ST_MLineFromWKB(ST_AsBinary(ST_MLineFromText('MULTILINESTRING((0 0,0 5,5 5),(1 1,2 2))')),4326)")
        # ST_MPointFromText
        pt = "MULTIPOINT(0 0,1 1)"
        self.compare(sqlfunc.ST_MPointFromText(pt), "ST_MPointFromText('MULTIPOINT(0 0,1 1)')")
        self.compare(sqlfunc.ST_MPointFromText(pt, 4326), "ST_MPointFromText('MULTIPOINT(0 0,1 1)',4326)")
        # ST_MPointFromWKB
        pt = "MULTIPOINT(0 0,1 1)"
        self.compare(sqlfunc.ST_MPointFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MPointFromText(pt))),
            "ST_MPointFromWKB(ST_AsBinary(ST_MPointFromText('MULTIPOINT(0 0,1 1)')))")
        self.compare(sqlfunc.ST_MPointFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MPointFromText(pt)), 4326),
            "ST_MPointFromWKB(ST_AsBinary(ST_MPointFromText('MULTIPOINT(0 0,1 1)')),4326)")
        # ST_MPolyFromText
        poly = "MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))"
        self.compare(sqlfunc.ST_MPolyFromText(poly),
            "ST_MPolyFromText('MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))')")
        self.compare(sqlfunc.ST_MPolyFromText(poly, 4326),
            "ST_MPolyFromText('MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))',4326)")
        # ST_MPolyFromWKB
        poly = "MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))"
        self.compare(sqlfunc.ST_MPolyFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MPolyFromText(poly))),
            "ST_MPolyFromWKB(ST_AsBinary(ST_MPolyFromText('MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))')))")
        self.compare(sqlfunc.ST_MPolyFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_MPolyFromText(poly)), 4326),
            "ST_MPolyFromWKB(ST_AsBinary(ST_MPolyFromText('MULTIPOLYGON(((0 0,0 5,5 5,0 0)),((1 1,2 2,2 1,1 1)))')),4326)")
        # ST_NumGeometries
        gc = "GeometryCollection(Point(1 1),LineString(2 2, 3 3))"
        self.compare(sqlfunc.ST_NumGeometries(sqlfunc.ST_GeomFromText(gc)),
            "ST_NumGeometries(ST_GeomFromText('GeometryCollection(Point(1 1),LineString(2 2, 3 3))'))")
        # ST_NumInteriorRing
        poly = "Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"
        self.compare(sqlfunc.ST_NumInteriorRing(sqlfunc.ST_GeomFromText(poly)),
            "ST_NumInteriorRing(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))'))")
        # ST_NumPoints
        ls = "LineString(1 1,2 2,3 3)"
        self.compare(sqlfunc.ST_NumPoints(sqlfunc.ST_GeomFromText(ls)),
            "ST_NumPoints(ST_GeomFromText('LineString(1 1,2 2,3 3)'))")
        # ST_Overlaps
        poly1, poly2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        self.compare(sqlfunc.ST_Overlaps(sqlfunc.ST_GeomFromText(poly1), sqlfunc.ST_GeomFromText(poly2)),
            "ST_Overlaps(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))")
        # ST_PointAtDistance
        ls = "LineString(0 0,0 5,5 5)"
        self.compare(sqlfunc.ST_PointAtDistance(sqlfunc.ST_GeomFromText(ls), 2.5),
            "ST_PointAtDistance(ST_GeomFromText('LineString(0 0,0 5,5 5)'),2.5)")
        # ST_PointFromGeoHash
        self.compare(sqlfunc.ST_PointFromGeoHash(sqlfunc.ST_GeoHash(45, -20, 10), 0),
            "ST_PointFromGeoHash(ST_GeoHash(45,-20,10),0)")
        # ST_PointFromText
        pt = "POINT(1 1)"
        self.compare(sqlfunc.ST_PointFromText(pt), "ST_PointFromText('POINT(1 1)')")
        self.compare(sqlfunc.ST_PointFromText(pt, 4326), "ST_PointFromText('POINT(1 1)',4326)")
        # ST_PointFromWKB
        pt = "POINT(1 1)"
        self.compare(sqlfunc.ST_PointFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_PointFromText(pt))),
            "ST_PointFromWKB(ST_AsBinary(ST_PointFromText('POINT(1 1)')))")
        self.compare(sqlfunc.ST_PointFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_PointFromText(pt)), 4326),
            "ST_PointFromWKB(ST_AsBinary(ST_PointFromText('POINT(1 1)')),4326)")
        # ST_PointN
        ls = "LineString(1 1,2 2,3 3)"
        self.compare(sqlfunc.ST_PointN(sqlfunc.ST_GeomFromText(ls), 2),
            "ST_PointN(ST_GeomFromText('LineString(1 1,2 2,3 3)'),2)")
        # ST_PolyFromText
        poly = "POLYGON((0 0,0 3,3 3,3 0,0 0))"
        self.compare(sqlfunc.ST_PolyFromText(poly),
            "ST_PolyFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))')")
        self.compare(sqlfunc.ST_PolyFromText(poly, 4326),
            "ST_PolyFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))',4326)")
        # ST_PolyFromWKB
        poly = "POLYGON((0 0,0 3,3 3,3 0,0 0))"
        self.compare(sqlfunc.ST_PolyFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_PolyFromText(poly))),
            "ST_PolyFromWKB(ST_AsBinary(ST_PolyFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))')))")
        self.compare(sqlfunc.ST_PolyFromWKB(sqlfunc.ST_AsBinary(sqlfunc.ST_PolyFromText(poly)), 4326),
            "ST_PolyFromWKB(ST_AsBinary(ST_PolyFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))')),4326)")
        # ST_Simplify
        ls = "LINESTRING(0 0,0 1,1 1,1 2,2 2,2 3,3 3)"
        self.compare(sqlfunc.ST_Simplify(sqlfunc.ST_GeomFromText(ls), 0.5),
            "ST_Simplify(ST_GeomFromText('LINESTRING(0 0,0 1,1 1,1 2,2 2,2 3,3 3)'),0.5)")
        self.compare(sqlfunc.ST_Simplify(sqlfunc.ST_GeomFromText(ls), 1.0),
            "ST_Simplify(ST_GeomFromText('LINESTRING(0 0,0 1,1 1,1 2,2 2,2 3,3 3)'),1.0)")
        # ST_SRID
        ls = "LineString(1 1,2 2)"
        self.compare(sqlfunc.ST_SRID(sqlfunc.ST_GeomFromText(ls, 0)),
            "ST_SRID(ST_GeomFromText('LineString(1 1,2 2)',0))")
        self.compare(sqlfunc.ST_SRID(sqlfunc.ST_GeomFromText(ls, 0), 4326),
            "ST_SRID(ST_GeomFromText('LineString(1 1,2 2)',0),4326)")
        # ST_StartPoint
        ls = "LineString(1 1,2 2,3 3)"
        self.compare(sqlfunc.ST_StartPoint(sqlfunc.ST_GeomFromText(ls)),
            "ST_StartPoint(ST_GeomFromText('LineString(1 1,2 2,3 3)'))")
        # ST_SwapXY
        pt = "LINESTRING(0 5,5 10,10 15)"
        self.compare(sqlfunc.ST_SwapXY(sqlfunc.ST_GeomFromText(pt)),
            "ST_SwapXY(ST_GeomFromText('LINESTRING(0 5,5 10,10 15)'))")
        # ST_SymDifference
        multi_p1, multi_p2 = "MULTIPOINT(5 0,15 10,15 25)", "MULTIPOINT(1 1,15 10,15 25)"
        self.compare(sqlfunc.ST_SymDifference(sqlfunc.ST_GeomFromText(multi_p1), sqlfunc.ST_GeomFromText(multi_p2)),
            "ST_SymDifference(ST_GeomFromText('MULTIPOINT(5 0,15 10,15 25)'),ST_GeomFromText('MULTIPOINT(1 1,15 10,15 25)'))")
        # ST_Touches
        poly1, poly2 = "Polygon((0 0,0 3,3 3,3 0,0 0))", "Polygon((1 1,1 2,2 2,2 1,1 1))"
        self.compare(sqlfunc.ST_Touches(sqlfunc.ST_GeomFromText(poly1), sqlfunc.ST_GeomFromText(poly2)),
            "ST_Touches(ST_GeomFromText('Polygon((0 0,0 3,3 3,3 0,0 0))'),ST_GeomFromText('Polygon((1 1,1 2,2 2,2 1,1 1))'))")
        # ST_Transform
        pt = "POINT(1 1)"
        self.compare(sqlfunc.ST_Transform(sqlfunc.ST_GeomFromText(pt, 4230), 4326),
            "ST_Transform(ST_GeomFromText('POINT(1 1)',4230),4326)")
        # ST_Union
        ls1, ls2 = "LineString(1 1, 3 3)", "LineString(1 3, 3 1)"
        self.compare(sqlfunc.ST_Union(sqlfunc.ST_GeomFromText(ls1), sqlfunc.ST_GeomFromText(ls2)),
            "ST_Union(ST_GeomFromText('LineString(1 1, 3 3)'),ST_GeomFromText('LineString(1 3, 3 1)'))")
        # ST_Validate
        ls1, ls2 = "LINESTRING(0 0)", "LINESTRING(0 0, 1 1)"
        self.compare(sqlfunc.ST_Validate(sqlfunc.ST_GeomFromText(ls1)),
            "ST_Validate(ST_GeomFromText('LINESTRING(0 0)'))")
        self.compare(sqlfunc.ST_Validate(sqlfunc.ST_GeomFromText(ls2)),
            "ST_Validate(ST_GeomFromText('LINESTRING(0 0, 1 1)'))")
        # ST_Within
        poly = "POLYGON((0 0,0 3,3 3,3 0,0 0))"
        p1, p2, p3 = "POINT(1 1)", "POINT(3 3)", "POINT(5 5)"
        self.compare(sqlfunc.ST_Within(sqlfunc.ST_GeomFromText(p1), sqlfunc.ST_GeomFromText(poly)),
            "ST_Within(ST_GeomFromText('POINT(1 1)'),ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'))")
        self.compare(sqlfunc.ST_Within(sqlfunc.ST_GeomFromText(p2), sqlfunc.ST_GeomFromText(poly)),
            "ST_Within(ST_GeomFromText('POINT(3 3)'),ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'))")
        self.compare(sqlfunc.ST_Within(sqlfunc.ST_GeomFromText(p3), sqlfunc.ST_GeomFromText(poly)),
            "ST_Within(ST_GeomFromText('POINT(5 5)'),ST_GeomFromText('POLYGON((0 0,0 3,3 3,3 0,0 0))'))")
        # ST_X
        pt = "POINT(56.7 53.34)"
        self.compare(sqlfunc.ST_X(sqlfunc.ST_GeomFromText(pt)),
            "ST_X(ST_GeomFromText('POINT(56.7 53.34)'))")
        self.compare(sqlfunc.ST_X(sqlfunc.ST_GeomFromText(pt), 10.5),
            "ST_X(ST_GeomFromText('POINT(56.7 53.34)'),10.5)")
        # ST_Y
        pt = "POINT(56.7 53.34)"
        self.compare(sqlfunc.ST_Y(sqlfunc.ST_GeomFromText(pt)),
            "ST_Y(ST_GeomFromText('POINT(56.7 53.34)'))")
        self.compare(sqlfunc.ST_Y(sqlfunc.ST_GeomFromText(pt), 10.5),
            "ST_Y(ST_GeomFromText('POINT(56.7 53.34)'),10.5)")
        # STR_TO_DATE
        self.compare(sqlfunc.STR_TO_DATE("01,5,2013", "%d,%m,%Y"), "STR_TO_DATE('01,5,2013','%d,%m,%Y')")
        self.compare(sqlfunc.STR_TO_DATE("May 1, 2013", "%M %d,%Y"), "STR_TO_DATE('May 1, 2013','%M %d,%Y')")
        # STRCMP
        self.compare(sqlfunc.STRCMP("text", "text2"), "STRCMP('text','text2')")
        self.compare(sqlfunc.STRCMP("text2", "text"), "STRCMP('text2','text')")
        self.compare(sqlfunc.STRCMP("text", "text"), "STRCMP('text','text')")
        # SUBDATE
        self.compare(sqlfunc.SUBDATE("2008-01-02", 31), "SUBDATE('2008-01-02',31)")
        self.compare(sqlfunc.SUBDATE(datetime.date(2008, 1, 2), sqlintvl.DAY(31)), "SUBDATE('2008-01-02',INTERVAL 31 DAY)")
        self.compare(sqlfunc.SUBDATE(datetime.datetime(2008, 1, 2), sqlintvl.DAY(31)), "SUBDATE('2008-01-02 00:00:00',INTERVAL 31 DAY)")
        # SUBSTRING
        self.compare(sqlfunc.SUBSTRING("Quadratically", 5), "SUBSTRING('Quadratically',5)")
        self.compare(sqlfunc.SUBSTRING("Quadratically", 5, 3), "SUBSTRING('Quadratically',5,3)")
        # SUBSTRING_INDEX
        self.compare(sqlfunc.SUBSTRING_INDEX("www.mysql.com", ".", 2), "SUBSTRING_INDEX('www.mysql.com','.',2)")
        self.compare(sqlfunc.SUBSTRING_INDEX("www.mysql.com", ".", -2), "SUBSTRING_INDEX('www.mysql.com','.',-2)")
        # SUBTIME
        self.compare(sqlfunc.SUBTIME("2007-12-31 23:59:59.999999", "1 1:1:1.000002"), "SUBTIME('2007-12-31 23:59:59.999999','1 1:1:1.000002')")
        self.compare(sqlfunc.SUBTIME("01:00:00.999999", "02:00:00.999998"), "SUBTIME('01:00:00.999999','02:00:00.999998')")
        self.compare(sqlfunc.SUBTIME(datetime.time(1, 0, 0, 999999), "02:00:00.999998"), "SUBTIME('01:00:00.999999','02:00:00.999998')")
        # SYSDATE
        self.compare(sqlfunc.SYSDATE(), "SYSDATE()")
        self.compare(sqlfunc.SYSDATE(3), "SYSDATE(3)")
        # TAN
        self.compare(sqlfunc.TAN(sqlfunc.PI()), "TAN(PI())")
        # TIME
        self.compare(sqlfunc.TIME("2003-12-31 01:02:03"), "TIME('2003-12-31 01:02:03')")
        self.compare(sqlfunc.TIME(datetime.datetime(2003, 12, 31, 1, 2, 3, 123)), "TIME('2003-12-31 01:02:03.000123')")
        # TIME_FORMAT
        self.compare(sqlfunc.TIME_FORMAT("100:00:00", "%H %k %h %I %l"), "TIME_FORMAT('100:00:00','%H %k %h %I %l')")
        self.compare(sqlfunc.TIME_FORMAT(datetime.time(10, 0, 0), "%H %k %h %I %l"), "TIME_FORMAT('10:00:00','%H %k %h %I %l')")
        # TIME_TO_SEC
        self.compare(sqlfunc.TIME_TO_SEC("22:23:00"), "TIME_TO_SEC('22:23:00')")
        self.compare(sqlfunc.TIME_TO_SEC(datetime.time(00, 39, 38)), "TIME_TO_SEC('00:39:38')")
        # TIMEDIFF
        self.compare(sqlfunc.TIMEDIFF("2000-01-01 00:00:00", "2000-01-01 00:00:00.000001"),
            "TIMEDIFF('2000-01-01 00:00:00','2000-01-01 00:00:00.000001')")
        self.compare(sqlfunc.TIMEDIFF(datetime.datetime(2008, 12, 31, 23, 59, 59, 1), datetime.datetime(2008, 12, 30, 1, 1, 1, 2)),
            "TIMEDIFF('2008-12-31 23:59:59.000001','2008-12-30 01:01:01.000002')")
        # TIMESTAMP
        self.compare(sqlfunc.TIMESTAMP("2003-12-31"), "TIMESTAMP('2003-12-31')")
        self.compare(sqlfunc.TIMESTAMP(datetime.datetime(2003, 12, 31, 12), "12:00:00"), "TIMESTAMP('2003-12-31 12:00:00','12:00:00')")
        # TIMESTAMPADD
        self.compare(sqlfunc.TIMESTAMPADD("MINUTE", 1, "2003-01-02"), "TIMESTAMPADD(MINUTE,1,'2003-01-02')")
        self.compare(sqlfunc.TIMESTAMPADD(sqlintvl.WEEK, 1, datetime.datetime(2003, 1, 2)), "TIMESTAMPADD(WEEK,1,'2003-01-02 00:00:00')")
        # TIMESTAMPDIFF
        self.compare(sqlfunc.TIMESTAMPDIFF("MONTH", "2003-02-01", "2003-05-01"), "TIMESTAMPDIFF(MONTH,'2003-02-01','2003-05-01')")
        self.compare(sqlfunc.TIMESTAMPDIFF(sqlintvl.YEAR, datetime.date(2002, 5, 1), "2001-01-01"),
            "TIMESTAMPDIFF(YEAR,'2002-05-01','2001-01-01')")
        self.compare(sqlfunc.TIMESTAMPDIFF(sqlintvl.MINUTE, "2003-02-01", datetime.datetime(2003, 5, 1, 12, 5, 55)),
            "TIMESTAMPDIFF(MINUTE,'2003-02-01','2003-05-01 12:05:55')")
        # TO_DAYS
        self.compare(sqlfunc.TO_DAYS(950501), "TO_DAYS(950501)")
        self.compare(sqlfunc.TO_DAYS("2007-10-07"), "TO_DAYS('2007-10-07')")
        self.compare(sqlfunc.TO_DAYS(datetime.datetime(2007, 10, 7, 0, 0, 0)), "TO_DAYS('2007-10-07 00:00:00')")
        # TO_SECONDS
        self.compare(sqlfunc.TO_SECONDS(950501), "TO_SECONDS(950501)")
        self.compare(sqlfunc.TO_SECONDS("2009-11-29"), "TO_SECONDS('2009-11-29')")
        self.compare(sqlfunc.TO_SECONDS(datetime.datetime(2009, 11, 29, 13, 43, 32)), "TO_SECONDS('2009-11-29 13:43:32')")
        # TRIM
        self.compare(sqlfunc.TRIM("  bar   "), "TRIM('  bar   ')")
        self.compare(sqlfunc.TRIM("xxxbarxxx", "x"), "TRIM('x' FROM 'xxxbarxxx')")
        self.compare(sqlfunc.TRIM("xxxbarxxx", "x", "LEADING"), "TRIM(LEADING 'x' FROM 'xxxbarxxx')")
        # TRUNCATE
        self.compare(sqlfunc.TRUNCATE(1.223, 1), "TRUNCATE(1.223,1)")
        self.compare(sqlfunc.TRUNCATE(1.999, 1), "TRUNCATE(1.999,1)")
        self.compare(sqlfunc.TRUNCATE(122, -2), "TRUNCATE(122,-2)")
        self.compare(sqlfunc.TRUNCATE("10.28", 0), "TRUNCATE('10.28',0)")
        # UNCOMPRESS
        self.compare(sqlfunc.UNCOMPRESS(sqlfunc.COMPRESS("any string")), "UNCOMPRESS(COMPRESS('any string'))")
        self.compare(sqlfunc.UNCOMPRESS("any string"), "UNCOMPRESS('any string')")
        # UNCOMPRESSED_LENGTH
        self.compare(sqlfunc.UNCOMPRESSED_LENGTH(sqlfunc.COMPRESS(sqlfunc.REPEAT('a',30))),
            "UNCOMPRESSED_LENGTH(COMPRESS(REPEAT('a',30)))")
        # UNHEX
        self.compare(sqlfunc.UNHEX("4D7953514C"), "UNHEX('4D7953514C')")
        self.compare(sqlfunc.UNHEX(sqlfunc.HEX("string")), "UNHEX(HEX('string'))")
        self.compare(sqlfunc.HEX(sqlfunc.UNHEX("1267")), "HEX(UNHEX('1267'))")
        # UPPER
        self.compare(sqlfunc.UPPER("hello"), "UPPER('hello')")
        # USER
        self.compare(sqlfunc.USER(), "USER()")
        # UTC_DATE
        self.compare(sqlfunc.UTC_DATE(), "UTC_DATE()")
        # UTC_TIME
        self.compare(sqlfunc.UTC_TIME(), "UTC_TIME()")
        self.compare(sqlfunc.UTC_TIME(3), "UTC_TIME(3)")
        # UTC_TIMESTAMP
        self.compare(sqlfunc.UTC_TIMESTAMP(), "UTC_TIMESTAMP()")
        self.compare(sqlfunc.UTC_TIMESTAMP(3), "UTC_TIMESTAMP(3)")
        # UUID
        self.compare(sqlfunc.UUID(), "UUID()")
        # UUID_SHORT
        self.compare(sqlfunc.UUID_SHORT(), "UUID_SHORT()")
        # UUID_TO_BIN
        uuid = "6ccd780c-baba-1026-9564-5b8c656024db"
        self.compare(sqlfunc.HEX(sqlfunc.UUID_TO_BIN(uuid)), "HEX(UUID_TO_BIN('6ccd780c-baba-1026-9564-5b8c656024db'))")
        self.compare(sqlfunc.HEX(sqlfunc.UUID_TO_BIN(uuid, 1)), "HEX(UUID_TO_BIN('6ccd780c-baba-1026-9564-5b8c656024db',1))")
        # VALIDATE_PASSWORD_STRENGTH
        self.compare(sqlfunc.VALIDATE_PASSWORD_STRENGTH("password"), "VALIDATE_PASSWORD_STRENGTH('password')")
        # VERSION
        self.compare(sqlfunc.VERSION(), "VERSION()")
        # WEEK
        self.compare(sqlfunc.WEEK("2008-02-20"), "WEEK('2008-02-20')")
        self.compare(sqlfunc.WEEK(datetime.date(2008, 2, 20), 1), "WEEK('2008-02-20',1)")
        # WEEKDAY
        self.compare(sqlfunc.WEEKDAY("2008-02-03 22:23:00"), "WEEKDAY('2008-02-03 22:23:00')")
        self.compare(sqlfunc.WEEKDAY(datetime.date(2007, 11, 6)), "WEEKDAY('2007-11-06')")
        # WEEKOFYEAR
        self.compare(sqlfunc.WEEKOFYEAR("2008-02-20"), "WEEKOFYEAR('2008-02-20')")
        self.compare(sqlfunc.WEEKOFYEAR(datetime.date(2008, 2, 20)), "WEEKOFYEAR('2008-02-20')")
        # YEAR
        self.compare(sqlfunc.YEAR("2008-02-20"), "YEAR('2008-02-20')")
        self.compare(sqlfunc.YEAR(datetime.date(2008, 2, 20)), "YEAR('2008-02-20')")
        # YEARWEEK
        self.compare(sqlfunc.YEARWEEK("2008-02-20"), "YEARWEEK('2008-02-20')")
        self.compare(sqlfunc.YEARWEEK(datetime.date(2008, 2, 20), 1), "YEARWEEK('2008-02-20',1)")

        # fmt: on
        self.log_ended("escape")


if __name__ == "__main__":
    TestCustomClass().test_all()
    TestSQLInterval().test_all()
    TestSQLFunction().test_all()
