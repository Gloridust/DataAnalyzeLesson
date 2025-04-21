-- 供销数据库SPJ的SQL创建和查询

CREATE DATABASE IF NOT EXISTS spj_db;
USE spj_db;

-- 供应商表
CREATE TABLE S (
    Sno CHAR(2) PRIMARY KEY, 
    Sname VARCHAR(20),        
    Status INT,               
    City VARCHAR(10)          
);

-- 零件表
CREATE TABLE P (
    Pno CHAR(2) PRIMARY KEY,
    Pname VARCHAR(20),
    Color VARCHAR(10),
    Weight INT 
);

-- 工程项目表
CREATE TABLE J (
    Jno CHAR(2) PRIMARY KEY,
    Jname VARCHAR(20),
    City VARCHAR(10) 
);

-- 供应情况表
CREATE TABLE SPJ (
    Sno CHAR(2),              
    Pno CHAR(2),              
    Jno CHAR(2),              
    Qty INT,                  
    PRIMARY KEY (Sno, Pno, Jno),
    FOREIGN KEY (Sno) REFERENCES S(Sno),
    FOREIGN KEY (Pno) REFERENCES P(Pno),
    FOREIGN KEY (Jno) REFERENCES J(Jno)
);

-- 插入供应商数据
INSERT INTO S VALUES ('S1', '精益', 20, '天津');
INSERT INTO S VALUES ('S2', '东方红', 10, '上海');
INSERT INTO S VALUES ('S3', '东方红', 30, '上海');
INSERT INTO S VALUES ('S4', '金叶', 10, '成都');
INSERT INTO S VALUES ('S5', '泰达', 20, '北京');

-- 插入零件数据
INSERT INTO P VALUES ('P1', '螺母', '红', 20);
INSERT INTO P VALUES ('P2', '螺栓', '绿', 12);
INSERT INTO P VALUES ('P3', '螺丝刀', '蓝', 18);
INSERT INTO P VALUES ('P4', '螺丝刀', '红', 16);
INSERT INTO P VALUES ('P5', '凸轮', '白', 23);

-- 插入工程项目数据
INSERT INTO J VALUES ('J1', '三建', '天津');
INSERT INTO J VALUES ('J2', '一汽', '长春');
INSERT INTO J VALUES ('J3', '造船厂', '上海');
INSERT INTO J VALUES ('J4', '机车厂', '成都');
INSERT INTO J VALUES ('J5', '弹簧厂', '南京');

-- 插入供应情况数据
INSERT INTO SPJ VALUES ('S1', 'P1', 'J1', 200);
INSERT INTO SPJ VALUES ('S1', 'P1', 'J3', 100);
INSERT INTO SPJ VALUES ('S1', 'P1', 'J4', 700);
INSERT INTO SPJ VALUES ('S1', 'P2', 'J2', 100);
INSERT INTO SPJ VALUES ('S2', 'P3', 'J1', 400);
INSERT INTO SPJ VALUES ('S2', 'P3', 'J3', 400);
INSERT INTO SPJ VALUES ('S2', 'P3', 'J4', 50);
INSERT INTO SPJ VALUES ('S2', 'P5', 'J2', 500);
INSERT INTO SPJ VALUES ('S2', 'P5', 'J1', 456);
INSERT INTO SPJ VALUES ('S3', 'P1', 'J4', 366);
INSERT INTO SPJ VALUES ('S3', 'P5', 'J3', 600);
INSERT INTO SPJ VALUES ('S4', 'P5', 'J3', 100);
INSERT INTO SPJ VALUES ('S4', 'P5', 'J2', 200);
INSERT INTO SPJ VALUES ('S4', 'P5', 'J4', 321);
INSERT INTO SPJ VALUES ('S5', 'P2', 'J4', 234);
INSERT INTO SPJ VALUES ('S2', 'P4', 'J4', 345);
INSERT INTO SPJ VALUES ('S3', 'P4', 'J3', 567);

-- 查询任务实现：

-- (1) 求供应工程J1零件的供应商的号码
SELECT DISTINCT Sno
FROM SPJ
WHERE Jno = 'J1';

-- (2) 求供应工程J1零件P1的供应商的号码
SELECT DISTINCT Sno
FROM SPJ
WHERE Jno = 'J1' AND Pno = 'P1';

-- (3) 求供应工程J1零件为红色的供应商的号码
SELECT DISTINCT SPJ.Sno
FROM SPJ, P
WHERE SPJ.Pno = P.Pno AND SPJ.Jno = 'J1' AND P.Color = '红';

-- (4) 求没有使用天津供应商生产的红色零件的工程号
SELECT DISTINCT Jno
FROM J
WHERE Jno NOT IN (
    SELECT DISTINCT SPJ.Jno
    FROM SPJ, S, P
    WHERE SPJ.Sno = S.Sno AND SPJ.Pno = P.Pno 
    AND S.City = '天津' AND P.Color = '红'
);

-- (5) 求至少使用了供应商S1所提供的全部零件的工程号
SELECT DISTINCT J1.Jno
FROM J J1
WHERE NOT EXISTS (
    SELECT *
    FROM SPJ SPJ1
    WHERE SPJ1.Sno = 'S1'
    AND NOT EXISTS (
        SELECT *
        FROM SPJ SPJ2
        WHERE SPJ2.Jno = J1.Jno
        AND SPJ2.Pno = SPJ1.Pno
    )
);

-- (6) 求与供应商同在一个城市的工程名称
SELECT DISTINCT J.Jname
FROM J, S
WHERE J.City = S.City;

-- (7) 找出所有供应商的姓名和所在城市
SELECT Sname, City
FROM S;

-- (8) 找出所有零件的名称、颜色、和重量
SELECT Pname, Color, Weight
FROM P;

-- (9) 找出使用供应商S1所供应零件的工程号码
SELECT DISTINCT Jno
FROM SPJ
WHERE Sno = 'S1';

-- (10) 找出工程项目J2使用的各种零件的名称及其数量
SELECT P.Pname, SPJ.Qty
FROM SPJ, P
WHERE SPJ.Pno = P.Pno AND SPJ.Jno = 'J2';

-- (11) 找出上海厂商供应的所有零件号码
SELECT DISTINCT SPJ.Pno
FROM SPJ, S
WHERE SPJ.Sno = S.Sno AND S.City = '上海';

-- (12) 找出使用上海产的零件的工程名称
SELECT DISTINCT J.Jname
FROM J, SPJ, S
WHERE J.Jno = SPJ.Jno AND SPJ.Sno = S.Sno AND S.City = '上海';

-- (13) 找出没有使用天津产的零件的工程号码
SELECT DISTINCT J.Jno
FROM J
WHERE J.Jno NOT IN (
    SELECT DISTINCT SPJ.Jno
    FROM SPJ, S
    WHERE SPJ.Sno = S.Sno AND S.City = '天津'
);