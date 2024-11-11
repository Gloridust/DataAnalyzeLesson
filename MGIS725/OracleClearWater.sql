--script to create Clearwater Traders database

DROP TABLE ct_order_line CASCADE CONSTRAINTS;
DROP TABLE ct_shipment_line CASCADE CONSTRAINTS;
DROP TABLE ct_shipment CASCADE CONSTRAINTS;
DROP TABLE ct_inventory CASCADE CONSTRAINTS;
DROP TABLE ct_color CASCADE CONSTRAINTS;
DROP TABLE ct_item CASCADE CONSTRAINTS;
DROP TABLE ct_category CASCADE CONSTRAINTS;
DROP TABLE ct_orders CASCADE CONSTRAINTS;
DROP TABLE ct_order_source CASCADE CONSTRAINTS;
DROP TABLE ct_customer CASCADE CONSTRAINTS;

CREATE TABLE ct_customer
(c_id NUMBER(5), 
c_last VARCHAR2(30),
c_first VARCHAR2(30),
c_mi CHAR(1),
c_birthdate DATE,
c_address VARCHAR2(30),
c_city VARCHAR2(30),
c_state CHAR(2),
c_zip VARCHAR2(10),
c_dphone VARCHAR2(10),
c_ephone VARCHAR2(10),
c_userid VARCHAR2(50),
c_password VARCHAR2(15),
CONSTRAINT customer_c_id_pk PRIMARY KEY (c_id));

CREATE TABLE ct_order_source
(os_id NUMBER(3),
os_desc VARCHAR2(30),
CONSTRAINT order_source_os_id_pk PRIMARY KEY(os_id));

CREATE TABLE ct_orders
(o_id NUMBER(8), 
o_date DATE,
o_methpmt VARCHAR2(10),
c_id NUMBER(5),
os_id NUMBER(3),
CONSTRAINT orders_o_id_pk PRIMARY KEY (o_id),
CONSTRAINT orders_c_id_fk FOREIGN KEY (c_id) REFERENCES ct_customer(c_id),
CONSTRAINT orders_os_id_fk FOREIGN KEY (os_id) REFERENCES ct_order_source(os_id));

CREATE TABLE ct_category
(cat_id NUMBER(2),
cat_desc VARCHAR2(20),
CONSTRAINT category_cat_id_pk PRIMARY KEY (cat_id));

CREATE TABLE ct_item
(item_id NUMBER(8),
item_desc VARCHAR2(30),
cat_id NUMBER(2),
item_image BLOB,
CONSTRAINT item_item_id_pk PRIMARY KEY (item_id),
CONSTRAINT item_cat_id_fk FOREIGN KEY (cat_id) REFERENCES ct_category(cat_id));

CREATE TABLE ct_color
(color VARCHAR2(20),
CONSTRAINT color_color_pk PRIMARY KEY (color));

CREATE TABLE ct_inventory
(inv_id NUMBER(10),
item_id NUMBER(8),
color VARCHAR2(20),
inv_size VARCHAR2(10),
inv_price NUMBER(6,2),
inv_qoh NUMBER(4),
CONSTRAINT inventory_inv_id_pk PRIMARY KEY (inv_id),
CONSTRAINT inventory_item_id_fk FOREIGN KEY (item_id) REFERENCES ct_item(item_id),
CONSTRAINT inventory_color_fk FOREIGN KEY (color) REFERENCES ct_color(color));

CREATE TABLE ct_shipment
(ship_id NUMBER(10),
ship_date_expected DATE,
CONSTRAINT shipment_ship_id_pk PRIMARY KEY (ship_id));

CREATE TABLE ct_shipment_line
(ship_id NUMBER(10), 
inv_id NUMBER(10),
sl_quantity NUMBER(4),
sl_date_received DATE, 
CONSTRAINT shipment_line_ship_id_fk FOREIGN KEY (ship_id) REFERENCES ct_shipment(ship_id),
CONSTRAINT shipment_line_inv_id_fk FOREIGN KEY (inv_id) REFERENCES ct_inventory(inv_id),
CONSTRAINT shipment_line_shipid_invid_pk PRIMARY KEY(ship_id, inv_id));

CREATE TABLE ct_order_line 
(o_id NUMBER(8), 
inv_id NUMBER(10), 
ol_quantity NUMBER(4) NOT NULL,  
CONSTRAINT order_line_o_id_fk FOREIGN KEY (o_id) REFERENCES ct_orders(o_id),
CONSTRAINT order_line_inv_id_fk FOREIGN KEY (inv_id) REFERENCES ct_inventory(inv_id),
CONSTRAINT order_line_oid_invid_pk PRIMARY KEY (o_id, inv_id));


--- inserting records into CUSTOMER
INSERT INTO CT_CUSTOMER VALUES
(1, 'Graham', 'Neal', 'R', to_date('12/10/1967', 'mm/dd/yyyy'), '9815 Circle Dr.', 'Tallahassee', 'FL', '32308', '9045551897', '904558599', 'grahamn', 'barbiecar');

INSERT INTO CT_CUSTOMER VALUES
(2, 'Sanchez', 'Myra', 'T', to_date('08/14/1958', 'mm/dd/yyyy'), '172 Alto Park', 'Seattle', 'WA','42180', '4185551791', '4185556643', 'sanchezmt', 'qwert5');

INSERT INTO CT_CUSTOMER VALUES
(3, 'Smith', 'Lisa', 'M', to_date('04/12/1960', 'mm/dd/yyyy'), '850 East Main', 'Santa Ana', 'CA', '51875', '3075557841', '3075559852', 'smithlm', 'joshua5');

INSERT INTO CT_CUSTOMER VALUES
(4, 'Phelp', 'Paul', NULL, to_date('01/18/1981', 'mm/dd/yyyy'), '994 Kirkman Rd.', 'Northpoint', 'NY', '11795', '4825554788', '4825558219', 'phelpp', 'hold98er');

INSERT INTO CT_CUSTOMER VALUES
(5, 'Lewis', 'Sheila', 'A', to_date('08/30/1978', 'mm/dd/yyyy'), '195 College Blvd.', 'Newton', 'GA', '37812', '3525554972', '3525551811', 'lewissa', '125pass');

INSERT INTO CT_CUSTOMER VALUES
(6, 'James', 'Thomas', 'E', to_date('06/01/1973', 'mm/dd/yyyy'), '348 Rice Lane', 'Radcliff', 'WY', '87195', '7615553485', '7615553319', 'jamest', 'nok$tell');

--- inserting records into ORDER_SOURCE
INSERT INTO ct_order_source VALUES (1, 'Winter 2005');
INSERT INTO ct_order_source VALUES (2, 'Spring 2006');
INSERT INTO ct_order_source VALUES (3, 'Summer 2006');
INSERT INTO ct_order_source VALUES (4, 'Outdoor 2006');
INSERT INTO ct_order_source VALUES (5, 'Children''s 2006');
INSERT INTO ct_order_source VALUES (6, 'Web Site');

--- inserting records into orders
INSERT INTO ct_orders VALUES
(1, TO_DATE('05/29/2006', 'MM/DD/YYYY'), 'CC', 1, 2);

INSERT INTO ct_orders VALUES
(2, TO_DATE('05/29/2006', 'MM/DD/YYYY'), 'CC', 5, 6);

INSERT INTO ct_orders VALUES
(3, TO_DATE('05/31/2006', 'MM/DD/YYYY'), 'CHECK', 2, 2);

INSERT INTO ct_orders VALUES
(4, TO_DATE('05/31/2006', 'MM/DD/YYYY'), 'CC', 3, 3);

INSERT INTO ct_orders VALUES
(5, TO_DATE('06/01/2006', 'MM/DD/YYYY'), 'CC', 4, 6);

INSERT INTO ct_orders VALUES
(6, TO_DATE('06/01/2006', 'MM/DD/YYYY'), 'CC', 4, 3);

--- inserting records into CATEGORY
INSERT INTO ct_category VALUES (1, 'Women''s Clothing');
INSERT INTO ct_category VALUES (2, 'Children''s Clothing');
INSERT INTO ct_category VALUES (3, 'Men''s Clothing');
INSERT INTO ct_category VALUES (4, 'Outdoor Gear');

--- inserting records into ITEM
INSERT INTO ct_item VALUES
(1, 'Men''s Expedition Parka', 3, EMPTY_BLOB());

INSERT INTO ct_item VALUES
(2, '3-Season Tent', 4, EMPTY_BLOB());

INSERT INTO ct_item VALUES
(3, 'Women''s Hiking Shorts', 1, EMPTY_BLOB());

INSERT INTO ct_item VALUES
(4, 'Women''s Fleece Pullover', 1, EMPTY_BLOB());

INSERT INTO ct_item VALUES
(5, 'Children''s Beachcomber Sandals', 2, EMPTY_BLOB());

INSERT INTO ct_item VALUES
(6, 'Boy''s Surf Shorts', 2, EMPTY_BLOB());

INSERT INTO ct_item VALUES
(7, 'Girl''s Soccer Tee', 2, EMPTY_BLOB());

--- inserting records into COLOR
INSERT INTO ct_color VALUES ('Sky Blue');
INSERT INTO ct_color VALUES ('Light Grey');
INSERT INTO ct_color VALUES ('Khaki');
INSERT INTO ct_color VALUES ('Navy');
INSERT INTO ct_color VALUES ('Royal');
INSERT INTO ct_color VALUES ('Eggplant');
INSERT INTO ct_color VALUES ('Blue');
INSERT INTO ct_color VALUES ('Red');
INSERT INTO ct_color VALUES ('Spruce');
INSERT INTO ct_color VALUES ('Turquoise');
INSERT INTO ct_color VALUES ('Bright Pink');
INSERT INTO ct_color VALUES ('White');

--- inserting records into INVENTORY
INSERT INTO ct_inventory VALUES
(1, 2, 'Sky Blue', NULL, 259.99, 16);

INSERT INTO ct_inventory VALUES
(2, 2, 'Light Grey', NULL, 259.99, 12);

INSERT INTO ct_inventory VALUES
(3, 3, 'Khaki', 'S', 29.95, 150);

INSERT INTO ct_inventory VALUES
(4, 3, 'Khaki', 'M', 29.95, 147);

INSERT INTO ct_inventory VALUES
(5, 3, 'Khaki', 'L', 29.95, 0);

INSERT INTO ct_inventory VALUES
(6, 3, 'Navy', 'S', 29.95, 139);

INSERT INTO ct_inventory VALUES
(7, 3, 'Navy', 'M', 29.95, 137);

INSERT INTO ct_inventory VALUES
(8, 3, 'Navy', 'L', 29.95, 115);

INSERT INTO ct_inventory VALUES
(9, 4, 'Eggplant', 'S', 59.95, 135);

INSERT INTO ct_inventory VALUES
(10, 4, 'Eggplant', 'M', 59.95, 168);

INSERT INTO ct_inventory VALUES
(11, 4, 'Eggplant', 'L', 59.95, 187);

INSERT INTO ct_inventory VALUES
(12, 4, 'Royal', 'S', 59.95, 0);

INSERT INTO ct_inventory VALUES
(13, 4, 'Royal', 'M', 59.95, 124);

INSERT INTO ct_inventory VALUES
(14, 4, 'Royal', 'L', 59.95, 112);

INSERT INTO ct_inventory VALUES
(15, 5, 'Turquoise', '10', 15.99, 121);

INSERT INTO ct_inventory VALUES
(16, 5, 'Turquoise', '11', 15.99, 111);

INSERT INTO ct_inventory VALUES
(17, 5, 'Turquoise', '12', 15.99, 113);

INSERT INTO ct_inventory VALUES
(18, 5, 'Turquoise', '1', 15.99, 121);

INSERT INTO ct_inventory VALUES
(19, 5, 'Bright Pink', '10', 15.99, 148);

INSERT INTO ct_inventory VALUES
(20, 5, 'Bright Pink', '11', 15.99, 137);

INSERT INTO ct_inventory VALUES
(21, 5, 'Bright Pink', '12', 15.99, 134);

INSERT INTO ct_inventory VALUES
(22, 5, 'Bright Pink', '1', 15.99, 123);

INSERT INTO ct_inventory VALUES
(23, 1, 'Spruce', 'S', 199.95, 114);

INSERT INTO ct_inventory VALUES
(24, 1,  'Spruce', 'M',199.95, 17);

INSERT INTO ct_inventory VALUES
(25, 1, 'Spruce', 'L', 209.95, 0);

INSERT INTO ct_inventory VALUES
(26, 1, 'Spruce', 'XL', 209.95, 12);

INSERT INTO ct_inventory VALUES
(27, 6, 'Blue', 'S', 15.95, 50);

INSERT INTO ct_inventory VALUES
(28, 6, 'Blue', 'M', 15.95, 100);

INSERT INTO ct_inventory VALUES
(29, 6, 'Blue', 'L', 15.95, 100);

INSERT INTO ct_inventory VALUES
(30, 7, 'White', 'S', 19.99, 100);

INSERT INTO ct_inventory VALUES
(31, 7, 'White', 'M', 19.99, 100);

INSERT INTO ct_inventory VALUES
(32, 7, 'White', 'L', 19.99, 100);

--inserting records into SHIPMENT

INSERT INTO ct_shipment VALUES
(1, TO_DATE('09/15/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment VALUES
(2, TO_DATE('11/15/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment VALUES
(3, TO_DATE('06/25/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment VALUES
(4, TO_DATE('06/25/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment VALUES
(5, TO_DATE('08/15/2006', 'MM/DD/YYYY'));

--inserting records into SHIPMENT_LINE
INSERT INTO ct_shipment_line VALUES
(1, 1, 25, TO_DATE('09/10/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment_line VALUES
(1, 2, 25, TO_DATE('09/10/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment_line VALUES
(2, 2, 25, NULL);

INSERT INTO ct_shipment_line VALUES
(3, 5, 200, NULL);

INSERT INTO ct_shipment_line VALUES
(3, 6, 200, NULL);

INSERT INTO ct_shipment_line VALUES
(3, 7, 200, NULL);

INSERT INTO ct_shipment_line VALUES
(4, 12, 100, TO_DATE('08/15/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment_line VALUES
(4, 13, 100, TO_DATE('08/25/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment_line VALUES
(5, 23, 50, TO_DATE('08/15/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment_line VALUES
(5, 24, 100, TO_DATE('08/15/2006', 'MM/DD/YYYY'));

INSERT INTO ct_shipment_line VALUES
(5, 25, 100, TO_DATE('08/15/2006', 'MM/DD/YYYY'));

--- inserting records into ORDER_LINE
INSERT INTO ct_order_line VALUES (1, 1, 1);
INSERT INTO ct_order_line VALUES (1, 14, 2);
INSERT INTO ct_order_line VALUES (2, 19, 1);
INSERT INTO ct_order_line VALUES (3, 24, 1);
INSERT INTO ct_order_line VALUES (3, 26, 1);
INSERT INTO ct_order_line VALUES (4, 12, 2);
INSERT INTO ct_order_line VALUES (5, 8, 1);
INSERT INTO ct_order_line VALUES (5, 13, 1);
INSERT INTO ct_order_line VALUES (6, 2, 1);
INSERT INTO ct_order_line VALUES (6, 7, 3);

COMMIT;
