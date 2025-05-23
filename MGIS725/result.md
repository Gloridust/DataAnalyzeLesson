# ClearWater Traders Database Analysis Report

## Introduction
This report presents the results of SQL analysis performed on the ClearWater Traders database, addressing four specific business questions. The analysis includes customer orders, shipment tracking, customer purchase patterns, and web sales performance.

## Analysis Tasks and Results

### 1. Customer Order #1 Analysis
**Question**: Determine which customer placed order #1 and the total amount of the order.

**SQL Query**:

```sql
SELECT 
    c.c_first || ' ' || c.c_last AS customer_name,
    SUM(i.inv_price * ol.ol_quantity) AS total_amount
FROM ct_customer c
JOIN ct_orders o ON c.c_id = o.c_id
JOIN ct_order_line ol ON o.o_id = ol.o_id
JOIN ct_inventory i ON ol.inv_id = i.inv_id
WHERE o.o_id = 1
GROUP BY c.c_first, c.c_last;
```

**Results**:

| Customer Name | Total Amount |
|--------------|--------------|
| Neal Graham  | $379.89      |

### 2. Pending Shipments Analysis
**Question**: Identify which shipments have not yet been received by Clearwater Traders and the items on each of those shipments.

**SQL Query**:

```sql
SELECT 
    s.ship_id,
    s.ship_date_expected,
    i.item_desc,
    sl.sl_quantity,
    inv.color,
    inv.inv_size
FROM ct_shipment s
JOIN ct_shipment_line sl ON s.ship_id = sl.ship_id
JOIN ct_inventory inv ON sl.inv_id = inv.inv_id
JOIN ct_item i ON inv.item_id = i.item_id
WHERE sl.sl_date_received IS NULL
ORDER BY s.ship_id;
```

**Results**:

| Ship ID | Expected Date | Item Description | Quantity | Color | Size |
|---------|---------------|------------------|----------|--------|------|
| 2 | 11/15/2006 | 3-Season Tent | 25 | Light Grey | - |
| 3 | 06/25/2006 | Women's Hiking Shorts | 200 | Navy | S |
| 3 | 06/25/2006 | Women's Hiking Shorts | 200 | Navy | M |
| 3 | 06/25/2006 | Women's Hiking Shorts | 200 | Khaki | L |

### 3. Customer Purchase Pattern Analysis
**Question**: Determine which customers have purchased the same items as customer #3.

**SQL Query**:

```sql
SELECT DISTINCT 
    c.c_id,
    c.c_first || ' ' || c.c_last AS customer_name,
    i.item_desc
FROM ct_customer c
JOIN ct_orders o ON c.c_id = o.c_id
JOIN ct_order_line ol ON o.o_id = ol.o_id
JOIN ct_inventory inv ON ol.inv_id = inv.inv_id
JOIN ct_item i ON inv.item_id = i.item_id
WHERE i.item_id IN (
    SELECT DISTINCT i2.item_id
    FROM ct_orders o2
    JOIN ct_order_line ol2 ON o2.o_id = ol2.o_id
    JOIN ct_inventory inv2 ON ol2.inv_id = inv2.inv_id
    JOIN ct_item i2 ON inv2.item_id = i2.item_id
    WHERE o2.c_id = 3
)
AND c.c_id != 3
ORDER BY c.c_id;
```

**Results**:

| Customer ID | Customer Name | Item Description |
|-------------|---------------|------------------|
| 1 | Neal Graham | Women's Fleece Pullover |
| 4 | Paul Phelp | Women's Fleece Pullover |

### 4. Web Sales Analysis
**Question**: Determine the total amount of sales generated by the company's Web site.

**SQL Query**:

```sql
SELECT 
    SUM(i.inv_price * ol.ol_quantity) AS total_web_sales
FROM ct_orders o
JOIN ct_order_source os ON o.os_id = os.os_id
JOIN ct_order_line ol ON o.o_id = ol.o_id
JOIN ct_inventory i ON ol.inv_id = i.inv_id
WHERE os.os_desc = 'Web Site';
```

**Results**:

| Total Web Sales |
|-----------------|
| $105.89         |

![](./Screenshot%202024-11-11%20at%2019.32.37.png)

## Key Findings

1. **Order Analysis**:
   - Customer Neal Graham placed order #1 with a total value of $379.89
   - This represents a significant individual order value

2. **Shipment Status**:
   - There are pending deliveries of both outdoor gear (tents) and clothing items
   - A large shipment of Women's Hiking Shorts (600 units total) is pending
   - The expected delivery dates range from June to November 2006

3. **Customer Behavior**:
   - Two customers (Neal Graham and Paul Phelp) have shown similar purchasing patterns to customer #3
   - The Women's Fleece Pullover appears to be a popular item among multiple customers

4. **Web Sales Performance**:
   - The web sales channel has generated $105.89 in total sales
   - This relatively low figure might indicate an opportunity for improving online sales performance

## Recommendations

1. **Inventory Management**:
   - Monitor the pending shipments, especially the large shipment of Women's Hiking Shorts
   - Consider implementing a better tracking system for delayed shipments

2. **Sales Channel Optimization**:
   - Investigate ways to improve web sales performance
   - Consider implementing marketing strategies to increase online presence

3. **Customer Analysis**:
   - Use the purchase pattern data to develop targeted marketing campaigns
   - Consider creating a customer loyalty program based on purchasing behaviors

4. **Product Strategy**:
   - Focus on popular items like the Women's Fleece Pullover
   - Consider expanding the product range in successful categories

## Conclusion
The analysis reveals important insights about ClearWater Traders' operations, highlighting areas of strength in individual sales while also identifying opportunities for improvement in online sales channels and inventory management.