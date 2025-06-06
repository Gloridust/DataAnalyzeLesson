import textwrap

def print_query_with_explanation(number, question, query, explanation):
    print(f"\n{'='*80}")
    print(f"Question {number}: {question}")
    print(f"\nSQL Query:")
    print(f"{'-'*80}")
    print(textwrap.dedent(query).strip())
    print(f"\nExplanation:")
    print(f"{'-'*80}")
    print(textwrap.dedent(explanation).strip())
    print(f"{'='*80}\n")

# Query 1: Customer who placed order #1 and total order amount
query1 = """
    SELECT c.c_first, c.c_last, o.o_id, 
           SUM(ol.ol_quantity * i.inv_price) as total_amount
    FROM ct_customer c
    JOIN ct_orders o ON c.c_id = o.c_id
    JOIN ct_order_line ol ON o.o_id = ol.o_id
    JOIN ct_inventory i ON ol.inv_id = i.inv_id
    WHERE o.o_id = 1
    GROUP BY c.c_first, c.c_last, o.o_id;
"""
explanation1 = """
    This query:
    1. Joins customer table with orders, order_line and inventory
    2. Filters for order #1
    3. Calculates total amount by multiplying quantity with price
    4. Returns customer name and total order amount
"""

# Query 2: Undelivered shipments and their items
query2 = """
    SELECT s.ship_id, s.ship_date_expected, 
           i.item_desc, sl.sl_quantity
    FROM ct_shipment s
    JOIN ct_shipment_line sl ON s.ship_id = sl.ship_id
    JOIN ct_inventory inv ON sl.inv_id = inv.inv_id
    JOIN ct_item i ON inv.item_id = i.item_id
    WHERE sl.sl_date_received IS NULL
    ORDER BY s.ship_id;
"""
explanation2 = """
    This query:
    1. Joins shipment with shipment_line, inventory and item tables
    2. Filters for shipments where received date is NULL
    3. Shows shipment details and items waiting to be received
"""

# Query 3: Customers who purchased same items as customer #3
query3 = """
    SELECT DISTINCT c.c_id, c.c_first, c.c_last
    FROM ct_customer c
    JOIN ct_orders o ON c.c_id = o.c_id
    JOIN ct_order_line ol ON o.o_id = ol.o_id
    JOIN ct_inventory i ON ol.inv_id = i.inv_id
    WHERE i.item_id IN (
        SELECT i2.item_id
        FROM ct_orders o2
        JOIN ct_order_line ol2 ON o2.o_id = ol2.o_id
        JOIN ct_inventory i2 ON ol2.inv_id = i2.inv_id
        WHERE o2.c_id = 3
    )
    AND c.c_id != 3;
"""
explanation3 = """
    This query:
    1. Uses a subquery to find all items purchased by customer #3
    2. Finds other customers who bought any of those same items
    3. Excludes customer #3 from the results
"""

# Query 4: Total sales from website
query4 = """
    SELECT os.os_desc, 
           SUM(ol.ol_quantity * i.inv_price) as total_sales
    FROM ct_orders o
    JOIN ct_order_source os ON o.os_id = os.os_id
    JOIN ct_order_line ol ON o.o_id = ol.o_id
    JOIN ct_inventory i ON ol.inv_id = i.inv_id
    WHERE os.os_desc = 'Web Site'
    GROUP BY os.os_desc;
"""
explanation4 = """
    This query:
    1. Joins orders with order_source, order_line and inventory
    2. Filters for orders from 'Web Site' source
    3. Calculates total sales by multiplying quantity with price
"""

def main():
    # Question 1
    print_query_with_explanation(
        1,
        "Determine which customer placed order #1 and the total amount of the order.",
        query1,
        explanation1
    )
    
    # Question 2
    print_query_with_explanation(
        2,
        "Identify which shipments have not yet been received by Clearwater Traders and the items on each of those shipments.",
        query2,
        explanation2
    )
    
    # Question 3
    print_query_with_explanation(
        3,
        "Determine which customers have purchased the same items as customer #3.",
        query3,
        explanation3
    )
    
    # Question 4
    print_query_with_explanation(
        4,
        "Determine the total amount of sales generated by the company's Web site.",
        query4,
        explanation4
    )

if __name__ == "__main__":
    main()