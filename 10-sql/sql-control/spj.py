import sqlite3
import os

def execute_sql_file(db_name="spj_database.db"):
    """
    Creates an SQLite database and executes the SQL file.
    Runs all queries and displays their results.
    """
    # Delete the database file if it exists
    if os.path.exists(db_name):
        os.remove(db_name)
    
    # Connect to the database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Read the SQL file
    with open("lab.sql", "r", encoding="utf-8") as file:
        sql_script = file.read()
    
    # First, execute the CREATE TABLE statements
    create_statements = [stmt for stmt in sql_script.split(';') 
                         if 'CREATE TABLE' in stmt.upper()]
    
    for statement in create_statements:
        stmt = statement.strip()
        if stmt:
            try:
                cursor.execute(stmt)
                conn.commit()
            except sqlite3.Error as e:
                print(f"Error creating table: {e}")
    
    # Then execute the INSERT statements, ignoring duplicate errors
    insert_statements = [stmt for stmt in sql_script.split(';') 
                        if 'INSERT INTO' in stmt.upper()]
    
    for statement in insert_statements:
        stmt = statement.strip()
        if stmt:
            try:
                cursor.execute(stmt)
                conn.commit()
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    print(f"Warning: Duplicate entry skipped - {stmt.split('VALUES')[1].strip()}")
                else:
                    print(f"Error inserting data: {e}")
    
    # Finally execute the SELECT statements
    select_statements = [stmt for stmt in sql_script.split(';') 
                         if stmt.strip().upper().startswith('SELECT')]
    
    for i, statement in enumerate(select_statements):
        stmt = statement.strip()
        if stmt:
            try:
                # Execute the statement
                cursor.execute(stmt)
                
                # Identify which query this is from the comments
                query_name = None
                lines = stmt.split('\n')
                for j in range(max(0, i-3), i+1):  # Look at a few lines before the query
                    if j < len(lines):
                        line = lines[j]
                        if '--' in line and '(' in line and ')' in line:
                            query_name = line.strip()
                            break
                
                # Fetch and print the results
                results = cursor.fetchall()
                if query_name:
                    print(f"\n{query_name}")
                else:
                    print(f"\nQuery {i+1} results:")
                
                # Print column names if available
                if cursor.description:
                    col_names = [desc[0] for desc in cursor.description]
                    print(f"Columns: {', '.join(col_names)}")
                
                # Print results
                if results:
                    for row in results:
                        print(row)
                else:
                    print("No results found.")
                
            except sqlite3.Error as e:
                print(f"Error executing query: {e}")
    
    # Close the connection
    conn.close()
    print("\nDatabase created and all queries executed successfully!")

def main():
    print("Starting SPJ database simulation...")
    execute_sql_file()
    print("Simulation completed.")

if __name__ == "__main__":
    main()