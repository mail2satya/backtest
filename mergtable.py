import sqlite3

def rebuild_merged_table_separate_db(source_db='fno_data.sqlite', target_db='merge_data.sqlite'):
    # Connect source and target databases
    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)

    # Attach source_db to the target connection for cross-db querying
    target_conn.execute(f"ATTACH DATABASE '{source_db}' AS source_db")

    cursor = target_conn.cursor()

    # Drop merged_data table if exists in target
    cursor.execute('DROP TABLE IF EXISTS merged_data;')

    # Create merged_data by joining tables in source_db
    create_query = '''
    CREATE TABLE merged_data AS
    SELECT
      o.Date AS date,
      o.Stock AS stock,
      o.Open AS open,
      o.High AS high,
      o.Low AS low,
      o.Close AS close,
      o.Volume AS volume,
      c.Action_Type AS action_type,
      c.Value AS value
    FROM
      source_db.ohlc_data o
    LEFT JOIN
      (
        SELECT
          substr(Date,1,10) AS Date,
          CASE
            WHEN Stock LIKE '%.NS' THEN substr(Stock, 1, length(Stock) - 3)
            ELSE Stock
          END AS Stock,
          Action_Type,
          Value
        FROM source_db.corporate_actions
      ) c
    ON o.Date = c.Date AND o.Stock = c.Stock;
    '''

    cursor.execute(create_query)
    target_conn.commit()

    # Detach source_db after use
    target_conn.execute('DETACH DATABASE source_db')

    # Close connections
    cursor.close()
    target_conn.close()
    source_conn.close()

    print(f"Merged table created successfully in database '{target_db}'.")

if __name__ == '__main__':
    rebuild_merged_table_separate_db()
