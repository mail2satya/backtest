import sqlite3
import pandas as pd

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect('upstox_data.db')
    return conn

def create_ohlc_table():
    """Creates the OHLC data table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlc (
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            PRIMARY KEY (symbol, timestamp)
        )
    ''')
    conn.commit()
    conn.close()

def save_ohlc_data(symbol, ohlc_data):
    """Saves a DataFrame of OHLC data to the database."""
    conn = get_db_connection()

    # Prepare data for insertion
    data_to_insert = []
    for candle in ohlc_data:
        # Assuming candle is a list: [timestamp, open, high, low, close, volume, open_interest]
        # Adjust indices based on the actual structure of the candle data
        timestamp = candle[0]
        open_price = candle[1]
        high_price = candle[2]
        low_price = candle[3]
        close_price = candle[4]
        data_to_insert.append((symbol, timestamp, open_price, high_price, low_price, close_price))

    if data_to_insert:
        cursor = conn.cursor()
        # Use INSERT OR REPLACE to avoid duplicate entries for the same symbol and timestamp
        cursor.executemany('''
            INSERT OR REPLACE INTO ohlc (symbol, timestamp, open, high, low, close)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        conn.commit()
        print(f"Saved {len(data_to_insert)} records for {symbol}.")

    conn.close()

if __name__ == '__main__':
    # Initialize the database and table
    create_ohlc_table()
    print("Database and 'ohlc' table are ready.")