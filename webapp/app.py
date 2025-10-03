from flask import Flask, jsonify, render_template, request
import sqlite3

app = Flask(__name__)

import os

# Build a robust path to the database file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'upstox_data.db')

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search')
def search():
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify([])

    conn = get_db_connection()
    # Use DISTINCT to avoid duplicate symbols
    symbols = conn.execute(
        "SELECT DISTINCT symbol FROM ohlc WHERE symbol LIKE ? ORDER BY symbol",
        (query.upper() + '%',)
    ).fetchall()
    conn.close()

    return jsonify([row['symbol'] for row in symbols])

@app.route('/api/ohlc')
def ohlc():
    symbol = request.args.get('symbol', '')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    conn = get_db_connection()
    # Order by timestamp to ensure data is chronological for charting
    ohlc_data = conn.execute(
        "SELECT timestamp, open, high, low, close FROM ohlc WHERE symbol = ? ORDER BY timestamp",
        (symbol.upper(),)
    ).fetchall()
    conn.close()

    # Convert rows to a list of dictionaries
    data = [dict(row) for row in ohlc_data]
    return jsonify(data)

if __name__ == '__main__':
    # Temporarily disabling reloader for stable verification
    app.run(debug=True, port=5001, use_reloader=False)