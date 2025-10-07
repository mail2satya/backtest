from flask import Flask, jsonify, render_template, request
import sqlite3

app = Flask(__name__)

import os

# Build a robust path to the database file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'merge_data.sqlite')

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
    # Use DISTINCT to avoid duplicate symbols from the merged_data table
    symbols = conn.execute(
        "SELECT DISTINCT stock FROM merged_data WHERE stock LIKE ? ORDER BY stock",
        (query.upper() + '%',)
    ).fetchall()
    conn.close()

    return jsonify([row['stock'] for row in symbols])

@app.route('/api/ohlc')
def ohlc():
    symbol = request.args.get('symbol', '')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    conn = get_db_connection()
    # Order by date to ensure data is chronological for charting
    # Fetch all relevant columns, including corporate actions
    ohlc_data = conn.execute(
        "SELECT date, open, high, low, close, action_type, value FROM merged_data WHERE stock = ? ORDER BY date",
        (symbol.upper(),)
    ).fetchall()
    conn.close()

    # Convert rows to a list of dictionaries
    data = [dict(row) for row in ohlc_data]
    return jsonify(data)

@app.route('/api/calculate_investment', methods=['POST'])
def calculate_investment():
    data = request.get_json()
    symbol = data.get('symbol')
    investment_amount = float(data.get('amount'))
    start_date = data.get('date')

    if not all([symbol, investment_amount, start_date]):
        return jsonify({"error": "Missing required parameters"}), 400

    conn = get_db_connection()
    # Fetch all data for the stock from the start date
    history = conn.execute(
        "SELECT date, open, high, low, close, action_type, value FROM merged_data WHERE stock = ? AND date >= ? ORDER BY date",
        (symbol.upper(), start_date)
    ).fetchall()
    conn.close()

    if not history:
        return jsonify({"error": "No data found for the given stock and date"}), 404

    # Initial investment
    initial_price = history[0]['close']
    if initial_price == 0:
        return jsonify({"error": "Initial price is zero, cannot calculate investment"}), 400

    shares = investment_amount / initial_price
    cash = 0  # To hold dividend payouts before reinvesting

    # Process historical data
    for i, day in enumerate(history):
        # Data source is pre-adjusted for splits/bonuses, so we only handle dividends.
        action = day['action_type'].lower() if day['action_type'] else ''
        if action == 'dividend':
            # Dividend: Add to cash, which is then reinvested
            cash += shares * day['value']

        # Reinvest cash from dividends on the same day
        if cash > 0:
            reinvestment_price = day['close']
            if reinvestment_price > 0:
                additional_shares = cash / reinvestment_price
                shares += additional_shares
                cash = 0  # Reset cash after reinvesting

    # Calculate final value
    final_price = history[-1]['close']
    final_value = shares * final_price

    return jsonify({
        "initial_investment": investment_amount,
        "final_value": round(final_value, 2),
        "start_date": history[0]['date'],
        "end_date": history[-1]['date'],
        "start_price": history[0]['close'],
        "end_price": final_price,
        "total_shares": round(shares, 4)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)