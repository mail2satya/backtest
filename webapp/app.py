from flask import Flask, jsonify, render_template, request
import sqlite3
from datetime import datetime

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
    from_date = data.get('from_date')
    to_date = data.get('to_date')

    if not all([symbol, investment_amount, from_date, to_date]):
        return jsonify({"error": "Missing required parameters"}), 400

    conn = get_db_connection()
    # Fetch all data for the stock within the specified date range
    history = conn.execute(
        "SELECT date, open, high, low, close, action_type, value FROM merged_data WHERE stock = ? AND date BETWEEN ? AND ? ORDER BY date",
        (symbol.upper(), from_date, to_date)
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
    total_dividends_received = 0

    # Process historical data
    for i, day in enumerate(history):
        # Data source is pre-adjusted for splits/bonuses, so we only handle dividends.
        action = day['action_type'].lower() if day['action_type'] else ''
        if action == 'dividend':
            # Calculate cash dividend and add to reinvestment pool and total tracker
            dividend_cash = shares * day['value']
            total_dividends_received += dividend_cash
            cash += dividend_cash

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

    # Calculate CAGR
    start_dt = datetime.strptime(history[0]['date'], '%Y-%m-%d')
    end_dt = datetime.strptime(history[-1]['date'], '%Y-%m-%d')
    years = (end_dt - start_dt).days / 365.25
    cagr = 0
    if years > 0 and investment_amount > 0:
        cagr = ((final_value / investment_amount) ** (1 / years)) - 1

    return jsonify({
        "initial_investment": investment_amount,
        "final_value": round(final_value, 2),
        "start_date": history[0]['date'],
        "end_date": history[-1]['date'],
        "start_price": history[0]['close'],
        "end_price": final_price,
        "total_dividends": round(total_dividends_received, 2),
        "cagr": round(cagr * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)