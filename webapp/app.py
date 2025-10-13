from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

app = Flask(__name__)
# In production, set a real, secure SECRET_KEY environment variable.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_fallback_secret_key_for_dev')

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to login page if user is not authenticated

class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user_data = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user_data:
        return User(id=user_data['id'], username=user_data['username'], password_hash=user_data['password_hash'])
    return None

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        user = User(id=user_data['id'], username=user_data['username'], password_hash=user_data['password_hash']) if user_data else None
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user_exists = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        if user_exists:
            flash('Username already exists.')
            conn.close()
            return redirect(url_for('register'))

        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
        conn.commit()
        conn.close()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Database setup
# In a production environment (like Render), set the DATABASE_URL environment variable
# to the path of your persistent disk, e.g., /var/data/merge_data.sqlite
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    DB_PATH = DATABASE_URL
else:
    # Local development uses the SQLite file in the repo
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, '..', 'merge_data.sqlite')

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
@login_required
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

@app.route('/scanner', methods=['GET', 'POST'])
@login_required
def scanner():
    if request.method == 'POST':
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        investment_amount = float(request.form.get('investment_amount', 10000))

        if not from_date or not to_date:
            return render_template('scanner.html', error="Please select both a start and end date.")

        results = calculate_stock_performance(investment_amount, from_date, to_date)
        return render_template('scanner.html', results=results)

    return render_template('scanner.html')

def calculate_stock_performance(investment_amount, from_date, to_date):
    """
    Calculates the performance of all stocks over a given period,
    reinvesting dividends.
    """
    conn = get_db_connection()
    # Get all distinct stocks from the database
    stocks = conn.execute("SELECT DISTINCT stock FROM merged_data").fetchall()

    results = []

    for stock_row in stocks:
        symbol = stock_row['stock']
        history = conn.execute(
            "SELECT date, close, action_type, value FROM merged_data WHERE stock = ? AND date BETWEEN ? AND ? ORDER BY date",
            (symbol, from_date, to_date)
        ).fetchall()

        if not history or not history[0]['close']:
            continue

        initial_price = history[0]['close']
        shares = investment_amount / initial_price
        cash = 0

        for day in history:
            action = day['action_type'].lower() if day['action_type'] else ''
            if action == 'dividend' and day['value'] > 0:
                cash += shares * day['value']

            if cash > 0 and day['close'] > 0:
                shares += cash / day['close']
                cash = 0

        final_price = history[-1]['close']
        final_value = shares * final_price

        start_dt = datetime.strptime(history[0]['date'], '%Y-%m-%d')
        end_dt = datetime.strptime(history[-1]['date'], '%Y-%m-%d')
        years = (end_dt - start_dt).days / 365.25
        cagr = 0
        if years > 0 and investment_amount > 0:
            cagr = ((final_value / investment_amount) ** (1 / years)) - 1

        results.append({
            "stock": symbol,
            "final_value": final_value,
            "cagr": cagr * 100
        })

    conn.close()

    # Sort results by final value in descending order
    sorted_results = sorted(results, key=lambda x: x['final_value'], reverse=True)

    # Add ranks
    for i, result in enumerate(sorted_results):
        result['rank'] = i + 1

    return sorted_results

@app.route('/low-volatility-scanner', methods=['GET', 'POST'])
@login_required
def low_volatility_scanner():
    if request.method == 'POST':
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        investment_amount = float(request.form.get('investment_amount', 10000))

        if not from_date or not to_date:
            return render_template('low_volatility_scanner.html', error="Please select both a start and end date.")

        results = calculate_low_volatility_performance(investment_amount, from_date, to_date)
        return render_template('low_volatility_scanner.html', results=results)

    return render_template('low_volatility_scanner.html')

def calculate_low_volatility_performance(investment_amount, from_date, to_date):
    """
    Calculates stock performance, including CAGR and the number of negative-return years.
    """
    conn = get_db_connection()
    stocks = conn.execute("SELECT DISTINCT stock FROM merged_data").fetchall()

    results = []

    for stock_row in stocks:
        symbol = stock_row['stock']
        history = conn.execute(
            "SELECT date, close, action_type, value FROM merged_data WHERE stock = ? AND date BETWEEN ? AND ? ORDER BY date",
            (symbol, from_date, to_date)
        ).fetchall()

        if not history or not history[0]['close']:
            continue

        # --- Standard Performance Calculation ---
        initial_price = history[0]['close']
        shares = investment_amount / initial_price
        cash = 0

        for day in history:
            action = day['action_type'].lower() if day['action_type'] else ''
            if action == 'dividend' and day['value'] > 0:
                cash += shares * day['value']
            if cash > 0 and day['close'] > 0:
                shares += cash / day['close']
                cash = 0

        final_price = history[-1]['close']
        final_value = shares * final_price

        start_dt = datetime.strptime(history[0]['date'], '%Y-%m-%d')
        end_dt = datetime.strptime(history[-1]['date'], '%Y-%m-%d')
        years = (end_dt - start_dt).days / 365.25
        cagr = ((final_value / investment_amount) ** (1 / years)) - 1 if years > 0 and investment_amount > 0 else 0

        # --- Yearly Performance Calculation ---
        yearly_performance = []
        prices = {datetime.strptime(day['date'], '%Y-%m-%d').date(): day['close'] for day in history}

        start_year = start_dt.year + 1
        end_year = end_dt.year

        for year in range(start_year, end_year):
            year_start_date = datetime(year, 1, 1).date()
            year_end_date = datetime(year, 12, 31).date()

            start_price_date = min((d for d in prices if d >= year_start_date), default=None)
            end_price_date = max((d for d in prices if d <= year_end_date), default=None)

            if start_price_date and end_price_date and start_price_date < end_price_date:
                year_start_price = prices[start_price_date]
                year_end_price = prices[end_price_date]
                is_positive = year_end_price >= year_start_price
                yearly_performance.append({"year": year, "is_positive": is_positive})

        negative_years = sum(1 for p in yearly_performance if not p['is_positive'])

        results.append({
            "stock": symbol,
            "final_value": final_value,
            "cagr": cagr * 100,
            "negative_years": negative_years,
            "yearly_performance": yearly_performance
        })

    conn.close()

    # Sort by CAGR (desc) and then by negative years (asc)
    sorted_results = sorted(results, key=lambda x: (-x['cagr'], x['negative_years']))

    for i, result in enumerate(sorted_results):
        result['rank'] = i + 1

    return sorted_results

if __name__ == '__main__':
    app.run(debug=True, port=5001)