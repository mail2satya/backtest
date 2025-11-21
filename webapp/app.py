from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
import psycopg2
import psycopg2.extras
import os
from datetime import datetime, timedelta
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
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return User(id=user_data['id'], username=user_data['username'], password_hash=user_data['password_hash'])
    return None

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user_data = cursor.fetchone()
        conn.close()
        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(id=user_data['id'], username=user_data['username'], password_hash=user_data['password_hash'])
            login_user(user)
            return redirect(url_for('dashboard'))
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
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT id FROM users WHERE username = %s', (username,))
        user_exists = cursor.fetchone()
        if user_exists:
            flash('Username already exists.')
            conn.close()
            return redirect(url_for('register'))

        password_hash = generate_password_hash(password, method='pbkdf2:sha256')
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (%s, %s)', (username, password_hash))
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
DATABASE_URL = os.environ.get('DATABASE_URL', "postgresql://postgres:admin@localhost:5432/postgres")

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    """Initializes the database and creates a test user if not present."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Check if the users table exists
    cursor.execute("SELECT to_regclass('public.users');")
    if cursor.fetchone()[0] is None:
        # Create the users table if it doesn't exist
        cursor.execute('''
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            );
        ''')
        conn.commit()

    # Check if the test user exists
    cursor.execute('SELECT id FROM users WHERE username = %s', ('testuser',))
    user = cursor.fetchone()
    if not user:
        password_hash = generate_password_hash('password', method='pbkdf2:sha256')
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (%s, %s)', ('testuser', password_hash))
        conn.commit()
    cursor.close()
    conn.close()

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/calculator')
@login_required
def calculator():
    return render_template('calculator.html')

@app.route('/api/search')
def search():
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify([])

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # Use DISTINCT to avoid duplicate symbols from the raw_data table
    cursor.execute(
        "SELECT DISTINCT symbol FROM raw_data WHERE symbol LIKE %s ORDER BY symbol",
        (query.upper() + '%',)
    )
    symbols = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify([row['symbol'] for row in symbols])

@app.route('/api/ohlc')
def ohlc():
    symbol = request.args.get('symbol', '')
    if not symbol:
        return jsonify({"error": "Symbol parameter is required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # Order by date to ensure data is chronological for charting
    # Fetch all relevant columns, including corporate actions
    cursor.execute(
        "SELECT date, open, high, low, close, dividends, stock_splits FROM raw_data WHERE symbol = %s ORDER BY date",
        (symbol.upper(),)
    )
    ohlc_data = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convert rows to a list of dictionaries and handle corporate actions
    data = []
    for row in ohlc_data:
        row_dict = dict(row)
        # Adapt dividends and stock_splits to the expected format
        row_dict['action_type'] = None
        row_dict['value'] = None
        if row_dict['dividends'] is not None and row_dict['dividends'] > 0:
            row_dict['action_type'] = 'dividend'
            row_dict['value'] = row_dict['dividends']
        elif row_dict['stock_splits'] is not None and row_dict['stock_splits'] > 0:
            row_dict['action_type'] = 'split'
            row_dict['value'] = row_dict['stock_splits']
        data.append(row_dict)

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
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # Fetch all data for the stock within the specified date range
    cursor.execute(
        "SELECT date, open, high, low, close, dividends, stock_splits FROM raw_data WHERE symbol = %s AND date BETWEEN %s AND %s ORDER BY date",
        (symbol.upper(), from_date, to_date)
    )
    history = cursor.fetchall()
    cursor.close()
    conn.close()

    if not history:
        return jsonify({"error": "No data found for the given stock and date"}), 404

    # Initial investment
    initial_price = float(history[0]['close'])
    if initial_price == 0:
        return jsonify({"error": "Initial price is zero, cannot calculate investment"}), 400

    shares = investment_amount / initial_price
    cash = 0  # To hold dividend payouts before reinvesting
    total_dividends_received = 0

    # Process historical data
    for i, day in enumerate(history):
        # Data source is pre-adjusted for splits/bonuses, so we only handle dividends.
        if day['dividends'] is not None and float(day['dividends']) > 0:
            # Calculate cash dividend and add to reinvestment pool and total tracker
            dividend_cash = shares * float(day['dividends'])
            total_dividends_received += dividend_cash
            cash += dividend_cash

        # Reinvest cash from dividends on the same day
        if cash > 0:
            reinvestment_price = float(day['close'])
            if reinvestment_price > 0:
                additional_shares = cash / reinvestment_price
                shares += additional_shares
                cash = 0  # Reset cash after reinvesting

    # Calculate final value
    final_price = float(history[-1]['close'])
    final_value = shares * final_price

    # Calculate CAGR
    start_dt = history[0]['date']
    end_dt = history[-1]['date']
    years = (end_dt - start_dt).days / 365.25
    cagr = 0
    if years > 0 and investment_amount > 0:
        cagr = ((final_value / investment_amount) ** (1 / years)) - 1

    return jsonify({
        "initial_investment": investment_amount,
        "final_value": round(final_value, 2),
        "start_date": history[0]['date'].strftime('%Y-%m-%d'),
        "end_date": history[-1]['date'].strftime('%Y-%m-%d'),
        "start_price": float(history[0]['close']),
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
        stock_symbol = request.form.get('stock', None)
        sort_by = request.form.get('sort_by', 'cagr')
        sort_order = request.form.get('sort_order', 'desc')

        if not from_date or not to_date:
            return render_template('scanner.html', error="Please select both a start and end date.")

        results = calculate_scanner_performance(investment_amount, from_date, to_date, stock_symbol, sort_by, sort_order)
        return render_template('scanner.html', results=results, stock=stock_symbol, sort_by=sort_by, sort_order=sort_order)

    return render_template('scanner.html')

def calculate_scanner_performance(investment_amount, from_date, to_date, stock_symbol=None, sort_by='cagr', sort_order='desc'):
    """
    Calculates stock performance, including CAGR and the number of negative-return years.
    If stock_symbol is provided, calculates for only that stock.
    Allows sorting by specified column.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if stock_symbol:
        cursor.execute("SELECT DISTINCT symbol FROM raw_data WHERE symbol = %s", (stock_symbol.upper(),))
    else:
        cursor.execute("SELECT DISTINCT symbol FROM raw_data")

    stocks = cursor.fetchall()
    results = []

    for stock_row in stocks:
        symbol = stock_row['symbol']
        cursor.execute(
            "SELECT date, close, dividends FROM raw_data WHERE symbol = %s AND date BETWEEN %s AND %s ORDER BY date",
            (symbol, from_date, to_date)
        )
        history = cursor.fetchall()

        if not history or not history[0]['close']:
            continue

        # --- Standard Performance Calculation ---
        initial_price = float(history[0]['close'])
        shares = investment_amount / initial_price
        cash = 0
        total_dividends = 0

        for day in history:
            if day['dividends'] is not None and float(day['dividends']) > 0:
                dividend_cash = shares * float(day['dividends'])
                total_dividends += dividend_cash
                cash += dividend_cash
            if cash > 0 and float(day['close']) > 0:
                shares += cash / float(day['close'])
                cash = 0

        final_price = float(history[-1]['close'])
        final_value = shares * final_price

        start_dt = history[0]['date']
        end_dt = history[-1]['date']
        years = (end_dt - start_dt).days / 365.25
        cagr = ((final_value / investment_amount) ** (1 / years)) - 1 if years > 0 and investment_amount > 0 else 0

        # --- Yearly Performance Calculation ---
        yearly_performance = []

        # Group data by year
        yearly_data = {}
        for day in history:
            year = day['date'].year
            if year not in yearly_data:
                yearly_data[year] = []
            yearly_data[year].append(day)

        for year, year_days in sorted(yearly_data.items()):
            if len(year_days) < 2:
                continue

            year_start_price = float(year_days[0]['close'])
            year_end_price = float(year_days[-1]['close'])

            if year_start_price > 0:
                performance = ((year_end_price / year_start_price) - 1) * 100

                # Check if the last day of the period is not the end of the calendar year
                is_ytd = (year == end_dt.year) and (end_dt.month != 12 or end_dt.day != 31)

                yearly_performance.append({
                    "year": f"{year}{' YTD' if is_ytd else ''}",
                    "return": performance
                })

        results.append({
            "stock": symbol,
            "final_value": final_value,
            "cagr": cagr * 100,
            "yearly_performance": yearly_performance,
            "total_dividends": total_dividends
        })

    cursor.close()
    conn.close()

    # Add negative years for volatility sorting
    for res in results:
        res['negative_years'] = sum(1 for y in res['yearly_performance'] if y['return'] < 0)

    # Dynamic sorting
    is_reverse = (sort_order == 'desc')

    # Define a primary sort key, with a secondary sort for volatility
    if sort_by in ['cagr', 'final_value', 'total_dividends']:
        sorted_results = sorted(results, key=lambda x: (x.get(sort_by, 0), x['negative_years']), reverse=is_reverse)
    else: # Default or other cases
        sorted_results = sorted(results, key=lambda x: (-x['cagr'], x['negative_years']))

    for i, result in enumerate(sorted_results):
        result['rank'] = i + 1

    return sorted_results

@app.route('/movement_scanner', methods=['GET', 'POST'])
@login_required
def movement_scanner():
    if request.method == 'POST':
        trigger_date = request.form.get('trigger_date')
        days_to_track = int(request.form.get('days_to_track', 5))
        scan_type = request.form.get('scan_type', 'bullish')
        min_price = float(request.form.get('min_price', 0))
        max_price = float(request.form.get('max_price', 10000))

        if not trigger_date:
            return render_template('movement_scanner.html', error="Please select a trigger date.")

        results = calculate_movement_performance(trigger_date, scan_type, days_to_track, min_price, max_price)

        return render_template('movement_scanner.html', results=results, trigger_date=trigger_date, 
                             days_to_track=days_to_track, scan_type=scan_type, min_price=min_price, max_price=max_price)

    return render_template('movement_scanner.html')

def calculate_movement_performance(trigger_date, scan_type, days_to_track, min_price=0, max_price=10000):
    """
    Finds stocks based on trigger pattern (open and close above/below previous day's CLOSE price)
    and tracks their intraday pattern (bullish/bearish) for the next N days.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Trigger logic based on previous day's CLOSE price
    if scan_type == 'bullish':
        condition = """
            AND current.open > previous.prev_close
            AND current.close > previous.prev_close
        """
    elif scan_type == 'bearish':
        condition = """
            AND current.open < previous.prev_close
            AND current.close < previous.prev_close
        """
    else:
        return []

    query = """
    WITH previous_data AS (
        SELECT 
            symbol, date, open as prev_open, close as prev_close
        FROM raw_data
    ),
    current_data AS (
        SELECT 
            symbol, date, open, close, high, low, volume
        FROM raw_data
    ),
    triggered_stocks AS (
        SELECT 
            current.symbol,
            current.date AS trigger_date,
            current.open AS trigger_open,
            current.close AS trigger_close,
            previous.prev_close AS prev_day_close,
            (current.close - current.open) AS trigger_intraday_move
        FROM current_data current
        JOIN previous_data previous ON current.symbol = previous.symbol 
            AND current.date = previous.date + INTERVAL '1 day'
        WHERE current.date = %s
            AND current.open BETWEEN %s AND %s
            {condition}
    ),
    performance_days AS (
        SELECT
            ts.symbol,
            ts.trigger_date,
            ts.trigger_close,
            ts.prev_day_close,
            ts.trigger_intraday_move,
            rd.date,
            rd.open,
            rd.close,
            rd.high,
            rd.low,
            rd.volume,
            (rd.close - rd.open) AS intraday_move,
            ROW_NUMBER() OVER (PARTITION BY ts.symbol ORDER BY rd.date) AS day_num
        FROM triggered_stocks ts
        JOIN raw_data rd ON ts.symbol = rd.symbol
        WHERE rd.date > ts.trigger_date
    )
    SELECT
        symbol,
        trigger_date,
        trigger_close,
        prev_day_close,
        trigger_intraday_move,
        date,
        open,
        close,
        high,
        low,
        volume,
        intraday_move,
        ROUND((close - trigger_close) / trigger_close * 100, 2) AS performance_pct,
        day_num
    FROM performance_days
    WHERE day_num <= %s
    ORDER BY symbol, date;
    """.format(condition=condition)

    cursor.execute(query, (trigger_date, min_price, max_price, days_to_track))
    all_data = cursor.fetchall()
    cursor.close()
    conn.close()

    # Process the data for template consumption
    results = {}
    for row in all_data:
        symbol = row['symbol']
        if symbol not in results:
            results[symbol] = {
                "stock": symbol,
                "trigger_date": row['trigger_date'].strftime('%Y-%m-%d'),
                "trigger_close": float(row['trigger_close']),
                "prev_day_close": float(row['prev_day_close']),
                "trigger_intraday_move": float(row['trigger_intraday_move']),
                "performance": []
            }

        # Calculate performance from trigger date and determine intraday pattern
        days_since_trigger = row['day_num']
        current_price = float(row['close']) if row['close'] is not None else 0
        trigger_price = results[symbol]['trigger_close']
        
        performance_pct = ((current_price / trigger_price) - 1) * 100 if trigger_price > 0 else 0
        intraday_move = float(row['intraday_move'])
        intraday_pattern = "bullish" if intraday_move > 0 else "bearish" if intraday_move < 0 else "neutral"

        results[symbol]['performance'].append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "day": days_since_trigger,
            "open": float(row['open']),
            "close": current_price,
            "high": float(row['high']),
            "low": float(row['low']),
            "volume": row['volume'],
            "return": round(performance_pct, 2),
            "intraday_move": intraday_move,
            "intraday_pattern": intraday_pattern
        })

    return list(results.values())

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5001)
