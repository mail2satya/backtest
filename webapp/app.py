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
    # Set default values for GET request
    results = None
    trigger_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    days_to_track = 5
    scan_type = 'bullish'
    stock_symbol = ''
    min_price = 0
    max_price = 10000

    if request.method == 'POST':
        trigger_date = request.form.get('trigger_date')
        days_to_track = int(request.form.get('days_to_track', 5))
        scan_type = request.form.get('scan_type', 'bullish')
        stock_symbol = request.form.get('stock_symbol', '').strip().upper()
        min_price_str = request.form.get('min_price', '0')
        max_price_str = request.form.get('max_price', '10000')

        min_price = float(min_price_str) if min_price_str else 0
        max_price = float(max_price_str) if max_price_str else 10000

        if not trigger_date:
            flash("Please select a trigger date.", "error")
        else:
            results = calculate_movement_performance(trigger_date, scan_type, days_to_track, stock_symbol, min_price, max_price)

    return render_template('movement_scanner.html',
                         results=results,
                         trigger_date=trigger_date,
                         days_to_track=days_to_track,
                         scan_type=scan_type,
                         stock_symbol=stock_symbol,
                         min_price=min_price,
                         max_price=max_price)


def calculate_movement_performance(trigger_date, scan_type, days_to_track, stock_symbol=None, min_price=0, max_price=10000):
    """
    Finds stocks based on a trigger pattern and tracks their daily intraday performance for N days.
    - Performance is calculated as the percentage change from that day's open to its close.
    - Includes an optional filter for a specific stock symbol and price range.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if scan_type == 'bullish':
        condition = "AND open > prev_day_close AND close > prev_day_close"
    elif scan_type == 'bearish':
        condition = "AND open < prev_day_close AND close < prev_day_close"
    else:
        return []

    # Build the query parameters dynamically
    params = [trigger_date, min_price, max_price]
    symbol_filter = ""
    if stock_symbol:
        symbol_filter = "AND symbol = %s"
        params.append(stock_symbol)

    query = f"""
    WITH daily_data_with_prev_close AS (
        SELECT
            symbol, date, open, close, high, low, volume,
            LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) AS prev_day_close
        FROM raw_data
        WHERE volume > 0 -- Ensure we are only considering trading days
    ),
    triggered_stocks AS (
        SELECT
            symbol,
            date AS trigger_date,
            close AS trigger_close,
            prev_day_close,
            (close - open) AS trigger_intraday_move
        FROM daily_data_with_prev_close
        WHERE date = %s
          AND open BETWEEN %s AND %s
          {symbol_filter}
          AND prev_day_close IS NOT NULL
          {condition}
    ),
    performance_days AS (
        SELECT
            ts.symbol,
            ts.trigger_date,
            ts.trigger_close,
            rd.date,
            rd.open,
            rd.close,
            rd.high,
            rd.low,
            rd.volume,
            -- Calculate the INTRA-DAY performance for each subsequent day
            CASE WHEN rd.open > 0 THEN
                ROUND(((rd.close - rd.open) / rd.open) * 100, 2)
            ELSE 0 END AS performance_pct,
            ROW_NUMBER() OVER (PARTITION BY ts.symbol ORDER BY rd.date) AS day_num
        FROM triggered_stocks ts
        JOIN raw_data rd ON ts.symbol = rd.symbol AND rd.date > ts.trigger_date
    )
    SELECT *
    FROM performance_days
    WHERE day_num <= %s
    ORDER BY symbol, date;
    """

    params.append(days_to_track)

    try:
        cursor.execute(query, tuple(params))
        all_data = cursor.fetchall()
    except psycopg2.Error as e:
        print(f"Database query failed: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

    # Group results by stock symbol
    results = {}
    for row in all_data:
        symbol = row['symbol']
        if symbol not in results:
            results[symbol] = {
                "stock": symbol,
                "trigger_date": row['trigger_date'].strftime('%Y-%m-%d'),
                "trigger_close": float(row['trigger_close']),
                "performance": []
            }

        # Determine the intraday pattern for dot coloring
        intraday_pattern = "bullish" if row['close'] > row['open'] else "bearish" if row['close'] < row['open'] else "neutral"

        results[symbol]['performance'].append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "day": row['day_num'],
            "return": row['performance_pct'],
            "intraday_pattern": intraday_pattern
        })

    return list(results.values())

def _parse_dynamic_conditions(form):
    """Helper to parse the dynamic trigger conditions and logical operators from the form."""
    conditions = {'trigger': []}
    trigger_indices = sorted([int(k.split('_')[-1]) for k in form if k.startswith('trigger_field1_')])

    for idx in trigger_indices:
        conditions['trigger'].append({
            'field1': form.get(f'trigger_field1_{idx}'),
            'day1': form.get(f'trigger_day1_{idx}'),
            'operator': form.get(f'trigger_operator_{idx}'),
            'field2': form.get(f'trigger_field2_{idx}'),
            'day2': form.get(f'trigger_day2_{idx}'),
            'logical': form.get(f'trigger_logical_{idx-1}', 'AND')
        })
    return conditions

@app.route('/pattern_analyzer', methods=['GET', 'POST'])
@login_required
def pattern_analyzer():
    results = None
    submitted_conditions = None
    from_date = request.form.get('from_date')
    to_date = request.form.get('to_date')
    stock_symbol = request.form.get('stock_symbol', None)
    days_to_track = int(request.form.get('days_to_track', 5))

    if request.method == 'POST':
        if not from_date or not to_date:
            return render_template('pattern_analyzer.html', error="Please select both a start and end date.")

        submitted_conditions = _parse_dynamic_conditions(request.form)

        if not submitted_conditions['trigger']:
            return render_template('pattern_analyzer.html', error="Please define at least one trigger condition.",
                                 submitted_conditions=submitted_conditions, days_to_track=days_to_track)

        results = analyze_pattern_performance(from_date, to_date, stock_symbol, submitted_conditions['trigger'], days_to_track)

    return render_template('pattern_analyzer.html', results=results, from_date=from_date,
                         to_date=to_date, stock_symbol=stock_symbol,
                         submitted_conditions=submitted_conditions, days_to_track=days_to_track)

def _build_sql_logic(conditions):
    """Translates a list of condition objects into a safe, combined SQL string with AND/OR logic."""
    allowed_fields = ['open', 'high', 'low', 'close']
    allowed_operators = ['>', '<', '=']

    sql_parts = []
    for i, condition in enumerate(conditions):
        field1 = condition['field1']
        day1 = int(condition['day1'])
        op = condition['operator']
        field2 = condition['field2']
        day2 = int(condition['day2'])
        logical = condition.get('logical', 'AND').upper()

        if field1 not in allowed_fields or field2 not in allowed_fields or op not in allowed_operators or logical not in ['AND', 'OR']:
            raise ValueError("Invalid condition parameters.")

        alias1 = f"t_{day1}".replace('-', 'minus_')
        alias2 = f"t_{day2}".replace('-', 'minus_')
        sql_part = f"{alias1}_{field1} {op} {alias2}_{field2}"

        if i > 0:
            sql_parts.append(logical)
        sql_parts.append(sql_part)

    return f"({ ' '.join(sql_parts) })" if sql_parts else "TRUE"

def analyze_pattern_performance(from_date, to_date, stock_symbol, trigger_conditions, days_to_track):
    """
    Analyzes the performance of stocks for N days following a user-defined trigger pattern.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    all_day_offsets = set([0])
    for condition in trigger_conditions:
        all_day_offsets.add(int(condition['day1']))
        all_day_offsets.add(int(condition['day2']))

    # Generate LAG/LEAD expressions for trigger condition validation
    trigger_window_expressions = []
    for offset in all_day_offsets:
        if offset == 0: continue
        func = 'LAG' if offset < 0 else 'LEAD'
        alias_prefix = f"t_{offset}".replace('-', 'minus_')
        for field in ['open', 'high', 'low', 'close']:
            trigger_window_expressions.append(
                f"{func}({field}, {abs(offset)}) OVER w AS {alias_prefix}_{field}"
            )

    # Generate LEAD expressions for performance tracking
    performance_window_expressions = []
    for i in range(1, days_to_track + 1):
        performance_window_expressions.append(
            f"LEAD(close, {i}) OVER w AS next_close_{i}"
        )

    try:
        trigger_sql = _build_sql_logic(trigger_conditions)
    except (ValueError, KeyError) as e:
        return {"error": f"Failed to build SQL logic: {e}"}

    symbol_filter = ""
    params = [from_date, to_date]
    if stock_symbol and stock_symbol.strip():
        symbol_filter = "AND symbol = %s"
        params.append(stock_symbol.upper().strip())

    query = f"""
    WITH trading_days AS (
        SELECT *
        FROM raw_data
        WHERE volume > 0
    ),
    daily_data_with_windows AS (
        SELECT
            date, symbol, close,
            open AS t_0_open, high AS t_0_high, low AS t_0_low, close AS t_0_close,
            {", ".join(trigger_window_expressions)},
            {", ".join(performance_window_expressions)}
        FROM trading_days
        WINDOW w AS (PARTITION BY symbol ORDER BY date)
    ),
    triggered_events AS (
        SELECT *
        FROM daily_data_with_windows
        WHERE date BETWEEN %s AND %s
          {symbol_filter}
          AND {trigger_sql}
    )
    SELECT * FROM triggered_events ORDER BY symbol, date;
    """

    try:
        cursor.execute(query, tuple(params))
        all_data = cursor.fetchall()
    except psycopg2.Error as e:
        print("Database Query Error:", e)
        return [] # Return empty list on error
    finally:
        cursor.close()
        conn.close()

    # Process results into a list of dictionaries for the template
    results = []
    for row in all_data:
        performance_pcts = []
        trigger_close = float(row['close'])
        for i in range(1, days_to_track + 1):
            next_close = row.get(f'next_close_{i}')
            if next_close is not None and trigger_close > 0:
                perf = ((float(next_close) / trigger_close) - 1) * 100
                performance_pcts.append(perf)
            else:
                performance_pcts.append(None) # Or 0, or some other indicator for missing data

        results.append({
            "symbol": row['symbol'],
            "trigger_date": row['date'],
            "performance": performance_pcts
        })

    return results

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5001)
