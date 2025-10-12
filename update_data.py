import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

def get_last_date(db_name='fno_data.sqlite'):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlc_data'")
        if cursor.fetchone() is None:
            conn.close()
            return datetime.strptime('2000-01-01', '%Y-%m-%d').date()

        cursor.execute("SELECT MAX(Date) FROM ohlc_data")
        last_date_str = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        last_date_str = None
    finally:
        if conn:
            conn.close()

    if last_date_str:
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
        return last_date + timedelta(days=1)
    else:
        return datetime.strptime('2000-01-01', '%Y-%m-%d').date()

def fetch_and_store_data(stock_symbols, start_date, db_name='fno_data.sqlite'):
    conn = sqlite3.connect(db_name)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS ohlc_data (
        Date TEXT,
        Stock TEXT,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Volume INTEGER,
        Adj_Close REAL,
        UNIQUE(Date, Stock)
    )
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS corporate_actions (
        Date TEXT,
        Stock TEXT,
        Action_Type TEXT,
        Value REAL,
        UNIQUE(Date, Stock, Action_Type, Value)
    )
    ''')
    conn.commit()

    for symbol in stock_symbols:
        yf_symbol = symbol if symbol.endswith('.NS') else symbol + '.NS'
        print(f"Processing {yf_symbol} from {start_date.strftime('%Y-%m-%d')}...")

        try:
            df = yf.download(yf_symbol, start=start_date,
                             end=datetime.today().strftime('%Y-%m-%d'),
                             progress=False, auto_adjust=False)
            if df.empty:
                print(f"No new OHLC data for {yf_symbol}")
                continue

            df.reset_index(inplace=True)
            adj_close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']

            ohlc_df = pd.DataFrame({
                'Date': df['Date'].dt.strftime('%Y-%m-%d'),
                'Stock': symbol,
                'Open': df['Open'].values.flatten().round(1),
                'High': df['High'].values.flatten().round(1),
                'Low': df['Low'].values.flatten().round(1),
                'Close': df['Close'].values.flatten().round(1),
                'Volume': df['Volume'].values.flatten(),
                'Adj_Close': adj_close.values.flatten().round(2)
            })

            ohlc_df.to_sql('ohlc_data_temp', conn, if_exists='replace', index=False)
            conn.execute("""
                INSERT OR IGNORE INTO ohlc_data (Date, Stock, Open, High, Low, Close, Volume, Adj_Close)
                SELECT Date, Stock, Open, High, Low, Close, Volume, Adj_Close FROM ohlc_data_temp
            """)
            conn.commit()
            print(f"Saved/updated OHLC for {symbol}")

            ticker = yf.Ticker(yf_symbol)

            divs = ticker.dividends
            if not divs.empty:
                dividends = divs.reset_index()
                dividends = dividends[dividends['Date'].dt.date >= start_date]
                if not dividends.empty:
                    dividends.rename(columns={'Dividends': 'Value'}, inplace=True)
                    dividends['Action_Type'] = 'Dividend'
                    dividends['Stock'] = symbol
                    dividends['Date'] = dividends['Date'].dt.strftime('%Y-%m-%d')
                    dividends = dividends[['Date', 'Stock', 'Action_Type', 'Value']]
                    dividends.to_sql('corporate_actions_temp', conn, if_exists='replace', index=False)
                    conn.execute("""
                        INSERT OR IGNORE INTO corporate_actions (Date, Stock, Action_Type, Value)
                        SELECT Date, Stock, Action_Type, Value FROM corporate_actions_temp
                    """)
                    conn.commit()
                    print(f"Saved {len(dividends)} new dividends for {symbol}")

            ticker_splits = ticker.splits
            if not ticker_splits.empty:
                splits = ticker_splits.reset_index()
                splits = splits[splits['Date'].dt.date >= start_date]
                if not splits.empty:
                    if 'Stock Splits' in splits.columns:
                        splits.rename(columns={'Stock Splits': 'Value'}, inplace=True)
                    else: # Fallback for different column names
                        splits.rename(columns={splits.columns[1]: 'Value'}, inplace=True)
                    splits['Action_Type'] = 'Split'
                    splits['Stock'] = symbol
                    splits['Date'] = splits['Date'].dt.strftime('%Y-%m-%d')
                    splits = splits[['Date', 'Stock', 'Action_Type', 'Value']]
                    splits.to_sql('corporate_actions_temp', conn, if_exists='replace', index=False)
                    conn.execute("""
                        INSERT OR IGNORE INTO corporate_actions (Date, Stock, Action_Type, Value)
                        SELECT Date, Stock, Action_Type, Value FROM corporate_actions_temp
                    """)
                    conn.commit()
                    print(f"Saved {len(splits)} new splits for {symbol}")

        except Exception as e:
            print(f"Error processing {yf_symbol}: {e}")

    conn.close()
    print("All data fetched and stored.")

def rebuild_merged_table(source_db='fno_data.sqlite', target_db='merge_data.sqlite'):
    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)
    target_conn.execute(f"ATTACH DATABASE '{source_db}' AS source_db")
    cursor = target_conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS merged_data;')
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
          Stock,
          Action_Type,
          Value
        FROM source_db.corporate_actions
      ) c
    ON o.Date = c.Date AND o.Stock = c.Stock;
    '''
    cursor.execute(create_query)
    target_conn.commit()
    target_conn.execute('DETACH DATABASE source_db')
    cursor.close()
    target_conn.close()
    source_conn.close()
    print(f"Merged table created successfully in database '{target_db}'.")

if __name__ == '__main__':
    symbols = [
        'ABCAPITAL','ADANIENSOL','INDHOTEL','INDIANB','INOXWIND','JINDALSTEL',
        'MPHASIS','NESTLEIND','NTPC','PRESTIGE','RBLBANK','SHREECEM','TATAELXSI',
        'TATASTEEL','TCS','AMBUJACEM','APLAPOLLO','AUROPHARMA','BSE','CDSL','CONCOR',
        'DALBHARAT','EICHERMOT','INDIGO','IRCTC','JIOFIN','KAYNES','KEI','MFSL','NYKAA',
        'PAGEIND','PERSISTENT','PNBHOUSING','SBILIFE','TIINDIA','TITAN','TRENT','UNIONBANK',
        'YESBANK','ABB','BDL','BEL','BHEL','CGPOWER','GMRAIRPORT','HAL','HCLTECH','JUBLFOOD',
        'OBEROIRLTY','ONGC','PNB','POWERGRID','SONACOMS','AXISBANK','ETERNAL','HINDPETRO',
        'IDFCFIRSTB','LODHA','TVSMOTOR','BAJAJ-AUTO','BOSCHLTD','CIPLA','DLF','DRREDDY',
        'EXIDEIND','GODREJPROP','KOTAKBANK','M&M','NATIONALUM','NHPC','PAYTM','PGEL','TECHM',
        'ULTRACEMCO','BANKINDIA','BIOCON','COALINDIA','COLPAL','FEDERALBNK','GODREJCP',
        'KPITTECH','LT','NMDC','OIL','PHOENIXLTD','PIIND','TATACONSUM','WIPRO','BHARATFORG',
        'CROMPTON','INDUSTOWER','PATANJALI','POWERINDIA','SOLARINDS','TATAMOTORS','TATAPOWER',
        'TITAGARH','VOLTAS','APOLLOHOSP','CYIENT','DELHIVERY','DIXON','FORTIS','GLENMARK',
        'HINDUNILVR','HUDCO','ICICIPRULI','IDEA','NBCC','POLYCAB','ALKEM','ASHOKLEY','ASTRAL',
        'BANKBARODA','BHARTIARTL','BPCL','CHOLAFIN','CUMMINSIND','DIVISLAB','IEX','INDUSINDBK',
        'JSWENERGY','KALYANKJIL','LAURUSLABS','LTF','MANAPPURAM','MARICO','MOTHERSON','RELIANCE',
        'SRF','ZYDUSLIFE','ADANIGREEN','ADANIPORTS','BAJAJFINSV','BLUESTARCO','DABUR','HDFCAMC',
        'IOC','ITC','MAXHEALTH','OFSS','PIDILITIND','SBIN','360ONE','ADANIENT','IGL','JSWSTEEL',
        'LICI','POLICYBZR','SAIL','SUZLON','UPL','VEDL','ASIANPAINT','BRITANNIA','GRASIM','HDFCLIFE',
        'HFCL','ICICIBANK','ICICIGI','INFY','KFINTECH','MARUTI','MUTHOOTFIN','PFC','SAMMAANCAP',
        'SUNPHARMA','SUPREMEIND','SYNGENE','TORNTPOWER','UNOMINDA','COFORGE','GAIL','HAVELLS',
        'HDFCBANK','HEROMOTOCO','MANKIND','MAZDOCK','MCX','NUVAMA','PETRONET','PPLPHARMA',
        'SIEMENS','UNITDSPR','AMBER','BAJFINANCE','CANBK','DMART','HINDZINC','IREDA','LICHSGFIN',
        'LTIM','LUPIN','RVNL','SHRIRAMFIN','ANGELONE','AUBANK','BANDHANBNK','CAMS','HINDALCO',
        'IIFL','IRFC','NAUKRI','NCC','RECLTD','SBICARD','TATATECH','TORNTPHARM','VBL'
    ]

    start_date = get_last_date()
    print(f"Starting data fetch from: {start_date.strftime('%Y-%m-%d')}")
    fetch_and_store_data(symbols, start_date=start_date)

    print("Rebuilding merged data table...")
    rebuild_merged_table()
    print("Data update process finished.")