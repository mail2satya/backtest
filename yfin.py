import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime

def fetch_and_store_data(stock_symbols, start_date='2000-01-01', db_name='fno_data.sqlite'):
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
        Adj_Close REAL
    )
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS corporate_actions (
        Date TEXT,
        Stock TEXT,
        Action_Type TEXT,
        Value REAL
    )
    ''')
    conn.commit()

    for symbol in stock_symbols:
        yf_symbol = symbol if symbol.endswith('.NS') else symbol + '.NS'
        print(f"Processing {yf_symbol}...")

        try:
            df = yf.download(yf_symbol, start=start_date,
                             end=datetime.today().strftime('%Y-%m-%d'),
                             progress=False, auto_adjust=False)
            if df.empty:
                print(f"No OHLC data for {yf_symbol}")
                continue

            df.reset_index(inplace=True)
            adj_close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']

            ohlc_df = pd.DataFrame({
                'Date': df['Date'].dt.strftime('%Y-%m-%d').values.flatten(),
                'Stock': [symbol]*len(df),
                'Open': df['Open'].values.flatten().round(1),
                'High': df['High'].values.flatten().round(1),
                'Low': df['Low'].values.flatten().round(1),
                'Close': df['Close'].values.flatten().round(1),
                'Volume': df['Volume'].values.flatten(),
                'Adj_Close': adj_close.values.flatten().round(2)
            })

            ohlc_df.to_sql('ohlc_data', conn, if_exists='append', index=False)
            print(f"Saved OHLC for {symbol}")

            ticker = yf.Ticker(yf_symbol)

            # Handle Dividends
            dividends = ticker.dividends.reset_index() if ticker.dividends is not None else pd.DataFrame()
            if not dividends.empty:
                dividends.rename(columns={'Dividends': 'Value'}, inplace=True)
                dividends['Action_Type'] = 'Dividend'
                dividends['Stock'] = yf_symbol
                dividends = dividends[['Date', 'Stock', 'Action_Type', 'Value']]
                dividends.to_sql('corporate_actions', conn, if_exists='append', index=False)
                print(f"Saved {len(dividends)} dividends for {symbol}")
            else:
                print(f"No dividends for {symbol}")

            # Handle Splits, dynamic column detection
            try:
                splits = ticker.splits.reset_index()
                if not splits.empty:
                    split_cols = [col for col in splits.columns if col != 'Date']
                    if len(split_cols) == 1:
                        splits.rename(columns={split_cols[0]: 'Value'}, inplace=True)
                    elif 'Splits' in splits.columns:
                        splits.rename(columns={'Splits': 'Value'}, inplace=True)
                    else:
                        raise KeyError(f"Expected split ratio column not found in splits cols: {splits.columns.tolist()}")

                    splits['Action_Type'] = 'Split'
                    splits['Stock'] = yf_symbol
                    splits = splits[['Date', 'Stock', 'Action_Type', 'Value']]
                    splits.to_sql('corporate_actions', conn, if_exists='append', index=False)
                    print(f"Saved {len(splits)} splits for {symbol}")
                else:
                    print(f"No splits for {symbol}")
            except Exception as ex:
                print(f"Error fetching splits for {symbol}: {ex}")

        except Exception as e:
            print(f"Error processing {yf_symbol}: {e}")

    conn.commit()
    conn.close()
    print("All data fetched and stored.")

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
]  # Add your full stock list here
    fetch_and_store_data(symbols, start_date='2000-01-01')
