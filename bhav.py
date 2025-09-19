# save as nse_backfill_store.py
import requests, zipfile, io, sqlite3, time, random, os
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser as dateparser
#from tqdm import tqdm

DB_PATH = "nse_options.db"
TABLE = "options_bhavcopy"
ERROR_TABLE = "options_bhavcopy_errors"

HEADERS_BASE = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

URL_TEMPLATE = (
    "https://www.nseindia.com/api/reports?archives="
    + '[{{"name":"F&O - UDiFF Common Bhavcopy Final (zip)","type":"archives","category":"derivatives","section":"equity"}}]'
    + "&date={date}&type=equity&mode=single"
)

# DB helpers
def init_db(conn):
    cur = conn.cursor()
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        trade_date TEXT NOT NULL,
        symbol TEXT NOT NULL,
        expiry TEXT,
        strike REAL,
        option_type TEXT,
        open REAL, high REAL, low REAL, close REAL,
        volume INTEGER, oi INTEGER,
        PRIMARY KEY (trade_date, symbol, expiry, strike, option_type)
    )""")
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {ERROR_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT,
        raw_row TEXT,
        error TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()

def upsert_rows(conn, df):
    # df must have columns matching table names
    cur = conn.cursor()
    inserted = 0
    for _, r in df.iterrows():
        try:
            cur.execute(f"""
            INSERT INTO {TABLE} (trade_date, symbol, expiry, strike, option_type, open, high, low, close, volume, oi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_date, symbol, expiry, strike, option_type) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume, oi=excluded.oi
            """, (
                r.trade_date, r.symbol, r.expiry, r.strike, r.option_type,
                r.open, r.high, r.low, r.close, r.volume, r.oi
            ))
            inserted += 1
        except Exception as e:
            cur.execute(f"INSERT INTO {ERROR_TABLE} (trade_date, raw_row, error) VALUES (?, ?, ?)",
                        (r.trade_date, str(dict(r)), str(e)))
    conn.commit()
    return inserted

# parsing helper - adapt depending on CSV columns
def parse_bhavcopy_csv_bytes(bytes_data, trade_date):
    # Try to read all CSV files in a zip
    z = zipfile.ZipFile(io.BytesIO(bytes_data))
    frames = []
    for name in z.namelist():
        # Accept .csv or .txt files
        if not name.lower().endswith(('.csv', '.txt')):
            continue
        with z.open(name) as f:
            try:
                df = pd.read_csv(f, low_memory=False)
            except Exception:
                continue
            # normalization: try to find required columns
            # Typical columns vary; attempt flexible mapping
            cols = {c.lower(): c for c in df.columns}
            def g(candidates):
                for c in candidates:
                    if c in cols:
                        return cols[c]
                return None

            symbol_col = g(['instrumentname','symbol','underlying'])
            expiry_col = g(['expiry','expirydate'])
            strike_col = g(['strikeprice','strike','strike_price'])
            option_col = g(['optiontype','optiontype','option','type'])
            ohlc_map = { 'open': g(['open','open_price']),
                         'high': g(['high','high_price']),
                         'low': g(['low','low_price']),
                         'close': g(['close','close_price','ltp','last_price']) }
            vol_col = g(['tradedqty','volume','qty'])
            oi_col = g(['openinterest','oi'])

            # if instrument table is combined equity & derivatives, attempt policy: keep rows where option_col present or strike present
            # Build normalized frame
            norm = []
            for _, row in df.iterrows():
                try:
                    symbol = row[symbol_col] if symbol_col else None
                    expiry = row[expiry_col] if expiry_col else None
                    strike = row[strike_col] if strike_col else None
                    option_type = row[option_col] if option_col else None
                    # Normalize values
                    if isinstance(expiry, str):
                        expiry = expiry.strip()
                    if isinstance(strike, str):
                        strike = strike.replace(',', '')
                    # ensure option_type is CE/PE or derive from columns
                    if pd.isna(symbol):
                        continue
                    # Only keep rows that look like options (strike numeric and option_type not null) OR instrument type hints
                    try:
                        strike_val = float(strike) if strike not in (None, '') and not pd.isna(strike) else None
                    except:
                        strike_val = None
                    option_type_val = None
                    if isinstance(option_type, str):
                        option_type_val = option_type.strip().upper()[:2]
                        if option_type_val not in ("CE","PE"):
                            option_type_val = None

                    o = {k: (row[v] if v and v in df.columns else None) for k,v in ohlc_map.items()}
                    vol = row[vol_col] if vol_col and vol_col in df.columns else None
                    oi = row[oi_col] if oi_col and oi_col in df.columns else None

                    norm.append({
                        "trade_date": trade_date,
                        "symbol": str(symbol).strip(),
                        "expiry": expiry,
                        "strike": strike_val,
                        "option_type": option_type_val,
                        "open": try_float(o.get('open')),
                        "high": try_float(o.get('high')),
                        "low": try_float(o.get('low')),
                        "close": try_float(o.get('close')),
                        "volume": try_int(vol),
                        "oi": try_int(oi)
                    })
                except Exception as e:
                    # skip row but could be logged
                    continue
            if norm:
                frames.append(pd.DataFrame(norm))
    if frames:
        out = pd.concat(frames, ignore_index=True)
        # drop rows without strike or option_type (option rows must have strike and option_type ideally)
        out = out.dropna(subset=['strike','option_type','symbol'])
        # Optionally coerce expiry to standardized format
        out['expiry'] = out['expiry'].apply(lambda x: normalize_expiry(x))
        return out
    return pd.DataFrame()

def try_float(x):
    try:
        if pd.isna(x): return None
        return float(str(x).replace(',',''))
    except:
        return None

def try_int(x):
    try:
        if pd.isna(x): return None
        return int(float(str(x).replace(',','')))
    except:
        return None

def normalize_expiry(x):
    if pd.isna(x) or x is None: return None
    try:
        d = dateparser.parse(str(x))
        return d.strftime("%d-%b-%Y")
    except:
        return str(x).strip()

# HTTP fetch with backoff
def fetch_with_backoff(url, session, max_retries=6):
    for attempt in range(1, max_retries+1):
        headers = HEADERS_BASE.copy()
        # add slight UA rotation
        headers['User-Agent'] = random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64)"
        ])
        try:
            resp = session.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.content
            # if 429 or 5xx -> backoff
            if resp.status_code in (429, 502, 503, 504, 500):
                sleep = (2 ** attempt) + random.random()
                print(f"HTTP {resp.status_code} — sleeping {sleep:.1f}s and retrying (attempt {attempt})")
                time.sleep(sleep)
                continue
            else:
                print(f"HTTP {resp.status_code} — giving up for url {url}")
                return None
        except Exception as e:
            sleep = (2 ** attempt) + random.random()
            print(f"Fetch error: {e} — sleeping {sleep:.1f}s")
            time.sleep(sleep)
    return None

def date_range(start_dt, end_dt):
    cur = start_dt
    while cur <= end_dt:
        yield cur
        cur += timedelta(days=1)

def run_backfill(start_date_str="01-Jan-2024", end_date_str="18-Sep-2025"):
    s = datetime.strptime(start_date_str, "%d-%b-%Y")
    e = datetime.strptime(end_date_str, "%d-%b-%Y")
    session = requests.Session()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    for dt in list(date_range(s,e)):
        date_text = dt.strftime("%d-%b-%Y")
        url = URL_TEMPLATE.format(date=date_text)
        print("Fetching", date_text, url)
        content = fetch_with_backoff(url, session)
        if not content:
            print("Failed to fetch", date_text)
            continue
        df = parse_bhavcopy_csv_bytes(content, date_text)
        if df is None or df.empty:
            print("No option rows parsed for", date_text)
            continue
        inserted = upsert_rows(conn, df)
        print(f"{date_text}: parsed {len(df)} rows, inserted/upserted {inserted}")
        # polite pause
        time.sleep(1.2 + random.random()*0.8)
    conn.close()

def run_incremental():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.cursor()
    cur.execute(f"SELECT MAX(trade_date) FROM {TABLE}")
    r = cur.fetchone()
    last = r[0] if r else None
    if last:
        next_dt = datetime.strptime(last, "%d-%b-%Y") + timedelta(days=1)
    else:
        next_dt = datetime.strptime("01-Jan-2024", "%d-%b-%Y")
    today = datetime.strptime("18-Sep-2025", "%d-%b-%Y")  # change if you want open-ended; here user gave end
    if next_dt > today:
        print("No new date to fetch. Latest in DB:", last)
        conn.close()
        return
    date_text = next_dt.strftime("%d-%b-%Y")
    print("Fetching next date:", date_text)
    session = requests.Session()
    url = URL_TEMPLATE.format(date=date_text)
    content = fetch_with_backoff(url, session)
    if not content:
        print("Failed to fetch", date_text)
        conn.close()
        return
    df = parse_bhavcopy_csv_bytes(content, date_text)
    if df is None or df.empty:
        print("No option rows parsed for", date_text)
        conn.close()
        return
    inserted = upsert_rows(conn, df)
    print(f"{date_text}: parsed {len(df)} rows, inserted/upserted {inserted}")
    conn.close()

if __name__ == "__main__":
    # example usage: backfill or incremental
    import sys
    mode = sys.argv[1] if len(sys.argv)>1 else "backfill"
    if mode == "backfill":
        run_backfill("01-Jan-2024","18-Sep-2025")
    elif mode=="incremental":
        run_incremental()
    else:
        print("Usage: python nse_backfill_store.py [backfill|incremental]")
# save as nse_backfill_store.py
import requests, zipfile, io, sqlite3, time, random, os
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser as dateparser
#from tqdm import tqdm

DB_PATH = "nse_options.db"
TABLE = "options_bhavcopy"
ERROR_TABLE = "options_bhavcopy_errors"

HEADERS_BASE = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
}

URL_TEMPLATE = (
    "https://www.nseindia.com/api/reports?archives="
    + '[{"name":"F&O - UDiFF Common Bhavcopy Final (zip)","type":"archives","category":"derivatives","section":"equity"}]'
    + "&date={date}&type=equity&mode=single"
)

# DB helpers
def init_db(conn):
    cur = conn.cursor()
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        trade_date TEXT NOT NULL,
        symbol TEXT NOT NULL,
        expiry TEXT,
        strike REAL,
        option_type TEXT,
        open REAL, high REAL, low REAL, close REAL,
        volume INTEGER, oi INTEGER,
        PRIMARY KEY (trade_date, symbol, expiry, strike, option_type)
    )""")
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {ERROR_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT,
        raw_row TEXT,
        error TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()

def upsert_rows(conn, df):
    # df must have columns matching table names
    cur = conn.cursor()
    inserted = 0
    for _, r in df.iterrows():
        try:
            cur.execute(f"""
            INSERT INTO {TABLE} (trade_date, symbol, expiry, strike, option_type, open, high, low, close, volume, oi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_date, symbol, expiry, strike, option_type) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume, oi=excluded.oi
            """, (
                r.trade_date, r.symbol, r.expiry, r.strike, r.option_type,
                r.open, r.high, r.low, r.close, r.volume, r.oi
            ))
            inserted += 1
        except Exception as e:
            cur.execute(f"INSERT INTO {ERROR_TABLE} (trade_date, raw_row, error) VALUES (?, ?, ?)",
                        (r.trade_date, str(dict(r)), str(e)))
    conn.commit()
    return inserted

# parsing helper - adapt depending on CSV columns
def parse_bhavcopy_csv_bytes(bytes_data, trade_date):
    # Try to read all CSV files in a zip
    z = zipfile.ZipFile(io.BytesIO(bytes_data))
    frames = []
    for name in z.namelist():
        # Accept .csv or .txt files
        if not name.lower().endswith(('.csv', '.txt')):
            continue
        with z.open(name) as f:
            try:
                df = pd.read_csv(f, low_memory=False)
            except Exception:
                continue
            # normalization: try to find required columns
            # Typical columns vary; attempt flexible mapping
            cols = {c.lower(): c for c in df.columns}
            def g(candidates):
                for c in candidates:
                    if c in cols:
                        return cols[c]
                return None

            symbol_col = g(['instrumentname','symbol','underlying'])
            expiry_col = g(['expiry','expirydate'])
            strike_col = g(['strikeprice','strike','strike_price'])
            option_col = g(['optiontype','optiontype','option','type'])
            ohlc_map = { 'open': g(['open','open_price']),
                         'high': g(['high','high_price']),
                         'low': g(['low','low_price']),
                         'close': g(['close','close_price','ltp','last_price']) }
            vol_col = g(['tradedqty','volume','qty'])
            oi_col = g(['openinterest','oi'])

            # if instrument table is combined equity & derivatives, attempt policy: keep rows where option_col present or strike present
            # Build normalized frame
            norm = []
            for _, row in df.iterrows():
                try:
                    symbol = row[symbol_col] if symbol_col else None
                    expiry = row[expiry_col] if expiry_col else None
                    strike = row[strike_col] if strike_col else None
                    option_type = row[option_col] if option_col else None
                    # Normalize values
                    if isinstance(expiry, str):
                        expiry = expiry.strip()
                    if isinstance(strike, str):
                        strike = strike.replace(',', '')
                    # ensure option_type is CE/PE or derive from columns
                    if pd.isna(symbol):
                        continue
                    # Only keep rows that look like options (strike numeric and option_type not null) OR instrument type hints
                    try:
                        strike_val = float(strike) if strike not in (None, '') and not pd.isna(strike) else None
                    except:
                        strike_val = None
                    option_type_val = None
                    if isinstance(option_type, str):
                        option_type_val = option_type.strip().upper()[:2]
                        if option_type_val not in ("CE","PE"):
                            option_type_val = None

                    o = {k: (row[v] if v and v in df.columns else None) for k,v in ohlc_map.items()}
                    vol = row[vol_col] if vol_col and vol_col in df.columns else None
                    oi = row[oi_col] if oi_col and oi_col in df.columns else None

                    norm.append({
                        "trade_date": trade_date,
                        "symbol": str(symbol).strip(),
                        "expiry": expiry,
                        "strike": strike_val,
                        "option_type": option_type_val,
                        "open": try_float(o.get('open')),
                        "high": try_float(o.get('high')),
                        "low": try_float(o.get('low')),
                        "close": try_float(o.get('close')),
                        "volume": try_int(vol),
                        "oi": try_int(oi)
                    })
                except Exception as e:
                    # skip row but could be logged
                    continue
            if norm:
                frames.append(pd.DataFrame(norm))
    if frames:
        out = pd.concat(frames, ignore_index=True)
        # drop rows without strike or option_type (option rows must have strike and option_type ideally)
        out = out.dropna(subset=['strike','option_type','symbol'])
        # Optionally coerce expiry to standardized format
        out['expiry'] = out['expiry'].apply(lambda x: normalize_expiry(x))
        return out
    return pd.DataFrame()

def try_float(x):
    try:
        if pd.isna(x): return None
        return float(str(x).replace(',',''))
    except:
        return None

def try_int(x):
    try:
        if pd.isna(x): return None
        return int(float(str(x).replace(',','')))
    except:
        return None

def normalize_expiry(x):
    if pd.isna(x) or x is None: return None
    try:
        d = dateparser.parse(str(x))
        return d.strftime("%d-%b-%Y")
    except:
        return str(x).strip()

# HTTP fetch with backoff
def fetch_with_backoff(url, session, max_retries=6):
    for attempt in range(1, max_retries+1):
        headers = HEADERS_BASE.copy()
        # add slight UA rotation
        headers['User-Agent'] = random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64)"
        ])
        try:
            resp = session.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.content
            # if 429 or 5xx -> backoff
            if resp.status_code in (429, 502, 503, 504, 500):
                sleep = (2 ** attempt) + random.random()
                print(f"HTTP {resp.status_code} — sleeping {sleep:.1f}s and retrying (attempt {attempt})")
                time.sleep(sleep)
                continue
            else:
                print(f"HTTP {resp.status_code} — giving up for url {url}")
                return None
        except Exception as e:
            sleep = (2 ** attempt) + random.random()
            print(f"Fetch error: {e} — sleeping {sleep:.1f}s")
            time.sleep(sleep)
    return None

def date_range(start_dt, end_dt):
    cur = start_dt
    while cur <= end_dt:
        yield cur
        cur += timedelta(days=1)

def run_backfill(start_date_str="01-Jan-2024", end_date_str="18-Sep-2025"):
    s = datetime.strptime(start_date_str, "%d-%b-%Y")
    e = datetime.strptime(end_date_str, "%d-%b-%Y")
    session = requests.Session()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    for dt in list(date_range(s,e)):
        date_text = dt.strftime("%d-%b-%Y")
        url = URL_TEMPLATE.format(date=date_text)
        print("Fetching", date_text, url)
        content = fetch_with_backoff(url, session)
        if not content:
            print("Failed to fetch", date_text)
            continue
        df = parse_bhavcopy_csv_bytes(content, date_text)
        if df is None or df.empty:
            print("No option rows parsed for", date_text)
            continue
        inserted = upsert_rows(conn, df)
        print(f"{date_text}: parsed {len(df)} rows, inserted/upserted {inserted}")
        # polite pause
        time.sleep(1.2 + random.random()*0.8)
    conn.close()

def run_incremental():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cur = conn.cursor()
    cur.execute(f"SELECT MAX(trade_date) FROM {TABLE}")
    r = cur.fetchone()
    last = r[0] if r else None
    if last:
        next_dt = datetime.strptime(last, "%d-%b-%Y") + timedelta(days=1)
    else:
        next_dt = datetime.strptime("01-Jan-2024", "%d-%b-%Y")
    today = datetime.strptime("18-Sep-2025", "%d-%b-%Y")  # change if you want open-ended; here user gave end
    if next_dt > today:
        print("No new date to fetch. Latest in DB:", last)
        conn.close()
        return
    date_text = next_dt.strftime("%d-%b-%Y")
    print("Fetching next date:", date_text)
    session = requests.Session()
    url = URL_TEMPLATE.format(date=date_text)
    content = fetch_with_backoff(url, session)
    if not content:
        print("Failed to fetch", date_text)
        conn.close()
        return
    df = parse_bhavcopy_csv_bytes(content, date_text)
    if df is None or df.empty:
        print("No option rows parsed for", date_text)
        conn.close()
        return
    inserted = upsert_rows(conn, df)
    print(f"{date_text}: parsed {len(df)} rows, inserted/upserted {inserted}")
    conn.close()

if __name__ == "__main__":
    # example usage: backfill or incremental
    import sys
    mode = sys.argv[1] if len(sys.argv)>1 else "backfill"
    if mode == "backfill":
        run_backfill("01-Jan-2024","18-Sep-2025")
    elif mode=="incremental":
        run_incremental()
    else:
        print("Usage: python nse_backfill_store.py [backfill|incremental]")
