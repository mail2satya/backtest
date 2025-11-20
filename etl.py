import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YFinanceToPostgreSQL:

    def __init__(self, db_url):
        """Initialize DB engine"""
        try:
            self.engine = create_engine(db_url)
            self.test_connection()
            logger.info("Connected to PostgreSQL successfully.")
        except Exception as e:
            logger.error(f"DB connection failed: {e}")
            raise

    def test_connection(self):
        """Test DB connection"""
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def create_raw_data_table(self):
        """Create table if it doesn't exist with new columns"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS raw_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(12,4),
            high DECIMAL(12,4),
            low DECIMAL(12,4),
            close DECIMAL(12,4),
            volume BIGINT,
            adj_close DECIMAL(12,4),
            dividends DECIMAL(10,6),
            stock_splits DECIMAL(10,6),
            UNIQUE(symbol, date)
        );

        CREATE INDEX IF NOT EXISTS idx_raw_data_symbol_date ON raw_data(symbol, date);
        """
        with self.engine.begin() as conn:
            conn.execute(text(create_table_sql))
        logger.info("raw_data table ready.")

    def get_last_date_for_symbol(self, symbol):
        """Get latest stored date"""
        query = text("SELECT MAX(date) FROM raw_data WHERE symbol = :symbol")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"symbol": symbol}).scalar()
                return result
        except Exception as e:
            logger.error(f"Error fetching last date for {symbol}: {e}")
            return None

    def download_incremental_data(self, symbol):
        """
        Download only missing data for each symbol.
        If no data exists -> download from 2000.
        """
        last_date = self.get_last_date_for_symbol(symbol)

        if last_date is None:
            start_date = "2000-01-01"
            logger.info(f"{symbol}: No existing data. Downloading full history from 2000.")
        else:
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"{symbol}: Downloading new data from {start_date} onwards.")

        end_date = datetime.today().strftime("%Y-%m-%d")

        if start_date > end_date:
            logger.info(f"{symbol}: Already up to date.")
            return None

        try:
            # Download OHLC data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
                actions=True  # This includes dividends and splits
            )

            if df.empty:
                logger.warning(f"{symbol}: No new data returned.")
                return None

            # Reset index and prepare data
            df = df.reset_index()
            
            # Debug: Log the columns we received
            logger.debug(f"{symbol}: Columns received - {df.columns.tolist()}")
            
            # Add symbol column
            df['symbol'] = symbol

            # Map columns - FIXED: Use exact column names from yfinance
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits',
                'symbol': 'symbol'
            }
            
            # Rename columns that exist in the dataframe
            existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=existing_columns)
            
            # Select only the columns we want
            final_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'dividends', 'stock_splits']
            available_columns = [col for col in final_columns if col in df.columns]
            
            # Create missing columns with None
            for col in final_columns:
                if col not in df.columns:
                    df[col] = None
                    logger.warning(f"{symbol}: Column {col} not found in data, setting to NULL")
            
            df = df[final_columns]

            # Ensure date is proper format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date

            # Log sample data to verify close column exists
            if not df.empty:
                sample_close = df['close'].iloc[0] if 'close' in df.columns and not df['close'].isna().all() else 'MISSING'
                logger.debug(f"{symbol}: Sample close value: {sample_close}")

            logger.info(f"{symbol}: Processed {len(df)} records with columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return None

    def store_dataframe(self, df):
        """Insert data into PostgreSQL with proper error handling"""
        if df is None or df.empty:
            logger.warning("No data to store")
            return

        try:
            # Ensure we have the required columns
            required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'dividends', 'stock_splits']
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns in final DataFrame: {missing_columns}")
                return
            
            # Remove any rows with null symbols or dates
            df = df.dropna(subset=['symbol', 'date'])
            
            # Insert data
            df.to_sql(
                'raw_data',
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            logger.info(f"Successfully inserted {len(df)} rows into raw_data table")
            
        except Exception as e:
            logger.error(f"DB insert error: {e}")
            # Log first few rows to debug
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"DataFrame dtypes: {df.dtypes}")
            if not df.empty:
                logger.debug(f"First row: {df.iloc[0].to_dict()}")

    def process_all_symbols(self, symbols):
        """Main loop: download and store data for all symbols"""
        successful = 0
        failed = 0
        
        for symbol in symbols:
            logger.info(f"Processing symbol: {symbol}")
            try:
                df = self.download_incremental_data(symbol)
                if df is not None and not df.empty:
                    self.store_dataframe(df)
                    successful += 1
                else:
                    logger.info(f"{symbol}: No data to process")
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                failed += 1
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")


# ----------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------

if __name__ == "__main__":

    DB_URL = "postgresql://postgres:admin@localhost:5432/postgres"

    etl = YFinanceToPostgreSQL(DB_URL)
    etl.create_raw_data_table()

    # Add all NSE symbols you want here
    nse_symbols = [
        'ABCAPITAL.NS','ADANIENSOL.NS','INDHOTEL.NS','INDIANB.NS','INOXWIND.NS','JINDALSTEL.NS',
        'MPHASIS.NS','NESTLEIND.NS','NTPC.NS','PRESTIGE.NS','RBLBANK.NS','SHREECEM.NS','TATAELXSI.NS',
        'TATASTEEL.NS','TCS.NS','AMBUJACEM.NS','APLAPOLLO.NS','AUROPHARMA.NS','BSE.NS','CDSL.NS','CONCOR.NS',
        'DALBHARAT.NS','EICHERMOT.NS','INDIGO.NS','IRCTC.NS','JIOFIN.NS','KAYNES.NS','KEI.NS','MFSL.NS','NYKAA.NS',
        'PAGEIND.NS','PERSISTENT.NS','PNBHOUSING.NS','SBILIFE.NS','TIINDIA.NS','TITAN.NS','TRENT.NS','UNIONBANK.NS',
        'YESBANK.NS','ABB.NS','BDL.NS','BEL.NS','BHEL.NS','CGPOWER.NS','GMRAIRPORT.NS','HAL.NS','HCLTECH.NS','JUBLFOOD.NS',
        'OBEROIRLTY.NS','ONGC.NS','PNB.NS','POWERGRID.NS','SONACOMS.NS','AXISBANK.NS','ETERNAL.NS','HINDPETRO.NS',
        'IDFCFIRSTB.NS','LODHA.NS','TVSMOTOR.NS','BAJAJ-AUTO.NS','BOSCHLTD.NS','CIPLA.NS','DLF.NS','DRREDDY.NS',
        'EXIDEIND.NS','GODREJPROP.NS','KOTAKBANK.NS','M&M.NS','NATIONALUM.NS','NHPC.NS','PAYTM.NS','PGEL.NS','TECHM.NS',
        'ULTRACEMCO.NS','BANKINDIA.NS','BIOCON.NS','COALINDIA.NS','COLPAL.NS','FEDERALBNK.NS','GODREJCP.NS',
        'KPITTECH.NS','LT.NS','NMDC.NS','OIL.NS','PHOENIXLTD.NS','PIIND.NS','TATACONSUM.NS','WIPRO.NS','BHARATFORG.NS',
        'CROMPTON.NS','INDUSTOWER.NS','PATANJALI.NS','POWERINDIA.NS','SOLARINDS.NS','TATAMOTORS.NS','TATAPOWER.NS',
        'TITAGARH.NS','VOLTAS.NS','APOLLOHOSP.NS','CYIENT.NS','DELHIVERY.NS','DIXON.NS','FORTIS.NS','GLENMARK.NS',
        'HINDUNILVR.NS','HUDCO.NS','ICICIPRULI.NS','IDEA.NS','NBCC.NS','POLYCAB.NS','ALKEM.NS','ASHOKLEY.NS','ASTRAL.NS',
        'BANKBARODA.NS','BHARTIARTL.NS','BPCL.NS','CHOLAFIN.NS','CUMMINSIND.NS','DIVISLAB.NS','IEX.NS','INDUSINDBK.NS',
        'JSWENERGY.NS','KALYANKJIL.NS','LAURUSLABS.NS','LTF.NS','MANAPPURAM.NS','MARICO.NS','MOTHERSON.NS','RELIANCE.NS',
        'SRF.NS','ZYDUSLIFE.NS','ADANIGREEN.NS','ADANIPORTS.NS','BAJAJFINSV.NS','BLUESTARCO.NS','DABUR.NS','HDFCAMC.NS',
        'IOC.NS','ITC.NS','MAXHEALTH.NS','OFSS.NS','PIDILITIND.NS','SBIN.NS','360ONE.NS','ADANIENT.NS','IGL.NS','JSWSTEEL.NS',
        'LICI.NS','POLICYBZR.NS','SAIL.NS','SUZLON.NS','UPL.NS','VEDL.NS','ASIANPAINT.NS','BRITANNIA.NS','GRASIM.NS','HDFCLIFE.NS',
        'HFCL.NS','ICICIBANK.NS','ICICIGI.NS','INFY.NS','KFINTECH.NS','MARUTI.NS','MUTHOOTFIN.NS','PFC.NS','SAMMAANCAP.NS',
        'SUNPHARMA.NS','SUPREMEIND.NS','SYNGENE.NS','TORNTPOWER.NS','UNOMINDA.NS','COFORGE.NS','GAIL.NS','HAVELLS.NS',
        'HDFCBANK.NS','HEROMOTOCO.NS','MANKIND.NS','MAZDOCK.NS','MCX.NS','NUVAMA.NS','PETRONET.NS','PPLPHARMA.NS',
        'SIEMENS.NS','UNITDSPR.NS','AMBER.NS','BAJFINANCE.NS','CANBK.NS','DMART.NS','HINDZINC.NS','IREDA.NS','LICHSGFIN.NS',
        'LTIM.NS','LUPIN.NS','RVNL.NS','SHRIRAMFIN.NS','ANGELONE.NS','AUBANK.NS','BANDHANBNK.NS','CAMS.NS','HINDALCO.NS',
        'IIFL.NS','IRFC.NS','NAUKRI.NS','NCC.NS','RECLTD.NS','SBICARD.NS','TATATECH.NS','TORNTPHARM.NS','VBL.NS'
    ]

    # Test with a few symbols first to verify the fix
    test_symbols = ['INFY.NS', 'TCS.NS', 'RELIANCE.NS']
    logger.info("Testing with 3 symbols first...")
    etl.process_all_symbols(test_symbols)

    # If test successful, process all symbols in batches
    if len(nse_symbols) > 3:
        batch_size = 10
        remaining_symbols = [s for s in nse_symbols if s not in test_symbols]
        
        for i in range(0, len(remaining_symbols), batch_size):
            batch = remaining_symbols[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(remaining_symbols)-1)//batch_size + 1}")
            etl.process_all_symbols(batch)

    logger.info("Incremental data load completed!")