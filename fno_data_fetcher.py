import upstox_client
from upstox_client.rest import ApiException
from utils.config import API_KEY
from utils.get_access_token import get_access_token
from utils.database import create_ohlc_table, save_ohlc_data
import pandas as pd
import requests
import io
import gzip
import json
import os
import datetime
import csv

def get_api_client():
    """
    Initializes and returns an Upstox API client.
    It will try to load the access token from a file, and if not found,
    it will trigger the authentication process.
    """
    token_file = "utils/access_token.json"
    access_token = None

    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token_data = json.load(f)
            access_token = token_data.get('access_token')

    if not access_token:
        print("Access token not found. Starting authentication process...")
        access_token = get_access_token()

    configuration = upstox_client.Configuration()
    configuration.access_token = access_token
    api_client = upstox_client.ApiClient(configuration)

    return api_client

def get_fno_stocks():
    """
    Returns a hardcoded list of F&O stocks and their instrument keys.
    """
    fno_instrument_keys = {
        "ABCAPITAL": "NSE_EQ|INE674K01013", "ADANIENSOL": "NSE_EQ|INE931S01010",
        "INDHOTEL": "NSE_EQ|INE053A01029", "INDIANB": "NSE_EQ|INE562A01011",
        "INOXWIND": "NSE_EQ|INE066P01011", "JINDALSTEL": "NSE_EQ|INE749A01030",
        "MPHASIS": "NSE_EQ|INE356A01018", "NESTLEIND": "NSE_EQ|INE239A01024",
        "NTPC": "NSE_EQ|INE733E01010", "PRESTIGE": "NSE_EQ|INE811K01011",
        "RBLBANK": "NSE_EQ|INE976G01028", "SHREECEM": "NSE_EQ|INE070A01015",
        "TATAELXSI": "NSE_EQ|INE670A01012", "TATASTEEL": "NSE_EQ|INE081A01020",
        "TCS": "NSE_EQ|INE467B01029", "AMBUJACEM": "NSE_EQ|INE079A01024",
        "APLAPOLLO": "NSE_EQ|INE702C01027", "AUROPHARMA": "NSE_EQ|INE406A01037",
        "BSE": "NSE_EQ|INE118H01025", "CDSL": "NSE_EQ|INE736A01011",
        "CONCOR": "NSE_EQ|INE111A01025", "DALBHARAT": "NSE_EQ|INE00R701025",
        "EICHERMOT": "NSE_EQ|INE066A01021", "INDIGO": "NSE_EQ|INE646L01027",
        "IRCTC": "NSE_EQ|INE335Y01020", "JIOFIN": "NSE_EQ|INE758E01017",
        "KAYNES": "NSE_EQ|INE918Z01012", "KEI": "NSE_EQ|INE878B01027",
        "MFSL": "NSE_EQ|INE180A01020", "NYKAA": "NSE_EQ|INE388Y01029",
        "PAGEIND": "NSE_EQ|INE761H01022", "PERSISTENT": "NSE_EQ|INE262H01021",
        "PNBHOUSING": "NSE_EQ|INE572E01012", "SBILIFE": "NSE_EQ|INE123W01016",
        "TIINDIA": "NSE_EQ|INE974X01010", "TITAN": "NSE_EQ|INE280A01028",
        "TRENT": "NSE_EQ|INE849A01020", "UNIONBANK": "NSE_EQ|INE692A01016",
        "YESBANK": "NSE_EQ|INE528G01035", "ABB": "NSE_EQ|INE117A01022",
        "BDL": "NSE_EQ|INE171Z01026", "BEL": "NSE_EQ|INE263A01024",
        "BHEL": "NSE_EQ|INE257A01026", "CGPOWER": "NSE_EQ|INE067A01029",
        "GMRAIRPORT": "NSE_EQ|INE776C01039", "HAL": "NSE_EQ|INE066F01020",
        "HCLTECH": "NSE_EQ|INE860A01027", "JUBLFOOD": "NSE_EQ|INE797F01020",
        "OBEROIRLTY": "NSE_EQ|INE093I01010", "ONGC": "NSE_EQ|INE213A01029",
        "PNB": "NSE_EQ|INE160A01022", "POWERGRID": "NSE_EQ|INE752E01010",
        "SONACOMS": "NSE_EQ|INE073K01018", "AXISBANK": "NSE_EQ|INE238A01034",
        "ETERNAL": "NSE_EQ|INE758T01015", "HINDPETRO": "NSE_EQ|INE094A01015",
        "IDFCFIRSTB": "NSE_EQ|INE092T01019", "LODHA": "NSE_EQ|INE670K01029",
        "TVSMOTOR": "NSE_EQ|INE494B01023", "BAJAJ-AUTO": "NSE_EQ|INE917I01010",
        "BOSCHLTD": "NSE_EQ|INE323A01026", "CIPLA": "NSE_EQ|INE059A01026",
        "DLF": "NSE_EQ|INE271C01023", "DRREDDY": "NSE_EQ|INE089A01031",
        "EXIDEIND": "NSE_EQ|INE302A01020", "GODREJPROP": "NSE_EQ|INE484J01027",
        "KOTAKBANK": "NSE_EQ|INE237A01028", "M&M": "NSE_EQ|INE101A01026",
        "NATIONALUM": "NSE_EQ|INE139A01034", "NHPC": "NSE_EQ|INE848E01016",
        "PAYTM": "NSE_EQ|INE982J01020", "PGEL": "NSE_EQ|INE457L01029",
        "TECHM": "NSE_EQ|INE669C01036", "ULTRACEMCO": "NSE_EQ|INE481G01011",
        "BANKINDIA": "NSE_EQ|INE084A01016", "BIOCON": "NSE_EQ|INE376G01013",
        "COALINDIA": "NSE_EQ|INE522F01014", "COLPAL": "NSE_EQ|INE259A01022",
        "FEDERALBNK": "NSE_EQ|INE171A01029", "GODREJCP": "NSE_EQ|INE102D01028",
        "KPITTECH": "NSE_EQ|INE04I401011", "LT": "NSE_EQ|INE018A01030",
        "NMDC": "NSE_EQ|INE584A01023", "OIL": "NSE_EQ|INE274J01014",
        "PHOENIXLTD": "NSE_EQ|INE211B01039", "PIIND": "NSE_EQ|INE603J01030",
        "TATACONSUM": "NSE_EQ|INE192A01020", "WIPRO": "NSE_EQ|INE075A01022",
        "BHARATFORG": "NSE_EQ|INE465A01025", "CROMPTON": "NSE_EQ|INE299U01018",
        "INDUSTOWER": "NSE_EQ|INE121J01017", "PATANJALI": "NSE_EQ|INE619A01035",
        "POWERINDIA": "NSE_EQ|INE07Y701011", "SOLARINDS": "NSE_EQ|INE343H01029",
        "TATAMOTORS": "NSE_EQ|INE155A01022", "TATAPOWER": "NSE_EQ|INE245A01021",
        "TITAGARH": "NSE_EQ|INE615H01020", "VOLTAS": "NSE_EQ|INE226A01021",
        "APOLLOHOSP": "NSE_EQ|INE437A01024", "CYIENT": "NSE_EQ|INE136B01020",
        "DELHIVERY": "NSE_EQ|INE148O01028", "DIXON": "NSE_EQ|INE935N01020",
        "FORTIS": "NSE_EQ|INE061F01013", "GLENMARK": "NSE_EQ|INE935A01035",
        "HINDUNILVR": "NSE_EQ|INE030A01027", "HUDCO": "NSE_EQ|INE031A01017",
        "ICICIPRULI": "NSE_EQ|INE726G01019", "IDEA": "NSE_EQ|INE669E01016",
        "NBCC": "NSE_EQ|INE095N01031", "POLYCAB": "NSE_EQ|INE455K01017",
        "ALKEM": "NSE_EQ|INE540L01014", "ASHOKLEY": "NSE_EQ|INE208A01029",
        "ASTRAL": "NSE_EQ|INE006I01046", "BANKBARODA": "NSE_EQ|INE028A01039",
        "BHARTIARTL": "NSE_EQ|INE397D01024", "BPCL": "NSE_EQ|INE029A01011",
        "CHOLAFIN": "NSE_EQ|INE121A01024", "CUMMINSIND": "NSE_EQ|INE298A01020",
        "DIVISLAB": "NSE_EQ|INE361B01024", "IEX": "NSE_EQ|INE022Q01020",
        "INDUSINDBK": "NSE_EQ|INE095A01012", "JSWENERGY": "NSE_EQ|INE121E01018",
        "KALYANKJIL": "NSE_EQ|INE303R01014", "LAURUSLABS": "NSE_EQ|INE947Q01028",
        "LTF": "NSE_EQ|INE498L01015", "MANAPPURAM": "NSE_EQ|INE522D01027",
        "MARICO": "NSE_EQ|INE196A01026", "MOTHERSON": "NSE_EQ|INE775A01035",
        "RELIANCE": "NSE_EQ|INE002A01018", "SRF": "NSE_EQ|INE647A01010",
        "ZYDUSLIFE": "NSE_EQ|INE010B01027", "ADANIGREEN": "NSE_EQ|INE364U01010",
        "ADANIPORTS": "NSE_EQ|INE742F01042", "BAJAJFINSV": "NSE_EQ|INE918I01026",
        "BLUESTARCO": "NSE_EQ|INE472A01039", "DABUR": "NSE_EQ|INE016A01026",
        "HDFCAMC": "NSE_EQ|INE127D01025", "IOC": "NSE_EQ|INE242A01010",
        "ITC": "NSE_EQ|INE154A01025", "MAXHEALTH": "NSE_EQ|INE027H01010",
        "OFSS": "NSE_EQ|INE881D01027", "PIDILITIND": "NSE_EQ|INE318A01026",
        "SBIN": "NSE_EQ|INE062A01020", "360ONE": "NSE_EQ|INE466L01038",
        "ADANIENT": "NSE_EQ|INE423A01024", "IGL": "NSE_EQ|INE203G01027",
        "JSWSTEEL": "NSE_EQ|INE019A01038", "LICI": "NSE_EQ|INE0J1Y01017",
        "POLICYBZR": "NSE_EQ|INE417T01026", "SAIL": "NSE_EQ|INE114A01011",
        "SUZLON": "NSE_EQ|INE040H01021", "UPL": "NSE_EQ|INE628A01036",
        "VEDL": "NSE_EQ|INE205A01025", "ASIANPAINT": "NSE_EQ|INE021A01026",
        "BRITANNIA": "NSE_EQ|INE216A01030", "GRASIM": "NSE_EQ|INE047A01021",
        "HDFCLIFE": "NSE_EQ|INE795G01014", "HFCL": "NSE_EQ|INE548A01028",
        "ICICIBANK": "NSE_EQ|INE090A01021", "ICICIGI": "NSE_EQ|INE765G01019",
        "INFY": "NSE_EQ|INE009A01021", "KFINTECH": "NSE_EQ|INE138Y01010",
        "MARUTI": "NSE_EQ|INE585B01010", "MUTHOOTFIN": "NSE_EQ|INE414G01012",
        "PFC": "NSE_EQ|INE134E01011", "SAMMAANCAP": "NSE_EQ|INE148I01020",
        "SUNPHARMA": "NSE_EQ|INE044A01036", "SUPREMEIND": "NSE_EQ|INE195A01028",
        "SYNGENE": "NSE_EQ|INE398R01022", "TORNTPOWER": "NSE_EQ|INE813H01021",
        "UNOMINDA": "NSE_EQ|INE405E01023", "COFORGE": "NSE_EQ|INE591G01025",
        "GAIL": "NSE_EQ|INE129A01019", "HAVELLS": "NSE_EQ|INE176B01034",
        "HDFCBANK": "NSE_EQ|INE040A01034", "HEROMOTOCO": "NSE_EQ|INE158A01026",
        "MANKIND": "NSE_EQ|INE634S01028", "MAZDOCK": "NSE_EQ|INE249Z01020",
        "MCX": "NSE_EQ|INE745G01035", "NUVAMA": "NSE_EQ|INE531F01015",
        "PETRONET": "NSE_EQ|INE347G01014", "PPLPHARMA": "NSE_EQ|INE0DK501011",
        "SIEMENS": "NSE_EQ|INE003A01024", "UNITDSPR": "NSE_EQ|INE854D01024",
        "AMBER": "NSE_EQ|INE371P01015", "BAJFINANCE": "NSE_EQ|INE296A01032",
        "CANBK": "NSE_EQ|INE476A01022", "DMART": "NSE_EQ|INE192R01011",
        "HINDZINC": "NSE_EQ|INE267A01025", "IREDA": "NSE_EQ|INE202E01016",
        "LICHSGFIN": "NSE_EQ|INE115A01026", "LTIM": "NSE_EQ|INE214T01019",
        "LUPIN": "NSE_EQ|INE326A01037", "RVNL": "NSE_EQ|INE415G01027",
        "SHRIRAMFIN": "NSE_EQ|INE721A01047", "ANGELONE": "NSE_EQ|INE732I01013",
        "AUBANK": "NSE_EQ|INE949L01017", "BANDHANBNK": "NSE_EQ|INE545U01014",
        "CAMS": "NSE_EQ|INE596I01012", "HINDALCO": "NSE_EQ|INE038A01020",
        "IIFL": "NSE_EQ|INE530B01024", "IRFC": "NSE_EQ|INE053F01010",
        "NAUKRI": "NSE_EQ|INE663F01032", "NCC": "NSE_EQ|INE868B01028",
        "RECLTD": "NSE_EQ|INE020B01018", "SBICARD": "NSE_EQ|INE018E01016",
        "TATATECH": "NSE_EQ|INE142M01025", "TORNTPHARM": "NSE_EQ|INE685A01028",
        "VBL": "NSE_EQ|INE200M01039",
    }
    return fno_instrument_keys

def get_all_ohlc_data(api_client, instrument_key):
    """
    Fetches all available daily OHLC data (up to 20 years back) for a given instrument
    by fetching data in chunks from the past towards the present.
    """
    history_api = upstox_client.HistoryApi(api_client)
    all_candles = []

    # Start from ~20 years ago
    from_date = datetime.date.today() - datetime.timedelta(days=365 * 20)

    while from_date < datetime.date.today():
        # Fetch in 2-year chunks
        to_date = from_date + datetime.timedelta(days=(365 * 2))
        if to_date > datetime.date.today():
            to_date = datetime.date.today()

        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')

        try:
            # The correct function is get_historical_candle_data1, which takes from_date
            api_response = history_api.get_historical_candle_data1(
                instrument_key=instrument_key,
                interval='day',
                from_date=from_date_str,
                to_date=to_date_str,
                api_version='v2'
            )

            if api_response.status == 'success' and hasattr(api_response.data, 'candles') and api_response.data.candles:
                # API returns data in ascending order, so we can just extend the list
                all_candles.extend(api_response.data.candles)

        except ApiException as e:
            if "No data found" in str(e.body) or "not found" in str(e.body).lower():
                pass # Ignore periods with no data and continue
            else:
                print(f"API Exception for {instrument_key} ({from_date_str} to {to_date_str}): {e}")
                break # Stop fetching for this instrument on other errors

        # Move to the next 2-year period
        from_date = to_date + datetime.timedelta(days=1)

    return all_candles

if __name__ == "__main__":
    try:
        # Initialize database
        create_ohlc_table()

        api_client = get_api_client()
        print("Successfully authenticated with Upstox API.")

        fno_stocks = get_fno_stocks()
        print(f"Found {len(fno_stocks)} F&O stocks to process.")

        print("\nFetching and storing all available OHLC data for F&O stocks...")
        for symbol, instrument_key in fno_stocks.items():
            print(f"--- Processing {symbol} ---")
            try:
                # Fetch all data
                all_ohlc_data = get_all_ohlc_data(api_client, instrument_key)
                if all_ohlc_data:
                    # Save data to database
                    save_ohlc_data(symbol, all_ohlc_data)
                else:
                    print(f"No OHLC data found for {symbol}.")
            except Exception as e:
                print(f"Could not process {symbol}: {e}")

        print("\n--- F&O data fetching complete! ---")

    except ValueError as e:
        print(e)
    except ApiException as e:
        if "UDAPI100050" in str(e.body):
            print("Invalid token detected. Deleting access_token.json. Please rerun the script to re-authenticate.")
            if os.path.exists("utils/access_token.json"):
                os.remove("utils/access_token.json")
        else:
            print(f"Upstox API Exception: {e}")