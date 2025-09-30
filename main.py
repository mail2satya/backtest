import upstox_client
from upstox_client.rest import ApiException
from utils.config import API_KEY
from utils.get_access_token import get_access_token
import pandas as pd
import requests
import io
import gzip
import json
import os

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

def get_nifty50_stocks():
    """
    Fetches the list of NIFTY50 stocks from the NSE website and maps them
    to Upstox instrument keys.
    """
    # URL of the Upstox instrument list
    instrument_url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"

    # Download and decompress the instrument file
    response = requests.get(instrument_url)
    compressed_file = io.BytesIO(response.content)
    decompressed_file = gzip.GzipFile(fileobj=compressed_file)

    # Read the CSV data into a pandas DataFrame
    instrument_df = pd.read_csv(decompressed_file)

    # --- Fetch NIFTY50 constituents from NSE ---
    nifty50_url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    session = requests.Session()
    # First, visit the main page to establish a session and get cookies
    session.get("https://www.nseindia.com", headers=headers)

    # Now, make the API call with the session
    nifty50_response = session.get(nifty50_url, headers=headers)
    nifty50_response.raise_for_status()  # Raise an exception for bad status codes

    nifty50_data = nifty50_response.json()['data']
    nifty50_symbols = [stock['symbol'] for stock in nifty50_data]

    # Filter the instrument DataFrame for NIFTY50 stocks
    nifty50_df = instrument_df[
        (instrument_df['exchange'] == 'NSE_EQ') &
        (instrument_df['tradingsymbol'].isin(nifty50_symbols))
    ]

    # Create a dictionary of symbol to instrument key
    nifty50_instrument_keys = pd.Series(
        nifty50_df.instrument_key.values,
        index=nifty50_df.tradingsymbol
    ).to_dict()

    return nifty50_instrument_keys

def get_ohlc_data(api_client, instrument_key, symbol):
    """
    Fetches the OHLC data for a given instrument key.
    """
    market_quote_api = upstox_client.MarketQuoteApi(api_client)
    api_response = market_quote_api.get_market_quote_ohlc(
        instrument_key=instrument_key,
        symbol=symbol,
        interval="1d",  # Use "1d" for daily OHLC
        api_version='v2'
    )
    return api_response.data

def place_dummy_order(api_client, instrument_key):
    """
    Places a dummy (test) order.

    This function places a simple BUY order for a single quantity of the
    specified instrument. This is for demonstration purposes and uses
    the 'MARKET' order type.
    """
    order_api = upstox_client.OrderApi(api_client)

    order_request = upstox_client.PlaceOrderRequest(
        quantity=1,
        product="D",  # Delivery
        validity="DAY",
        price=0,  # Market order
        instrument_key=instrument_key,
        order_type="MARKET",
        transaction_type="BUY",
        disclosed_quantity=0,
        trigger_price=0,
        is_amo=False
    )

    try:
        api_response = order_api.place_order(body=order_request, api_version='v2')
        print(f"Successfully placed order. Order ID: {api_response.data.order_id}")
        return api_response.data
    except ApiException as e:
        print(f"Failed to place order: {e}")
        return None

if __name__ == "__main__":
    try:
        api_client = get_api_client()
        print("Successfully authenticated with Upstox API.")

        # Fetch NIFTY50 stocks
        nifty50_stocks = get_nifty50_stocks()

        # --- Fetch OHLC Data ---
        print("\nFetching OHLC data for NIFTY50 stocks...")
        for symbol, instrument_key in nifty50_stocks.items():
            try:
                ohlc_data = get_ohlc_data(api_client, instrument_key, symbol)
                print(f"--- {symbol} ---")
                print(f"  OHLC: O={ohlc_data.ohlc.open}, H={ohlc_data.ohlc.high}, L={ohlc_data.ohlc.low}, C={ohlc_data.ohlc.close}")
            except ApiException as e:
                print(f"Could not fetch OHLC for {symbol}: {e}")

        # --- Place a Dummy Order ---
        print("\nPlacing a dummy order...")
        first_stock_instrument_key = list(nifty50_stocks.values())[0]
        place_dummy_order(api_client, first_stock_instrument_key)

    except ValueError as e:
        print(e)
    except ApiException as e:
        print(f"Upstox API Exception: {e}")