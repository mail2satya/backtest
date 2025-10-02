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
import datetime

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
    Returns a hardcoded list of NIFTY50 stocks and their instrument keys.
    This avoids dependency on the unstable NSE website API.
    """
    nifty50_instrument_keys = {
        "RELIANCE": "NSE_EQ|INE002A01018",
        "TCS": "NSE_EQ|INE467B01029",
        "HDFCBANK": "NSE_EQ|INE040A01034",
        "ICICIBANK": "NSE_EQ|INE090A01021",
        "INFY": "NSE_EQ|INE009A01021",
        "HINDUNILVR": "NSE_EQ|INE030A01027",
        "ITC": "NSE_EQ|INE154A01025",
        "SBIN": "NSE_EQ|INE062A01020",
        "BHARTIARTL": "NSE_EQ|INE397D01024",
        "LICI": "NSE_EQ|INE0J1Y01017",
    }
    return nifty50_instrument_keys

def get_ohlc_data(api_client, instrument_key):
    """
    Fetches the latest daily OHLC data using the History API.
    """
    history_api = upstox_client.HistoryApi(api_client)
    to_date = datetime.date.today().strftime('%Y-%m-%d')

    api_response = history_api.get_historical_candle_data(
        instrument_key=instrument_key,
        interval='day',
        to_date=to_date,
        api_version='v2'
    )
    # The API returns a list of candles, we'll take the most recent one.
    latest_candle = api_response.data.candles[-1]
    return {
        'open': latest_candle[1],
        'high': latest_candle[2],
        'low': latest_candle[3],
        'close': latest_candle[4]
    }

def place_dummy_order(api_client, instrument_key):
    """
    Places a dummy (test) order.
    """
    order_api = upstox_client.OrderApi(api_client)

    order_request = upstox_client.PlaceOrderRequest(
        quantity=1,
        product="D",
        validity="DAY",
        price=0,
        instrument_token=instrument_key,
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

        nifty50_stocks = get_nifty50_stocks()

        print("\nFetching OHLC data for NIFTY50 stocks...")
        for symbol, instrument_key in nifty50_stocks.items():
            try:
                ohlc_data = get_ohlc_data(api_client, instrument_key)
                print(f"--- {symbol} ---")
                print(f"  OHLC: O={ohlc_data['open']}, H={ohlc_data['high']}, L={ohlc_data['low']}, C={ohlc_data['close']}")
            except ApiException as e:
                print(f"Could not fetch OHLC for {symbol}: {e}")

        print("\nPlacing a dummy order...")
        first_stock_instrument_key = list(nifty50_stocks.values())[0]
        place_dummy_order(api_client, first_stock_instrument_key)

    except ValueError as e:
        print(e)
    except ApiException as e:
        print(f"Upstox API Exception: {e}")