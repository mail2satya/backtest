import os
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
            # The correct function is get_historical_candle_data, which takes from_date
            api_response = history_api.get_historical_candle_data(
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
        # Initialize database
        create_ohlc_table()

        api_client = get_api_client()
        print("Successfully authenticated with Upstox API.")

        nifty50_stocks = get_nifty50_stocks()

        print("\nFetching and storing all available OHLC data for NIFTY50 stocks...")
        for symbol, instrument_key in nifty50_stocks.items():
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

        prompt = input("\nDo you want to place a dummy order? (Y/N): ")
        if prompt.lower() == 'y':
            print("\nPlacing a dummy order...")
            first_stock_instrument_key = list(nifty50_stocks.values())[0]
            place_dummy_order(api_client, first_stock_instrument_key)
        else:
            print("\nSkipping dummy order placement.")

    except ValueError as e:
        print(e)
    except ApiException as e:
        if "UDAPI100050" in str(e.body):
            print("Invalid token detected. Deleting access_token.json. Please rerun the script to re-authenticate.")
            if os.path.exists("utils/access_token.json"):
                os.remove("utils/access_token.json")
        else:
            print(f"Upstox API Exception: {e}")