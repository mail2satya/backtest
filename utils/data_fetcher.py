import upstox_client
from upstox_client.rest import ApiException
import datetime

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