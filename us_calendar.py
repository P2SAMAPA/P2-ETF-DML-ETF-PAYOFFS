"""
us_calendar.py  —  US Trading Days Calendar
"""

import pandas as pd
from pandas_market_calendars import get_calendar


def get_us_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Get US trading days between start_date and end_date."""
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index


def get_last_n_trading_days(n: int, end_date: str = None) -> pd.DatetimeIndex:
    """Get the last N trading days up to end_date."""
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # Get a large enough window
    start_date = pd.Timestamp(end_date) - pd.Timedelta(days=n * 2)
    all_days = get_us_trading_days(start_date.strftime('%Y-%m-%d'), end_date)
    
    # Return last N days
    return all_days[-n:]


def is_trading_day(date: str) -> bool:
    """Check if a given date is a US trading day."""
    try:
        trading_days = get_us_trading_days(date, date)
        return len(trading_days) > 0
    except:
        return False


def get_next_trading_day(date: str) -> str:
    """Get the next US trading day after the given date."""
    from datetime import datetime, timedelta
    
    d = pd.Timestamp(date)
    # Check up to 10 days ahead
    for i in range(1, 11):
        check_date = (d + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        if is_trading_day(check_date):
            return check_date
    
    # Fallback: use calendar
    days = get_us_trading_days(date, (d + pd.Timedelta(days=15)).strftime('%Y-%m-%d'))
    if len(days) > 0:
        return days[0].strftime('%Y-%m-%d')
    return date
