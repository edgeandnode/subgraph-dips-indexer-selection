"""
A module for time-related functions.
"""

from datetime import date, datetime, timedelta
from typing import NewType, Optional, Tuple

DateStr = NewType("DateStr", str)
TimestampStr = NewType("TimestampStr", str)


def derive_timestamps(
    num_days: int, end_date: Optional[date] = None
) -> Tuple[date, date, TimestampStr, TimestampStr]:
    """
    Derive start and end timestamps for a data collection period based on the current date.

    This function calculates a date range ending at the current date and starting 'num_days' ago.
    It returns both datetime objects and formatted string timestamps.

    :param num_days: Number of days to look back from the current date. Must be a non-negative integer.
    :param end_date: The end date of the range. If not provided, the current date is used.
    :return: A tuple containing four elements:
        - start_date: The start date of the range.
        - end_date: The end date of the range.
        - start_ts: Formatted string of the start date (YYYY-MM-DDTHH:MM:SSZ).
        - end_ts: Formatted string of the end date (YYYY-MM-DDTHH:MM:SSZ).
    :raises ValueError: If num_days is negative or not an integer.
    """
    if not isinstance(num_days, int) or num_days < 0:
        raise ValueError("num_days must be a non-negative integer")

    end_date = end_date or datetime.today()
    start_date = end_date - timedelta(days=num_days)

    start_ts = TimestampStr(start_date.strftime("%Y-%m-%dT%H:%M:%SZ"))
    end_ts = TimestampStr(end_date.strftime("%Y-%m-%dT%H:%M:%SZ"))

    return start_date, end_date, start_ts, end_ts
