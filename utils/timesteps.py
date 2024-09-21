import pandas as pd
from dateutil.relativedelta import relativedelta

def generate_date_range(start_date: str, T: float, num_steps: int) -> pd.DatetimeIndex:
    """
    Generates a date range from the start date with a duration of T years and num_steps number of steps.

    Parameters
    ----------
    start_date : str
        The start date in 'YYYY-MM-DD' format.
    T : float
        The duration of the date range in years.
    num_steps : int
        The number of dates in the date range.

    Returns
    -------
    pd.DatetimeIndex
        A Pandas DatetimeIndex containing the date range.
    """

    start = pd.to_datetime(start_date)
     
    days_in_year = 365.25  # Taking leap years into account
    total_days = T * days_in_year
    total_ds = int(total_days * 24 * 60 * 60 * 10)
    #total_days = int(T * days_in_year * 24 * 60 * 60)

    end = start + pd.Timedelta(seconds=total_ds, unit='ds')

    date_range = pd.date_range(start=start, end=end, periods=num_steps)
    
    return date_range

def generate_date_range_with_granularity(start_date: str, end_date: str, granularity) -> pd.DatetimeIndex:
    """
    Generate a date range between start and end dates based on a given granularity.

    Parameters
    ----------
    start_date : str
        The start date in 'YYYY-MM-DD' or 'YYYY-MM-DD hh:mm:ss' format.
    end_date : str
        The end date in 'YYYY-MM-DD' or 'YYYY-MM-DD hh:mm:ss' format.
    granularity : str
        The frequency of the date range in the format f"{n}{u}" or f"{u}", where:
        - n is the integer number of time units (optional).
        - u is a Pandas frequency string, which defines the time unit.

        Examples of valid granularity values includes, but not limited to:
        - '10T': Every 10 minutes.
        - 'T': Every minute.
        - 'H': Every hour.
        - 'D': Every day.
        - 'W': Every week.
        - 'M': Every month.
        - '3H': Every 3 hours.
        - '2D': Every 2 days.

    Returns
    -------
    pd.DatetimeIndex
        A Pandas DatetimeIndex containing the date range.

    Examples
    --------
    Generate a date range every 10 minutes:
    >>> generate_date_range_with_granularity('2023-09-01', '2023-09-01 03:00:00', '10T')

    Generate a date range every minute:
    >>> generate_date_range_with_granularity('2023-09-01', '2023-09-01 01:00:00', 'T')

    Generate a date range every 2 days:
    >>> generate_date_range_with_granularity('2023-01-01', '2023-01-10', '2D')
    """
    return pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq=granularity)

def date_range_duration(range: pd.DatetimeIndex) -> float:
    start = range[0]
    end = range[-1]

    # Calculate the difference using relativedelta
    delta = relativedelta(end, start)

    # Calculate the proportion of years, including months, days, hours, minutes, and seconds
    T = (
        delta.years
        + delta.months / 12
        + delta.days / 365.25
        + delta.hours / (365.25 * 24)
        + delta.minutes / (365.25 * 24 * 60)
        + delta.seconds / (365.25 * 24 * 60 * 60)
    )
    
    return T