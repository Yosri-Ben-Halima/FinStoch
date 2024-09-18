import pandas as pd

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
    end = start + pd.DateOffset(years=T)
    
    date_range = pd.date_range(start=start, end=end, periods=num_steps)
    
    return date_range