from sspipe import p, px

def ts_to_strings(ts):
    """
    Returns a list of strings from XRP timestamps.

    Parameters:
    -----------
    ts: pandas timestamp col
    """
    return [str(x)[:19] for x in ts];

def strings_to_datetime(col):
    """
    Returns a Pandas column of datetime objects from list of strings.

    Parameters:
    -----------
    col: list
      List of strings representing datetimes
    """
    return [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in col];
