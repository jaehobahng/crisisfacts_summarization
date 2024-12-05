import pandas as pd

def clean_summary(df):
    """
    Cleans the summary data by applying necessary transformations.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be cleaned.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """

    df = df[['Event Name', 'summary', 'question','unix_timestamp', 'avg_importance']]

    # Convert 'unix_timestamp' to datetime and date formats
    df['datetime'] = df['unix_timestamp'].apply(lambda x: pd.to_datetime(x, unit='s'))
    df['date'] = df['datetime'].apply(lambda x: x.date())

    df = df.sort_values(by=['date','avg_importance'], ascending=[True,False])

    df['request'] = df['Event Name']
    df['summary_xsum_detail'] = df['summary']

    # Ensure consistent data types
    df['request'] = df['request'].astype(str)
    df['date'] = df['date'].astype(str)
    df['datetime'] = df['datetime'].astype(str)
    df['summary_xsum_detail'] = df['summary_xsum_detail'].astype(str)

    df = df[['request', 'date', 'datetime', 'question', 'summary_xsum_detail', 'avg_importance']]

    return df

def clean_original(df):
    """
    Cleans the original data by applying necessary transformations.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be cleaned.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """

    # docno,text,query,question,unix_timestamp,EventName,importance,source


    # Subset the DataFrame
    df = df[['docno', 'query', 'text', 'EventName', 'unix_timestamp', 'question', 'importance', 'source']]

    # 
    df['request'] = df['EventName']

    # Convert 'unix_timestamp' to datetime and date formats
    df['datetime'] = df['unix_timestamp'].apply(lambda x: pd.to_datetime(x, unit='s'))
    df['date'] = df['datetime'].apply(lambda x: x.date())

    df = df.sort_values(by=['date','importance'], ascending=[True,False])

    # Ensure consistent data types
    df['request'] = df['request'].astype(str)
    df['date'] = df['date'].astype(str)
    df['datetime'] = df['datetime'].astype(str)

    df = df[['query', 'text', 'request', 'datetime', 'date', 'question', 'importance', 'source']]

    return df