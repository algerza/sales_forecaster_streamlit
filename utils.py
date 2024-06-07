import pandas as pd

def map_week_category(day_list):
    """
    Map day of the week values to categories.

    Parameters:
    - day_list (list): List of day_of_week values.

    Returns:
    - list: List of corresponding week categories.
    """
    week_mapping = {
        1: 'Beginning of week',
        2: 'Beginning of week',
        3: 'Mid-end of week',
        4: 'Mid-end of week',
        5: 'Mid-end of week',
        6: 'Weekend',
        7: 'Weekend',
    }
    return [week_mapping[day] for day in day_list]

def add_day_of_week(df):
    """
    Add 'day_of_week' and 'week_category' columns to DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame.
    """
    df['Started Date'] = pd.to_datetime(df['Started Date'])
    df['day_of_week'] = df['Started Date'].dt.dayofweek + 1
    df['week_category'] = map_week_category(df['day_of_week'].tolist())

def calculate_perc_columns(df):
    """
    Calculate 'amount_plus_20_perc' and 'amount_minus_20_perc' columns.

    Parameters:
    - df (DataFrame): Input DataFrame.
    """
    df['amount_plus_20_perc'] = df['Amount'] * 1.2
    df['amount_minus_20_perc'] = df['Amount'] * 0.8

def find_matching_rows(row, df):
    """
    Find matching rows based on amount thresholds and description.

    Parameters:
    - row (Series): Row from the DataFrame.
    - df (DataFrame): Input DataFrame.

    Returns:
    - list: List of matched row indices.
    """
    if row['Amount'] < 0:
        matching_rows = df[
            (row['amount_minus_20_perc'] >= df['Amount']) & 
            (df['Amount'] >= row['amount_plus_20_perc']) & 
            (df.index != row.name) & 
            (df['Description'] == row['Description'])
        ]
        matched_indices = [row.name] + matching_rows.index.tolist()

    else:
        matching_rows = df[
            (row['amount_minus_20_perc'] <= df['Amount']) & 
            (df['Amount'] <= row['amount_plus_20_perc']) & 
            (df.index != row.name) & 
            (df['Description'] == row['Description'])
        ]
        matched_indices = [row.name] + matching_rows.index.tolist()

    return matched_indices if len(matched_indices) > 1 else float('nan')



def find_matching_column(row, df, column_name):
    """
    Find matching values in a specific column based on amount thresholds and description.

    Parameters:
    - row (Series): Row from the DataFrame.
    - df (DataFrame): Input DataFrame.
    - column_name (str): Name of the column to check.

    Returns:
    - list: List of matched values in the specified column.
    """
    if row['Amount'] < 0:
        matching_values = df[
            (row['amount_minus_20_perc'] >= df['Amount']) & 
            (df['Amount'] >= row['amount_plus_20_perc']) & 
            (df.index != row.name) & 
            (df['Description'] == row['Description'])
        ]
        matched_values = [row[column_name]] + matching_values[column_name].tolist()

    else:
        matching_values = df[
            (row['amount_minus_20_perc'] <= df['Amount']) & 
            (df['Amount'] <= row['amount_plus_20_perc']) & 
            (df.index != row.name) & 
            (df['Description'] == row['Description'])
        ]
        matched_values = [row[column_name]] + matching_values[column_name].tolist()
    return matched_values if len(matched_values) > 1 else []


# Create mapping to show on the table instead of numbers
day_mapping = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday"
}