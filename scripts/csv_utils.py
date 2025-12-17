"""
Utility functions for reading and cleaning CSV flight data files.
"""
import pandas as pd


def read_flight_data_csv(file_path):
    """
    Read and clean a flight data CSV file.
    
    Handles:
    - Whitespace in values (especially Dep_time_min)
    - Empty/Unnamed columns
    - Proper data type conversion
    - Column name normalization
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Cleaned pandas DataFrame
    """
    df = pd.read_csv(file_path, skipinitialspace=True)
    
    # Remove empty/Unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Normalize column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Ensure proper data types
    if 'Dep_time_min' in df.columns:
        df['Dep_time_min'] = pd.to_numeric(df['Dep_time_min'], errors='coerce').fillna(0)
    
    if 'N_passengers' in df.columns:
        df['N_passengers'] = pd.to_numeric(df['N_passengers'], errors='coerce').fillna(0).astype(int)
    
    if 'Airline' in df.columns:
        df['Airline'] = df['Airline'].astype(str).str.strip()
    
    if 'Type' in df.columns:
        df['Type'] = df['Type'].astype(str).str.strip()
    
    return df

