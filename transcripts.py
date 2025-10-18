from typing import List
import pandas as pd
import numpy as np

def sample_calls(df: pd.DataFrame, max_calls=10000) -> pd.DataFrame:
    """
    Sample a subset of calls from a DataFrame based on the 'Filename' field.

    Parameters:
    - df: DataFrame containing at least a 'Filename' column
    - max_calls: Maximum number of unique calls (i.e., unique 'Filename's) to sample

    Returns:
    - DataFrame containing rows from the sampled calls only
    """
    grouped = df.groupby("Filename")
    keys = list(grouped.groups.keys())  # Ensure it's a list, not dict_keys
    sample_size = min(max_calls, len(keys))  # Prevent ValueError if too large
    selected_groups = np.random.choice(keys, size=sample_size, replace=False)

    sampled_df = pd.concat([grouped.get_group(k) for k in selected_groups])
    return sampled_df.reset_index(drop=True)

def clean_transcripts(selected_df: pd.DataFrame, max_gap=2.0, junk_phrases: List[str] = ['<unk>.', '[laughter].', 'Huh.', '<unk> <unk>.', '<unk>.< <unk>.']) -> pd.DataFrame:
    """
    Clean transcripts by merging consecutive rows based on specified conditions.  Logic particular to NX transcripts.

    Args:
        selected_df (pd.DataFrame): The input DataFrame containing the transcripts.
        max_gap (float, optional): The maximum time gap (in seconds) allowed between consecutive rows to be merged. Defaults to 2.0.
        junk_phrases (List[str], optional): A list of junk phrases to be excluded from the merged transcripts. Defaults to ['<unk>.', '[laughter].', 'Huh.', '<unk> <unk>.', '<unk>.< <unk>.'].

    Returns:
        pd.DataFrame: The cleaned DataFrame with merged transcripts.

    """
    # Calculate the time gap between consecutive rows
    selected_df['Gap'] = (selected_df['StartOffset (sec)'] - selected_df['EndOffset (sec)'].shift(1)).fillna(0)
    
    # Check if the party is the same as the previous row
    selected_df['SameParty'] = (selected_df['Party'] == selected_df['Party'].shift(1)).fillna(False)
    
    # Determine if the rows should not be merged based on gap and speaker changes
    selected_df['NoMerge'] = ~((selected_df['Gap'] < max_gap) & (selected_df['Gap'] >= 0) & (selected_df.SameParty))
    
    # Assign line numbers to the rows within each filename
    selected_df['LineNumber'] = selected_df.groupby('Filename')['NoMerge'].transform('cumsum')
    
    # Merge the consecutive rows based on the line number, party, and specified aggregation functions
    merged_df = selected_df.groupby(['Filename', 'LineNumber', 'Party']).agg({'Text': ' '.join, 'StartOffset (sec)': 'min', 'EndOffset (sec)': 'max'}).reset_index()
    
    # Exclude junk phrases from the merged transcripts
    junk_phrases_df = pd.DataFrame({'Text': junk_phrases})
    merged_df = pd.merge(merged_df, junk_phrases_df, how='left', indicator='how').query('how=="left_only"')
    
    # Assign new line numbers to the merged rows within each filename
    merged_df['LineNumber'] = merged_df.groupby('Filename').transform('cumcount')
    
    # Remove the indicator column
    return merged_df.drop(['how'], axis=1)

def load_whisper_as_nx(filename:str) ->pd.DataFrame:
    ''' 
    Load Whisper CSV output as NX dataframe consistent with NXFormatTranscript
    '''
    col_mapping = {'Path':'Filename', 'party':'Party','start':'StartOffset (sec)','end':'EndOffset (sec)', 'text':'Text'}
    df = pd.read_csv(filename, sep='\t').rename(col_mapping, axis=1)
    return df      

def load_and_clean_nxtranscript(filename:str) ->pd.DataFrame:
    ''' 
    Load NXFormatTranscript transcript and perform additional post-processing
    '''
    df = pd.read_csv(filename, sep='\t')
    return clean_transcripts(df)

def transcript2DataFrame(transcript:str, filename:str) -> pd.DataFrame:
    lines = transcript.split("\n")
    rows = []
    for line in lines:
        parts = line.split(": ")
        if len(parts) == 3:
            rows.append({"LineNumber": int(parts[0]), "Party": parts[1], "Text": parts[2]})
    df = pd.DataFrame(rows)
    df['Filename'] = filename
    return df

