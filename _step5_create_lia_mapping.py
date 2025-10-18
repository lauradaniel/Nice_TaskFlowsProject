import argparse
import os
import sys
import logging
import logging.config
import pandas as pd
from intents import categories2dataframe
from openpyxl import Workbook
from tqdm import tqdm

# Logging setup
os.makedirs('./logs', exist_ok=True)
logging.config.fileConfig('./logger.ini')
logger = logging.getLogger("research")

def write_mapping_excel(output_file, df):
    logger.info(f"Writing to Excel: {output_file}")
    df = df.copy()
    df['Category_Orig'] = df['Level2_Topic_Mapped']
    df['Topic_Orig'] = df['Level3_CallIntents_Mapped']
    df['AgentTaskLabel_Mapped'] = '*drop*'
    df['anti_terms'] = ''
    df['call_driver'] = ''

    columns = [
        'Category_Orig', 'Topic_Orig', 'Level1_Category_Mapped',
        'Level2_Topic_Mapped', 'Level3_CallIntents_Mapped',
        'AgentTaskLabel_Mapped', 'terms', 'anti_terms', 'call_driver'
    ]
    df = df[columns].sort_values(['Level1_Category_Mapped', 'Level2_Topic_Mapped', 'Level3_CallIntents_Mapped'])
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Excel LIA mapping spreadsheet based on user-provided mapping")
    parser.add_argument('--mapping_file', required=True, help='Excel/TSV mapping file with L1-L2-L3 and phrases')
    parser.add_argument('--output_excel', required=True, help='Final output Excel mapping')
    parser.add_argument('--is_excel', action='store_true', help='Set this flag if input mapping is in Excel format')

    args = parser.parse_args()
    logger.info(f'Running with args: {args}')

    try:
        if args.is_excel:
            df = pd.read_excel(args.mapping_file)
        else:
            df = pd.read_csv(args.mapping_file, sep='\t')

        # Check required columns
        required_cols = ['Level1_Category_Mapped', 'Level2_Topic_Mapped', 'Level3_CallIntents_Mapped', 'terms']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        logger.info(f"Loaded mapping file with {len(df)} entries")
        write_mapping_excel(args.output_excel, df)
        logger.info("Completed successfully")

    except Exception as e:
        logger.exception(f"Failed with error: {e}")