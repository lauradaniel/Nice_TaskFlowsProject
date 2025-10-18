from io import StringIO
import os
os.makedirs('./logs', exist_ok=True)
import logging
import logging.config
logging.config.fileConfig('./logger.ini')
logger = logging.getLogger("research")
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import bedrock
from intents import categories2dataframe
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    # Parse command line
    parser = argparse.ArgumentParser(description="Assign pre-defined L1 > L2 > L3 categories to freeform intents")
    parser.add_argument('--intents_filename', required=True, help='Input: TSV file with generated freeform intents')
    parser.add_argument('--categories_txt', required=True, help='Input: YAML-style file with predefined categories')
    parser.add_argument('--output_mapping_csv', required=True, help='Output: TSV file with intent-to-category mapping')
    parser.add_argument('--model_id', required=False, default='anthropic.claude-3-sonnet-20240229-v1:0', help='Bedrock model ID')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for batching intent assignment')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of parallel threads')
    args = parser.parse_args()

    logger.info(f'Running: {" ".join(sys.argv)}')
    logger.info(f'Arguments: {args}')

    try:
        client = bedrock.get_client(region="us-east-1")

        logger.info(f'Loading categories from {args.categories_txt}')
        cat_df = categories2dataframe(args.categories_txt)

        logger.info(f'Loading intents from {args.intents_filename}')
        reasons_df = pd.read_csv(args.intents_filename, sep='\t')
        reasons_df = reasons_df.dropna(subset=['Intent']).copy()
        reasons_df['Ind'] = range(len(reasons_df))

        # Merge with pre-defined categories (manual mapping only)
        logger.info("Using pre-defined category mapping only (no LLM assignment)")
        merged_df = pd.merge(reasons_df, cat_df, how='left', left_on='Intent', right_on='L3')
        if merged_df[['L1', 'L2', 'L3']].isnull().any().any():
            logger.warning("Some intents in reasons file do not match any categories in mapping file")

        merged_df['Intent_Category'] = merged_df['L1'] + ' - ' + merged_df['L2'] + ' - ' + merged_df['L3']
        merged_df['L1_Score'] = 5
        merged_df['L2_Score'] = 5
        merged_df['L3_Score'] = 5

        logger.info(f'Writing output mapping to {args.output_mapping_csv}')
        merged_df.to_csv(args.output_mapping_csv, sep='\t', index=False)

        logger.info(f"Successfully Completed {os.path.basename(__file__)}")
    except Exception:
        logger.exception(f'Failed {os.path.basename(__file__)}')
