from io import StringIO
import os
os.makedirs('./logs',exist_ok=True)
import logging
import logging.config
logging.config.fileConfig('./logger.ini')
logger = logging.getLogger("research")
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import bedrock
from intents import IntentBuilder, categories2dataframe
import pandas as pd
from tqdm import tqdm
import threading

if __name__ == "__main__":
    # Parse command line
    parser = argparse.ArgumentParser(description="Assign pre-defined L1 > L2 > L3 categories to freeform intents")
    parser.add_argument('--intents_filename', required=True, help='Input: TSV file with generated freeform intents')
    parser.add_argument('--categories_txt', required=True, help='Input: YAML-style file with predefined categories')
    parser.add_argument('--assign_prompt_filename', required=True, help='Prompt file for assigning categories')
    parser.add_argument('--output_mapping_csv', required=True, help='Output: TSV file with intent-to-category mapping')
    parser.add_argument('--model_id', required=False, default='anthropic.claude-3-sonnet-20240229-v1:0', help='Bedrock model ID')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for batching intent assignment')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of parallel threads')
    args = parser.parse_args()

    logger.info(f'Running: {" ".join(sys.argv)}')
    logger.info(f'Arguments: {args}')

    try:
        client = bedrock.get_client(region="us-east-1")

        logger.info(f'Loading prompts from {args.assign_prompt_filename}')
        with open(args.assign_prompt_filename) as f:
            assign_prompt = f.read()

        logger.info(f'Loading categories from {args.categories_txt}')
        cat_df = categories2dataframe(args.categories_txt)

        logger.info(f'Loading intents from {args.intents_filename}')
        reasons_df = pd.read_csv(args.intents_filename, sep='\t')
        reasons_df = reasons_df.dropna(subset=['Intent']).copy()
        reasons_df['Ind'] = range(len(reasons_df))

        builder = IntentBuilder(client, cluster_prompt="", assign_prompt=assign_prompt)

        output_mapping = args.output_mapping_csv
        if os.path.exists(output_mapping):
            os.remove(output_mapping)
        lock = threading.Lock()

        logger.info(f"Assigning categories to {len(reasons_df)} intents")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(builder.assign_reasons, 
                                       args.categories_txt, 
                                       list(reasons_df.iloc[start:start+args.chunk_size]['Intent']),
                                       args.model_id,
                                       start)
                       for start in range(0, len(reasons_df), args.chunk_size)]

            for response in tqdm(as_completed(futures), total=len(futures)):
                result = response.result()
                data_io = StringIO(result)
                assign_cols = ['Ind','Intent_Input','Intent_Category','L3_Score','L2_Score','L1_Score']
                df = pd.read_csv(data_io, names=assign_cols, on_bad_lines='skip')

                # ✅ Ignore any header rows accidentally included
                df = df[~df['Ind'].astype(str).str.lower().str.contains('ind', na=False)]

                # Ensure types are correct
                score_cols = ['Ind','L3_Score', 'L2_Score','L1_Score']
                for col in score_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=score_cols)
                for col in score_cols:
                    df[col] = df[col].astype(int)

                df.drop(['Intent_Input'], axis=1, inplace=True)
                merged_df = pd.merge(reasons_df, df, on='Ind')
                merged_df['Intent_Category'] = merged_df['Intent_Category'].str.replace(r',.*', '', regex=True)
                with lock:
                    header = not os.path.exists(output_mapping)
                    merged_df.to_csv(output_mapping, sep='\t', index=False, mode='a', header=header)

        logger.info(f"Total Input Tokens: {builder.input_token_counter}")
        logger.info(f"Total Output Tokens: {builder.output_token_counter}")
        logger.info(f'Successfully Completed {os.path.basename(__file__)}')
    except Exception:
        logger.exception(f'Failed {os.path.basename(__file__)}')
