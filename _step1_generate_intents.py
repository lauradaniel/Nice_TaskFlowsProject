import argparse
import os
import sys
import logging
import logging.config
import json
from tqdm import tqdm
import pandas as pd
from intents import IntentGenerator, categories2dataframe
from transcripts import transcript2DataFrame
from annotations import save_convs, get_conversations
import bedrock

# Logging setup
os.makedirs('./logs', exist_ok=True)
logging.config.fileConfig('./logger.ini')
logger = logging.getLogger("research")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate intents using Bedrock LLM and provided category mapping")
    parser.add_argument('--input_json', required=True, help='Input: JSON transcript (from step0)')
    parser.add_argument('--output_json', required=True, help='Output: JSON with LLM-generated reasons')
    parser.add_argument('--output_csv', required=True, help='Output: CSV file with extracted intents')
    parser.add_argument('--categories_txt', required=True, help='Input: Mapping file in YAML-style format')
    parser.add_argument('--prompt_filename', required=True, help='Input: Prompt file to use with LLM')
    parser.add_argument('--model_id', required=False, default='anthropic.claude-3-sonnet-20240229-v1:0', help='Bedrock model ID')
    parser.add_argument('--max_interactions', type=int, default=1000, help='Maximum number of interactions to process')
    parser.add_argument('--num_lines', type=int, default=10, help='Max lines to include from conversation')
    parser.add_argument('--min_words', type=int, default=5, help='Min number of words in generated intent')
    parser.add_argument('--max_words', type=int, default=10, help='Max number of words in generated intent')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files')

    args = parser.parse_args()

    logger.info(f'Running: {" ".join(sys.argv)}')
    logger.info(f'ArgParse: {args}')

    if not args.force and os.path.exists(args.output_json):
        logger.info(f"Output {args.output_json} already exists. Use --force to overwrite.")
        sys.exit(0)

    try:
        client = bedrock.get_client(region="us-east-1")

        logger.info(f'Loading categories from {args.categories_txt}')
        cat_df = categories2dataframe(args.categories_txt)
        categories_str = open(args.categories_txt).read()

        logger.info(f'Loading prompt from {args.prompt_filename}')
        with open(args.prompt_filename) as f:
            prompt = f.read()

        generator = IntentGenerator(
            bedrock_client=client,
            prompt=prompt,
            categories=categories_str,
            num_lines=args.num_lines,
            model_id=args.model_id,
            min_words=args.min_words,
            max_words=args.max_words,
            max_workers=10,
            max_tokens=256
        )

        logger.info(f'Generating intents for up to {args.max_interactions} interactions')
        generator.collect_reasons(
            input_json=args.input_json,
            output_json=args.output_json,
            max_interactions=args.max_interactions
        )

        logger.info(f'Writing intent CSV to {args.output_csv}')
        generator.create_intent_csv(args.output_json, args.output_csv, additional_columns=[])

        logger.info(f'Successfully completed {os.path.basename(__file__)}')

    except Exception:
        logger.exception(f'Failed {os.path.basename(__file__)}')
