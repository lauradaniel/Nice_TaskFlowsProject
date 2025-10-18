import os
import sys
os.makedirs('./logs',exist_ok=True)
import logging
import logging.config
logging.config.fileConfig('./logger.ini')
logger = logging.getLogger("research")
import argparse
import pandas as pd
from annotations import save_convs, get_conversations
from transcripts import load_whisper_as_nx, load_and_clean_nxtranscript, sample_calls

if __name__ == "__main__":
    # Parse command line
    parser = argparse.ArgumentParser(description="Convert ASR transcripts to internal format")
    parser.add_argument('--input_csv', required=True, help='Input: ASR in CSV format (NXFormatTranscript or )')
    parser.add_argument('--output_json', required=True, help='Output: Json file ')
    parser.add_argument('--output_csv', required=True, help='Output: Sampled ASR data')
    parser.add_argument('--filename_txt', required=False, default=None, help='Input: File with list of filenames to process')
    parser.add_argument("--format", default="whisper", choices=["nx", "whisper"], help="Choose ASR Format: 'nx' or 'whisper'. Defaults to 'whisper'.")
    parser.add_argument('--max_interactions', type=int, default=10_000,  help='Maximum number of interactions to process')

    args = parser.parse_args()
    
    logger.info(f'Running: {" ".join(sys.argv)}')
    logger.info(f'ArgParse: {args}')
    try:
        os.makedirs('./data', exist_ok=True)

        if args.format == 'whisper':
            df = load_whisper_as_nx(args.input_csv)
        else:
            # Nx-formatted
            df = load_and_clean_nxtranscript(args.input_csv)

            num_merged_files = len(set(df.Filename))
        logger.info(f'Loaded {len(df)} lines from {len(set(df.Filename))} files in {args.input_csv}')

        # Remove path and extension
        df['Filename'] = df['Filename'].str.replace(r'.*[\\/]([^\\/\.]+)\..*', r'\1', regex=True)

        if args.filename_txt is not None:
            files_df = pd.read_csv(args.filename_txt, names=['Filename'])
            logger.info(f'Found {len(files_df)} files in {args.filename_txt}')
            df = pd.merge(files_df, df)
            num_merged_files = len(set(df.Filename))
            logger.info(f'After merge, found {len(df)} lines and {num_merged_files} files')

        # Sample calls from the input DataFrame
        sampled_df = sample_calls(df, max_calls=args.max_interactions)

        # Save the sampled calls to a CSV file
        sampled_df.to_csv(args.output_csv, index=False, sep='\t')

        # Extract conversations from the sampled DataFrame in JSON format
        conversations = get_conversations(sampled_df)
        save_convs(output_fn=args.output_json, prompt=args.input_csv, convs=conversations, save_path=True)

        logger.info(f'Successfully Completed {os.path.basename(__file__)} {args}')
    except Exception:
        logger.exception(f'Failed {os.path.basename(__file__)} {args}')
