import argparse
import os
import sys
os.makedirs('./logs', exist_ok=True)
import logging
import logging.config
logging.config.fileConfig('./logger.ini')
logger = logging.getLogger("research")
import json
from tqdm import tqdm
import pandas as pd
import re
from intents import clean_intents, categories2dataframe
from transcripts import transcript2DataFrame
from bedrock import simple_prompt


def extract_fn(path):
    return re.sub(r'^.*[\\/]([^\\/\\.]*).*$', r'\1', path.strip())

def create_excel_sheet(output_path: str, filename2media: dict, with_times_df: pd.DataFrame, additional_columns: list):
    output_col = ['Best', 'Accept', 'Notes', 'Intent', 'L1', 'L2', 'L3', 'L3_Score'] + additional_columns + ['Party', 'Start_sec', 'Text']
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        startrow = 3
        for filename, intent_df in tqdm(with_times_df.groupby('Filename')):
            mediafile = filename2media.get(filename)
            df = intent_df.head(30).copy()
            df['Best'] = ''
            df['Accept'] = ''
            df['Notes'] = ''
            df[output_col].to_excel(writer, startrow=startrow, index=False)
            link_formula = f'=HYPERLINK("file:///{mediafile}", "{filename}")'
            sheet = writer.book.active
            sheet.cell(row=startrow, column=1).value = link_formula
            startrow += len(df.index) + 3
        sheet.column_dimensions['D'].width = 35
        sheet.column_dimensions['E'].width = 25
        sheet.column_dimensions['F'].width = 25
        sheet.column_dimensions['G'].width = 30
        sheet.column_dimensions['I'].width = 10

def sample_files(df, filenames_txt, num_interactions, random_state=42):
    df_filtered = df.query('L3_Score >= 4').copy()
    df_filtered['NumIntent'] = df_filtered.groupby('Filename')['Filename'].transform('count')
    df_filtered = df_filtered[(df_filtered.NumIntent > 2) | (df_filtered['L2'] != "Personal Details")].copy()
    if filenames_txt:
        with open(filenames_txt) as f:
            filename_sample = {line.strip() for line in f.readlines()}
        logger.info(f'Selecting {len(filename_sample)} files from {filenames_txt}')
    else:
        filename_sample = set(df_filtered['Filename'].sample(n=num_interactions, random_state=random_state, replace=False))
        logger.info(f'Selecting {len(filename_sample)} samples randomly from original {len(set(df_filtered.Filename))}')
    df_sampled = df_filtered[df_filtered['Filename'].isin(filename_sample)]
    return filename_sample, df_sampled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Excel validation spreadsheet")
    parser.add_argument('--categorization_map_csv', required=True, help='TSV file from _step3_map_intents.py')
    parser.add_argument('--reasons_eval_json', required=True, help='JSON file with reason assignments')
    parser.add_argument('--categories_txt', required=True, help='YAML-style category mapping')
    parser.add_argument('--media_files_unc', required=False, default=None, help='Optional list of UNC paths to media')
    parser.add_argument('--asr_csv_filename', required=False, default=None, help='Optional ASR file with timestamp info')
    parser.add_argument('--output_excel', required=True, help='Excel file to write output')
    parser.add_argument('--filenames_txt', type=str, default=None, help='Optional list of filenames to include')
    parser.add_argument('--num_interactions', type=int, default=100, help='Number of interactions to sample')
    parser.add_argument('--additional_columns', nargs='*', default=[], help='Additional columns to include')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--model_id', required=False, default='anthropic.claude-3-sonnet-20240229-v1:0', help='Bedrock model ID')
    args = parser.parse_args()

    logger.info(f'Running: {" ".join(sys.argv)}')
    try:
        logger.info(f'Loading {args.reasons_eval_json}')
        with open(args.reasons_eval_json) as f:
            j = json.load(f)

        logger.info(f'Loading {args.categorization_map_csv}')
        df = pd.read_csv(args.categorization_map_csv, sep='\t').dropna()
        logger.info(f'Loading {args.categories_txt}')
        cat_df = categories2dataframe(args.categories_txt)
        filename2media = {}
        if args.media_files_unc:
            logger.info(f'Loading media file paths from {args.media_files_unc}')
            with open(args.media_files_unc) as f:
                for line in f:
                    filename2media[extract_fn(line)] = line.strip()
        else:
            logger.info('Skipping media file mapping')
        if args.asr_csv_filename:
            logger.info(f'Loading offsets from {args.asr_csv_filename}')
            orig_asr_df = pd.read_csv(args.asr_csv_filename, sep='\t')
            orig_asr_df = orig_asr_df.rename({
                'StartOffset (sec)': 'Start_sec',
                'start': 'Start_sec',
                'Path': 'Filename',
                'Line': 'LineNumber'
            }, axis=1)[['Filename', 'LineNumber', 'Start_sec']]
            orig_asr_df['Filename'] = orig_asr_df['Filename'].str.replace(r'.*[\\/]([^\\.\\/]+)\\..*', r'\1', regex=True)
            orig_asr_df.drop_duplicates(['Filename', 'LineNumber'], inplace=True)
        else:
            logger.info('Skipping ASR offset mapping')
            orig_asr_df = None

        logger.info('Creating spreadsheet')
        df = clean_intents(df, cat_df)
        filename_sample, df_sampled = sample_files(df, args.filenames_txt, args.num_interactions, args.random_state)
        responses = [r for r in j['Responses'] if extract_fn(r['Conv']['Path']) in filename_sample]
        transcript_df = pd.concat([transcript2DataFrame(r['Conv']['Conv'], extract_fn(r['Conv']['Path'])) for r in responses])
        keep_col = ['LineNumber', 'Party', 'Text', 'Filename', 'Intent', 'L1', 'L2', 'L3', 'L2_Score', 'L3_Score'] + args.additional_columns
        df_merged = pd.merge(transcript_df, df_sampled, on=['Filename', 'LineNumber'], how='left')[keep_col]

        logger.info(f'Validating L3 assignments using {args.model_id}')
        for i, row in df_merged.iterrows():
            prompt = f"""
Assign a confidence score (1-5) for how well the following text fits into the category: {row['L3']}

Category context:
- L1: {row['L1']}
- L2: {row['L2']}
- L3: {row['L3']}

Text:
""" + row['Text']
            try:
                response, _, _ = simple_prompt(None, prompt, model_id=args.model_id, max_tokens=100)
                score = int(re.search(r'[1-5]', response).group())
                df_merged.at[i, 'L3_Score'] = score
            except:
                df_merged.at[i, 'L3_Score'] = 3

        if orig_asr_df is None:
            with_times_df = df_merged.copy()
            with_times_df['Start_sec'] = with_times_df['LineNumber']
        else:
            with_times_df = pd.merge(df_merged, orig_asr_df, how='left')
            with_times_df['Start_sec'].fillna(with_times_df['LineNumber'])

        create_excel_sheet(args.output_excel, filename2media, with_times_df, args.additional_columns)
        logger.info(f'Successfully Completed {os.path.basename(__file__)}')
    except Exception:
        logger.exception(f'Failed {os.path.basename(__file__)}')
