import io
import json
import pandas as pd
import bedrock
from typing import Any, Dict, List, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
from io import StringIO
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger("research")
import csv

def categories2dataframe(categories_filename: str) -> pd.DataFrame:
    """
    Converts a file containing categories into a pandas DataFrame.

    Args:
        categories_filename (str): The path to the file containing the categories.
                                    File format is yaml-like structure

    Returns:
        pd.DataFrame: A DataFrame containing the categories, one row per L3

    """
    records = []
    with open(categories_filename) as f:
        l1=''
        l2=''
        for l in f.readlines():
            l = l.rstrip()
            m_l1 =  re.match('^- (.*)',l)
            m_l2 =  re.match('^    - (.*)',l)
            m_l3 =  re.match('^        - (.*)',l)
            if m_l1:
                l1 = m_l1.group(1)
            if m_l2:
                l2 = m_l2.group(1)
            if m_l3:
                l3 = m_l3.group(1)
                records.append({'L1':l1,'L2':l2,'L3':l3})
    cat_df = pd.DataFrame(records)
    return cat_df


def categories_dataframe2text(cat_df: pd.DataFrame) -> str:
    '''
    Convert DataFram version of categorization to yaml-style string
    '''
    L1=''
    L2=''
    outputs=[]    
    for row in cat_df.itertuples():
        if L1 != row.Level1_Category_Mapped:
            L1 = row.Level1_Category_Mapped
            outputs.append(f'- {L1}')
        if L2 != row.Level2_Topic_Mapped:
            L2 = row.Level2_Topic_Mapped
            outputs.append(f'    - {L2}')
        outputs.append(f'        - {row.Level3_CallIntents_Mapped}')
    return '\n'.join(outputs)


class IntentBuilder:
    def __init__(self, bedrock_client, cluster_prompt, assign_prompt, example_reasons:str=None, example_output:str=None) -> None:
        self.bedrock_client = bedrock_client
        self.cluster_prompt = cluster_prompt
        self.assign_prompt = assign_prompt
        self.example_reasons = example_reasons
        self.example_output = example_output
        # Count processesed tokens
        self.input_token_counter = 0
        self.output_token_counter = 0


    def cluster_reasons(self, reasons:List[str], model_id, max_l3=70) -> str:
        reasons_str = '\n'.join(reasons)
        data = {"reasons_str": reasons_str, "max_l3": max_l3}
        prompt = self.cluster_prompt.format(**data)

        categories, input_tokens, output_tokens = bedrock.simple_prompt(self.bedrock_client, prompt, model_id, max_tokens=5000)
        self.input_token_counter += input_tokens
        self.output_token_counter += output_tokens

        return categories

    def assign_reasons(self, categories_filename:str, reasons: List[str], model_id: str, start_ind=0)-> str:
        '''
            Assigns reasons from categorization and Bedrock model
            Parameters:
            categories_filename:    Path to categories
            reasons:                List of freeform reasons
            model_id:               Name of Bedrock Model
            start_ind:              Start index of reason
        '''
        cat_lines = []
        # Read categories, throw out anything that doesn't start with "-""
        with open(categories_filename) as f:
            for line in f.readlines():
                if re.match(r'^\s*-', line):
                    cat_lines.append(line.rstrip())
        if len(cat_lines) < 10:
            raise Exception(f'Invalid categorization, #lines = {len(cat_lines)}')
        categories = '\n'.join(cat_lines)
        reasons_str = '\n'.join([f'{i+start_ind},{reason}' for i,reason in enumerate(reasons)])
        data = {"reasons_str": reasons_str, "categories": categories}
        prompt = self.assign_prompt.format(**data)
        if self.example_reasons is None:
            classified, input_tokens, output_tokens = bedrock.simple_prompt(self.bedrock_client, prompt, model_id, max_tokens=5000)
        else:
            classified, input_tokens, output_tokens = bedrock.few_shot(self.bedrock_client, prompt, [(self.example_reasons, self.example_output)], model_id, max_tokens=5000)

        self.input_token_counter += input_tokens
        self.output_token_counter += output_tokens

        return classified


class IntentGenerator:
    '''
    Class to generate intents using Bedrock LLM.

    Parameters:
    - bedrock_client: The Bedrock client object.
    - prompt: The prompt template used for generating intents.
    - categories: Optional string specifying the categories in prompt.
    - num_lines: The number of lines to consider from the conversation.
    - model_id: The Bedrock ID of the model to use for intent generation.
    - min_words: The target minimum number of words in an intent.
    - max_words: The target maximum number of words in an intent.
    - max_workers: The maximum number of worker threads to use for intent generation.
    - max_tokens: The maximum number of tokens to generate for an intent.
    - additional_instructions: Additional instructions for intent generation (added to prompt).

    Attributes:
    - input_token_counter: The count of processed input tokens.
    - output_token_counter: The count of processed output tokens.
    '''
    def __init__(self, bedrock_client, prompt: str, categories:Optional[str], num_lines:int, model_id:str, min_words=5, max_words=10,max_workers = 10, max_tokens=256, additional_instructions='') -> None:
        self.client = bedrock_client
        self.categories = categories
        self.prompt = prompt
        self.num_lines=num_lines
        self.min_words = min_words
        self.max_words = max_words
        self.model_id = model_id
        self.max_workers = max_workers
        self.additional_instructions = additional_instructions
        # Count processesed tokens
        self.input_token_counter = 0
        self.output_token_counter = 0

    def generate_reason(self, record:Dict) -> Dict[str, Any]:
        '''
        Get the intents using full categorization of intents.

        Parameters:
        - record: The record containing conversation data.

        Returns:
        - A dictionary containing the conversation record and the generated intent.
        '''
        conv = '\n'.join(record['Conv'].split('\n')[0:self.num_lines])

        # Data for formatting prompt
        data = {"categories": self.categories, "conv": conv, "min_words": self.min_words, 
                                "max_words": self.max_words, "additional_instructions":self.additional_instructions}
        prompt_filled = self.prompt.format(**data)
        max_tokens = 256
        try:
            intent, input_tokens, output_tokens = bedrock.simple_prompt(self.client, prompt_filled, self.model_id, max_tokens=max_tokens)
            self.input_token_counter += input_tokens
            self.output_token_counter += output_tokens
        except Exception as ex:
            return {'Conv': record, 'Intent': f"Error: {ex}"}
        return {'Conv': record, 'Intent': intent}

    def collect_reasons(self, input_json, output_json, max_interactions=1000):
        '''
        Read input transcript json and add 'Intent' to each record in output json.

        Parameters:
        - input_json: The path to the input transcript JSON file.
        - output_json: The path to the output JSON file.
        - max_interactions: The maximum number of interactions to process.
        '''
        with open(input_json) as f:
            j = json.load(f)
        output = {'Prompt': self.prompt, "Responses":[]}
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.generate_reason, record) 
                                                        for i,record in enumerate(j['Responses'][0:max_interactions])]
            for i,response in enumerate(tqdm(as_completed(futures), total=len(futures))):
                result = response.result()
                if not "Customer:" in result['Intent']:
                    output['Responses'].append(result)
                if (i + 1) % 50 == 0:
                    with lock:
                        with open(output_json, 'wt') as f:
                            json.dump(output, f, indent=4)
        with open(output_json, 'wt') as f:
            json.dump(output, f, indent=4)


    def create_intent_csv(self, json_filename: str, csv_filename:str, additional_columns: List[str]) -> None:
        '''
        Loads JSON intents and converts to CSV table.

        Parameters:
        - json_filename: The path to the JSON file containing intents.
        - csv_filename: The path to the CSV file to be created.
        - additional_columns: Additional columns to include in the CSV file (after Intent,LineNumber).
        '''

        def remove_extra_elements(input_string: str, num_initial: int, num_final: int) -> str:
            # Split the input string into a list of elements
            elements = list(csv.reader(StringIO(input_string)))[0]

            if len(elements) < num_initial + num_final:
                raise ValueError('Not enough values for remove_extra_elements')

            # Put quotes around everyting
            elements = [f'"{e}"' for e in elements]

            # Join the final elements into a string and return it
            if num_final == 0:
                return ','.join(elements[:num_initial])
            else:
                return ','.join(elements[:num_initial]+elements[-num_final:])

        def convert_record(record:Dict, additional_columns:List[str])-> Optional[pd.DataFrame]:
            ''' Convert Intent output to DataFrame '''
            try:
                # Just get the relevant lines, strip off any additional lines except the first
                intent_lines = [ remove_extra_elements(line, 2, len(additional_columns))
                                            for line in record['Intent'].split('\n')
                                            if re.match(r'^[^,]+,\d+.*',line)]   
                intent_io = StringIO('\n'.join(intent_lines))
                # Construct dataframe, in order of interaction, only keep first example of any intent
                df = (pd.read_csv(intent_io, names=['Intent','LineNumber'] + additional_columns,
                                dtype={'Intent':str, 'LineNumber':int}, on_bad_lines='skip')
                    .sort_values('LineNumber')
                    .drop_duplicates('Intent')
                    )
                # Clean up "Intent" language
                df['Intent'] = (df['Intent'].str.replace('_',' ')
                                        .str.replace(r'^# *','',regex=True)
                                        .str.replace(r'^[Ii]ntent( for | to )?[ :0-9]*','', regex=True)
                                )
                df['Intent'] = df['Intent'].astype(str).map(lambda x: x[0].upper() + x[1:] if len(x) > 0 else x)
                # Throw out rows with no real intents
                df = df[df['Intent'].str.len() > 5]
                # And one last drop_duplicates if edits have introduced new dups
                df.drop_duplicates('Intent', inplace=True)
                # All lines have same filename
                df['Filename'] = re.sub(r'.*[\\/]([^\\/\.]+)\..*', r'\1', record['Conv']['Path'])
            except Exception as ex:
                return None
            return df

        with open(json_filename) as f:
            j = json.load(f)
        
        df = pd.concat([convert_record(r, additional_columns) for r in j['Responses']])
        logger.info(f'Writing {len(df)} lines to {csv_filename}')
        df.to_csv(csv_filename, sep='\t', index=False)

class LiaMappingGenerator:
    """
    A class that generates Text-Tools Excel mapping file

    Args:
        bedrock_client: The Bedrock client object.
        prompt (str): The prompt string for generating phrasings with verbatims.
        prompt_noverbatims (str): The prompt string for generating phrasings without verbatims.
        categories_df (pd.DataFrame): The DataFrame containing the categories.
        model_id (str): The ID of the model to use for generating phrasings.
        max_workers (int, optional): The maximum number of worker threads to use for parallel processing. Defaults to 10.
        max_tokens (int, optional): The maximum number of tokens to use for generating phrasings. Defaults to 256.
        max_verbatims_per_prompt (int, optional): The maximum number of verbatims to use per prompt. Defaults to 50.
        max_phrase_count (int, optional): The maximum number of phrases to generate per intent. Defaults to 10.
    """
    def __init__(self, bedrock_client, prompt: str, prompt_noverbatims: str, categories_df:pd.DataFrame, model_id:str, 
                 example_data:List[Dict]=[], max_workers = 10, max_tokens=256, max_verbatims_per_prompt = 50, max_phrase_count=10,
                 min_phrase_len = 2, max_phrase_len = 4) -> None:
        self.client = bedrock_client
        self.categories_df = categories_df
        self.prompt = prompt
        self.prompt_noverbatims = prompt_noverbatims
        self.model_id = model_id
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.max_verbatims_per_prompt = max_verbatims_per_prompt
        self.max_phrase_count = max_phrase_count
        self.min_phrase_len = min_phrase_len
        self.max_phrase_len = max_phrase_len
        self.example_data = example_data
        # Constants
        self.excel_category_cols = ['Level1_Category_Mapped',  'Level2_Topic_Mapped', 'Level3_CallIntents_Mapped']
        self.excel_output_cols = ['Category_Orig','Topic_Orig','Level1_Category_Mapped','Level2_Topic_Mapped',
                        'Level3_CallIntents_Mapped','AgentTaskLabel_Mapped','terms','anti_terms','call_driver']

        # Count processesed tokens
        self.input_token_counter = 0
        self.output_token_counter = 0
        # Additional tracking
        self.no_verbatim_intents: List[str] = []
        # TODO: Make this configurable!
        self.prompt_nosuggest = '''
I have a 3-level categorization of phrases in a call center in the Categories section.  
I also have a csv ind the Data section in the format: line number, phrase.
Please create a headerless CSV with the following fields: line, phrase, level 3 category, score 1-5 of certainty
No additional commentary please!
# Categories
{categories}
# Data
{data}
'''


    def get_verbatims(self, L1: str, L2: str, L3: str, intent_merged_df: pd.DataFrame)-> str:
        """
        Retrieves verbatims from the given DataFrame based on the provided L1, L2, and L3 values.

        Parameters:
        - L1 (str): Level 1 string to find.
        - L2 (str): Level 2 string to find.
        - L3 (str): Level 3 string to find.
        - intent_merged_df (pd.DataFrame): The output of _step3_map_intents with verbatims.

        Returns:
        - str: A string containing the matched verbatims joined by newline characters.

        """
        # Get matches to L1, L2, L3, and not NaN
        matched_rows = intent_merged_df[intent_merged_df[['L1','L2','L3']].eq([L1, L2, L3]).all(axis=1)].dropna()
        if len(matched_rows) == 0:
            return ''
        else:
            # Try limiting to L3_Score == 5 if there are a reasonable number of them
            if len(matched_rows) > self.max_verbatims_per_prompt:
                high_score_rows = matched_rows[matched_rows['L3_Score'] == 5]
                if len(high_score_rows) > 0.3 * len(matched_rows):
                    matched_rows = high_score_rows
                # Sample new set down to max verbatims per prompt
                if len(matched_rows) > self.max_verbatims_per_prompt:
                    matched_rows = matched_rows.sample(n=self.max_verbatims_per_prompt)
            return '\n'.join(matched_rows['Text'])

    def valid_phrase(self, phrase:str) -> bool:
        """
        Check if a phrase is valid for Text-Tools (no numbers, etc.).

        Args:
            phrase (str): The phrase to be checked.

        Returns:
            bool: True if the phrase is valid, False otherwise.
        """        
        if re.match('.*[0-9].*', phrase):
            return False
        return True

    def build_prompt(self, L1, L2, L3, intents_merged_df: pd.DataFrame) -> str:
        '''
        Build prompt for the intent
        '''
        intent = [L1, L2, L3]
        all_verbatims = self.get_verbatims(L1,L2,L3,intents_merged_df)
        all_others = '\n'.join(set(intents_merged_df.L3.dropna()).difference([L3]))
        intent_name = '-'.join(intent)
        data = {"all_others": all_others, "intent_name": intent_name, "all_verbatims": all_verbatims,
                "min_phrase_len": self.min_phrase_len, "max_phrase_len": self.max_phrase_len, "max_phrase_count": self.max_phrase_count}
        if len(all_verbatims) == 0:
            prompt_filled = self.prompt_noverbatims.format(**data)
            self.no_verbatim_intents.append(intent_name)
        else:
            prompt_filled = self.prompt.format(**data)
        return prompt_filled

    def create_shots(self, intent_merged_df: pd.DataFrame)->List[Tuple[str,str]]:
        """
        Convert examples into prompt-response pairs for multi-shot prompting
        """
        return [(self.build_prompt(element['L1'],element['L2'],element['L3'],intent_merged_df), 
                                                        element['Phrases']) 
                                                        for element in self.example_data]


    def get_phrasings(self, L1: str, L2: str, L3: str,intent_merged_df: pd.DataFrame) -> Dict[str,str]:
        """
        Retrieves phrasings for a given intent.

        Args:
            L1 (str): The first level of the intent.
            L2 (str): The second level of the intent.
            L3 (str): The third level of the intent.
            intent_merged_df (pd.DataFrame): The output of _step3_map_intents with verbatims.

        Returns:
            dict: A dictionary containing the intent levels and the normalized phrasings.
        """
        prompt_filled = self.build_prompt(L1, L2, L3, intent_merged_df)
        examples = self.create_shots(intent_merged_df)
        phrasings, input_tokens, output_tokens = bedrock.few_shot(self.client, prompt_filled,examples, self.model_id, max_tokens=5000)
        self.input_token_counter += input_tokens
        self.output_token_counter += output_tokens
        all_phrasings = [p.strip().lower() for p in phrasings.split(',')]
        filtered_phrasings = [p for p in all_phrasings if self.valid_phrase(p)][0:self.max_phrase_count]
        normalized_phrasings = ','.join(filtered_phrasings)
        return {'L1': L1, 'L2': L2, 'L3':L3, 'phrasings' : normalized_phrasings}


    def get_phrasings_nothrow(self, L1: str, L2: str, L3: str,intent_merged_df: pd.DataFrame) -> Dict[str,str]:
        """
        Retrieves phrasings based on the provided L1, L2, L3 values from the intent_merged_df DataFrame.
        If an exception occurs during the retrieval process, it logs a warning and returns a dictionary
        with the failed L1, L2, L3 values and the error message.

        Parameters:
        - L1 (str): The value for L1.
        - L2 (str): The value for L2.
        - L3 (str): The value for L3.
        - intent_merged_df (pd.DataFrame): The output of _step3_map_intents with verbatims.

        Returns:
        - dict: A dictionary containing the retrieved phrasings or the failed L1, L2, L3 values and error message.
        """
        try:
            return self.get_phrasings(L1, L2, L3, intent_merged_df)
        except Exception as ex:
            logger.warning(f'Failed on intent {L1}-{L2}-{L3}: {ex}')
            return {'L1': L1, 'L2': L2, 'L3':L3, 'phrasings' : f'Failed {ex}'}

    def collect_phrases(self, intent_merged_df: pd.DataFrame) -> List[Dict[str,str]]:
        """
        Collects phrases using multiple threads and returns a list of dictionaries.

        Args:
            intent_merged_df (pd.DataFrame): The merged dataframe containing intent data.

        Returns:
            List[Dict[str,str]]: A list of dictionaries containing the collected phrases.
        """
        output=[]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.get_phrasings_nothrow, row.L1,row.L2,row.L3,intent_merged_df) 
                                                        for row in self.categories_df.itertuples()]
            for response in tqdm(as_completed(futures), total=len(futures)):
                output.append(response.result())
        return output

    def choose_category(self, term:str, group: pd.DataFrame)-> Dict[str,str]:
        """
        Choose a category for a given term.

        Args:
            term (str): The term for which a category needs to be chosen.
            group (pd.DataFrame): The DataFrame containing the categories.

        Returns:
            dict: A dictionary containing the chosen category and its corresponding columns, along with the term.

        """
        prompt= '''
I have a term "{term}" that I need to choose a category.  Please output the best matching line without additional commentary.  The output choinces are given below one per line:
{categories}
'''
        categories = '\n'.join(( group[self.excel_category_cols[0]] + ' - ' + 
                                                group[self.excel_category_cols[1]] + ' - ' + 
                                                group[self.excel_category_cols[2]]))
        data = {'term': term, 'categories': categories}
        prompt_filled = prompt.format(**data)
        category,it,ot = bedrock.simple_prompt(self.client, prompt_filled, self.model_id, max_tokens=500)
        self.input_token_counter += it
        self.output_token_counter += ot
        category_split = category.split(' - ')
        return {self.excel_category_cols[0]: category_split[0], 
                self.excel_category_cols[1]: category_split[1], 
                self.excel_category_cols[2]: category_split[2],
                'terms_split':term}
    
    def remove_duplicates_mapping(self, input_df:pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate mappings from the input DataFrame representing Text-Tools mapping file.

        Args:
            input_df (pd.DataFrame): The input DataFrame containing Text-Tool mapping data.

        Returns:
            pd.DataFrame: The output DataFrame with duplicate mappings removed.
        """
        df = input_df.copy()
        df['terms_split'] = df['terms'].str.split(',')
        # Explode out the terms per category
        exploded_df = df[self.excel_category_cols + ['terms_split']].explode('terms_split')
        exploded_df['num_intents'] = exploded_df.groupby('terms_split')['terms_split'].transform('size')
        # Get terms that occur in multiple categories
        repeats_df = exploded_df[exploded_df['num_intents'] > 1].sort_values(['num_intents','terms_split'])

        # For each of the repeated terms, use LLM to select best category
        logger.info(f'Pick categories for {len(set(repeats_df.terms_split))} ambiguous phrases')
        category_records=[]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.choose_category, term, group) 
                                                        for term,group in repeats_df.groupby('terms_split')]
            for response in tqdm(as_completed(futures), total=len(futures)):
                category_records.append(response.result())
        category_choices = pd.DataFrame(category_records)
        # Merge the category selection data with the exploded terms frame
        merged_df = pd.merge(exploded_df, category_choices, how='left', 
                                        on=self.excel_category_cols+['terms_split'], indicator='how')
        # Select terms that are unique or selected from group
        merged_df['unique']=merged_df['num_intents']==1
        merged_df['selected']=merged_df['how']=='both'
        exploded_nodups_df = merged_df.query('unique or selected')

        # Create DataFrame with the updated terms
        new_terms_df = exploded_nodups_df.groupby(self.excel_category_cols)['terms_split'].agg(','.join).reset_index()
        if len(new_terms_df) != len(input_df):
            logger.warning(f'Lost categories in repeat removal ({len(new_terms_df)} != {len(input_df)})')
        updated_df = pd.merge(input_df, new_terms_df, on=self.excel_category_cols, validate='1:1')
        updated_df['terms'] = updated_df['terms_split']
        # Put back to correct column order
        output_df = updated_df[self.excel_output_cols].copy()
        return output_df


    def check_phrases(self, phrase_df: pd.DataFrame, cat_str:str, offset:int, num_phrases:int)-> pd.DataFrame:
        '''
        Take existing phrases with categories and ask LLM to produce category and scores
        
        Args:
            phrase_df: Must include line and terms_split
            cat_str: L1 - L2 - L3 formatted categorization
            offset: offset into phrase_df
            num_phrases: Number of phrases to process
        '''
        # Get the disjoint example input phrases
        num_examples = 10
        example_input_df = phrase_df[(phrase_df.line < offset)|(phrase_df.line > offset+num_phrases)].sample(num_examples)
        example_input_string = io.StringIO()
        example_input_df[['line','terms_split']].to_csv(example_input_string, index=False, header=False)
        example_input_nosuggest = example_input_string.getvalue()
        prompt_example_nosuggest = self.prompt_nosuggest.format(**{'categories':cat_str, 'data':example_input_nosuggest})
        output_string = io.StringIO()
        # For the example output, just assume that the output is correct and set the score to 5
        example_input_df['score'] = 5
        example_input_df[['line','terms_split','Level3_CallIntents_Mapped', 'score']].to_csv(output_string, index=False, header=False)
        short_output = output_string.getvalue()
        example_shot = [(prompt_example_nosuggest,short_output)]

        labels_string = io.StringIO()
        subset_df = phrase_df.iloc[offset:(offset+num_phrases)][['line','terms_split']]
        subset_df.to_csv(labels_string, index=False, header=False)
        all_input_nosuggest = labels_string.getvalue()
        prompt_filled_nosuggest = self.prompt_nosuggest.format(**{'categories':cat_str, 'data':all_input_nosuggest})
        recat_1shot_nosuggest,input_tokens,output_tokens=bedrock.few_shot(self.client,prompt_filled_nosuggest,example_shot,max_tokens=2000)
        self.input_token_counter += input_tokens
        self.output_token_counter += output_tokens
        recat_1shot_nosuggest_df = pd.read_csv(io.StringIO(recat_1shot_nosuggest), names=['line', 'terms_split', 'llm_L3_Label','llm_score'])
        merged_df = pd.merge(phrase_df, recat_1shot_nosuggest_df, on=['line','terms_split'])
        return merged_df


    def clean_phrase_mapping(self, input_df:pd.DataFrame) -> pd.DataFrame:
        """
        Remaps phrases based on categorization.  Follow the logic below on output of LLM categorization
        * Drop `llm_score` <= 3
        * Drop `llm_score` == 4 and changed
        * Drop if `llm_L3_Label` is not in original L3 set or is not equal to `Level2_Topic_Mapped`
        * Final output is `Level_CallIntents_Mapped` if in L3 set or `Level3_CallIntents_Mapped` (keep changes if score is 5)
        * Notate any dropped topics

        Args:
            input_df (pd.DataFrame): The input DataFrame containing Text-Tool mapping data.

        Returns:
            pd.DataFrame: The output DataFrame with updating phrases.

        """
        input_df['terms_split'] = input_df['terms'].str.split(',')
        cat_terms_df = input_df.explode('terms_split')
        cat_terms_df['line'] = range(len(cat_terms_df))
        cat_flat = '\n'.join(input_df['Level1_Category_Mapped'] + ' - ' + input_df['Level2_Topic_Mapped'] + ' - ' + input_df['Level3_CallIntents_Mapped'])

        output_dfs = []
        chunk_size = 50
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.check_phrases, 
                                                                cat_terms_df, cat_flat, offset, chunk_size)
                                            for offset in range(0,len(cat_terms_df),chunk_size)]
            for response in tqdm(as_completed(futures), total=len(futures)):
                output_dfs.append(response.result())

        update_phrases_df = pd.concat(output_dfs)[['terms_split','Level1_Category_Mapped','Level3_CallIntents_Mapped','Level2_Topic_Mapped','llm_L3_Label','llm_score']]
        update_phrases_df['llm_L3_Label'] = update_phrases_df['llm_L3_Label'].str.replace('.*- ','', regex=True)
        update_phrases_df['change'] = update_phrases_df['Level3_CallIntents_Mapped'] != update_phrases_df['llm_L3_Label']
        valid_l3 = set(update_phrases_df['Level3_CallIntents_Mapped'])
        update_phrases_df['valid_change'] = update_phrases_df['llm_L3_Label'].isin(valid_l3)
        keep_phrases =  (((update_phrases_df.llm_score == 5) & ((update_phrases_df.llm_L3_Label.isin(valid_l3)) | 
                                                                (update_phrases_df.llm_L3_Label==update_phrases_df.Level2_Topic_Mapped))) |
                                                                ((update_phrases_df.llm_score == 4) & (~update_phrases_df.change)))
        final_phrases_df = update_phrases_df[keep_phrases].copy()
        # If changing to L2, keep L3 label
        final_phrases_df['Final_L3_Label'] = np.where(final_phrases_df.llm_L3_Label.isin(valid_l3), final_phrases_df.llm_L3_Label, final_phrases_df.Level3_CallIntents_Mapped)
        logger.info(f'Dropped {len(update_phrases_df)-len(final_phrases_df)} and changed {final_phrases_df.change.sum()} of {len(update_phrases_df)} phrases')
        logger.info(f'Dropped {len(valid_l3)-len(set(final_phrases_df.Final_L3_Label))} topics')
        # Need to drop the L1 and L2 and re-infer from the cat_df
        final_phrases_df['Level3_CallIntents_Mapped'] = final_phrases_df['Final_L3_Label']
        final_phrases_df.drop(['Level1_Category_Mapped','Level2_Topic_Mapped'], axis=1, inplace=True)
        final_phrases_df = pd.merge(final_phrases_df, input_df[[ 'Level1_Category_Mapped', 'Level2_Topic_Mapped','Level3_CallIntents_Mapped']], on='Level3_CallIntents_Mapped')
        final_phrases_df['terms'] = final_phrases_df['terms_split']
        df = final_phrases_df.groupby(['Level1_Category_Mapped', 'Level2_Topic_Mapped','Level3_CallIntents_Mapped'])['terms'].agg(','.join).reset_index()
        # create LIA output
        # Make category names more consistent with LIA
        df['Category_Orig'] = df['Level2_Topic_Mapped']
        df['Topic_Orig'] = df['Level3_CallIntents_Mapped']
        df['AgentTaskLabel_Mapped'] = '*drop*'
        df['anti_terms'] = ''
        df['call_driver'] = ''
        # Sort output and get the correct coluns
        output_df = df.sort_values(['Level1_Category_Mapped', 'Level2_Topic_Mapped','Level3_CallIntents_Mapped'])[self.excel_output_cols]
        # output_df.drop_duplicates('Level3_CallIntents_Mapped', inplace=True)
        logger.info(f'Original topics = {len(valid_l3)}, updated: {len(output_df)}')
        # Write to Excel
        return output_df

    def create_excel_mapping(self, output_filename: str, phrasings: List[Dict[str,str]]) -> None:
        """
        Create an Excel Text-Tools mapping file based on the provided phrasings.

        Args:
            output_filename (str): The name of the output TT Excel file.
            phrasings (List[Dict[str,str]]): A list of dictionaries mapping categories to phrasings.

        Returns:
            None
        """
        df = pd.DataFrame(phrasings).drop_duplicates(['L1','L2','L3'],keep='last')
        # create LIA output
        # Make category names more consistent with LIA
        df['Category_Orig'] = df['L2']
        df['Topic_Orig'] = df['L3']
        df['Level1_Category_Mapped'] = df['L1'].str.upper()
        df['Level2_Topic_Mapped'] = df['L2']
        df['Level3_CallIntents_Mapped'] = df['L3']
        df['AgentTaskLabel_Mapped'] = '*drop*'
        df['terms'] = df['phrasings']
        df['anti_terms'] = ''
        df['call_driver'] = ''
        # Sort output and get the correct coluns
        output_df = df.sort_values(['L1','L2','L3'])[self.excel_output_cols]

        # Remove duplicate terms
        cleaned_df = self.remove_duplicates_mapping(output_df)
        
        # Remap phrases
        remapped_df = self.clean_phrase_mapping(cleaned_df)

        # Write to Excel
        remapped_df.to_excel(output_filename, index=False)


def clean_intents(intents_df: pd.DataFrame, categories_df:pd.DataFrame) -> pd.DataFrame:
    """
    Updates intents DataFrame by performing various clean-up operations.

        1) Remove multiple intents (end up in L3 with a " / ")
        2) Fix incorrect L2,L1.  In this case L3 is uniquely determined from L1/L2, may not always be case
        3) Missing L3, makes it NaN but leave L2 an L1 intact
        4) L3 == L2, should be treated same as L3 == NaN
        5) Various forms of "Don't know"
        6) New L3 intents
        7) Correctness of L1,L2,L3 not monotonic
        Fix 4) by setting L3 to NaN if equal to L2

    Args:
        intents_df (pd.DataFrame): The DataFrame containing the intents data.
        categories_df (pd.DataFrame): The DataFrame containing the categories data.

    Returns:
        pd.DataFrame: The cleaned DataFrame.

    """
    cleaned_df = intents_df.copy()

    # Intents Split on " - "
    cleaned_df[['L1','L2','L3']] = cleaned_df['Intent_Category'].str.strip().str.split(' - ', expand=True,n=2)
    cleaned_df.loc[cleaned_df['L2'] == cleaned_df['L3'], 'L3'] = np.nan
    # Fux 1) by Remove extra intents
    cleaned_df['L3'] = cleaned_df['L3'].str.replace(r' / .*','', regex=True)
    # Fix 2) by merging with categorization to find correct L1,L2
    catdata_df = cleaned_df.groupby(['L1','L2','L3']).size().reset_index()
    cat_combined = pd.merge(categories_df,catdata_df, how='right',indicator='how').sort_values(['L1','L2','L3'])
    mask = cat_combined['how'] == 'both'
    # Make mapping from L3 to correct L1 and L2
    correct_l2 = cat_combined[mask].groupby('L3')['L2'].first()
    correct_l1 = cat_combined[mask].groupby('L3')['L1'].first()
    # Map the correct L1 value to the original DataFrame
    # Fix it in the df only if L3 is notna (3)
    cleaned_df.loc[cleaned_df['L3'].notna(), 'L1'] = cleaned_df['L3'].dropna().map(correct_l1)
    cleaned_df.loc[cleaned_df['L3'].notna(), 'L2'] = cleaned_df['L3'].dropna().map(correct_l2)
    # Get updated cat_combined
    catdata_df = cleaned_df.groupby(['L1','L2','L3']).size().reset_index()
    cat_combined = pd.merge(categories_df,catdata_df, how='right',indicator='how').sort_values(['L1','L2','L3'])
    return cleaned_df
