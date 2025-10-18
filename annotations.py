
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import asdict, dataclass
import pandas as pd

@dataclass
class Call:
    """
    Represents a call with its path and conversation.

    Attributes:
        Path (str): The path of the call.
        Conv (str): The conversation of the call.
    """
    Path: str
    Conv: str

@dataclass
class AnnotatedCall(Call):
    """
    Represents an annotated call with a summary.

    Attributes:
        Summary (str): The summary of the annotated call.
    """
    Summary: str

@dataclass
class TranscriptTurn:
    """
    Represents a single turn in a transcript.

    Attributes:
        Line (int): The line number of the turn.
        Party (str): The party associated with the turn.
        Text (str): The text of the turn.
    """
    Line: int
    Party: str
    Text: str

@dataclass
class Transcript:
    """
    Represents a transcript.

    Attributes:
        Path (str): The path of the transcript.
        Turns (List[TranscriptTurn]): The list of transcript turns.
    """
    Path: str
    Turns: List[TranscriptTurn]

@dataclass
class Annotation:
    """
    Represents an annotation for a specific section of transcript.

    Attributes:
        Path (str): The path of the file containing the code snippet.
        Lines (List[int]): The line numbers of the code snippet.
        Annotation (str): The annotation message.
        Type (str): The type of the annotation.
    """
    Path: str
    Lines: List[int]
    Annotation: str
    Type: str

def text2turns(text_transcript: str) -> List[TranscriptTurn]:
    ''' Convert text formatted transcript into TranscriptTurns

    Args:
        text_transcript (str): The text formatted transcript to convert.

    Returns:
        List[TranscriptTurn]: A list of TranscriptTurn objects.
    '''
    tuples = []
    for t in text_transcript.split('\n'):
        match = re.match(r'^(\d+): (\w+): (.*)', t)
        if match is not None:
            tuples.append(match.groups())
    return [TranscriptTurn(int(l), p, t) for l, p, t in tuples]

def call2dataframe(call: Call) -> pd.DataFrame:
    """
    Convert a Call object to a pandas DataFrame.
    """
    df = pd.DataFrame(text2turns(call.Conv))
    df['Path'] = call.Path
    return df

def line2annotation(summary_line: str, path: str, type: str="") -> Optional[Annotation]:
    '''
    Create an annotation from a raw summary line: <text> (Lines 3-5) <text)

    Args:
        summary_line (str): The raw summary line containing the annotation information.
        path (str): The path of the file associated with the annotation.
        type (str, optional): The type of the annotation. Defaults to "".

    Returns:
        Annotation: The created annotation object (or None if can't parse).

    Raises:
        None
    '''
    try:
        # Regular expression to match the pattern in the data
        # The three capturing groups are set up to extract the text before the parenthesized section, 
        # the number or range within the parentheses, and the text after the parentheses
        pattern = r'(?:[ \*]*)(.*?) \((?:[lL](?:ine|ines)? )?([\d, \-]+)\)(.*)'
        # Process each item in the data
        # Handle (<>)
        summary_line = re.sub(r'\(<','(',summary_line)
        summary_line = re.sub(r'>\)',')',summary_line)
        match = re.search(pattern, summary_line)
        line_nums = []
        if match:
            text, line_info, post_text = match.groups()
            text = text + post_text
            for line_sub in re.split('[, ]+', line_info):
                # Check if line_info contains a range or multiple line numbers
                if '-' in line_sub:  # It's a range
                    line_data = [int(num) for num in line_sub.split('-') if num]
                    if len(line_data) == 2:
                        start, end = line_data
                        line_nums.extend(range(start, end + 1))
                else:  # It's individual line numbers
                    lines = [int(num) for num in re.split('[, ]+', line_sub) if num]
                    line_nums.extend(lines)
            return Annotation(path, line_nums, text, type)
        else:
            return None
    except Exception as ex:
        print(ex)
        return None

def call2annotations_notype(call: AnnotatedCall) -> List[Annotation]:
    ''' Get all annotations from a call '''
    # Split and convert to annotations
    annotations = [line2annotation(line, call.Path) for line in call.Summary.split('\n')]
    # Remove the Nones
    return [a for a in annotations if a is not None]

def annotations2dataframe(annotations: List[Annotation]) -> pd.DataFrame:
    ''' Convert list of annotations into a dataframe'''
    # Get row per annotation, explode to row per line, then rename Lines to Line
    if len(annotations) == 0:
        return pd.DataFrame(columns=['Path','Line','Annotation'])
    return pd.DataFrame(annotations).explode('Lines').rename({'Lines':'Line'}, axis=1)

def annotatedcall2dataframe(call: AnnotatedCall, how='left') -> pd.DataFrame:
    ''' Merge a single call into a merged transcript with annotations'''
    annotations = call2annotations(call)
    annotation_df = annotations2dataframe(annotations)
    transcript_df = call2dataframe(call)
    merged_df = pd.merge(transcript_df,annotation_df, on=['Path','Line'], how = 'left')
    return merged_df[['Path','Text','Party','Annotation']]

def call2annotations(interaction: AnnotatedCall) -> List[Annotation]:
    """
    Converts an AnnotatedCall object into a list of Annotation objects.

    Args:
        interaction (AnnotatedCall): An object representing an annotated call.

    Returns:
        List[Annotation]: A list of Annotation objects.
    """
    records = []
    valid_sections = set(['Good Work','Need Improvement'])
    invalid_reasons = set(['Missed opportunity for questions','Relevant question'])
    lines = interaction.Summary.split('\n')
    current_section = ""
    for line in lines:
        # Check if there is a header to start a new "Good Work" or "Need Improvement" section
        match = re.search(r'^[ \t+*]*([A-Z][^:]*): *(.*)', line)
        if match:
            # Make sure this isn't a "Line 4: ..." entry
            if 'Line' not in match.group(1):
                current_section = match.group(1)
            example_text = match.group(2)
        else:
            example_text = line
        if current_section in valid_sections:
            # Strip headers like (Relevant question) from output
            for reason in invalid_reasons:
                example_text = example_text.replace(reason,'')
            example_text = example_text.strip(' -*+\t')
            if len(example_text) > 10 and example_text not in invalid_reasons:
                records.append(line2annotation(example_text,interaction.Path, current_section))
    return records    

def get_conversations(df: pd.DataFrame) -> List[Call]:
    """
    Extracts conversations from a DataFrame and returns a list of dictionaries.

    Args:
        df (pd.DataFrame): The DataFrame containing the conversations.

    Returns:
        List[Call]: A list of calls representing the conversations.
    """
    output = []
    for key,g_df in df.groupby('Filename'):
        g_df['Line'] = range(len(g_df))
        conv = Call(str(key), "\n".join([f'{row.Line}: {row.Party}: {row.Text}' 
                                                for row in g_df.itertuples()]))
        output.append(conv)
    return output

def save_convs(output_fn: str, prompt: str, convs:List[Call], save_path=False) -> None:
    """
    Save conversations to a file in internal JSON format.

    Args:
        output_fn (str): The filename to save the conversations to.
        prompt (str): The prompt for the conversations.
        convs (list): A list of conversation objects.
        save_path (bool, optional): Whether to save the conversation paths or just the conversation text. 
                                    Defaults to False.
    """
    if save_path:
        output = {'Prompt': prompt, "Responses": [asdict(c) for c in convs]}
    else:
        output = {'Prompt': prompt, "Responses": [c.Conv for c in convs]}

    with open(output_fn, 'wt') as f:
        json.dump(output, f, indent=4)
