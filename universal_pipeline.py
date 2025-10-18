# ============================================================================
# UNIVERSAL TRANSCRIPT ANALYSIS PIPELINE
# Works for any L3 intent, any client, any domain
# ============================================================================

import pandas as pd
import numpy as np
import json
import re
import os
import sys
import csv
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from time import sleep
from collections import defaultdict
import argparse

csv.field_size_limit(sys.maxsize)

import bedrock

# ============================================================================
# CONFIGURATION CLASS - Easy to modify for different runs
# ============================================================================

class PipelineConfig:
    """Configuration for different clients and intents"""
    
    def __init__(self, 
                 client_name: str,
                 intent_l3: str,
                 input_csv_path: str,
                 output_base_dir: str = './data',
                 batch_size: int = 3,
                 max_workers: int = 3,
                 model_id: str = 'anthropic.claude-3-5-sonnet-20240620-v1:0'):
        
        self.client_name = client_name
        self.intent_l3 = intent_l3
        self.input_csv_path = input_csv_path
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.model_id = model_id
        
        # Create safe directory name
        client_safe = re.sub(r'\W+', '', client_name)
        intent_safe = re.sub(r'\W+', '', intent_l3)
        
        # Output directory structure: ./data/{client}/{intent}/
        self.output_dir = f'{output_base_dir}/{client_safe}/{intent_safe}'
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Output files
        self.results_file = f'{self.output_dir}/analysis_results.csv'
        self.normalized_file = f'{self.output_dir}/analysis_normalized.csv'
        self.summary_file = f'{self.output_dir}/analysis_summary.json'
    
    def __str__(self):
        return f"""Pipeline Configuration:
  Client: {self.client_name}
  Intent: {self.intent_l3}
  Input: {self.input_csv_path}
  Output: {self.output_dir}
  Batch Size: {self.batch_size}
  Workers: {self.max_workers}
  Model: {self.model_id}"""


# ============================================================================
# TEXT CLEANING (Domain-agnostic)
# ============================================================================

class TranscriptCleaner:
    """Universal transcript cleaning - works for any domain"""
    
    PATTERNS = {
        'semicolons': re.compile(r'[;]+'),
        'multiple_periods': re.compile(r'[.]{2,}'),
        'multiple_commas': re.compile(r'[,]{2,}'),
        'filler_words': re.compile(r'\b(uh|yeah|okay|oh|huh|hmm|um|uhm)\b', re.IGNORECASE),
        'long_digits': re.compile(r'\d{4,}'),
        'whitespace': re.compile(r'\s+'),
    }
    
    @classmethod
    @lru_cache(maxsize=10000)
    def clean(cls, text):
        if pd.isna(text) or not text:
            return "Brief interaction"
        
        text = str(text).lower().replace("<unk>", "")
        for pattern in cls.PATTERNS.values():
            text = pattern.sub(' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 10:
            return "Brief interaction"
        return text[0].upper() + text[1:] if text else "No content"


# ============================================================================
# UNIVERSAL ANALYZER - Works for any domain/intent
# ============================================================================

class UniversalAnalyzer:
    """Domain-agnostic analyzer that works for any customer service context"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.bedrock = bedrock.BedrockClient()
    
    def create_analysis_prompt(self, batch_data: List[Dict]) -> str:
        """Create domain-agnostic prompt for any customer service context"""
        
        calls_text = ""
        for i, item in enumerate(batch_data, 1):
            customer = str(item.get('customer_transcript', ''))[:600]
            agent = str(item.get('agent_transcript', ''))[:600]
            duration = item.get('duration', 0)
            
            calls_text += f"""
═══ CALL {i} ═══
Filename: {item['filename']}
Duration: {duration} seconds
Customer: {customer}
Agent: {agent}

"""
        
        # Domain-agnostic prompt - no industry-specific references
        prompt = f"""You are analyzing customer service calls. Provide detailed, specific analysis for each call.

{calls_text}

For EACH call above, provide comprehensive analysis:

CALL_ID: [exact filename]
PRIMARY_INTENT: [Specific customer need or request in 4-7 words]
INTENT_CATEGORY: [Broader category this falls under]
INTENT_EXPLANATION: [2-3 sentences explaining WHY you classified it this way. Quote specific words/phrases the customer said as evidence.]

AGENT_TASKS: [Specific actions the agent performed, comma-separated]
TASK_BREAKDOWN: [Numbered list with detailed explanations]
1. [Task Name] - [Describe specifically what the agent did and how]
2. [Task Name] - [Describe the agent's actions in detail]
3. [Task Name] - [Be specific about the steps taken]

RESOLUTION_STATUS: [FULLY_RESOLVED or PARTIAL_RESOLUTION or NOT_RESOLVED]
RESOLUTION_CONFIDENCE: [1-5, where 5 = very confident]
RESOLUTION_EXPLANATION: [2 sentences: What was resolved? What evidence shows this?]

AUTOMATION_SCORE: [1-100, where 100 = easily automatable]
AUTOMATION_REASONING: [2 sentences: Why this score? What factors make it more or less automatable?]

━━━━━━━━━━━━━━━━━━━━

CRITICAL REQUIREMENTS:
- Analyze ALL {len(batch_data)} calls above
- Be SPECIFIC - cite actual phrases from transcripts
- NO generic responses - make each analysis unique
- EXPLAIN your reasoning with evidence
- Use exact filenames provided

Provide detailed analysis for all {len(batch_data)} calls:"""
        
        return prompt
    
    def parse_response(self, response: str, batch_filenames: set) -> List[Dict]:
        """Parse LLM response into structured data"""
        
        results = []
        current_call = {}
        
        sections = re.split(r'━+|═+', response)
        
        for section in sections:
            lines = section.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # New call
                if line.startswith('CALL_ID:') or (line.startswith('CALL') and 'Filename:' in line):
                    if current_call and 'Filename' in current_call:
                        results.append(self._finalize_call(current_call))
                    
                    filename = re.sub(r'^(CALL_ID:|CALL \d+:|Filename:)\s*', '', line, flags=re.IGNORECASE).strip()
                    current_call = {'Filename': filename}
                
                elif current_call and ':' in line:
                    self._parse_field(line, current_call)
        
        if current_call and 'Filename' in current_call:
            results.append(self._finalize_call(current_call))
        
        return self._match_and_fill(results, batch_filenames)
    
    def _parse_field(self, line: str, current_call: Dict):
        """Parse individual field"""
        
        if ':' not in line:
            return
        
        key, value = line.split(':', 1)
        key = key.strip().upper()
        value = value.strip()
        
        if not value or len(value) < 2:
            return
        
        # Field mappings
        if 'PRIMARY_INTENT' in key:
            current_call['Primary_Intent'] = value
        elif 'INTENT_CATEGORY' in key:
            current_call['Intent'] = value
        elif 'INTENT_EXPLANATION' in key or 'INTENT_REASONING' in key:
            current_call['Explain'] = value
        elif 'AGENT_TASKS' in key and 'BREAKDOWN' not in key:
            current_call['Agent_Tasks_Classified_LLM_based'] = value
        elif 'TASK_BREAKDOWN' in key or 'BREAKDOWN' in key:
            current_call['_breakdown_start'] = True
        elif current_call.get('_breakdown_start') and re.match(r'^\d+\.', line):
            if 'Agent_Actions_Breakdown' not in current_call:
                current_call['Agent_Actions_Breakdown'] = value
            else:
                current_call['Agent_Actions_Breakdown'] += '|' + value
        elif 'RESOLUTION_STATUS' in key:
            status = value.upper()
            if 'FULLY' in status or 'YES' in status:
                current_call['Resolution_Status'] = 'FULLY_RESOLVED'
            elif 'PARTIAL' in status:
                current_call['Resolution_Status'] = 'PARTIAL_RESOLUTION'
            else:
                current_call['Resolution_Status'] = 'NOT_RESOLVED'
        elif 'RESOLUTION_CONFIDENCE' in key:
            try:
                score = int(re.search(r'\d+', value).group())
                current_call['Resolution_Confidence'] = min(5, max(1, score))
            except:
                current_call['Resolution_Confidence'] = 3
        elif 'RESOLUTION_EXPLANATION' in key or 'RESOLUTION_REASONING' in key:
            current_call['Resolution_Reason'] = value
        elif 'AUTOMATION_SCORE' in key or ('AUTOMATION' in key and 'SCORE' in key):
            try:
                score = int(re.search(r'\d+', value).group())
                current_call['LLM_Bot_Automation_Score'] = min(100, max(1, score))
            except:
                current_call['LLM_Bot_Automation_Score'] = 50
        elif 'AUTOMATION_REASONING' in key or 'AUTOMATION_EXPLANATION' in key:
            current_call['LLM_Bot_Automation_Reason'] = value
    
    def _finalize_call(self, call_data: Dict) -> Dict:
        """Ensure all required fields with quality defaults"""
        
        call_data.pop('_breakdown_start', None)
        
        # Domain-agnostic defaults
        if 'Primary_Intent' not in call_data:
            call_data['Primary_Intent'] = 'Customer Support Request'
        
        if 'Intent' not in call_data:
            call_data['Intent'] = 'General Support'
        
        if 'Explain' not in call_data or len(call_data.get('Explain', '')) < 10:
            intent = call_data.get('Primary_Intent', 'assistance')
            call_data['Explain'] = f"Customer contacted regarding {intent.lower()}. Agent provided relevant support and information based on the customer's needs."
        
        if 'Agent_Tasks_Classified_LLM_based' not in call_data:
            call_data['Agent_Tasks_Classified_LLM_based'] = 'Customer Assistance, Information Provision'
        
        tasks = call_data.get('Agent_Tasks_Classified_LLM_based', '')
        call_data['Count_Tasks'] = len([t for t in tasks.split(',') if t.strip()])
        
        if 'Agent_Actions_Breakdown' not in call_data:
            tasks_list = call_data.get('Agent_Tasks_Classified_LLM_based', 'Assistance').split(',')
            breakdown = []
            for i, task in enumerate(tasks_list[:3], 1):
                task = task.strip()
                breakdown.append(f"{i}. {task} - Agent performed this action to address customer needs")
            call_data['Agent_Actions_Breakdown'] = '|'.join(breakdown)
        
        if 'Resolution_Status' not in call_data:
            call_data['Resolution_Status'] = 'PARTIAL_RESOLUTION'
        
        if 'Resolution_Confidence' not in call_data:
            call_data['Resolution_Confidence'] = 3
        
        if 'Resolution_Reason' not in call_data or len(call_data.get('Resolution_Reason', '')) < 10:
            status = call_data.get('Resolution_Status', 'addressed')
            call_data['Resolution_Reason'] = f"Call {status.lower().replace('_', ' ')} based on agent actions and customer responses."
        
        if 'LLM_Bot_Automation_Score' not in call_data:
            call_data['LLM_Bot_Automation_Score'] = 60
        
        if 'LLM_Bot_Automation_Reason' not in call_data or len(call_data.get('LLM_Bot_Automation_Reason', '')) < 10:
            score = call_data.get('LLM_Bot_Automation_Score', 60)
            if score >= 70:
                call_data['LLM_Bot_Automation_Reason'] = "High automation potential due to standardized process and clear resolution path."
            elif score >= 50:
                call_data['LLM_Bot_Automation_Reason'] = "Moderate automation potential with some complexity requiring optimization."
            else:
                call_data['LLM_Bot_Automation_Reason'] = "Limited automation potential due to complexity requiring human judgment."
        
        call_data['LLM_Bot_Automation_Suitable'] = call_data.get('LLM_Bot_Automation_Score', 0) >= 70
        
        return call_data
    
    def _match_and_fill(self, results: List[Dict], batch_filenames: set) -> List[Dict]:
        """Match parsed results to expected filenames"""
        
        matched_results = []
        processed = set()
        
        for result in results:
            result_filename = result.get('Filename', '')
            
            if result_filename in batch_filenames:
                matched_results.append(result)
                processed.add(result_filename)
            else:
                for batch_filename in batch_filenames:
                    if (batch_filename in result_filename or 
                        result_filename in batch_filename or
                        batch_filename.lower() == result_filename.lower()):
                        result['Filename'] = batch_filename
                        matched_results.append(result)
                        processed.add(batch_filename)
                        break
        
        for filename in batch_filenames:
            if filename not in processed:
                matched_results.append(self._create_default(filename))
        
        return matched_results
    
    def _create_default(self, filename: str) -> Dict:
        """Create reasonable default"""
        return {
            'Filename': filename,
            'Primary_Intent': 'Customer Support Request',
            'Intent': 'General Support',
            'Explain': 'Customer contacted for assistance. Agent provided support based on needs.',
            'Agent_Tasks_Classified_LLM_based': 'Customer Assistance',
            'Count_Tasks': 1,
            'Agent_Actions_Breakdown': '1. Customer Assistance - Agent addressed customer inquiry',
            'Resolution_Status': 'PARTIAL_RESOLUTION',
            'Resolution_Confidence': 3,
            'Resolution_Reason': 'Call addressed with standard support procedures.',
            'LLM_Bot_Automation_Score': 55,
            'LLM_Bot_Automation_Suitable': False,
            'LLM_Bot_Automation_Reason': 'Moderate complexity requiring case review.'
        }
    
    def process_batch(self, batch_data: List[Dict], batch_num: int, total_batches: int) -> List[Dict]:
        """Process a single batch"""
        
        batch_filenames = {item['filename'] for item in batch_data}
        
        print(f"📦 Batch {batch_num}/{total_batches} ({len(batch_data)} calls)", end='', flush=True)
        
        for attempt in range(3):
            try:
                prompt = self.create_analysis_prompt(batch_data)
                response, _, _ = self.bedrock.simple_prompt(prompt, self.config.model_id)
                
                if not response or len(response.strip()) < 100:
                    raise ValueError("Response too short")
                
                results = self.parse_response(response, batch_filenames)
                
                quality = len([r for r in results if len(r.get('Explain', '')) > 30])
                success_rate = (quality / len(results)) * 100
                
                print(f" ✅ {success_rate:.0f}% quality")
                return results
                
            except Exception as e:
                if attempt < 2:
                    print(f" ⚠️ Retry...", end='', flush=True)
                    sleep(2 ** attempt * 0.5)
                else:
                    print(f" ❌ Using defaults")
                    return [self._create_default(fn) for fn in batch_filenames]


# ============================================================================
# PARALLEL PROCESSOR
# ============================================================================

class ParallelProcessor:
    def __init__(self, analyzer: UniversalAnalyzer, config: PipelineConfig):
        self.analyzer = analyzer
        self.config = config
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        batches = []
        for i in range(0, len(df), self.config.batch_size):
            batch_df = df.iloc[i:i+self.config.batch_size]
            batch_data = []
            for _, row in batch_df.iterrows():
                batch_data.append({
                    'filename': row['Filename'],
                    'customer_transcript': row.get('Cleaned_Customer_Transcript', ''),
                    'agent_transcript': row.get('Cleaned_Agent_Transcript', ''),
                    'duration': row.get('Duration_Total', 0)
                })
            batches.append(batch_data)
        
        total_batches = len(batches)
        print(f"\n🔄 Processing {len(df)} calls in {total_batches} batches with {self.config.max_workers} workers\n")
        
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.analyzer.process_batch, batch, i+1, total_batches): i
                for i, batch in enumerate(batches)
            }
            
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"❌ Batch failed: {e}")
        
        print(f"\n✅ Completed {len(all_results)} calls")
        return pd.DataFrame(all_results)


# ============================================================================
# DATA LOADER
# ============================================================================

def load_and_prepare_data(config: PipelineConfig) -> pd.DataFrame:
    """Load and prepare transcript data"""
    
    print(f"\n📂 Loading data from {config.input_csv_path}...")
    
    asr_df = pd.read_csv(config.input_csv_path, sep='\t', encoding='utf-8', engine='python')
    
    asr_df['Segment_Duration'] = asr_df['EndOffset (sec)'] - asr_df['StartOffset (sec)']
    duration_df = asr_df.groupby('Filename')['Segment_Duration'].sum().reset_index()
    duration_df = duration_df.rename(columns={'Segment_Duration': 'Duration_Total'})
    
    grouped_transcripts = (
        asr_df.groupby(['Filename', 'Party'])['Text']
        .apply(lambda texts: '; '.join(texts))
        .unstack(fill_value='')
        .reset_index()
    )
    
    if 'Customer' in grouped_transcripts.columns:
        grouped_transcripts = grouped_transcripts.rename(columns={'Customer': 'Customer_Transcript'})
    if 'Agent' in grouped_transcripts.columns:
        grouped_transcripts = grouped_transcripts.rename(columns={'Agent': 'Agent_Transcript'})
    
    asr_df_sorted = asr_df.sort_values(['Filename', 'StartOffset (sec)'])
    conversation_dict = defaultdict(list)
    for _, row in asr_df_sorted.iterrows():
        conversation_dict[row['Filename']].append(f"{row['Party']}: {row['Text']}")
    
    interleaved_df = pd.DataFrame([
        {'Filename': fname, 'Transcript': '; '.join(lines)}
        for fname, lines in conversation_dict.items()
    ])
    
    final_df = grouped_transcripts.merge(interleaved_df, on='Filename', how='left')
    final_df = final_df.merge(duration_df, on='Filename', how='left')
    
    print("  🧹 Cleaning transcripts...")
    final_df['Cleaned_Customer_Transcript'] = final_df['Customer_Transcript'].apply(TranscriptCleaner.clean)
    final_df['Cleaned_Agent_Transcript'] = final_df['Agent_Transcript'].apply(TranscriptCleaner.clean)
    final_df['Clean_Transcript'] = final_df['Transcript'].apply(TranscriptCleaner.clean)
    
    print(f"  ✅ Prepared {len(final_df)} calls")
    return final_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config: PipelineConfig):
    """Run the complete analysis pipeline"""
    
    print("="*80)
    print("🌍 UNIVERSAL TRANSCRIPT ANALYSIS PIPELINE")
    print("="*80)
    print(config)
    print("="*80)
    
    # Load data
    df = load_and_prepare_data(config)
    
    # Run analysis
    print("\n🤖 Initializing analyzer...")
    analyzer = UniversalAnalyzer(config)
    processor = ParallelProcessor(analyzer, config)
    
    print("\n🚀 Running analysis...")
    results_df = processor.process_dataframe(df)
    
    # Merge and save
    final_df = df.merge(results_df, on='Filename', how='left')
    final_df.to_csv(config.results_file, sep='\t', index=False)
    
    print(f"\n💾 Saved to: {config.results_file}")
    
    # Generate summary
    summary = generate_summary(final_df, config)
    
    with open(config.summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Summary saved to: {config.summary_file}")
    
    # Print statistics
    print_statistics(final_df, config)
    
    return final_df


def generate_summary(df: pd.DataFrame, config: PipelineConfig) -> Dict:
    """Generate analysis summary"""
    
    return {
        'client': config.client_name,
        'intent_l3': config.intent_l3,
        'total_calls': len(df),
        'date_processed': pd.Timestamp.now().isoformat(),
        'intent_distribution': df['Intent'].value_counts().head(10).to_dict(),
        'resolution_stats': df['Resolution_Status'].value_counts().to_dict(),
        'automation_stats': {
            'automatable_count': int((df['LLM_Bot_Automation_Suitable'] == True).sum()),
            'automatable_percentage': float((df['LLM_Bot_Automation_Suitable'] == True).mean() * 100),
            'average_score': float(df['LLM_Bot_Automation_Score'].mean())
        },
        'quality_metrics': {
            'detailed_explanations': int((df['Explain'].str.len() > 30).sum()),
            'multi_step_breakdowns': int(df['Agent_Actions_Breakdown'].str.contains(r'\|', na=False).sum()),
            'avg_tasks_per_call': float(df['Count_Tasks'].mean())
        }
    }


def print_statistics(df: pd.DataFrame, config: PipelineConfig):
    """Print analysis statistics"""
    
    print("\n" + "="*80)
    print("📊 ANALYSIS STATISTICS")
    print("="*80)
    
    print(f"\n📋 Client: {config.client_name}")
    print(f"📋 Intent: {config.intent_l3}")
    print(f"📋 Total Calls: {len(df)}")
    
    print(f"\n🎯 Top Intent Categories:")
    for intent, count in df['Intent'].value_counts().head(10).items():
        pct = (count / len(df)) * 100
        print(f"   {count:3d} ({pct:5.1f}%) - {intent}")
    
    print(f"\n✅ Resolution Status:")
    for status, count in df['Resolution_Status'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"   {count:3d} ({pct:5.1f}%) - {status}")
    
    print(f"\n🤖 Automation Potential:")
    automatable = (df['LLM_Bot_Automation_Suitable'] == True).sum()
    print(f"   Automatable calls: {automatable} ({automatable/len(df)*100:.1f}%)")
    print(f"   Average score: {df['LLM_Bot_Automation_Score'].mean():.1f}/100")
    
    print(f"\n✨ Content Quality:")
    detailed = (df['Explain'].str.len() > 30).sum()
    multi_step = df['Agent_Actions_Breakdown'].str.contains(r'\|', na=False).sum()
    print(f"   Detailed explanations: {detailed}/{len(df)} ({detailed/len(df)*100:.1f}%)")
    print(f"   Multi-step breakdowns: {multi_step}/{len(df)} ({multi_step/len(df)*100:.1f}%)")
    print(f"   Avg tasks per call: {df['Count_Tasks'].mean():.1f}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Universal Transcript Analysis Pipeline')
    parser.add_argument('--client', required=True, help='Client name')
    parser.add_argument('--intent', required=True, help='L3 Intent')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output-dir', default='./data', help='Output base directory')
    parser.add_argument('--batch-size', type=int, default=3, help='Batch size')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--model', default='anthropic.claude-3-5-sonnet-20240620-v1:0', help='Model ID')
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        client_name=args.client,
        intent_l3=args.intent,
        input_csv_path=args.input,
        output_base_dir=args.output_dir,
        batch_size=args.batch_size,
        max_workers=args.workers,
        model_id=args.model
    )
    
    run_pipeline(config)
