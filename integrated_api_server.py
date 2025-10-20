#!/usr/bin/env python3
"""
Integrated API Server with Steps 0, 1, 2
NEW WORKFLOW: Upload CSV → Upload Mapping → Generate Intents → Select Intent → Run Analysis
"""

import http.server
import socketserver
import json
import os
import csv
import subprocess
import threading
import urllib.parse
from pathlib import Path
import io
import sys
import time
import re
import pandas as pd
import openpyxl

# Import step modules
try:
    from annotations import save_convs, get_conversations
    from transcripts import load_whisper_as_nx, load_and_clean_nxtranscript, sample_calls
    from intents import IntentGenerator, categories2dataframe
    import bedrock
    from default_prompts import STEP1_GENERATE_INTENTS_PROMPT, STEP2_ASSIGN_CATEGORIES_PROMPT
except ImportError as e:
    print(f"⚠️  Warning: Could not import required modules: {e}")
    print("Make sure all supporting files are in the same directory")

csv.field_size_limit(sys.maxsize)

PORT = 5000
UPLOAD_FOLDER = './uploads'
WORKING_FOLDER = './data'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(WORKING_FOLDER).mkdir(exist_ok=True)
Path('./logs').mkdir(exist_ok=True)

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request Handler with CORS support"""
    
    def end_headers(self):
        """Add CORS headers to all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/upload-asr':
            self.handle_upload_asr()
        elif self.path == '/api/upload-mapping':
            self.handle_upload_mapping()
        elif self.path == '/api/generate-intents':
            self.handle_generate_intents()
        elif self.path == '/api/filter-and-run':
            self.handle_filter_and_run()
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/api/health':
            self.handle_health()
        elif self.path.startswith('/api/download/'):
            self.handle_download()
        else:
            self.send_error(404, "Endpoint not found")
    
    def handle_health(self):
        """Health check endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {'status': 'healthy', 'message': 'API server is running'}
        self.wfile.write(json.dumps(response).encode())
    
    def parse_multipart(self):
        """Parse multipart form data and return file content and filename"""
        content_type = self.headers['Content-Type']
        if not content_type.startswith('multipart/form-data'):
            return None, None
        
        boundary = content_type.split('boundary=')[1].encode()
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        
        parts = body.split(b'--' + boundary)
        
        for part in parts:
            if b'Content-Disposition' in part:
                if b'filename=' in part:
                    lines = part.split(b'\r\n')
                    filename = None
                    for line in lines:
                        if b'filename=' in line:
                            filename = line.split(b'filename=')[1].strip(b'"\r\n ')
                            filename = filename.decode('utf-8')
                            break
                    
                    if filename:
                        content_start = part.find(b'\r\n\r\n') + 4
                        file_data = part[content_start:].rstrip(b'\r\n')
                        return file_data, filename
        
        return None, None
    
    def handle_upload_asr(self):
        """Upload the ASR CSV file"""
        try:
            file_data, filename = self.parse_multipart()
            
            if not file_data or not filename:
                self.send_error(400, "No file uploaded")
                return
            
            timestamp = int(time.time())
            file_path = os.path.join(UPLOAD_FOLDER, f'input_asr_{timestamp}.csv')
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # Quick validation
            try:
                df = pd.read_csv(file_path, sep='\t', nrows=5)
                row_count = len(pd.read_csv(file_path, sep='\t'))
            except:
                self.send_error(400, "Invalid CSV format")
                return
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'success': True,
                'file_path': file_path,
                'filename': filename,
                'row_count': row_count,
                'message': f'Uploaded {filename} with {row_count} rows'
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, f"Upload failed: {str(e)}")
    
    def handle_upload_mapping(self):
        """Upload the L123 Intent Mapping Excel file"""
        try:
            file_data, filename = self.parse_multipart()
            
            if not file_data or not filename:
                self.send_error(400, "No file uploaded")
                return
            
            timestamp = int(time.time())
            file_path = os.path.join(UPLOAD_FOLDER, f'mapping_{timestamp}.xlsx')
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # Convert Excel to categories format
            categories_txt_path = self.convert_mapping_to_categories(file_path)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'success': True,
                'file_path': file_path,
                'categories_txt': categories_txt_path,
                'filename': filename,
                'message': f'Uploaded and converted {filename}'
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, f"Upload failed: {str(e)}")
    
    def convert_mapping_to_categories(self, excel_path):
        """Convert L123 Excel mapping to YAML-style categories text file"""
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            
            # Expected columns (flexible matching)
            l1_col = [c for c in df.columns if 'level1' in c.lower() or 'l1' in c.lower()][0]
            l2_col = [c for c in df.columns if 'level2' in c.lower() or 'l2' in c.lower()][0]
            l3_col = [c for c in df.columns if 'level3' in c.lower() or 'l3' in c.lower()][0]
            
            # Get unique categories
            df_unique = df[[l1_col, l2_col, l3_col]].drop_duplicates().sort_values([l1_col, l2_col, l3_col])
            
            # Convert to YAML-style format
            lines = []
            current_l1 = None
            current_l2 = None
            
            for _, row in df_unique.iterrows():
                l1 = str(row[l1_col]).strip()
                l2 = str(row[l2_col]).strip()
                l3 = str(row[l3_col]).strip()
                
                if l1 != current_l1:
                    lines.append(f'- {l1}')
                    current_l1 = l1
                    current_l2 = None
                
                if l2 != current_l2:
                    lines.append(f'    - {l2}')
                    current_l2 = l2
                
                lines.append(f'        - {l3}')
            
            # Save to text file
            output_path = excel_path.replace('.xlsx', '_categories.txt')
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"✅ Converted mapping to categories: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to convert mapping: {e}")
            raise
    
    def handle_generate_intents(self):
        """Run Steps 0, 1, 2 to generate intent mapping"""
        try:
            # Read JSON body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Send SSE headers
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            # Extract parameters
            input_csv = data['input_csv']
            categories_txt = data['categories_txt']
            company_name = data['company_name']
            company_description = data['company_description']
            prompt_template = data.get('prompt_template', STEP1_GENERATE_INTENTS_PROMPT)
            csv_format = data.get('format', 'whisper')
            max_calls = data.get('max_calls', 10000)
            
            self.send_sse({'type': 'progress', 'message': '🚀 Starting Intent Generation Pipeline'})
            self.send_sse({'type': 'progress', 'message': '=' * 80})
            
            # Create working directory
            timestamp = int(time.time())
            work_dir = f"{WORKING_FOLDER}/intent_generation_{timestamp}"
            Path(work_dir).mkdir(exist_ok=True, parents=True)
            
            # Define intermediate files
            step0_json = f"{work_dir}/step0_conversations.json"
            step0_csv = f"{work_dir}/step0_sampled.csv"
            step1_json = f"{work_dir}/step1_intents.json"
            step1_csv = f"{work_dir}/step1_intents.csv"
            step2_csv = f"{work_dir}/step2_intent_mapping.csv"
            
            # STEP 0: Prepare Transcripts
            self.send_sse({'type': 'progress', 'message': '\n📋 STEP 0: Preparing Transcripts'})
            self.send_sse({'type': 'progress', 'message': f'  Input: {input_csv}'})
            self.send_sse({'type': 'progress', 'message': f'  Format: {csv_format}'})
            
            try:
                # Load data
                if csv_format == 'whisper':
                    df = load_whisper_as_nx(input_csv)
                else:
                    df = load_and_clean_nxtranscript(input_csv)
                
                self.send_sse({'type': 'progress', 'message': f'  Loaded {len(df)} rows from {len(set(df.Filename))} files'})
                
                # Clean filenames
                df['Filename'] = df['Filename'].str.replace(r'.*[\\/]([^\\/\.]+)\..*', r'\1', regex=True)
                
                # Sample calls
                sampled_df = sample_calls(df, max_calls=max_calls)
                sampled_df.to_csv(step0_csv, index=False, sep='\t')
                self.send_sse({'type': 'progress', 'message': f'  Sampled {len(set(sampled_df.Filename))} calls'})
                
                # Extract conversations
                conversations = get_conversations(sampled_df)
                save_convs(output_fn=step0_json, prompt=input_csv, convs=conversations, save_path=True)
                self.send_sse({'type': 'progress', 'message': f'  ✅ Step 0 Complete: {len(conversations)} conversations ready'})
                
            except Exception as e:
                self.send_sse({'type': 'error', 'message': f'Step 0 failed: {str(e)}'})
                return
            
            # STEP 1: Generate Intents
            self.send_sse({'type': 'progress', 'message': '\n🤖 STEP 1: Generating Intents with AI'})
            self.send_sse({'type': 'progress', 'message': f'  Model: Claude 3.5 Sonnet'})
            
            try:
                client = bedrock.get_client(region="us-east-1")
                
                # Read categories
                categories_str = open(categories_txt).read()
                
                # Create custom prompt with company info
                custom_prompt = prompt_template.format(
                    company_name=company_name,
                    company_description=company_description,
                    conv="{conv}",
                    categories="{categories}",
                    min_words="{min_words}",
                    max_words="{max_words}",
                    additional_instructions="{additional_instructions}"
                )
                
                generator = IntentGenerator(
                    bedrock_client=client,
                    prompt=custom_prompt,
                    categories=categories_str,
                    num_lines=10,
                    model_id='anthropic.claude-3-5-sonnet-20240620-v1:0',
                    min_words=5,
                    max_words=10,
                    max_workers=10,
                    max_tokens=256
                )
                
                self.send_sse({'type': 'progress', 'message': f'  Processing {len(conversations)} conversations...'})
                
                generator.collect_reasons(
                    input_json=step0_json,
                    output_json=step1_json,
                    max_interactions=max_calls
                )
                
                self.send_sse({'type': 'progress', 'message': f'  Creating intent CSV...'})
                generator.create_intent_csv(step1_json, step1_csv, additional_columns=[])
                
                self.send_sse({'type': 'progress', 'message': f'  ✅ Step 1 Complete: Intents generated'})
                self.send_sse({'type': 'progress', 'message': f'  Tokens used - Input: {generator.input_token_counter:,}, Output: {generator.output_token_counter:,}'})
                
            except Exception as e:
                self.send_sse({'type': 'error', 'message': f'Step 1 failed: {str(e)}'})
                return
            
            # STEP 2: Map to Categories
            self.send_sse({'type': 'progress', 'message': '\n🏷️  STEP 2: Mapping Intents to Categories'})
            
            try:
                from intents import IntentBuilder
                
                # Read assign prompt (using default)
                assign_prompt = STEP2_ASSIGN_CATEGORIES_PROMPT
                
                # Load categories and intents
                cat_df = categories2dataframe(categories_txt)
                reasons_df = pd.read_csv(step1_csv, sep='\t').dropna(subset=['Intent']).copy()
                reasons_df['Ind'] = range(len(reasons_df))
                
                self.send_sse({'type': 'progress', 'message': f'  Categorizing {len(reasons_df)} intents...'})
                
                builder = IntentBuilder(client, cluster_prompt="", assign_prompt=assign_prompt)
                
                # Process in chunks
                chunk_size = 100
                all_results = []
                
                for start in range(0, len(reasons_df), chunk_size):
                    end = min(start + chunk_size, len(reasons_df))
                    chunk_intents = list(reasons_df.iloc[start:end]['Intent'])
                    
                    result = builder.assign_reasons(
                        categories_txt,
                        chunk_intents,
                        'anthropic.claude-3-5-sonnet-20240620-v1:0',
                        start
                    )
                    
                    # Parse results
                    from io import StringIO
                    data_io = StringIO(result)
                    assign_cols = ['Ind', 'Intent_Input', 'Intent_Category', 'L3_Score', 'L2_Score', 'L1_Score']
                    df = pd.read_csv(data_io, names=assign_cols, on_bad_lines='skip')
                    
                    # Clean data
                    df = df[~df['Ind'].astype(str).str.lower().str.contains('ind', na=False)]
                    score_cols = ['Ind', 'L3_Score', 'L2_Score', 'L1_Score']
                    for col in score_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=score_cols)
                    for col in score_cols:
                        df[col] = df[col].astype(int)
                    
                    all_results.append(df)
                    
                    progress_pct = int((end / len(reasons_df)) * 100)
                    self.send_sse({'type': 'progress', 'message': f'  Progress: {progress_pct}% ({end}/{len(reasons_df)})'})
                
                # Combine all results
                combined_df = pd.concat(all_results, ignore_index=True)
                combined_df.drop(['Intent_Input'], axis=1, inplace=True)
                
                # Merge with original data
                merged_df = pd.merge(reasons_df, combined_df, on='Ind')
                merged_df['Intent_Category'] = merged_df['Intent_Category'].str.replace(r',.*', '', regex=True)
                
                # Save final mapping
                merged_df.to_csv(step2_csv, sep='\t', index=False)
                
                self.send_sse({'type': 'progress', 'message': f'  ✅ Step 2 Complete: Intent mapping created'})
                self.send_sse({'type': 'progress', 'message': f'  Tokens used - Input: {builder.input_token_counter:,}, Output: {builder.output_token_counter:,}'})
                
            except Exception as e:
                self.send_sse({'type': 'error', 'message': f'Step 2 failed: {str(e)}'})
                return
            
            # FINAL: Extract and send intent statistics
            self.send_sse({'type': 'progress', 'message': '\n📊 Extracting Intent Statistics'})
            
            try:
                intents = self.extract_intents(step2_csv)
                
                self.send_sse({'type': 'progress', 'message': '=' * 80})
                self.send_sse({'type': 'progress', 'message': f'✅ PIPELINE COMPLETE!'})
                self.send_sse({'type': 'progress', 'message': f'  Total intents found: {len(intents)}'})
                self.send_sse({'type': 'progress', 'message': f'  Output file: {step2_csv}'})
                
                self.send_sse({
                    'type': 'complete',
                    'results': {
                        'intents': intents,
                        'intent_mapping_file': step2_csv,
                        'total_intents': len(intents),
                        'work_dir': work_dir
                    }
                })
                
            except Exception as e:
                self.send_sse({'type': 'error', 'message': f'Failed to extract intents: {str(e)}'})
                return
            
        except Exception as e:
            import traceback
            self.send_sse({'type': 'error', 'message': f'Pipeline failed: {str(e)}'})
            self.send_sse({'type': 'progress', 'message': traceback.format_exc()})
    
    def extract_intents(self, intent_file_path):
        """Extract unique intents with volume from the mapping file"""
        intents_dict = {}
        
        # Read the intent mapping file
        df = pd.read_csv(intent_file_path, sep='\t')
        
        # Filter high confidence (score = 5)
        if 'L3_Score' in df.columns:
            df = df[df['L3_Score'] == 5]
        
        # Count by Intent_Category
        if 'Intent_Category' in df.columns:
            intent_counts = df['Intent_Category'].value_counts()
            
            total = len(df)
            intents = []
            
            for intent, volume in intent_counts.items():
                parts = intent.split(' - ')
                intents.append({
                    'intent': intent,
                    'volume': int(volume),
                    'percentage': round((volume / total * 100), 1) if total > 0 else 0,
                    'level1': parts[0].strip() if len(parts) > 0 else 'General',
                    'level2': parts[1].strip() if len(parts) > 1 else 'Support',
                    'level3': parts[2].strip() if len(parts) > 2 else 'Inquiry'
                })
            
            return sorted(intents, key=lambda x: x['volume'], reverse=True)
        
        return []
    
    def handle_filter_and_run(self):
        """Filter ASR data by intent and run pipeline"""
        try:
            # Read JSON body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Send SSE headers
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            # Extract parameters
            intent = data['intent']
            intent_mapping_file = data['intent_mapping_file']
            asr_file = data['asr_file']
            client = data['client']
            output_dir = data['output_dir']
            batch_size = data.get('batch_size', 3)
            workers = data.get('workers', 3)
            
            self.send_sse({'type': 'progress', 'message': f"Starting analysis for intent: {intent}"})
            self.send_sse({'type': 'progress', 'message': '=' * 80})
            
            # Step 1: Filter by intent
            self.send_sse({'type': 'progress', 'message': 'Step 1: Filtering ASR data by intent...'})
            
            # Get filenames for this intent
            filenames = self.get_filenames_for_intent(intent_mapping_file, intent)
            self.send_sse({'type': 'progress', 'message': f'  Found {len(filenames)} calls for this intent'})
            
            # Filter ASR file
            filtered_asr_path = self.filter_asr_by_filenames(asr_file, filenames, intent)
            self.send_sse({'type': 'progress', 'message': f'  Created filtered ASR file: {filtered_asr_path}'})
            self.send_sse({'type': 'progress', 'message': '  ✅ Step 1 complete'})
            self.send_sse({'type': 'progress', 'message': '=' * 80})
            
            # Step 2: Run pipeline
            self.send_sse({'type': 'progress', 'message': 'Step 2: Running analysis pipeline...'})
            
            # Check if pipeline exists
            if not os.path.exists('universal_pipeline.py'):
                self.send_sse({'type': 'error', 'message': 'ERROR: universal_pipeline.py not found'})
                return
            
            # Build command
            cmd = [
                sys.executable,
                'universal_pipeline.py',
                '--client', client,
                '--intent', intent,
                '--input', filtered_asr_path,
                '--output-dir', output_dir,
                '--batch-size', str(batch_size),
                '--workers', str(workers)
            ]
            
            self.send_sse({'type': 'progress', 'message': f"  Command: {' '.join(cmd)}"})
            self.send_sse({'type': 'progress', 'message': '-' * 80})
            
            # Execute pipeline
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            output_lines = []
            for line in process.stdout:
                line = line.strip()
                if line:
                    output_lines.append(line)
                    self.send_sse({'type': 'progress', 'message': f"  {line}"})
            
            # Wait for completion
            return_code = process.wait()
            
            self.send_sse({'type': 'progress', 'message': '-' * 80})
            self.send_sse({'type': 'progress', 'message': f"  Process exit code: {return_code}"})
            
            if return_code == 0:
                # Calculate output path
                client_safe = re.sub(r'\W+', '', client)
                intent_safe = re.sub(r'\W+', '', intent)
                output_path = f"{output_dir}/{client_safe}/{intent_safe}"
                
                # Count results
                results_file = f"{output_path}/analysis_results.csv"
                total_calls = 0
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        total_calls = sum(1 for _ in f) - 1
                
                self.send_sse({'type': 'progress', 'message': '=' * 80})
                self.send_sse({'type': 'progress', 'message': f"✅ Pipeline completed successfully!"})
                self.send_sse({'type': 'progress', 'message': f"  Processed {total_calls} calls"})
                self.send_sse({'type': 'progress', 'message': f"  Results saved to: {output_path}"})
                
                self.send_sse({
                    'type': 'complete',
                    'results': {
                        'output_dir': output_path,
                        'total_calls': total_calls,
                        'batches': batch_size,
                        'filtered_asr_file': filtered_asr_path
                    }
                })
            else:
                error_msg = f'Pipeline execution failed with exit code {return_code}'
                if output_lines:
                    error_msg += f'. Last output: {output_lines[-1]}'
                self.send_sse({'type': 'error', 'message': error_msg})
            
        except Exception as e:
            import traceback
            self.send_sse({'type': 'error', 'message': f'Exception: {str(e)}'})
            self.send_sse({'type': 'progress', 'message': traceback.format_exc()})
    
    def get_filenames_for_intent(self, intent_file, target_intent):
        """Get list of filenames for a specific intent"""
        filenames = set()
        
        df = pd.read_csv(intent_file, sep='\t')
        
        # Filter by intent category and high score
        if 'Intent_Category' in df.columns and 'Filename' in df.columns:
            filtered = df[df['Intent_Category'] == target_intent]
            if 'L3_Score' in df.columns:
                filtered = filtered[filtered['L3_Score'] == 5]
            filenames = set(filtered['Filename'].tolist())
        
        return filenames
    
    def filter_asr_by_filenames(self, asr_file, filenames, intent):
        """Filter ASR file to only include specified filenames"""
        # Create output directory
        intent_safe = re.sub(r'\W+', '', intent)
        output_dir = f"{WORKING_FOLDER}/filtered_{intent_safe}"
        Path(output_dir).mkdir(exist_ok=True)
        
        filtered_path = f"{output_dir}/asr_filtered.csv"
        
        # Read and filter
        df = pd.read_csv(asr_file, sep='\t')
        
        if 'Filename' in df.columns:
            # Clean filenames for comparison
            df['Filename_clean'] = df['Filename'].str.replace(r'.*[\\/]([^\\/\.]+)\..*', r'\1', regex=True)
            filenames_clean = {re.sub(r'.*[\\/]([^\\/\.]+)\..*', r'\1', fn) for fn in filenames}
            
            filtered_df = df[df['Filename_clean'].isin(filenames_clean)]
            filtered_df = filtered_df.drop('Filename_clean', axis=1)
            filtered_df.to_csv(filtered_path, sep='\t', index=False)
        
        return filtered_path
    
    def send_sse(self, data):
        """Send Server-Sent Event"""
        message = f"data: {json.dumps(data)}\n\n"
        self.wfile.write(message.encode())
        self.wfile.flush()
    
    def handle_download(self):
        """Handle file download"""
        try:
            parsed = urllib.parse.urlparse(self.path)
            file_type = parsed.path.split('/')[-1]
            query = urllib.parse.parse_qs(parsed.query)
            
            output_dir = query.get('path', [''])[0]
            if not output_dir:
                self.send_error(400, "No path provided")
                return
            
            file_mapping = {
                'results': 'analysis_results.csv',
                'normalized': 'analysis_normalized.csv',
                'summary': 'analysis_summary.json'
            }
            
            if file_type not in file_mapping:
                self.send_error(400, "Invalid file type")
                return
            
            file_path = os.path.join(output_dir, file_mapping[file_type])
            
            if not os.path.exists(file_path):
                self.send_error(404, "File not found")
                return
            
            # Send file
            self.send_response(200)
            content_type = 'application/json' if file_type == 'summary' else 'text/csv'
            self.send_header('Content-type', content_type)
            self.send_header('Content-Disposition', f'attachment; filename="{file_mapping[file_type]}"')
            self.end_headers()
            
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
            
        except Exception as e:
            self.send_error(500, f"Download failed: {str(e)}")

def main():
    """Start the server"""
    print("=" * 80)
    print("🚀 INTEGRATED TRANSCRIPT ANALYSIS API SERVER")
    print("   NEW WORKFLOW: Generate Intents → Select → Analyze")
    print("=" * 80)
    print(f"📂 Upload folder: {UPLOAD_FOLDER}")
    print(f"📂 Working folder: {WORKING_FOLDER}")
    print(f"🌐 Server running on http://localhost:{PORT}")
    print("=" * 80)
    print("\nNew Workflow:")
    print("  1. Upload ASR CSV file (asr-whisper or nx_transcripts)")
    print("  2. Upload L123 Intent Mapping Excel file")
    print("  3. Enter company name and description")
    print("  4. Edit AI prompt (optional)")
    print("  5. Run Steps 0→1→2 to generate intent mapping")
    print("  6. Select intent from table")
    print("  7. Run detailed analysis")
    print("\nPress Ctrl+C to stop the server\n")
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")
            sys.exit(0)

if __name__ == '__main__':
    main()
