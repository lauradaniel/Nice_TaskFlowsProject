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

# Import step modules
try:
    from annotations import save_convs, get_conversations
    from transcripts import load_whisper_as_nx, load_and_clean_nxtranscript, sample_calls
    from intents import IntentGenerator, IntentBuilder, categories2dataframe
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
        print(f"📥 POST request to: {self.path}")
        
        if self.path == '/api/upload-asr':
            self.handle_upload_asr()
        elif self.path == '/api/upload-mapping':
            self.handle_upload_mapping()
        elif self.path == '/api/generate-intents':
            self.handle_generate_intents()
        elif self.path == '/api/filter-and-run':
            self.handle_filter_and_run()
        else:
            print(f"❌ Unknown endpoint: {self.path}")
            self.send_error(404, f"Endpoint not found: {self.path}")
    
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
            
            print(f"📁 Uploaded ASR file: {file_path}")
            
            row_count = 0
            error_msg = None
            
            try:
                print("   Reading CSV with tab delimiter...")
                df = pd.read_csv(file_path, sep='\t', nrows=10)
                print(f"   Original columns: {list(df.columns)}")
                
                # Normalize column names to standard format
                column_mapping = {
                    'Path': 'Filename',
                    'path': 'Filename',
                    'party': 'Party',
                    'Party': 'Party',
                    'text': 'Text',
                    'Text': 'Text',
                    'start': 'StartOffset (sec)',
                    'StartOffset (sec)': 'StartOffset (sec)',
                    'end': 'EndOffset (sec)',
                    'EndOffset (sec)': 'EndOffset (sec)'
                }
                
                df_renamed = df.rename(columns=column_mapping)
                print(f"   Normalized columns: {list(df_renamed.columns)}")
                
                # Check for required columns
                required = ['Filename', 'Party', 'Text', 'StartOffset (sec)', 'EndOffset (sec)']
                missing = [col for col in required if col not in df_renamed.columns]
                
                if missing:
                    error_msg = f"Missing columns: {missing}"
                    print(f"   ⚠️  {error_msg}")
                    row_count = len(df)
                else:
                    # Read full file and normalize
                    df_full = pd.read_csv(file_path, sep='\t')
                    df_full = df_full.rename(columns=column_mapping)
                    
                    # Save normalized version
                    normalized_path = file_path.replace('.csv', '_normalized.csv')
                    df_full.to_csv(normalized_path, sep='\t', index=False)
                    
                    row_count = len(df_full)
                    print(f"   ✅ Normalized {row_count} rows")
                    print(f"   Saved: {normalized_path}")
                    
                    # Use normalized file for further processing
                    file_path = normalized_path
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"   ❌ {error_msg}")
                import traceback
                traceback.print_exc()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'success': True,
                'file_path': file_path,
                'filename': filename,
                'row_count': row_count,
                'error_msg': error_msg,
                'message': f'Uploaded {filename}' + (f' with {row_count} rows' if row_count > 0 else '')
            }
            
            print(f"   Response: row_count={row_count}")
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Upload failed: {str(e)}")
    
    def handle_upload_mapping(self):
        """Upload the L123 Intent Mapping Excel file"""
        print("🔵 handle_upload_mapping called")
        try:
            file_data, filename = self.parse_multipart()
            
            if not file_data or not filename:
                self.send_error(400, "No file uploaded")
                return
            
            timestamp = int(time.time())
            file_path = os.path.join(UPLOAD_FOLDER, f'mapping_{timestamp}.xlsx')
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            print(f"📄 Uploaded mapping file: {file_path}")
            
            try:
                categories_txt_path = self.convert_mapping_to_categories(file_path)
                print(f"✅ Conversion successful: {categories_txt_path}")
            except Exception as conv_error:
                print(f"❌ Conversion failed: {conv_error}")
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Failed to convert mapping: {str(conv_error)}")
                return
            
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
            print(f"❌ Upload failed: {e}")
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Upload failed: {str(e)}")
    
    def convert_mapping_to_categories(self, excel_path):
        """Convert L123 Excel mapping to YAML-style categories text file"""
        print(f"🔄 Converting {excel_path}...")
        
        df = pd.read_excel(excel_path)
        print(f"   Read {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        
        l1_col = None
        l2_col = None
        l3_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'level1' in col_lower or 'l1' in col_lower or 'category_mapped' in col_lower:
                l1_col = col
            if 'level2' in col_lower or 'l2' in col_lower or 'topic_mapped' in col_lower:
                l2_col = col
            if 'level3' in col_lower or 'l3' in col_lower or 'intent' in col_lower:
                l3_col = col
        
        if not l1_col:
            raise ValueError(f"Cannot find L1 column. Available: {list(df.columns)}")
        if not l2_col:
            raise ValueError(f"Cannot find L2 column. Available: {list(df.columns)}")
        if not l3_col:
            raise ValueError(f"Cannot find L3 column. Available: {list(df.columns)}")
        
        print(f"   ✅ Found L1: '{l1_col}'")
        print(f"   ✅ Found L2: '{l2_col}'")
        print(f"   ✅ Found L3: '{l3_col}'")
        
        df_unique = df[[l1_col, l2_col, l3_col]].drop_duplicates().sort_values([l1_col, l2_col, l3_col])
        print(f"   Unique combinations: {len(df_unique)}")
        
        lines = []
        current_l1 = None
        current_l2 = None
        
        for _, row in df_unique.iterrows():
            l1 = str(row[l1_col]).strip()
            l2 = str(row[l2_col]).strip()
            l3 = str(row[l3_col]).strip()
            
            if l1 == 'nan' or l2 == 'nan' or l3 == 'nan':
                continue
            if not l1 or not l2 or not l3:
                continue
            
            if l1 != current_l1:
                lines.append(f'- {l1}')
                current_l1 = l1
                current_l2 = None
            
            if l2 != current_l2:
                lines.append(f'    - {l2}')
                current_l2 = l2
            
            lines.append(f'        - {l3}')
        
        if len(lines) == 0:
            raise ValueError("No valid L1/L2/L3 combinations found")
        
        output_path = excel_path.replace('.xlsx', '_categories.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"   ✅ Converted {len(lines)} lines to: {output_path}")
        
        preview = '\n'.join(lines[:10])
        print(f"\n   Preview:\n{preview}\n   ...")
        
        return output_path
    
    def handle_generate_intents(self):
        """Run Steps 0, 1, 2 to generate intent mapping"""
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            input_csv = data['input_csv']
            categories_txt = data['categories_txt']
            company_name = data['company_name']
            company_description = data['company_description']
            prompt_template = data.get('prompt_template', STEP1_GENERATE_INTENTS_PROMPT)
            csv_format = data.get('format', 'whisper')
            max_calls = data.get('max_calls', 10000)
            
            self.send_sse({'type': 'progress', 'message': '🚀 Starting Intent Generation Pipeline'})
            self.send_sse({'type': 'progress', 'message': '=' * 80})
            
            timestamp = int(time.time())
            work_dir = f"{WORKING_FOLDER}/intent_generation_{timestamp}"
            Path(work_dir).mkdir(exist_ok=True, parents=True)
            
            step0_json = f"{work_dir}/step0_conversations.json"
            step0_csv = f"{work_dir}/step0_sampled.csv"
            step1_json = f"{work_dir}/step1_intents.json"
            step1_csv = f"{work_dir}/step1_intents.csv"
            step2_csv = f"{work_dir}/step2_intent_mapping.csv"
            
            # STEP 0
            self.send_sse({'type': 'progress', 'message': '\n📋 STEP 0: Preparing Transcripts'})
            
            try:
                if csv_format == 'whisper':
                    df = load_whisper_as_nx(input_csv)
                else:
                    df = load_and_clean_nxtranscript(input_csv)
                
                self.send_sse({'type': 'progress', 'message': f'  Loaded {len(df)} rows'})
                
                df['Filename'] = df['Filename'].str.replace(r'.*[\\/]([^\\/\.]+)\..*', r'\1', regex=True)
                sampled_df = sample_calls(df, max_calls=max_calls)
                sampled_df.to_csv(step0_csv, index=False, sep='\t')
                
                conversations = get_conversations(sampled_df)
                save_convs(output_fn=step0_json, prompt=input_csv, convs=conversations, save_path=True)
                self.send_sse({'type': 'progress', 'message': f'  ✅ Step 0 Complete: {len(conversations)} conversations'})
                
            except Exception as e:
                self.send_sse({'type': 'error', 'message': f'Step 0 failed: {str(e)}'})
                return
            
            # STEP 1
            self.send_sse({'type': 'progress', 'message': '\n🤖 STEP 1: Generating Intents'})
            
            try:
                client = bedrock.get_client(region="us-east-1")
                categories_str = open(categories_txt).read()
                
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
                
                generator.collect_reasons(input_json=step0_json, output_json=step1_json, max_interactions=max_calls)
                generator.create_intent_csv(step1_json, step1_csv, additional_columns=[])
                
                self.send_sse({'type': 'progress', 'message': f'  ✅ Step 1 Complete'})
                
            except Exception as e:
                self.send_sse({'type': 'error', 'message': f'Step 1 failed: {str(e)}'})
                return
            
            # STEP 2
            self.send_sse({'type': 'progress', 'message': '\n🏷️  STEP 2: Mapping to Categories'})

            try:
                assign_prompt = STEP2_ASSIGN_CATEGORIES_PROMPT
                cat_df = categories2dataframe(categories_txt)
                reasons_df = pd.read_csv(step1_csv, sep='\t').dropna(subset=['Intent']).copy()
                
                if len(reasons_df) == 0:
                    self.send_sse({'type': 'error', 'message': 'Step 1 produced no intents!'})
                    return
                
                reasons_df['Ind'] = range(len(reasons_df))
                
                self.send_sse({'type': 'progress', 'message': f'  Categorizing {len(reasons_df)} intents...'})
                
                builder = IntentBuilder(client, cluster_prompt="", assign_prompt=assign_prompt)
                
                chunk_size = 100
                all_results = []
                
                for start in range(0, len(reasons_df), chunk_size):
                    end = min(start + chunk_size, len(reasons_df))
                    chunk_intents = list(reasons_df.iloc[start:end]['Intent'])
                    
                    try:
                        result = builder.assign_reasons(categories_txt, chunk_intents, 
                                                    'anthropic.claude-3-5-sonnet-20240620-v1:0', start)
                        
                        # Debug: show what LLM returned
                        print(f"\n=== LLM Response for chunk {start}-{end} ===")
                        print(result[:500])  # First 500 chars
                        
                        from io import StringIO
                        data_io = StringIO(result)
                        assign_cols = ['Ind', 'Intent_Input', 'Intent_Category', 'L3_Score', 'L2_Score', 'L1_Score']
                        df = pd.read_csv(data_io, names=assign_cols, on_bad_lines='skip')
                        
                        print(f"   Parsed {len(df)} rows")
                        
                        # Clean data
                        df = df[~df['Ind'].astype(str).str.lower().str.contains('ind', na=False)]
                        score_cols = ['Ind', 'L3_Score', 'L2_Score', 'L1_Score']
                        for col in score_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        df = df.dropna(subset=score_cols)
                        for col in score_cols:
                            df[col] = df[col].astype(int)
                        
                        print(f"   After cleaning: {len(df)} rows")
                        
                        if len(df) > 0:
                            all_results.append(df)
                        else:
                            print(f"   ⚠️  Chunk {start}-{end} produced no valid rows")
                        
                    except Exception as chunk_error:
                        print(f"   ❌ Chunk {start}-{end} failed: {chunk_error}")
                        self.send_sse({'type': 'progress', 'message': f'  ⚠️  Chunk {start}-{end} failed'})
                    
                    progress_pct = int((end / len(reasons_df)) * 100)
                    self.send_sse({'type': 'progress', 'message': f'  Progress: {progress_pct}%'})
                
                # Check if we have ANY results
                if len(all_results) == 0:
                    self.send_sse({'type': 'error', 'message': 'Step 2: No valid categorizations produced. Check server logs.'})
                    return
                
                # Combine results
                combined_df = pd.concat(all_results, ignore_index=True)
                print(f"\n✅ Combined {len(combined_df)} total rows from {len(all_results)} chunks")
                
                combined_df.drop(['Intent_Input'], axis=1, inplace=True, errors='ignore')
                
                merged_df = pd.merge(reasons_df, combined_df, on='Ind')
                merged_df['Intent_Category'] = merged_df['Intent_Category'].str.replace(r',.*', '', regex=True)
                merged_df.to_csv(step2_csv, sep='\t', index=False)
                
                self.send_sse({'type': 'progress', 'message': f'  ✅ Step 2 Complete: {len(merged_df)} intents categorized'})
                
            except Exception as e:
                print(f"\n❌ Step 2 failed: {e}")
                import traceback
                traceback.print_exc()
                self.send_sse({'type': 'error', 'message': f'Step 2 failed: {str(e)}'})
                return
            
            # Extract intents
            try:
                intents = self.extract_intents(step2_csv)
                self.send_sse({'type': 'progress', 'message': f'✅ COMPLETE! {len(intents)} intents'})
                self.send_sse({'type': 'complete', 'results': {
                    'intents': intents,
                    'intent_mapping_file': step2_csv,
                    'total_intents': len(intents),
                    'work_dir': work_dir
                }})
            except Exception as e:
                self.send_sse({'type': 'error', 'message': f'Failed: {str(e)}'})
            
        except Exception as e:
            import traceback
            self.send_sse({'type': 'error', 'message': f'Pipeline failed: {str(e)}'})
            self.send_sse({'type': 'progress', 'message': traceback.format_exc()})
    
    def extract_intents(self, intent_file_path):
        """Extract unique intents with volume from the mapping file"""
        df = pd.read_csv(intent_file_path, sep='\t')
        
        if 'L3_Score' in df.columns:
            df = df[df['L3_Score'] == 5]
        
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
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            
            intent = data['intent']
            intent_mapping_file = data['intent_mapping_file']
            asr_file = data['asr_file']
            client = data['client']
            output_dir = data['output_dir']
            batch_size = data.get('batch_size', 3)
            workers = data.get('workers', 3)
            
            self.send_sse({'type': 'progress', 'message': f"Starting analysis for: {intent}"})
            self.send_sse({'type': 'progress', 'message': '=' * 80})
            
            self.send_sse({'type': 'progress', 'message': 'Filtering ASR data...'})
            
            filenames = self.get_filenames_for_intent(intent_mapping_file, intent)
            self.send_sse({'type': 'progress', 'message': f'  Found {len(filenames)} calls'})
            
            filtered_asr_path = self.filter_asr_by_filenames(asr_file, filenames, intent)
            self.send_sse({'type': 'progress', 'message': f'  ✅ Filtered ASR created'})
            
            self.send_sse({'type': 'progress', 'message': 'Running analysis pipeline...'})
            
            if not os.path.exists('universal_pipeline.py'):
                self.send_sse({'type': 'error', 'message': 'universal_pipeline.py not found'})
                return
            
            cmd = [
                sys.executable, 'universal_pipeline.py',
                '--client', client,
                '--intent', intent,
                '--input', filtered_asr_path,
                '--output-dir', output_dir,
                '--batch-size', str(batch_size),
                '--workers', str(workers)
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.send_sse({'type': 'progress', 'message': f"  {line}"})
            
            return_code = process.wait()
            
            if return_code == 0:
                client_safe = re.sub(r'\W+', '', client)
                intent_safe = re.sub(r'\W+', '', intent)
                output_path = f"{output_dir}/{client_safe}/{intent_safe}"
                
                results_file = f"{output_path}/analysis_results.csv"
                total_calls = 0
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        total_calls = sum(1 for _ in f) - 1
                
                self.send_sse({'type': 'progress', 'message': f"✅ Complete! {total_calls} calls processed"})
                self.send_sse({'type': 'complete', 'results': {
                    'output_dir': output_path,
                    'total_calls': total_calls,
                    'batches': batch_size,
                    'filtered_asr_file': filtered_asr_path
                }})
            else:
                self.send_sse({'type': 'error', 'message': f'Pipeline failed: exit code {return_code}'})
            
        except Exception as e:
            import traceback
            self.send_sse({'type': 'error', 'message': f'Exception: {str(e)}'})
            self.send_sse({'type': 'progress', 'message': traceback.format_exc()})
    
    def get_filenames_for_intent(self, intent_file, target_intent):
        """Get list of filenames for a specific intent"""
        df = pd.read_csv(intent_file, sep='\t')
        
        if 'Intent_Category' in df.columns and 'Filename' in df.columns:
            filtered = df[df['Intent_Category'] == target_intent]
            if 'L3_Score' in df.columns:
                filtered = filtered[filtered['L3_Score'] == 5]
            return set(filtered['Filename'].tolist())
        
        return set()
    
    def filter_asr_by_filenames(self, asr_file, filenames, intent):
        """Filter ASR file to only include specified filenames"""
        intent_safe = re.sub(r'\W+', '', intent)
        output_dir = f"{WORKING_FOLDER}/filtered_{intent_safe}"
        Path(output_dir).mkdir(exist_ok=True)
        
        filtered_path = f"{output_dir}/asr_filtered.csv"
        
        df = pd.read_csv(asr_file, sep='\t')
        
        if 'Filename' in df.columns:
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
    print("  1. Upload ASR CSV file")
    print("  2. Upload L123 Intent Mapping Excel")
    print("  3. Enter company info")
    print("  4. Edit AI prompt (optional)")
    print("  5. Generate intents (Steps 0→1→2)")
    print("  6. Select intent from table")
    print("  7. Run detailed analysis")
    print("\nPress Ctrl+C to stop\n")
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")
            sys.exit(0)


if __name__ == '__main__':
    main()
