#!/usr/bin/env python3
"""
Simplified API Server - Matches the Jupyter Notebook Workflow
Step 1: Filter ASR data by intent
Step 2: Run pipeline on filtered data
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
import cgi
import time
import re

csv.field_size_limit(sys.maxsize)

PORT = 5000
UPLOAD_FOLDER = './uploads'
WORKING_FOLDER = './data'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(WORKING_FOLDER).mkdir(exist_ok=True)

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
        if self.path == '/api/upload-intent-mapping':
            self.handle_upload_intent_mapping()
        elif self.path == '/api/upload-asr':
            self.handle_upload_asr()
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
    
    def handle_upload_intent_mapping(self):
        """Upload the intent mapping file (reasons_mapped_001.csv)"""
        try:
            file_data, filename = self.parse_multipart()
            
            if not file_data or not filename:
                self.send_error(400, "No file uploaded")
                return
            
            # Save file
            timestamp = int(time.time())
            file_path = os.path.join(UPLOAD_FOLDER, f'intent_mapping_{timestamp}.csv')
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # Analyze intents
            intents = self.extract_intents(file_path)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'success': True,
                'file_path': file_path,
                'intents': intents,
                'total_intents': len(intents)
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, f"Upload failed: {str(e)}")
    
    def handle_upload_asr(self):
        """Upload the full ASR file (asr-whisper_001.csv)"""
        try:
            file_data, filename = self.parse_multipart()
            
            if not file_data or not filename:
                self.send_error(400, "No file uploaded")
                return
            
            # Save file
            timestamp = int(time.time())
            file_path = os.path.join(UPLOAD_FOLDER, f'asr_full_{timestamp}.csv')
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'success': True,
                'file_path': file_path,
                'message': 'ASR file uploaded successfully'
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error(500, f"Upload failed: {str(e)}")
    
    def extract_intents(self, intent_file_path):
        """Extract unique intents from the mapping file"""
        intents_dict = {}
        
        print(f"📊 Reading intent file: {intent_file_path}")
        
        # Try to detect delimiter
        with open(intent_file_path, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            delimiter = '\t' if '\t' in sample else ','
            delimiter_name = 'TAB' if delimiter == '\t' else 'COMMA'
            print(f"  Detected delimiter: {delimiter_name}")
        
        with open(intent_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Print column names
            print(f"  Columns found: {', '.join(reader.fieldnames or [])}")
            
            # Find the right column names
            score_col = None
            intent_col = None
            
            for col in reader.fieldnames or []:
                if 'score' in col.lower() and 'l3' in col.lower():
                    score_col = col
                if 'intent' in col.lower() and 'category' in col.lower():
                    intent_col = col
            
            print(f"  Using score column: {score_col}")
            print(f"  Using intent column: {intent_col}")
            
            if not intent_col:
                print("  ⚠️ WARNING: Could not find Intent_Category column")
                return []
            
            row_count = 0
            high_conf_count = 0
            
            for row in reader:
                row_count += 1
                
                # Check score if available
                if score_col:
                    score = row.get(score_col, '')
                    if score not in ['5', '5.0']:
                        continue
                
                high_conf_count += 1
                intent = row.get(intent_col, 'Unknown')
                
                if intent not in intents_dict:
                    intents_dict[intent] = 0
                intents_dict[intent] += 1
            
            print(f"  Total rows: {row_count}")
            print(f"  High confidence rows (score=5): {high_conf_count}")
            print(f"  Unique intents: {len(intents_dict)}")
        
        # Convert to list format
        total = sum(intents_dict.values())
        intents = []
        
        for intent, volume in sorted(intents_dict.items(), key=lambda x: x[1], reverse=True):
            parts = intent.split('-')
            intents.append({
                'intent': intent,
                'volume': volume,
                'percentage': round((volume / total * 100), 1) if total > 0 else 0,
                'level1': parts[0].strip() if len(parts) > 0 else 'General',
                'level2': parts[1].strip() if len(parts) > 1 else 'Support',
                'level3': parts[2].strip() if len(parts) > 2 else 'Inquiry'
            })
        
        print(f"  ✅ Returning {len(intents)} intents\n")
        return intents
    
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
        
        with open(intent_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                if row.get('Intent_Category') == target_intent and row.get('L3_Score') == '5':
                    filenames.add(row.get('Filename'))
        
        return filenames
    
    def filter_asr_by_filenames(self, asr_file, filenames, intent):
        """Filter ASR file to only include specified filenames"""
        # Create output directory
        intent_safe = re.sub(r'\W+', '', intent)
        output_dir = f"{WORKING_FOLDER}/002_improve_actions_{intent_safe}"
        Path(output_dir).mkdir(exist_ok=True)
        
        filtered_path = f"{output_dir}/asr_filtered.csv"
        
        # Read and filter
        with open(asr_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile, delimiter='\t')
            fieldnames = reader.fieldnames
            
            with open(filtered_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()
                
                for row in reader:
                    if row.get('Filename') in filenames:
                        writer.writerow(row)
        
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
    print("🚀 TRANSCRIPT ANALYSIS API SERVER")
    print("   Matches Jupyter Notebook Workflow")
    print("=" * 80)
    print(f"📂 Upload folder: {UPLOAD_FOLDER}")
    print(f"📂 Working folder: {WORKING_FOLDER}")
    print(f"🌐 Server running on http://localhost:{PORT}")
    print(f"📊 Ready to process transcript files!")
    print("=" * 80)
    print("\nWorkflow:")
    print("  1. Upload intent mapping file (reasons_mapped_001.csv)")
    print("  2. Upload full ASR file (asr-whisper_001.csv)")
    print("  3. Select intent → filters ASR → runs pipeline")
    print("\nPress Ctrl+C to stop the server\n")
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")
            sys.exit(0)

if __name__ == '__main__':
    main()