<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Transcript Analysis Pipeline - Integrated</title>
  <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      font-family: 'Open Sans', sans-serif;
      background: #E1EAF2;
      min-height: 100vh;
    }
    
    .header {
      background: white;
      border-bottom: 1px solid #e2e8f0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .header-content {
      max-width: 100%;
      margin: 0 auto;
      padding: 1rem 32px;
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    
    .logo {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: bold;
      font-size: 20px;
    }
    
    .title h1 {
      font-size: 1.5rem;
      font-weight: 700;
      color: #0f172a;
    }
    
    .title p {
      font-size: 0.875rem;
      color: #64748b;
    }
    
    .container {
      max-width: 100%;
      margin: 0 auto;
      padding: 32px;
    }
    
    .card {
      background: white;
      border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      border: 1px solid #e2e8f0;
      margin-bottom: 1.5rem;
    }
    
    .card-header {
      padding: 32px;
      border-bottom: 1px solid #e2e8f0;
      background: #F6F9FB;
    }
    
    .card-header h2 {
      font-size: 13px;
      font-weight: 600;
      color: #526B7A;
    }
    
    .card-header p {
      font-size: 13px;
      color: #526B7A;
      margin-top: 0.25rem;
    }
    
    .card-body {
      padding: 32px;
    }
    
    .upload-area {
      text-align: center;
      padding: 32px;
      border: 2px dashed #cbd5e1;
      border-radius: 8px;
      margin-bottom: 1rem;
    }
    
    .upload-icon {
      width: 48px;
      height: 48px;
      background: #dbeafe;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 0.75rem;
      color: #3b82f6;
      font-size: 1.5rem;
    }
    
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      font-weight: 500;
      cursor: pointer;
      border: none;
      font-size: 1rem;
      transition: all 0.2s;
    }
    
    .btn-primary {
      background: #3b82f6;
      color: white;
    }
    
    .btn-primary:hover:not(:disabled) {
      background: #2563eb;
    }
    
    .btn-secondary {
      background: white;
      color: #475569;
      border: 1px solid #cbd5e1;
    }
    
    .btn-secondary:hover:not(:disabled) {
      background: #f8fafc;
    }
    
    .btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    
    .form-group {
      margin-bottom: 1.5rem;
    }
    
    .form-group label {
      display: block;
      font-size: 0.875rem;
      font-weight: 500;
      color: #334155;
      margin-bottom: 0.5rem;
    }
    
    .form-group input, .form-group textarea {
      width: 100%;
      padding: 0.625rem 1rem;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 13px;
      font-family: inherit;
    }
    
    .form-group textarea {
      font-family: 'Courier New', monospace;
      resize: vertical;
      min-height: 200px;
    }
    
    .form-group input:focus, .form-group textarea:focus {
      outline: none;
      border-color: #3b82f6;
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    table {
      width: 100%;
      border-collapse: collapse;
    }
    
    thead {
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
    }
    
    th {
      padding: 12px 32px;
      text-align: left;
      font-size: 13px;
      font-weight: 600;
      color: #526B7A;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      cursor: pointer;
      user-select: none;
    }
    
    th:hover {
      background: #f1f5f9;
    }
    
    th.sortable::after {
      content: ' ↕';
      opacity: 0.3;
    }
    
    th.sorted-asc::after {
      content: ' ↑';
      opacity: 1;
    }
    
    th.sorted-desc::after {
      content: ' ↓';
      opacity: 1;
    }
    
    th.text-right {
      text-align: right;
    }
    
    td {
      padding: 0 32px;
      height: 40px;
      border-bottom: 1px solid #e2e8f0;
      font-size: 13px;
      color: #2e2e2e;
    }
    
    td.text-right {
      text-align: right;
    }
    
    tbody tr:hover {
      background: #F6F9FB;
    }
    
    .alert {
      padding: 1rem;
      border-radius: 8px;
      margin-bottom: 1.5rem;
      display: flex;
      gap: 0.75rem;
      align-items: start;
    }
    
    .alert-error {
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #991b1b;
    }
    
    .alert-info {
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      color: #1e40af;
    }
    
    .alert-success {
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      color: #166534;
    }
    
    .spinner {
      display: inline-block;
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    
    .log-container {
      background: #1e293b;
      color: #e2e8f0;
      padding: 1.5rem;
      border-radius: 8px;
      max-height: 400px;
      overflow-y: auto;
      font-family: 'Courier New', monospace;
      font-size: 0.875rem;
    }
    
    .log-line {
      margin-bottom: 0.25rem;
    }
    
    .hidden {
      display: none;
    }
    
    .flex-between {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .btn-group {
      display: flex;
      gap: 1rem;
      justify-content: flex-end;
      margin-top: 1.5rem;
    }
    
    input[type="file"] {
      display: none;
    }
    
    .status-badge {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
    }
    
    .status-uploaded {
      background: #dcfce7;
      color: #166534;
    }
    
    .form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="header-content">
      <div class="logo">📊</div>
      <div class="title">
        <h1>Transcript Analysis Pipeline</h1>
        <p>Generate intents → Select intent → Analyze calls</p>
      </div>
    </div>
  </div>
  
  <div class="container">
    <!-- Error/Info Display -->
    <div id="errorAlert" class="alert alert-error hidden">
      <span>⚠️</span>
      <div>
        <strong>Error</strong>
        <p id="errorMessage"></p>
      </div>
    </div>
    
    <div id="infoAlert" class="alert alert-info hidden">
      <span>ℹ️</span>
      <div id="infoMessage"></div>
    </div>
    
    <!-- PHASE 1: Upload and Generate Intents -->
    <div id="phase1" class="phase">
      <!-- Upload ASR File -->
      <div class="card">
        <div class="card-header">
          <h2>📁 Step 1: Upload ASR/Transcript File</h2>
          <p>Upload your asr-whisper_001.csv or nx_transcripts.csv file</p>
        </div>
        <div class="card-body">
          <div class="upload-area">
            <div class="upload-icon" id="asrIcon">📁</div>
            <h3 id="asrTitle" style="margin-bottom: 0.5rem;">Select ASR File</h3>
            <p style="color: #64748b; font-size: 0.875rem;">CSV file with transcript data</p>
            <label for="asrFileInput" class="btn btn-primary" style="margin-top: 1rem;">
              <span>📤</span> Select File
            </label>
            <input type="file" id="asrFileInput" accept=".csv">
          </div>
          <div id="asrStatus" class="hidden" style="margin-top: 1rem;">
            <span class="status-badge status-uploaded">✓ Uploaded</span>
            <span style="margin-left: 1rem; color: #64748b; font-size: 0.875rem;" id="asrFileName"></span>
          </div>
        </div>
      </div>
      
      <!-- Upload Mapping File -->
      <div class="card">
        <div class="card-header">
          <h2>📋 Step 2: Upload Intent Mapping File</h2>
          <p>Upload your L123Intent_AgentTaskLabel_Mapping.xlsx file</p>
        </div>
        <div class="card-body">
          <div class="upload-area">
            <div class="upload-icon" id="mappingIcon">📄</div>
            <h3 id="mappingTitle" style="margin-bottom: 0.5rem;">Select Mapping File</h3>
            <p style="color: #64748b; font-size: 0.875rem;">Excel file with L1/L2/L3 taxonomy</p>
            <label for="mappingFileInput" class="btn btn-primary" style="margin-top: 1rem;">
              <span>📤</span> Select File
            </label>
            <input type="file" id="mappingFileInput" accept=".xlsx,.xls">
          </div>
          <div id="mappingStatus" class="hidden" style="margin-top: 1rem;">
            <span class="status-badge status-uploaded">✓ Uploaded</span>
            <span style="margin-left: 1rem; color: #64748b; font-size: 0.875rem;" id="mappingFileName"></span>
          </div>
        </div>
      </div>
      
      <!-- Company Information -->
      <div id="companyCard" class="card hidden">
        <div class="card-header">
          <h2>🏢 Step 3: Enter Company Information</h2>
          <p>Provide context for AI intent generation</p>
        </div>
        <div class="card-body">
          <div class="form-grid">
            <div class="form-group">
              <label>Company Name *</label>
              <input type="text" id="companyName" placeholder="e.g., Acme Financial Services">
            </div>
            <div class="form-group">
              <label>File Format</label>
              <select id="formatSelect" style="width: 100%; padding: 0.625rem 1rem; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 13px;">
                <option value="whisper">Whisper Format</option>
                <option value="nx">NX Format</option>
              </select>
            </div>
          </div>
          <div class="form-group">
            <label>Company Description *</label>
            <input type="text" id="companyDescription" placeholder="e.g., A leading provider of banking and investment services">
          </div>
          <div class="form-grid">
            <div class="form-group">
              <label>Maximum Calls to Process *</label>
              <input type="number" id="maxCalls" value="100" min="10" max="50000" placeholder="100">
              <p style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">
                Start with 100 calls (~30 min). Increase after testing.
              </p>
            </div>
            <div class="form-group">
              <label>Estimated Time</label>
              <div style="padding: 0.625rem 1rem; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                <span id="estimatedTime" style="font-weight: 600; color: #3b82f6;">~30 minutes</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Prompt Editor -->
      <div id="promptCard" class="card hidden">
        <div class="card-header">
          <h2>✏️ Step 4: Review AI Prompt (Optional)</h2>
          <p>Customize the prompt used for intent generation</p>
        </div>
        <div class="card-body">
          <div class="form-group">
            <label>Intent Generation Prompt</label>
            <textarea id="promptTemplate"></textarea>
            <p style="font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">
              Variables: {company_name}, {company_description}, {conv}
            </p>
          </div>
          <div class="btn-group">
            <button class="btn btn-secondary" onclick="resetToDefaultPrompt()">Reset to Default</button>
            <button class="btn btn-primary" onclick="generateIntents()">
              <span>🚀</span> Generate Intents
            </button>
          </div>
        </div>
      </div>
      
      <!-- Progress Log -->
      <div id="progressCard" class="card hidden">
        <div class="card-header">
          <h2 id="progressTitle">⏳ Generating Intents...</h2>
        </div>
        <div class="card-body">
          <div class="log-container" id="logContainer"></div>
        </div>
      </div>
    </div>
    
    <!-- PHASE 2: Intent Selection -->
    <div id="phase2" class="phase hidden">
      <button class="btn btn-secondary" onclick="startOver()" style="margin-bottom: 1rem;">
        ← Start Over
      </button>
      
      <div class="card">
        <div class="card-header">
          <div class="flex-between">
            <div>
              <h2>📊 Select Intent to Analyze</h2>
              <p id="intentCount"></p>
            </div>
          </div>
        </div>
        <div style="overflow-x: auto;">
          <table id="resultsTable">
            <thead>
              <tr>
                <th class="sortable sorted-desc" data-sort="level1" onclick="sortTable('level1')">Category</th>
                <th class="sortable" data-sort="level2" onclick="sortTable('level2')">Topic</th>
                <th class="sortable" data-sort="level3" onclick="sortTable('level3')">Intent</th>
                <th class="sortable text-right" data-sort="volume" onclick="sortTable('volume')">Volume</th>
                <th class="sortable text-right" data-sort="percentage" onclick="sortTable('percentage')">% of Total</th>
                <th class="text-right">Action</th>
              </tr>
            </thead>
            <tbody id="resultsBody"></tbody>
          </table>
        </div>
      </div>
    </div>
    
    <!-- PHASE 3: Run Analysis -->
    <div id="phase3" class="phase hidden">
      <button class="btn btn-secondary" onclick="backToIntentSelection()" style="margin-bottom: 1rem;">
        ← Back to Intent Selection
      </button>
      
      <!-- Selected Intent Display -->
      <div class="card" style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-color: #93c5fd;">
        <div class="card-body">
          <p style="font-size: 0.875rem; color: #1e40af; font-weight: 500; margin-bottom: 0.5rem;">Selected Intent</p>
          <h2 style="font-size: 1.5rem; font-weight: 700; color: #0f172a;" id="selectedIntentName"></h2>
          <div style="display: flex; gap: 1.5rem; margin-top: 1rem; font-size: 0.875rem;">
            <span style="color: #64748b;">
              <strong id="selectedIntentVolume"></strong> calls
            </span>
            <span style="color: #64748b;">
              <strong id="selectedIntentPercent"></strong> of total
            </span>
          </div>
        </div>
      </div>
      
      <!-- Results Display -->
      <div id="resultsDisplay" class="card hidden">
        <div class="card-body">
          <div style="display: flex; align-items: start; gap: 1rem;">
            <span style="font-size: 2rem;">✅</span>
            <div style="flex: 1;">
              <h3 style="font-size: 1.125rem; font-weight: 700; color: #166534; margin-bottom: 0.5rem;">
                Analysis Complete!
              </h3>
              <p style="font-size: 0.875rem; color: #15803d;" id="resultsSummary"></p>
            </div>
          </div>
          
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1.5rem;">
            <button class="btn btn-secondary" onclick="downloadFile('results')" style="width: 100%;">
              📥 Download Results CSV
            </button>
            <button class="btn btn-secondary" onclick="downloadFile('normalized')" style="width: 100%;">
              📥 Download Normalized CSV
            </button>
            <button class="btn btn-secondary" onclick="downloadFile('summary')" style="width: 100%;">
              📥 Download Summary JSON
            </button>
          </div>
        </div>
      </div>
      
      <!-- Configuration -->
      <div id="configForm">
        <div class="card">
          <div class="card-header">
            <h2>⚙️ Analysis Configuration</h2>
          </div>
          <div class="card-body">
            <div class="form-grid">
              <div class="form-group">
                <label>Client Name *</label>
                <input type="text" id="clientName" placeholder="Enter client name">
              </div>
              <div class="form-group">
                <label>Output Folder *</label>
                <input type="text" id="outputFolder" value="./data">
              </div>
              <div class="form-group">
                <label>Batch Size</label>
                <input type="number" id="batchSize" value="3" min="1" max="10">
              </div>
              <div class="form-group">
                <label>Parallel Workers</label>
                <input type="number" id="workers" value="3" min="1" max="10">
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Analysis Progress -->
      <div id="analysisProgressCard" class="card hidden">
        <div class="card-header">
          <h2 id="analysisProgressTitle">⏳ Running Analysis...</h2>
        </div>
        <div class="card-body">
          <div class="log-container" id="analysisLogContainer"></div>
        </div>
      </div>
      
      <!-- Action Buttons -->
      <div class="btn-group" id="actionButtons">
        <button class="btn btn-secondary" onclick="backToIntentSelection()">Cancel</button>
        <button class="btn btn-primary" onclick="runAnalysis()">
          ▶️ Run Analysis
        </button>
      </div>
    </div>
  </div>
  
  <script>
    const API_BASE = 'http://localhost:5000';
    
    // State variables
    let asrFile = null;
    let mappingFile = null;
    let categoriesTxt = null;
    let intentData = null;
    let selectedIntent = null;
    let pipelineResults = null;
    let currentSort = { column: 'volume', direction: 'desc' };
    
    // Default prompt
    const defaultPrompt = `You are analyzing customer service conversations for {company_name}.

Company Description: {company_description}

Please analyze the following conversation and identify the primary customer intent in 5-10 words.

The intent should:
- Be specific to what the customer wants to accomplish
- Focus on the customer's goal, not the agent's actions
- Be phrased from the customer's perspective
- Be concise and actionable

Conversation:
{conv}

Respond with ONLY the intent phrase, no additional commentary.

Examples of good intents:
- "Update billing address for account"
- "Report unauthorized transaction on credit card"
- "Request refund for cancelled service"
- "Inquire about product warranty coverage"

Your response (intent only):`;
    
    // Initialize
    document.addEventListener('DOMContentLoaded', function() {
      document.getElementById('promptTemplate').value = defaultPrompt;
      
      // Update estimated time when max calls changes
      document.getElementById('maxCalls').addEventListener('input', updateEstimatedTime);
      updateEstimatedTime(); // Initial calculation
    });
    
    function updateEstimatedTime() {
      const maxCalls = parseInt(document.getElementById('maxCalls').value) || 100;
      const minutesPerCall = 0.3; // ~20 seconds per call average
      const totalMinutes = Math.ceil(maxCalls * minutesPerCall);
      
      let timeText;
      if (totalMinutes < 60) {
        timeText = `~${totalMinutes} minutes`;
      } else {
        const hours = Math.floor(totalMinutes / 60);
        const mins = totalMinutes % 60;
        timeText = `~${hours}h ${mins}m`;
      }
      
      document.getElementById('estimatedTime').textContent = timeText;
    }
    
    // File upload handlers
    document.getElementById('asrFileInput').addEventListener('change', handleASRFileUpload);
    document.getElementById('mappingFileInput').addEventListener('change', handleMappingFileUpload);
    
    async function handleASRFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      
      showLoading('asr');
      hideError();
      
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await fetch(`${API_BASE}/api/upload-asr`, {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const result = await response.json();
        asrFile = result.file_path;
        
        document.getElementById('asrStatus').classList.remove('hidden');
        document.getElementById('asrFileName').textContent = `${file.name} (${result.row_count.toLocaleString()} rows)`;
        hideLoading('asr');
        
        showInfo(`✓ ASR file uploaded: ${result.row_count.toLocaleString()} rows`);
        checkBothFilesUploaded();
        
      } catch (error) {
        showError('Failed to upload ASR file: ' + error.message);
        hideLoading('asr');
      }
    }
    
    async function handleMappingFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      
      showLoading('mapping');
      hideError();
      
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await fetch(`${API_BASE}/api/upload-mapping`, {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const result = await response.json();
        mappingFile = result.file_path;
        categoriesTxt = result.categories_txt;
        
        document.getElementById('mappingStatus').classList.remove('hidden');
        document.getElementById('mappingFileName').textContent = file.name;
        hideLoading('mapping');
        
        showInfo('✓ Mapping file uploaded and converted');
        checkBothFilesUploaded();
        
      } catch (error) {
        showError('Failed to upload mapping file: ' + error.message);
        hideLoading('mapping');
      }
    }
    
    function checkBothFilesUploaded() {
      if (asrFile && mappingFile) {
        document.getElementById('companyCard').classList.remove('hidden');
        document.getElementById('promptCard').classList.remove('hidden');
      }
    }
    
    async function generateIntents() {
      const companyName = document.getElementById('companyName').value.trim();
      const companyDescription = document.getElementById('companyDescription').value.trim();
      const promptTemplate = document.getElementById('promptTemplate').value;
      const format = document.getElementById('formatSelect').value;
      const maxCalls = parseInt(document.getElementById('maxCalls').value) || 100;
      
      if (!companyName || !companyDescription) {
        showError('Please enter company name and description');
        return;
      }
      
      if (maxCalls < 10 || maxCalls > 50000) {
        showError('Max calls must be between 10 and 50,000');
        return;
      }
      
      // Warn for large numbers
      if (maxCalls > 1000) {
        const proceed = confirm(
          `You're about to process ${maxCalls.toLocaleString()} calls.\n` +
          `This will take approximately ${calculateTime(maxCalls)}.\n\n` +
          `We recommend starting with 100-500 calls for testing.\n\n` +
          `Continue with ${maxCalls.toLocaleString()} calls?`
        );
        if (!proceed) return;
      }
      
      hideError();
      document.getElementById('companyCard').classList.add('hidden');
      document.getElementById('promptCard').classList.add('hidden');
      document.getElementById('progressCard').classList.remove('hidden');
      document.getElementById('logContainer').innerHTML = '';
      
      const requestData = {
        input_csv: asrFile,
        categories_txt: categoriesTxt,
        company_name: companyName,
        company_description: companyDescription,
        prompt_template: promptTemplate,
        format: format,
        max_calls: maxCalls
      };
      
      try {
        const response = await fetch(`${API_BASE}/api/generate-intents`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestData)
        });
        
        if (!response.ok) throw new Error('Intent generation failed');
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'progress') {
                  addLogLine(data.message);
                } else if (data.type === 'complete') {
                  intentData = data.results.intents;
                  showIntentSelection(data.results);
                } else if (data.type === 'error') {
                  showError(data.message);
                  document.getElementById('progressCard').classList.add('hidden');
                  document.getElementById('companyCard').classList.remove('hidden');
                  document.getElementById('promptCard').classList.remove('hidden');
                }
              } catch (e) {
                console.error('Failed to parse SSE:', e);
              }
            }
          }
        }
      } catch (error) {
        showError('Failed to generate intents: ' + error.message);
        document.getElementById('progressCard').classList.add('hidden');
        document.getElementById('companyCard').classList.remove('hidden');
        document.getElementById('promptCard').classList.remove('hidden');
      }
    }
    
    function calculateTime(calls) {
      const minutes = Math.ceil(calls * 0.3);
      if (minutes < 60) return `${minutes} minutes`;
      const hours = Math.floor(minutes / 60);
      const mins = minutes % 60;
      return `${hours} hour${hours > 1 ? 's' : ''} ${mins} minutes`;
    }
    
    function showIntentSelection(results) {
      document.getElementById('phase1').classList.add('hidden');
      document.getElementById('phase2').classList.remove('hidden');
      document.getElementById('intentCount').textContent = 
        `${results.total_intents} unique intents identified`;
      
      // Store the mapping file for later use
      window.generatedMappingFile = results.intent_mapping_file;
      
      renderTable(intentData);
    }
    
    function renderTable(intents) {
      const tbody = document.getElementById('resultsBody');
      tbody.innerHTML = '';
      
      intents.forEach((intent) => {
        const row = tbody.insertRow();
        row.dataset.intentData = JSON.stringify(intent);
        row.innerHTML = `
          <td>${intent.level1}</td>
          <td>${intent.level2}</td>
          <td>${intent.level3}</td>
          <td class="text-right">${intent.volume}</td>
          <td class="text-right">${intent.percentage}%</td>
          <td class="text-right">
            <button class="btn btn-primary" style="height: 28px; padding: 0 1rem; font-size: 13px; border-radius: 4px;" onclick="selectIntent(this)">
              ▶️ Analyze
            </button>
          </td>
        `;
      });
    }
    
    function selectIntent(button) {
      const row = button.closest('tr');
      selectedIntent = JSON.parse(row.dataset.intentData);
      
      document.getElementById('phase2').classList.add('hidden');
      document.getElementById('phase3').classList.remove('hidden');
      
      document.getElementById('selectedIntentName').textContent = selectedIntent.intent;
      document.getElementById('selectedIntentVolume').textContent = selectedIntent.volume;
      document.getElementById('selectedIntentPercent').textContent = selectedIntent.percentage + '%';
    }
    
    async function runAnalysis() {
      const clientName = document.getElementById('clientName').value.trim();
      const outputFolder = document.getElementById('outputFolder').value.trim();
      const batchSize = parseInt(document.getElementById('batchSize').value);
      const workers = parseInt(document.getElementById('workers').value);
      
      if (!clientName) {
        showError('Please enter client name');
        return;
      }
      
      hideError();
      document.getElementById('configForm').classList.add('hidden');
      document.getElementById('actionButtons').classList.add('hidden');
      document.getElementById('analysisProgressCard').classList.remove('hidden');
      document.getElementById('analysisLogContainer').innerHTML = '';
      
      const requestData = {
        intent: selectedIntent.intent,
        intent_mapping_file: window.generatedMappingFile,
        asr_file: asrFile,
        client: clientName,
        output_dir: outputFolder,
        batch_size: batchSize,
        workers: workers
      };
      
      try {
        const response = await fetch(`${API_BASE}/api/filter-and-run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestData)
        });
        
        if (!response.ok) throw new Error('Analysis failed');
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'progress') {
                  addAnalysisLogLine(data.message);
                } else if (data.type === 'complete') {
                  pipelineResults = data.results;
                  showAnalysisComplete(data.results);
                } else if (data.type === 'error') {
                  showError(data.message);
                  document.getElementById('analysisProgressCard').classList.add('hidden');
                  document.getElementById('configForm').classList.remove('hidden');
                  document.getElementById('actionButtons').classList.remove('hidden');
                }
              } catch (e) {
                console.error('Failed to parse SSE:', e);
              }
            }
          }
        }
      } catch (error) {
        showError('Failed to run analysis: ' + error.message);
        document.getElementById('analysisProgressCard').classList.add('hidden');
        document.getElementById('configForm').classList.remove('hidden');
        document.getElementById('actionButtons').classList.remove('hidden');
      }
    }
    
    function showAnalysisComplete(results) {
      document.getElementById('analysisProgressTitle').textContent = '✅ Analysis Complete';
      document.getElementById('analysisProgressCard').classList.add('hidden');
      document.getElementById('resultsDisplay').classList.remove('hidden');
      document.getElementById('resultsSummary').textContent = 
        `Successfully processed ${results.total_calls} calls`;
    }
    
    async function downloadFile(fileType) {
      if (!pipelineResults) return;
      
      try {
        const response = await fetch(
          `${API_BASE}/api/download/${fileType}?path=${encodeURIComponent(pipelineResults.output_dir)}`
        );
        
        if (!response.ok) throw new Error('Download failed');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${fileType}.${fileType === 'summary' ? 'json' : 'csv'}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } catch (error) {
        showError('Failed to download file: ' + error.message);
      }
    }
    
    // Navigation
    function startOver() {
      window.location.reload();
    }
    
    function backToIntentSelection() {
      document.getElementById('phase3').classList.add('hidden');
      document.getElementById('phase2').classList.remove('hidden');
      document.getElementById('resultsDisplay').classList.add('hidden');
      document.getElementById('analysisProgressCard').classList.add('hidden');
      document.getElementById('configForm').classList.remove('hidden');
      document.getElementById('actionButtons').classList.remove('hidden');
    }
    
    function resetToDefaultPrompt() {
      document.getElementById('promptTemplate').value = defaultPrompt;
    }
    
    // Table sorting
    function sortTable(column) {
      if (!intentData) return;
      
      if (currentSort.column === column) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
      } else {
        currentSort.column = column;
        currentSort.direction = (column === 'volume' || column === 'percentage') ? 'desc' : 'asc';
      }
      
      intentData.sort((a, b) => {
        let aVal = a[column];
        let bVal = b[column];
        
        if (column === 'volume' || column === 'percentage') {
          aVal = parseFloat(aVal);
          bVal = parseFloat(bVal);
        } else {
          aVal = String(aVal).toLowerCase();
          bVal = String(bVal).toLowerCase();
        }
        
        if (currentSort.direction === 'asc') {
          return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
        } else {
          return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
        }
      });
      
      renderTable(intentData);
      updateSortIndicators();
    }
    
    function updateSortIndicators() {
      document.querySelectorAll('th.sortable').forEach(th => {
        th.classList.remove('sorted-asc', 'sorted-desc');
      });
      
      const currentHeader = document.querySelector(`th[data-sort="${currentSort.column}"]`);
      if (currentHeader) {
        currentHeader.classList.add(currentSort.direction === 'asc' ? 'sorted-asc' : 'sorted-desc');
      }
    }
    
    // UI Helpers
    function showLoading(type) {
      const icon = document.getElementById(type + 'Icon');
      const title = document.getElementById(type + 'Title');
      icon.innerHTML = '<div class="spinner"></div>';
      title.textContent = 'Processing...';
    }
    
    function hideLoading(type) {
      const icon = document.getElementById(type + 'Icon');
      const title = document.getElementById(type + 'Title');
      icon.textContent = type === 'asr' ? '📁' : '📄';
      title.textContent = type === 'asr' ? 'Select ASR File' : 'Select Mapping File';
    }
    
    function addLogLine(message) {
      const logContainer = document.getElementById('logContainer');
      const line = document.createElement('div');
      line.className = 'log-line';
      line.textContent = message;
      logContainer.appendChild(line);
      logContainer.scrollTop = logContainer.scrollHeight;
    }
    
    function addAnalysisLogLine(message) {
      const logContainer = document.getElementById('analysisLogContainer');
      const line = document.createElement('div');
      line.className = 'log-line';
      line.textContent = message;
      logContainer.appendChild(line);
      logContainer.scrollTop = logContainer.scrollHeight;
    }
    
    function showError(message) {
      document.getElementById('errorMessage').textContent = message;
      document.getElementById('errorAlert').classList.remove('hidden');
      setTimeout(() => {
        document.getElementById('errorAlert').classList.add('hidden');
      }, 10000);
    }
    
    function hideError() {
      document.getElementById('errorAlert').classList.add('hidden');
    }
    
    function showInfo(message) {
      document.getElementById('infoMessage').innerHTML = message;
      document.getElementById('infoAlert').classList.remove('hidden');
    }
    
    function hideInfo() {
      document.getElementById('infoAlert').classList.add('hidden');
    }
  </script>
</body>
</html>
