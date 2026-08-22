/**
 * Voice-Enabled Guardrailed RAG - Frontend Dashboard Controller
 */

let currentLang = 'hi';
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// DOM Elements
const recordBtn = document.getElementById('record-btn');
const recordingStatus = document.getElementById('recording-status');
const textInput = document.getElementById('text-query-input');
const sendTextBtn = document.getElementById('send-text-btn');
const langChips = document.querySelectorAll('.lang-chip');

// Stepper Timers
const timerListen = document.getElementById('timer-listen');
const timerStt = document.getElementById('timer-stt');
const timerRetrieval = document.getElementById('timer-retrieval');
const timerGen = document.getElementById('timer-gen');

// Nodes
const stepListen = document.getElementById('step-listen');
const stepStt = document.getElementById('step-stt');
const stepRetrieval = document.getElementById('step-retrieval');
const stepGen = document.getElementById('step-gen');

// Outputs
const transcriptDisplay = document.getElementById('transcript-display');
const detectedLangChip = document.getElementById('detected-lang-chip');
const guardrailBadge = document.getElementById('guardrail-badge');
const answerDisplay = document.getElementById('answer-display');
const confidenceBadge = document.getElementById('confidence-badge');
const totalTimeBadge = document.getElementById('total-time-badge');
const sourcesList = document.getElementById('sources-list');
const sourcesCount = document.getElementById('sources-count');
const toggleSourcesHeader = document.getElementById('toggle-sources-header');

// Modal Elements
const toggleDashboardBtn = document.getElementById('toggle-dashboard-btn');
const dashboardModal = document.getElementById('dashboard-modal');
const closeModalBtn = document.getElementById('close-modal-btn');

// Latency Bars
const barStt = document.getElementById('bar-stt');
const valStt = document.getElementById('val-stt');
const barRetrieval = document.getElementById('bar-retrieval');
const valRetrieval = document.getElementById('val-retrieval');
const barGen = document.getElementById('bar-gen');
const valGen = document.getElementById('val-gen');
const barE2e = document.getElementById('bar-e2e');
const valE2e = document.getElementById('val-e2e');


// 1. Language Selector
langChips.forEach(chip => {
  chip.addEventListener('click', () => {
    langChips.forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    currentLang = chip.dataset.lang;
  });
});

// 2. Sources Collapsible Toggle
toggleSourcesHeader.addEventListener('click', () => {
  sourcesList.classList.toggle('collapsed');
  if (sourcesList.style.display === 'none') {
    sourcesList.style.display = 'flex';
  } else {
    sourcesList.style.display = 'none';
  }
});

// 3. Modal Toggles
toggleDashboardBtn.addEventListener('click', () => {
  dashboardModal.classList.remove('hidden');
});

closeModalBtn.addEventListener('click', () => {
  dashboardModal.classList.add('hidden');
});

dashboardModal.addEventListener('click', (e) => {
  if (e.target === dashboardModal) {
    dashboardModal.classList.add('hidden');
  }
});


// 4. Stepper Reset & Animation
function resetStepper() {
  [stepListen, stepStt, stepRetrieval, stepGen].forEach(node => {
    node.classList.remove('active', 'completed');
  });
  timerListen.textContent = '-- ms';
  timerStt.textContent = '-- ms';
  timerRetrieval.textContent = '-- ms';
  timerGen.textContent = '-- ms';
}

function setStepActive(stepNode) {
  [stepListen, stepStt, stepRetrieval, stepGen].forEach(n => n.classList.remove('active'));
  stepNode.classList.add('active');
}

function setStepCompleted(stepNode, timerEl, durationMs) {
  stepNode.classList.remove('active');
  stepNode.classList.add('completed');
  timerEl.textContent = `${durationMs.toFixed(1)} ms`;
}


// 5. Audio Recorder Setup
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      await processAudioQuery(audioBlob);
      stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    recordBtn.classList.add('recording');
    recordingStatus.textContent = 'RECORDING // SPEAK NOW... CLICK AGAIN TO FINISH';
    resetStepper();
    setStepActive(stepListen);

  } catch (err) {
    console.error('Microphone error:', err);
    recordingStatus.textContent = 'MIC ACCESS DENIED // ENTER TEXT QUERY';
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    recordBtn.classList.remove('recording');
    recordingStatus.textContent = 'PROCESSING AUDIO SIGNAL...';
  }
}

recordBtn.addEventListener('click', () => {
  if (!isRecording) {
    startRecording();
  } else {
    stopRecording();
  }
});


// 6. Text Query Handler
sendTextBtn.addEventListener('click', () => {
  const query = textInput.value.trim();
  if (query) {
    processTextQuery(query);
  }
});

textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const query = textInput.value.trim();
    if (query) processTextQuery(query);
  }
});


// 7. API Processing
async function processAudioQuery(audioBlob) {
  setStepCompleted(stepListen, timerListen, 120.0);
  setStepActive(stepStt);

  const formData = new FormData();
  formData.append('file', audioBlob, 'voice_query.webm');
  formData.append('language', currentLang);
  formData.append('top_k', 5);

  try {
    const t0 = performance.now();
    const res = await fetch('/api/query', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    renderPipelineResponse(data);
  } catch (err) {
    renderError(err.message);
  }
}

async function processTextQuery(query) {
  resetStepper();
  setStepCompleted(stepListen, timerListen, 0.0);
  setStepCompleted(stepStt, timerStt, 0.0);
  setStepActive(stepRetrieval);

  try {
    const res = await fetch('/api/query-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, language: currentLang, top_k: 5 })
    });
    const data = await res.json();
    renderPipelineResponse(data);
  } catch (err) {
    renderError(err.message);
  }
}


// 8. Render Response
function renderPipelineResponse(data) {
  const lat = data.latency_breakdown;
  const guards = data.guardrails;

  // Complete Stepper
  setStepCompleted(stepStt, timerStt, lat.stt_ms);
  setStepCompleted(stepRetrieval, timerRetrieval, lat.retrieval_ms);
  setStepCompleted(stepGen, timerGen, lat.generation_ms);

  // Transcript Box
  transcriptDisplay.textContent = data.query_text;
  detectedLangChip.textContent = `LANG: ${data.language.toUpperCase()}`;

  // Guardrail Badge
  if (!guards.input_guard_passed || !guards.retrieval_guard_passed) {
    guardrailBadge.textContent = `GUARD: REFUSED (${guards.confidence_level})`;
    guardrailBadge.className = 'badge badge-guard refused mono';
  } else {
    guardrailBadge.textContent = 'GUARD: PASSED';
    guardrailBadge.className = 'badge badge-guard mono';
  }

  // Answer Box
  answerDisplay.textContent = data.answer;
  confidenceBadge.textContent = `CONFIDENCE: ${data.confidence.toUpperCase()}`;
  const confLower = data.confidence.toLowerCase();
  if (confLower.includes('high')) {
    confidenceBadge.className = 'badge badge-confidence high mono';
  } else if (confLower.includes('fallback')) {
    confidenceBadge.className = 'badge badge-confidence fallback mono';
  } else if (confLower.includes('refusal') || confLower.includes('low')) {
    confidenceBadge.className = 'badge badge-confidence refused mono';
  } else {
    confidenceBadge.className = 'badge badge-confidence fallback mono';
  }

  totalTimeBadge.textContent = `E2E: ${lat.total_e2e_ms.toFixed(1)} ms`;
  recordingStatus.textContent = 'STANDBY // READY FOR NEXT QUERY';

  // Render Sources
  renderSources(data.sources);

  // Update Latency Bars
  updateLatencyDashboard(lat);
}

function renderSources(sources) {
  sourcesCount.textContent = sources.length;
  if (!sources || sources.length === 0) {
    sourcesList.innerHTML = '<div class="empty-sources mono">No passages retrieved or query refused.</div>';
    return;
  }

  sourcesList.innerHTML = sources.map((src, idx) => `
    <div class="source-item">
      <div class="source-meta mono">
        <div class="source-tags">
          <span class="source-tag">[Source ${idx + 1}]</span>
          <span class="source-tag">${src.lang.toUpperCase()}</span>
          <span class="source-tag">${src.chunk_strategy.toUpperCase()}</span>
          ${src.is_selected === 1 ? '<span class="source-tag" style="color:var(--accent-emerald);">SELECTED</span>' : ''}
        </div>
        <span class="source-score">SCORE: ${src.score.toFixed(3)}</span>
      </div>
      <div class="source-body">${escapeHtml(src.text)}</div>
    </div>
  `).join('');
}

function updateLatencyDashboard(lat) {
  valStt.textContent = `${lat.stt_ms.toFixed(1)} ms`;
  valRetrieval.textContent = `${lat.retrieval_ms.toFixed(1)} ms`;
  valGen.textContent = `${lat.generation_ms.toFixed(1)} ms`;
  valE2e.textContent = `${lat.total_e2e_ms.toFixed(1)} ms`;

  const maxVal = Math.max(lat.total_e2e_ms, 100);
  barStt.style.width = `${Math.min(100, (lat.stt_ms / maxVal) * 100)}%`;
  barRetrieval.style.width = `${Math.min(100, (lat.retrieval_ms / maxVal) * 100)}%`;
  barGen.style.width = `${Math.min(100, (lat.generation_ms / maxVal) * 100)}%`;
  barE2e.style.width = `100%`;
}

function renderError(msg) {
  answerDisplay.textContent = `[Error]: ${msg}`;
  recordingStatus.textContent = 'ERROR ENCOUNTERED // RETRY';
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
