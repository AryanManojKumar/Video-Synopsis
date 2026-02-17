// ── State ──────────────────────────────────────────────────────────
let selectedFile = null;
let jobId = null;
let summaryData = null;

// ── Elements ───────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const screenUpload = $('screenUpload');
const screenProcessing = $('screenProcessing');
const screenResults = $('screenResults');
const dropZone = $('dropZone');
const fileInput = $('fileInput');
const fileInfo = $('fileInfo');
const compressionSlider = $('compressionSlider');
const compressionVal = $('compressionVal');
const btnGenerate = $('btnGenerate');
const processingStatus = $('processingStatus');
const progressFill = $('progressFill');
const navStats = $('navStats');
const statsGrid = $('statsGrid');
const tubeTable = $('tubeTable');
const videoSynopsis = $('videoSynopsis');
const videoOriginal = $('videoOriginal');
const synTime = $('synTime');
const origTime = $('origTime');
const btnDownload = $('btnDownload');

// ── Screen Navigation ──────────────────────────────────────────────
function showScreen(screen) {
    [screenUpload, screenProcessing, screenResults].forEach(s => s.classList.add('hidden'));
    screen.classList.remove('hidden');
}

// ── Upload Flow ────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    const validExts = ['.mp4', '.avi', '.mov'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExts.includes(ext)) {
        alert('Please upload a video file (.mp4, .avi, or .mov)');
        return;
    }
    selectedFile = file;
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    fileInfo.textContent = `✓ ${file.name} (${sizeMB} MB)`;
    fileInfo.style.display = 'block';
    btnGenerate.disabled = false;
}

compressionSlider.addEventListener('input', () => {
    compressionVal.textContent = compressionSlider.value;
});

btnGenerate.addEventListener('click', startProcessing);

// ── Processing ─────────────────────────────────────────────────────
async function startProcessing() {
    if (!selectedFile) return;

    showScreen(screenProcessing);
    processingStatus.textContent = 'Uploading video…';
    progressFill.style.animation = 'progress-pulse 2s ease-in-out infinite';

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('compression_ratio', compressionSlider.value);

    try {
        const res = await fetch('/api/synopsis', { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }
        const data = await res.json();
        jobId = data.job_id;
        processingStatus.textContent = 'Processing — detecting and tracking objects…';
        pollStatus();
    } catch (err) {
        processingStatus.textContent = `Error: ${err.message}`;
        progressFill.style.animation = 'none';
        progressFill.style.width = '100%';
        progressFill.style.background = 'var(--red)';
    }
}

async function pollStatus() {
    try {
        const res = await fetch(`/api/status/${jobId}`);
        const data = await res.json();

        if (data.status === 'completed') {
            summaryData = data.summary;
            showResults();
            return;
        }

        if (data.status === 'failed') {
            processingStatus.textContent = `Error: ${data.error || 'Processing failed'}`;
            progressFill.style.animation = 'none';
            progressFill.style.width = '100%';
            progressFill.style.background = 'var(--red)';
            return;
        }

        // Still processing — poll again
        processingStatus.textContent = 'Processing — this may take a few minutes…';
        setTimeout(pollStatus, 3000);
    } catch {
        setTimeout(pollStatus, 5000);
    }
}

// ── Results ────────────────────────────────────────────────────────
function showResults() {
    showScreen(screenResults);

    // Videos
    videoSynopsis.src = `/api/video/synopsis/${jobId}`;
    videoOriginal.src = `/api/video/original/${jobId}`;
    btnDownload.href = `/api/video/synopsis/${jobId}`;

    // Nav stats
    navStats.classList.remove('hidden');
    navStats.innerHTML = `
    <span>Original<span class="val">${summaryData.original_duration}s</span></span>
    <span>Synopsis<span class="val">${summaryData.synopsis_duration}s</span></span>
    <span>Ratio<span class="val">${(summaryData.compression_ratio * 100).toFixed(1)}%</span></span>
    <span>Objects<span class="val">${summaryData.total_tubes}</span></span>
  `;

    // Stats cards
    const ratio = (summaryData.compression_ratio * 100).toFixed(1);
    statsGrid.innerHTML = `
    <div class="stat-card">
      <div class="stat-label">Original Duration</div>
      <div class="stat-value">${summaryData.original_duration}s</div>
      <div class="stat-sub">${Math.round(summaryData.original_duration * summaryData.fps)} frames</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Synopsis Duration</div>
      <div class="stat-value">${summaryData.synopsis_duration}s</div>
      <div class="stat-sub">${Math.round(summaryData.synopsis_duration * summaryData.fps)} frames</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Compression</div>
      <div class="stat-value">${ratio}%</div>
      <div class="stat-sub">${ratio < 100 ? 'smaller' : 'larger'} than original</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Objects Tracked</div>
      <div class="stat-value">${summaryData.total_tubes}</div>
      <div class="stat-sub">unique activities</div>
    </div>
  `;

    // Table
    renderTable();
}

function renderTable() {
    const tubes = summaryData.tubes;
    const maxEnd = summaryData.synopsis_duration || 1;

    tubeTable.innerHTML = tubes.map((t, i) => {
        const barLeft = (t.synopsis_from / maxEnd * 100).toFixed(1);
        const barWidth = Math.max(1, ((t.synopsis_to - t.synopsis_from) / maxEnd * 100)).toFixed(1);
        const durSec = (t.duration_frames / summaryData.fps).toFixed(1);

        return `
      <tr data-idx="${i}">
        <td>${i + 1}</td>
        <td><span class="cell-track">#${t.track_id}</span></td>
        <td><span class="cell-class">${t.class_name}</span></td>
        <td>
          <span class="cell-time original" data-seek-orig="${t.original_from}">
            ${t.original_from}s<span class="arrow">→</span>${t.original_to}s
          </span>
        </td>
        <td>
          <span class="cell-time synopsis" data-seek-syn="${t.synopsis_from}">
            ${t.synopsis_from}s<span class="arrow">→</span>${t.synopsis_to}s
          </span>
        </td>
        <td><span class="cell-dur">${durSec}s</span></td>
        <td class="cell-bar">
          <div class="bar-bg">
            <div class="bar-fill" style="left:${barLeft}%; width:${barWidth}%"></div>
          </div>
        </td>
      </tr>
    `;
    }).join('');

    // Click handlers — original timestamps
    tubeTable.querySelectorAll('.cell-time.original').forEach(el => {
        el.addEventListener('click', e => {
            e.stopPropagation();
            seekVideo(videoOriginal, parseFloat(el.dataset.seekOrig));
            highlightRow(el.closest('tr'));
        });
    });

    // Click handlers — synopsis timestamps
    tubeTable.querySelectorAll('.cell-time.synopsis').forEach(el => {
        el.addEventListener('click', e => {
            e.stopPropagation();
            seekVideo(videoSynopsis, parseFloat(el.dataset.seekSyn));
            highlightRow(el.closest('tr'));
        });
    });

    // Row click — seek both
    tubeTable.querySelectorAll('tr').forEach(tr => {
        tr.addEventListener('click', () => {
            const idx = parseInt(tr.dataset.idx);
            const t = summaryData.tubes[idx];
            seekVideo(videoOriginal, t.original_from);
            seekVideo(videoSynopsis, t.synopsis_from);
            highlightRow(tr);
        });
    });
}

function seekVideo(video, time) {
    if (!video.src) return;
    video.currentTime = time;
    video.play().catch(() => { }); // ignore autoplay restrictions
}

function highlightRow(tr) {
    tubeTable.querySelectorAll('tr.active').forEach(r => r.classList.remove('active'));
    tr.classList.add('active');
}

// ── Live Time Display ──────────────────────────────────────────────
function formatTime(sec) {
    if (isNaN(sec)) return '0:00.0';
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1);
    return `${m}:${s.padStart(4, '0')}`;
}

videoOriginal.addEventListener('timeupdate', () => {
    origTime.textContent = formatTime(videoOriginal.currentTime);
});

videoSynopsis.addEventListener('timeupdate', () => {
    synTime.textContent = formatTime(videoSynopsis.currentTime);

    // Auto-highlight active rows
    if (!summaryData) return;
    const ct = videoSynopsis.currentTime;
    tubeTable.querySelectorAll('tr').forEach(tr => {
        const idx = parseInt(tr.dataset.idx);
        const t = summaryData.tubes[idx];
        if (ct >= t.synopsis_from && ct <= t.synopsis_to) {
            tr.classList.add('active');
        } else {
            tr.classList.remove('active');
        }
    });
});
