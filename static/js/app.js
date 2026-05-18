/* ============================================
   Interview Transcriber v5 — Client-Side Logic
   ============================================ */

(function () {
  'use strict';

  // --- DOM References ---
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const fileInfo = document.getElementById('fileInfo');
  const fileName = document.getElementById('fileName');
  const fileSize = document.getElementById('fileSize');
  const fileClear = document.getElementById('fileClear');
  const uploadProgress = document.getElementById('uploadProgress');
  const uploadProgressFill = document.getElementById('uploadProgressFill');
  const uploadProgressText = document.getElementById('uploadProgressText');
  const languageSelect = document.getElementById('language');
  const numSpeakersInput = document.getElementById('numSpeakers');
  const btnTranscribe = document.getElementById('btnTranscribe');
  const jobsEmpty = document.getElementById('jobsEmpty');
  const jobsList = document.getElementById('jobsList');
  const modalOverlay = document.getElementById('modalOverlay');
  const modalTitle = document.getElementById('modalTitle');
  const modalContent = document.getElementById('modalContent');
  const modalDownload = document.getElementById('modalDownload');
  const modalClose = document.getElementById('modalClose');
  const jobGroupSelect = document.getElementById('jobGroupSelect');

  // Sidebar elements
  const sidebarNav = document.getElementById('sidebarNav');
  const sidebarNewGroupBtn = document.getElementById('sidebarNewGroupBtn');
  const sidebarAddGroup = document.getElementById('sidebarAddGroup');
  const sidebarGroupInput = document.getElementById('sidebarGroupInput');
  const sidebarGroupConfirm = document.getElementById('sidebarGroupConfirm');

  // --- State ---
  let selectedFile = null;
  let jobs = [];
  let groups = ['Ungrouped'];
  let activeGroup = 'All Jobs'; // 'All Jobs' or a group name
  let pollTimer = null;
  let currentModalDownloadUrl = null;

  // --- Utility ---
  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function formatDuration(seconds) {
    if (!seconds && seconds !== 0) return '—';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + 'm ' + s + 's';
  }

  function formatTimestamp(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = String(str || '');
    return div.innerHTML;
  }

  // --- System Status ---
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status fetch failed');
      const data = await res.json();
      updateStatusDot('status-gpu', data.gpu);
      updateStatusDot('status-whisper', data.whisper);
      updateStatusDot('status-diarization', data.diarization);
      // Update VRAM monitor
      if (data.gpu_detail) updateVramMonitor(data.gpu_detail);
    } catch (e) {
      console.error('Failed to fetch status:', e);
      ['status-gpu', 'status-whisper', 'status-diarization'].forEach(id => {
        updateStatusDot(id, false);
      });
    }
  }

  function updateStatusDot(elementId, isOk) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const dot = el.querySelector('.status-dot');
    dot.className = 'status-dot ' + (isOk ? 'status-dot--ok' : 'status-dot--error');
  }

  // --- Advanced settings toggle ---
  const advToggle = document.getElementById('advToggle');
  const advBody   = document.getElementById('advancedSettingsBody');
  if (advToggle && advBody) {
    advToggle.addEventListener('click', () => {
      const open = !advBody.hidden;
      advBody.hidden = open;
      advToggle.setAttribute('aria-expanded', String(!open));
    });
  }

  // --- Advanced settings preset wiring ---
  const advPreset = document.getElementById('advDiarizationPreset');
  if (advPreset) {
    const presets = {
      interview:   { minTurn: 0.8, minSilence: 0.5, minCluster: 60 },
      focus_group: { minTurn: 0.5, minSilence: 0.3, minCluster: 40 },
      panel:       { minTurn: 0.3, minSilence: 0.2, minCluster: 25 },
      monologue:   { minTurn: 2.0, minSilence: 1.5, minCluster: 120 },
    };
    advPreset.addEventListener('change', () => {
      const p = presets[advPreset.value];
      if (!p) return;
      const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
      set('advMinTurnDuration', p.minTurn);
      set('advMinSilence', p.minSilence);
      set('advMinClusterSize', p.minCluster);
    });
    // Set custom when numeric fields changed
    ['advMinTurnDuration','advMinSilence','advMinClusterSize'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', () => { advPreset.value = 'custom'; });
    });
  }

  // --- VRAM Monitor ---
  const vramLabel   = document.getElementById('vramLabel');
  const vramBarFill = document.getElementById('vramBarFill');

  function updateVramMonitor(gpuDetail) {
    if (!vramLabel || !vramBarFill) return;
    const used  = gpuDetail.vram_used  || '0 GB';
    const total = gpuDetail.vram_total || '? GB';
    const pct   = gpuDetail.vram_pct   || 0;
    vramLabel.textContent = `${used} / ${total}`;
    vramBarFill.style.width = pct + '%';
    // Color: green <50%, amber 50-80%, red >80%
    vramBarFill.style.background = pct > 80 ? 'var(--error)'
      : pct > 50 ? '#d97706' : 'var(--success)';
  }

  // --- File Selection ---
  function setSelectedFile(file) {
    if (!file) return clearSelectedFile();
    const allowed = ['.mp3', '.wav', '.m4a', '.ogg', '.flac'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
      alert('Unsupported file type. Please select: ' + allowed.join(', '));
      return;
    }
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    fileInfo.hidden = false;
    dropZone.classList.add('drop-zone--has-file');
    btnTranscribe.disabled = false;
  }

  function clearSelectedFile() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.hidden = true;
    dropZone.classList.remove('drop-zone--has-file');
    btnTranscribe.disabled = true;
    uploadProgress.hidden = true;
  }

  // Drop zone events
  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drop-zone--active');
  });

  dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drop-zone--active');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drop-zone--active');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      setSelectedFile(files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      setSelectedFile(fileInput.files[0]);
    }
  });

  fileClear.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  // --- Upload / Transcribe ---
  btnTranscribe.addEventListener('click', startTranscription);

  async function startTranscription() {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('language', languageSelect.value);

    const numSpeakers = numSpeakersInput.value;
    if (numSpeakers) {
      formData.append('num_speakers', parseInt(numSpeakers, 10));
    }

    // Include selected group
    const selectedGroup = jobGroupSelect.value || 'Ungrouped';
    formData.append('group', selectedGroup);

    const whisperModelSelect = document.getElementById('whisperModel');
    if (whisperModelSelect && whisperModelSelect.value) {
      formData.append('whisper_model', whisperModelSelect.value);
    }

    // Secondary language
    const secondaryLang = document.getElementById('secondaryLanguage');
    if (secondaryLang && secondaryLang.value) {
      formData.append('secondary_language', secondaryLang.value);
    }

    // Auto-select language when a language-specific model is chosen
    if (whisperModelSelect && languageSelect) {
      const m = whisperModelSelect.value;
      if ((m.includes('-th-') || m.includes('/whisper-th')) && languageSelect.value === 'auto') {
        formData.set('language', 'th');
      }
    }

    // Advanced settings
    const adv = (id, fallback) => { const el = document.getElementById(id); return el ? el.value : fallback; };
    const advCb = (id) => { const el = document.getElementById(id); return el ? el.checked : false; };
    formData.append('adv_beam_size',           adv('advBeamSize', '1'));
    formData.append('adv_temperature',          adv('advTemperature', '0'));
    formData.append('adv_chunk_length',         adv('advChunkLength', '30'));
    formData.append('adv_no_speech_threshold',  adv('advNoSpeechThreshold', '0.6'));
    formData.append('adv_hotwords',             adv('advHotwords', ''));
    formData.append('adv_condition_on_prev',    advCb('advConditionOnPrev'));
    formData.append('adv_min_turn_duration',    adv('advMinTurnDuration', '1.5'));
    formData.append('adv_min_silence',          adv('advMinSilence', '0.5'));
    formData.append('adv_min_cluster_size',     adv('advMinClusterSize', '75'));
    formData.append('adv_min_word_count',       adv('advMinWordCount', '4'));
    formData.append('adv_diarization_preset',   adv('advDiarizationPreset', 'interview'));

    // Disable button during upload
    btnTranscribe.disabled = true;
    uploadProgress.hidden = false;
    uploadProgressFill.style.width = '0%';
    uploadProgressText.textContent = 'Uploading… 0%';

    try {
      const xhr = new XMLHttpRequest();
      const result = await new Promise((resolve, reject) => {
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            uploadProgressFill.style.width = pct + '%';
            uploadProgressText.textContent = 'Uploading… ' + pct + '%';
          }
        });

        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(new Error('Upload failed: ' + xhr.status));
          }
        });

        xhr.addEventListener('error', () => reject(new Error('Network error')));
        xhr.addEventListener('abort', () => reject(new Error('Upload aborted')));

        xhr.open('POST', '/api/transcribe');
        xhr.send(formData);
      });

      // Success
      uploadProgressFill.style.width = '100%';
      uploadProgressText.textContent = 'Upload complete';

      clearSelectedFile();
      setTimeout(() => { uploadProgress.hidden = true; }, 1500);

      // Refresh jobs immediately
      await fetchJobs();
      startPolling();

      // Open live transcript SSE stream for the new job
      const newJobId = result.job_id;
      if (newJobId) {
        startLiveTranscript(newJobId);
      }

    } catch (e) {
      console.error('Transcription upload failed:', e);
      uploadProgressText.textContent = 'Upload failed';
      uploadProgressFill.style.width = '0%';
      btnTranscribe.disabled = false;
    }
  }

  // --- Language helpers ---
  function isCJKorThai(text) {
    for (let i = 0; i < text.length; i++) {
      const cp = text.codePointAt(i);
      if (cp >= 0x0E00 && cp <= 0x0E7F) return true;  // Thai
      if (cp >= 0x4E00 && cp <= 0x9FFF) return true;  // CJK
    }
    return false;
  }

  function joinSegments(segs) {
    if (!segs.length) return '';
    const sample = segs.slice(0, 5).join('');
    return isCJKorThai(sample) ? segs.join('') : segs.join(' ');
  }

  // --- Live Transcript (SSE) ---
  const liveTranscripts = {}; // job_id -> array of segment texts (in-progress)
  const liveEventSources = {}; // job_id -> EventSource
  const completedTranscripts = {}; // job_id -> full transcript text (loaded from disk)

  function buildCompletedTranscriptPanel(panelId, text) {
    const lines = text.trim().split('\n').filter(l => l.trim());
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    return `<div class="live-transcript-panel" id="${panelId}">`
      + '<div class="live-transcript-header">'
      + '<span>\uD83D\uDCC4</span> Transcript'
      + `<span class="live-status">${wordCount.toLocaleString()} words · ${lines.length} segments</span>`
      + '</div>'
      + `<div class="live-text">${escapeHtml(text)}</div>`
      + '</div>';
  }

  async function loadCompletedTranscript(jobId, panelId) {
    if (completedTranscripts[jobId] !== undefined) return;
    completedTranscripts[jobId] = null; // mark as fetching
    try {
      const res = await fetch('/api/preview/' + jobId + '/txt');
      if (!res.ok) throw new Error('fetch failed');
      const data = await res.json();
      const text = data.content || '';
      completedTranscripts[jobId] = text;
      const panel = document.getElementById(panelId);
      if (panel) {
        panel.outerHTML = buildCompletedTranscriptPanel(panelId, text);
      }
    } catch (e) {
      completedTranscripts[jobId] = null;
      const panel = document.getElementById(panelId);
      if (panel) {
        const statusEl = panel.querySelector('.live-status');
        const textEl = panel.querySelector('.live-text');
        if (statusEl) statusEl.textContent = 'Failed to load';
        if (textEl) textEl.textContent = 'Could not load transcript. Try downloading the .txt file.';
      }
    }
  }

  function startLiveTranscript(jobId) {
    if (liveEventSources[jobId]) return; // already connected
    liveTranscripts[jobId] = [];

    const es = new EventSource('/api/live/' + jobId);
    liveEventSources[jobId] = es;

    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.error || data.done) {
        es.close();
        delete liveEventSources[jobId];
        renderLivePanel(jobId, true);
        return;
      }
      liveTranscripts[jobId].push(data.text);
      renderLivePanel(jobId, false);
    };

    es.onerror = () => {
      es.close();
      delete liveEventSources[jobId];
    };
  }

  function renderLivePanel(jobId, done) {
  const panel = document.getElementById('live-panel-' + jobId);
  if (!panel) return;

  const segs = liveTranscripts[jobId] || [];
  const textEl = panel.querySelector('.live-text');
  const statusEl = panel.querySelector('.live-status');

  if (textEl) {
    textEl.textContent = joinSegments(segs);
  }

  if (statusEl) {
    statusEl.textContent = done
      ? `Done — ${segs.length} segments`
      : `Transcribing… ${segs.length} segments`;
  }

  requestAnimationFrame(() => {
    panel.scrollTop = panel.scrollHeight;
  });
}
  // ============================================================
  // Groups
  // ============================================================

  async function fetchGroups() {
    try {
      const res = await fetch('/api/groups');
      if (!res.ok) throw new Error('Groups fetch failed');
      const data = await res.json();
      groups = Array.isArray(data) ? data : ['Ungrouped'];
      if (!groups.includes('Ungrouped')) groups.unshift('Ungrouped');
      renderSidebar();
      populateGroupSelects();
    } catch (e) {
      console.error('Failed to fetch groups:', e);
      groups = ['Ungrouped'];
      renderSidebar();
      populateGroupSelects();
    }
  }

  function renderSidebar() {
    const allCount = jobs.length;
    const groupCounts = {};
    jobs.forEach(j => {
      const g = j.group || 'Ungrouped';
      groupCounts[g] = (groupCounts[g] || 0) + 1;
    });

    let html = `<li class="sidebar-nav-item ${activeGroup === 'All Jobs' ? 'sidebar-nav-item--active' : ''}"
        data-group="All Jobs">
        All Jobs
        <span class="sidebar-nav-count">${allCount}</span>
      </li>`;

    groups.forEach(g => {
      const count = groupCounts[g] || 0;
      html += `<li class="sidebar-nav-item ${activeGroup === g ? 'sidebar-nav-item--active' : ''}"
          data-group="${escapeHtml(g)}">
          ${escapeHtml(g)}
          <span class="sidebar-nav-count">${count}</span>
        </li>`;
    });

    sidebarNav.innerHTML = html;

    sidebarNav.querySelectorAll('.sidebar-nav-item').forEach(item => {
      item.addEventListener('click', () => {
        activeGroup = item.dataset.group;
        renderSidebar();
        renderJobs();
      });
    });
  }

  function populateGroupSelects() {
    const current = jobGroupSelect.value;
    jobGroupSelect.innerHTML = '';
    groups.forEach(g => {
      const opt = document.createElement('option');
      opt.value = g;
      opt.textContent = g;
      if (g === current) opt.selected = true;
      jobGroupSelect.appendChild(opt);
    });
  }

  // Sidebar: new group
  sidebarNewGroupBtn.addEventListener('click', () => {
    sidebarAddGroup.style.display = 'flex';
    sidebarNewGroupBtn.style.display = 'none';
    sidebarGroupInput.focus();
  });

  sidebarGroupConfirm.addEventListener('click', createGroup);
  sidebarGroupInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') createGroup();
    if (e.key === 'Escape') cancelNewGroup();
  });

  function cancelNewGroup() {
    sidebarAddGroup.style.display = 'none';
    sidebarNewGroupBtn.style.display = '';
    sidebarGroupInput.value = '';
  }

  async function createGroup() {
    const name = sidebarGroupInput.value.trim();
    if (!name) return;
    try {
      const fd = new FormData();
      fd.append('name', name);
      const res = await fetch('/api/groups', { method: 'POST', body: fd });
      if (!res.ok) throw new Error('Create group failed');
      groups = await res.json();
      cancelNewGroup();
      renderSidebar();
      populateGroupSelects();
    } catch (e) {
      alert('Failed to create group: ' + e.message);
    }
  }

  // --- Jobs ---
  async function fetchJobs() {
    try {
      const res = await fetch('/api/jobs');
      if (!res.ok) throw new Error('Jobs fetch failed');
      jobs = await res.json();
      renderSidebar();
      renderJobs();
    } catch (e) {
      console.error('Failed to fetch jobs:', e);
    }
  }

  function renderJobs() {
    let visibleJobs = jobs;
    if (activeGroup !== 'All Jobs') {
      visibleJobs = jobs.filter(j => (j.group || 'Ungrouped') === activeGroup);
    }

    if (!visibleJobs || visibleJobs.length === 0) {
      jobsEmpty.hidden = false;
      jobsList.innerHTML = '';
      return;
    }

    jobsEmpty.hidden = true;

    // Sort: most recent first
    const sorted = [...visibleJobs].sort((a, b) => {
      const ta = new Date(a.created_at || 0).getTime();
      const tb = new Date(b.created_at || 0).getTime();
      return tb - ta;
    });

    if (activeGroup !== 'All Jobs') {
      jobsList.innerHTML = sorted.map(job => renderJobCard(job)).join('');
    } else {
      // Group by group name
      const byGroup = {};
      sorted.forEach(job => {
        const g = job.group || 'Ungrouped';
        if (!byGroup[g]) byGroup[g] = [];
        byGroup[g].push(job);
      });

      let html = '';
      Object.entries(byGroup).forEach(([groupName, groupJobs]) => {
        html += `<div class="group-jobs" data-group-container="${escapeHtml(groupName)}">
          <div class="group-header" data-group-toggle="${escapeHtml(groupName)}">
            <svg class="group-header-chevron" width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span class="group-header-name">${escapeHtml(groupName)}</span>
            <span class="group-header-count">${groupJobs.length}</span>
          </div>
          <div class="jobs-list">${groupJobs.map(job => renderJobCard(job)).join('')}</div>
        </div>`;
      });
      jobsList.innerHTML = html;

      // Group collapse toggle
      jobsList.querySelectorAll('[data-group-toggle]').forEach(header => {
        header.addEventListener('click', () => {
          const container = header.closest('[data-group-container]');
          container.classList.toggle('group-jobs--collapsed');
          header.classList.toggle('group-header--collapsed');
        });
      });
    }

    // Attach event listeners to action buttons
    jobsList.querySelectorAll('[data-action]').forEach(btn => {
      btn.addEventListener('click', handleJobAction);
    });

    // Move-to-group selects
    jobsList.querySelectorAll('.job-move-select').forEach(sel => {
      sel.addEventListener('change', async (e) => {
        e.stopPropagation();
        const jobId = sel.dataset.jobId;
        const newGroup = sel.value;
        if (!jobId || !newGroup) return;
        try {
          const fd = new FormData();
          fd.append('group', newGroup);
          const res = await fetch(`/api/jobs/${jobId}/move`, { method: 'POST', body: fd });
          if (!res.ok) throw new Error('Move failed');
          await fetchJobs();
        } catch (err) {
          alert('Failed to move job: ' + err.message);
        }
      });
    });
  }

  function getStatusCategory(status) {
    if (!status) return 'queued';
    const s = status.toLowerCase();
    if (s === 'complete' || s === 'completed') return 'complete';
    if (s === 'error' || s === 'failed') return 'error';
    if (s === 'queued') return 'queued';
    return 'processing';
  }

  function buildMoveGroupSelect(job) {
    let opts = groups.map(g => {
      const sel = (g === (job.group || 'Ungrouped')) ? ' selected' : '';
      return `<option value="${escapeHtml(g)}"${sel}>${escapeHtml(g)}</option>`;
    }).join('');
    return `<select class="job-move-select" data-job-id="${job.id}" title="Move to group">${opts}</select>`;
  }

  function buildFileBrowser(job) {
    const files = job.files || {};
    const fileTypes = [
      { key: 'txt', label: 'Transcript', icon: '📄', ext: '.txt' },
      { key: 'srt', label: 'Subtitles', icon: '🎬', ext: '.srt' },
      { key: 'json', label: 'Word timestamps', icon: '🔢', ext: '.json' },
    ];

    let rows = '';
    fileTypes.forEach(ft => {
      if (!files[ft.key]) return;
      rows += `<div class="file-item">
        <span class="file-item-icon">${ft.icon}</span>
        <div class="file-item-info">
          <div class="file-item-name">${escapeHtml(ft.label)}</div>
          <div class="file-item-size">${escapeHtml(ft.ext)}</div>
        </div>
        <div class="file-item-actions">
          <button class="file-item-btn" data-action="preview" data-job-id="${job.id}" data-type="${ft.key}" title="Preview">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="4.5" stroke="currentColor" stroke-width="1.3"/><circle cx="6" cy="6" r="1.5" fill="currentColor"/></svg>
          </button>
          <a class="file-item-btn" href="/api/download/${job.id}/${ft.key}" download title="Download">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M6 2V8M6 8L4 6M6 8L8 6" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 9V10C2 10.5523 2.44772 11 3 11H9C9.55228 11 10 10.5523 10 10V9" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>
          </a>
        </div>
      </div>`;
    });

    if (!rows) {
      rows = '<div class="file-item" style="color:var(--text-tertiary);font-size:0.8rem;justify-content:center;">No files yet</div>';
    }

    const actions = `<div class="file-browser-action">
      <a class="btn-job" href="/editor/${job.id}" target="_blank" style="text-decoration:none;">✏️ Edit Transcript</a>
    </div>`;

    return `<div class="file-browser">
      <div class="file-browser-header">Files</div>
      ${rows}
      ${actions}
    </div>`;
  }

  function renderJobCard(job) {
    const category = getStatusCategory(job.status);
    const cardClass = category === 'complete' ? 'job-card--complete'
      : category === 'error' ? 'job-card--error'
      : category === 'queued' ? '' : 'job-card--processing';

    const badgeClass = category === 'complete' ? 'status-badge--complete'
      : category === 'error' ? 'status-badge--error'
      : category === 'queued' ? 'status-badge--queued'
      : 'status-badge--processing';

    const badgeDot = category === 'processing'
      ? '<span class="badge-dot"></span>' : '';

    let body = '';

    // Step text + live transcript panel for non-complete/error jobs
    if (category === 'processing' || category === 'queued') {
      const step = job.step || job.status || 'Waiting…';
      body += '<p class="job-card-step">' + escapeHtml(step) + '</p>';
      if (category === 'processing') {
        body += '<div class="job-progress"><div class="job-progress-fill"></div></div>';
        body += `<button class="btn-delete-job" style="margin-top:6px;" data-action="cancel" data-job-id="${job.id}">⏹ Cancel</button>`;
      }
      // Live transcript panel — shown during transcription
      const isTranscribing = job.status === 'transcribing';
      const hasLive = liveTranscripts[job.id] && liveTranscripts[job.id].length > 0;
      if (isTranscribing || hasLive) {
        const segs = liveTranscripts[job.id] || [];
        const liveStatus = isTranscribing
          ? `Transcribing… ${segs.length} segments`
          : `Done — ${segs.length} segments`;
        body += '<div class="live-transcript-panel" id="live-panel-' + job.id + '">'
          + '<div class="live-transcript-header">'
          + '<span class="live-dot"></span> Live transcript'
          + '<span class="live-status">' + escapeHtml(liveStatus) + '</span>'
          + '</div>'
          + '<div class="live-text">' + escapeHtml(joinSegments(segs)) + '</div>'
          + '</div>';
      }
    }

    // Completed transcript panel — load from disk and show
    if (category === 'complete' && job.files && job.files.txt) {
      const cached = completedTranscripts[job.id];
      const panelId = 'completed-transcript-' + job.id;
      if (cached) {
        body += buildCompletedTranscriptPanel(panelId, cached);
      } else {
        body += `<div class="live-transcript-panel" id="${panelId}">`
          + '<div class="live-transcript-header">'
          + '<span>📄</span> Transcript'
          + '<span class="live-status">Loading…</span>'
          + '</div>'
          + '<div class="live-text">Loading transcript…</div>'
          + '</div>';
        loadCompletedTranscript(job.id, panelId);
      }
    }

    // Result info for complete jobs
    if (category === 'complete') {
      const duration = job.duration ? formatDuration(job.duration) : null;
      const speakers = job.num_speakers_detected || job.num_speakers || null;
      if (duration || speakers) {
        body += '<div class="job-card-result">';
        if (duration) {
          body += '<span class="job-card-result-item">'
            + '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="5.5" stroke="currentColor" stroke-width="1.2"/><path d="M7 4.5V7L9 8.5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>'
            + escapeHtml(duration)
            + '</span>';
        }
        if (speakers) {
          body += '<span class="job-card-result-item">'
            + '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="5" cy="4.5" r="2" stroke="currentColor" stroke-width="1.2"/><path d="M1 11.5C1 9.29 2.79 7.5 5 7.5C5.7 7.5 6.36 7.68 6.93 8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/><circle cx="10" cy="5.5" r="1.8" stroke="currentColor" stroke-width="1.2"/><path d="M7.5 12C7.5 10.07 8.79 8.5 10.5 8.5C12.21 8.5 13.5 10.07 13.5 12" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>'
            + escapeHtml(speakers + ' speaker' + (speakers > 1 ? 's' : ''))
            + '</span>';
        }
        body += '<span style="flex:1"></span>';
        body += buildMoveGroupSelect(job);
        body += '</div>';
      } else {
        body += '<div class="job-card-result" style="justify-content:flex-end;">';
        body += buildMoveGroupSelect(job);
        body += '</div>';
      }

      // File browser — single panel (no chat)
      body += `<div class="job-detail">
        ${buildFileBrowser(job)}
      </div>`;
    }

    // Error message
    if (category === 'error' && job.error) {
      body += '<p class="job-card-step" style="color: var(--accent-red);">' + escapeHtml(job.error) + '</p>';
    }

    // Delete button
    const canDelete = category === 'error' || category === 'complete';
    const deleteBtn = canDelete
      ? `<button class="btn-delete-job" data-action="delete" data-job-id="${job.id}" title="Delete job">&#x2715; Delete</button>`
      : '';

    return `<div class="job-card ${cardClass}" data-job-id-card="${job.id}">`
      + '<div class="job-card-header">'
      + '<div class="job-card-info">'
      + '<div class="job-card-filename">' + escapeHtml(job.filename || 'Unknown file') + '</div>'
      + '<div class="job-card-time">' + escapeHtml(formatTimestamp(job.created_at)) + '</div>'
      + '</div>'
      + '<div style="display:flex;align-items:center;gap:8px;">'
      + deleteBtn
      + '<span class="status-badge ' + badgeClass + '">'
      + badgeDot
      + escapeHtml(job.status || 'unknown')
      + '</span>'
      + '</div>'
      + '</div>'
      + body
      + '</div>';
  }

  // --- Job Actions ---
  async function handleJobAction(e) {
    const btn = e.currentTarget;
    const action = btn.dataset.action;
    const jobId = btn.dataset.jobId;
    const type = btn.dataset.type;

    if (action === 'preview') {
      await openPreview(jobId, type);
    } else if (action === 'cancel') {
      await cancelJob(jobId);
    } else if (action === 'delete') {
      await deleteJob(jobId);
    }
  }

  async function cancelJob(jobId) {
    try {
      const res = await fetch('/api/jobs/' + jobId + '/cancel', { method: 'POST' });
      if (!res.ok) throw new Error('Cancel failed');
      await fetchJobs();
    } catch (e) {
      console.error('Failed to cancel job:', e);
      alert('Failed to cancel job. Use: pkill -f whisper_worker.py');
    }
  }

  async function deleteJob(jobId) {
    try {
      const res = await fetch('/api/jobs/' + jobId, { method: 'DELETE' });
      if (!res.ok) throw new Error('Delete failed');
      jobs = jobs.filter(j => j.id !== jobId);
      delete completedTranscripts[jobId];
      delete liveTranscripts[jobId];
      if (liveEventSources[jobId]) {
        liveEventSources[jobId].close();
        delete liveEventSources[jobId];
      }
      renderSidebar();
      renderJobs();
    } catch (e) {
      console.error('Failed to delete job:', e);
      alert('Failed to delete job.');
    }
  }

  async function openPreview(jobId, type) {
    modalOverlay.hidden = false;
    modalTitle.textContent = 'Loading…';
    modalContent.textContent = '';
    currentModalDownloadUrl = '/api/download/' + jobId + '/' + type;

    try {
      const res = await fetch('/api/preview/' + jobId + '/' + type);
      if (!res.ok) throw new Error('Preview fetch failed');
      const data = await res.json();
      modalTitle.textContent = data.filename || (type.toUpperCase() + ' Preview');
      modalContent.textContent = data.content || '(empty)';
    } catch (e) {
      modalTitle.textContent = 'Error';
      modalContent.textContent = 'Failed to load preview.';
    }
  }

  // --- Modal ---
  modalClose.addEventListener('click', closeModal);
  modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) closeModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modalOverlay.hidden) closeModal();
  });

  modalDownload.addEventListener('click', () => {
    if (currentModalDownloadUrl) {
      const a = document.createElement('a');
      a.href = currentModalDownloadUrl;
      a.download = '';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  });

  function closeModal() {
    modalOverlay.hidden = true;
    modalContent.textContent = '';
    currentModalDownloadUrl = null;
  }

  // --- Polling ---
  function startPolling() {
    stopPolling();
    pollTimer = setInterval(async () => {
      await fetchJobs();
      const hasActive = jobs.some(j => {
        const cat = getStatusCategory(j.status);
        return cat === 'processing' || cat === 'queued';
      });
      if (!hasActive) {
        stopPolling();
      }
    }, 2000);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  // --- Init ---
  async function init() {
    await fetchStatus();
    await fetchGroups();
    await fetchJobs();

    // Re-attach live transcript SSE for any job still transcribing
    jobs.forEach(j => {
      if (j.status === 'transcribing') {
        startLiveTranscript(j.id);
      }
    });

    // Start polling if there are active jobs
    const hasActive = jobs.some(j => {
      const cat = getStatusCategory(j.status);
      return cat === 'processing' || cat === 'queued';
    });
    if (hasActive) {
      startPolling();
    }
  }

  init();

})();
