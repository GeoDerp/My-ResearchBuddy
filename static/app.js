'use strict';

marked.use({ breaks: true });

// ── localStorage utilities ────────────────────────────────────────
const STORAGE_KEY = 'research_reports';

function saveReport(query, report) {
  try {
    const reports = loadReports();
    const id = Date.now();
    reports.push({ id, query, report, timestamp: new Date().toISOString() });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(reports));
    renderSidebar();
  } catch (e) {
    console.error('Failed to save report:', e);
  }
}

function loadReports() {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch (e) {
    console.error('Failed to load reports:', e);
    return [];
  }
}

function deleteReport(id) {
  try {
    const reports = loadReports().filter(r => r.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(reports));
    renderSidebar();
  } catch (e) {
    console.error('Failed to delete report:', e);
  }
}

function loadSavedReport(id) {
  const reports = loadReports();
  const report = reports.find(r => r.id === id);
  if (!report) return;

  clearChat();
  
  // Add user query
  const userRow = document.createElement('div');
  userRow.className = 'row user';
  const userBubble = document.createElement('div');
  userBubble.className = 'bubble';
  userBubble.textContent = report.query;
  userRow.appendChild(userBubble);
  chat.appendChild(userRow);

  // Add bot report
  const botRow = document.createElement('div');
  botRow.className = 'row bot';
  const botBubble = document.createElement('div');
  botBubble.className = 'bubble';
  
  const reportDiv = document.createElement('div');
  reportDiv.className = 'report';
  reportDiv.innerHTML = DOMPurify.sanitize(marked.parse(report.report));
  reportDiv.querySelectorAll('a').forEach(a => {
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
  });
  
  botBubble.appendChild(reportDiv);
  botRow.appendChild(botBubble);
  chat.appendChild(botRow);
  
  scrollEnd();
}

// ── UI controls ───────────────────────────────────────────────────

function clearChat() {
  if (es) { es.close(); es = null; }
  chat.innerHTML = '<div id="empty"><svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" /></svg><p>Ask a deep research question to get started</p></div>';
  inp.value = '';
  inp.style.height = 'auto';
  lock(false);
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  sidebar.classList.toggle('open');
}

function renderSidebar() {
  const list = document.getElementById('reports-list');
  const reports = loadReports();
  
  if (reports.length === 0) {
    list.innerHTML = '<div class="no-reports">No saved reports yet</div>';
    return;
  }
  
  list.innerHTML = reports.reverse().map(r => {
    const date = new Date(r.timestamp);
    const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const queryPreview = r.query.length > 60 ? r.query.substring(0, 60) + '...' : r.query;
    return `
      <div class="report-item" data-id="${r.id}">
        <div class="report-item-content">
          <div class="report-query">${escapeHtml(queryPreview)}</div>
          <div class="report-time">${timeStr}</div>
        </div>
        <button class="delete-btn" title="Delete report">✕</button>
      </div>
    `;
  }).join('');
  
  // Add event listeners to report items
  list.querySelectorAll('.report-item').forEach(item => {
    const id = parseInt(item.dataset.id);
    item.querySelector('.report-item-content').addEventListener('click', () => loadSavedReport(id));
    item.querySelector('.delete-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteReport(id);
    });
  });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Provider badge
fetch('/health')
  .then(r => r.json())
  .then(d => { document.getElementById('badge').textContent = d.provider; })
  .catch(() => { document.getElementById('badge').textContent = 'offline'; });

const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const btn = document.getElementById('btn');
let es = null;

// Initialize sidebar on page load
renderSidebar();

// Add event listeners for new controls
document.getElementById('new-chat-btn').addEventListener('click', clearChat);
document.getElementById('sidebar-toggle').addEventListener('click', toggleSidebar);
document.querySelector('.close-sidebar').addEventListener('click', toggleSidebar);

inp.addEventListener('input', () => {
  inp.style.height = 'auto';
  inp.style.height = Math.min(inp.scrollHeight, 160) + 'px';
});

inp.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

btn.addEventListener('click', send);

function lock(on) {
  inp.disabled = on;
  btn.disabled = on;
  if (!on) inp.focus();
}

function scrollEnd() {
  chat.scrollTo({ top: chat.scrollHeight, behavior: 'smooth' });
}

function removeEmpty() {
  const el = document.getElementById('empty');
  if (el) el.remove();
}

function addUser(text) {
  removeEmpty();
  const row = document.createElement('div');
  row.className = 'row user';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  row.appendChild(b);
  chat.appendChild(row);
  scrollEnd();
}

function addBot() {
  const row = document.createElement('div');
  row.className = 'row bot';
  const b = document.createElement('div');
  b.className = 'bubble';

  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.innerHTML = '<i></i><i></i><i></i>';
  b.appendChild(typing);

  const details = document.createElement('details');
  details.className = 'log-wrap';
  details.open = true;
  const summary = document.createElement('summary');
  summary.textContent = 'Research progress';
  const logBody = document.createElement('div');
  logBody.className = 'log-body';
  details.append(summary, logBody);
  b.appendChild(details);

  const report = document.createElement('div');
  report.className = 'report';
  report.hidden = true;
  b.appendChild(report);

  row.appendChild(b);
  chat.appendChild(row);
  scrollEnd();
  return { typing, details, logBody, report, bubble: b };
}

function send() {
  const q = inp.value.trim();
  if (!q || btn.disabled) return;

  if (es) { es.close(); es = null; }

  inp.value = '';
  inp.style.height = 'auto';
  lock(true);

  addUser(q);
  const { typing, details, logBody, report, bubble } = addBot();

  es = new EventSource('/stream?query=' + encodeURIComponent(q));

  es.onmessage = ({ data }) => {
    let d;
    try { d = JSON.parse(data); } catch { return; }

    if (d.type === 'progress') {
      logBody.textContent += d.message + '\n';
      logBody.scrollTop = logBody.scrollHeight;
      scrollEnd();

    } else if (d.type === 'result') {
      typing.remove();
      details.open = false;
      report.hidden = false;
      report.innerHTML = DOMPurify.sanitize(marked.parse(d.report));
      report.querySelectorAll('a').forEach(a => {
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
      });
      // Auto-save completed report
      saveReport(q, d.report);
      scrollEnd();

    } else if (d.type === 'done') {
      es.close(); es = null;
      lock(false);
      scrollEnd();

    } else if (d.type === 'error') {
      typing.remove();
      details.open = false;
      const err = document.createElement('div');
      err.className = 'err';
      err.textContent = '\u26a0 ' + d.message;
      bubble.appendChild(err);
      es.close(); es = null;
      lock(false);
      scrollEnd();
    }
  };

  es.onerror = () => {
    if (es && es.readyState === EventSource.CLOSED) return;
    typing.remove();
    details.open = false;
    const err = document.createElement('div');
    err.className = 'err';
    err.textContent = '\u26a0 Connection lost. The server may have restarted.';
    bubble.appendChild(err);
    if (es) { es.close(); es = null; }
    lock(false);
    scrollEnd();
  };
}
