'use strict';

marked.use({ breaks: true });

// Provider badge
fetch('/health')
  .then(r => r.json())
  .then(d => { document.getElementById('badge').textContent = d.provider; })
  .catch(() => { document.getElementById('badge').textContent = 'offline'; });

const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const btn = document.getElementById('btn');
let es = null;

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
