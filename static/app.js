/* Frontend script: robust/fault-tolerant with small UX improvements */
const analyzeBtn = document.getElementById('analyzeBtn');
const reviewEl = document.getElementById('review');
const resultEl = document.getElementById('result');
const labelBadge = document.getElementById('labelBadge');
const scoreBar = document.getElementById('scoreBar');
const scoreText = document.getElementById('scoreText');
const otherScores = document.getElementById('otherScores');
const scoreList = document.getElementById('scoreList');

const ORIGINAL_BTN_HTML = analyzeBtn.innerHTML;

/* Bootstrap spinner HTML */
function spinnerHtml() {
  return `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>`;
}

/* Heuristic label map fallback (common for LabelEncoder outputs) */
const FALLBACK_LABEL_MAP = { '1': 'positive', '0': 'negative' };

/* Utility: format number to 2 decimals, clamp 0..100 */
function formatPercent(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '0.00';
  let n = Number(v);
  if (n <= 1.01 && n > 0) n = n * 100; // support 0..1 probabilities
  n = Math.max(0, Math.min(100, n));
  return n.toFixed(2);
}

/* Make fetch tolerant of both new and old API responses */
function parseApiResponse(data) {
  // Try new keys first
  let label = data.sentiment ?? data.label ?? null;
  let confidence = data.confidence ?? data.score ?? null;
  let scores = data.all_scores ?? data.scores ?? null;

  // If label missing, try to infer from scores (pick highest)
  if ((label === null || label === undefined) && scores && Object.keys(scores).length) {
    let best = null, bestVal = -Infinity;
    for (const k of Object.keys(scores)) {
      const v = Number(scores[k]);
      if (!Number.isFinite(v)) continue;
      if (v > bestVal) { bestVal = v; best = k; }
    }
    if (best !== null) label = best;
  }

  // If label looks numeric (0/1), map it if possible
  if (label !== null && !isNaN(Number(label))) {
    const key = String(label);
    if (FALLBACK_LABEL_MAP[key]) label = FALLBACK_LABEL_MAP[key];
  }

  // If confidence missing but scores exist, use the label's score
  if ((confidence === null || confidence === undefined) && scores) {
    const key = label ?? Object.keys(scores)[0];
    if (key in scores) confidence = scores[key];
  }

  // Normalize scores values to percentages (0..100)
  const normalizedScores = {};
  if (scores) {
    for (const [k, v] of Object.entries(scores)) {
      let n = Number(v);
      if (!Number.isFinite(n)) continue;
      if (n <= 1.01 && n >= 0) n = n * 100;
      normalizedScores[String(k)] = Number(n.toFixed(2));
    }
  }

  // Ensure confidence is percent number
  if (confidence !== null && confidence !== undefined) {
    let c = Number(confidence);
    if (c <= 1.01 && c >= 0) c = c * 100;
    confidence = Number(Number(c).toFixed(2));
  }

  return { label, confidence, scores: normalizedScores };
}

async function analyze() {
  const text = reviewEl.value.trim();
  if (!text) { alert("Please enter text to analyze."); reviewEl.focus(); return; }

  // set loading state
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `${spinnerHtml()}Analyzing...`;
  resultEl.classList.add('opacity-50');

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    // better error reporting
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(txt || `Server error: ${resp.status}`);
    }

    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    const parsed = parseApiResponse(data);
    const label = parsed.label ?? 'unknown';
    const score = parsed.confidence ?? 0;
    const scores = parsed.scores ?? {};

    // Badge color (Bootstrap classes)
    let badgeClass = "bg-secondary";
    const labelLower = String(label).toLowerCase();
    if (labelLower === "positive") badgeClass = "bg-success";
    else if (labelLower === "negative") badgeClass = "bg-danger";
    else if (labelLower === "neutral") badgeClass = "bg-warning";

    // Show label
    labelBadge.textContent = String(label);
    labelBadge.className = `ms-3 px-3 py-1 rounded-pill text-white shadow-sm ${badgeClass}`;

    // Progress bar (clamped)
    const pct = Math.max(0, Math.min(100, Number(score) || 0));
    scoreBar.style.width = pct + '%';
    scoreBar.textContent = formatPercent(pct) + '%';
    scoreBar.className = `progress-bar d-flex justify-content-center align-items-center fw-semibold ${badgeClass}`;

    // Confidence text
    scoreText.textContent = `Confidence: ${formatPercent(pct)}%`;

    // All scores
    if (scores && Object.keys(scores).length > 0) {
      otherScores.classList.remove('d-none');
      scoreList.innerHTML = '';
      // Prefer deterministic ordering: highest first
      const entries = Object.entries(scores).sort((a,b)=>Number(b[1])-Number(a[1]));
      for (const [k, v] of entries) {
        const pctVal = formatPercent(v);
        const li = document.createElement('li');
        // Use fallback map for numeric keys
        const displayKey = (k in FALLBACK_LABEL_MAP) ? FALLBACK_LABEL_MAP[k] : k;
        li.textContent = `${displayKey}: ${pctVal}%`;
        scoreList.appendChild(li);
      }
    } else {
      otherScores.classList.add('d-none');
      scoreList.innerHTML = '';
    }

    // reveal result
    resultEl.classList.remove('d-none');
    resultEl.classList.remove('opacity-50');
    resultEl.scrollIntoView({ behavior: 'smooth', block: 'center' });

  } catch (err) {
    console.error(err);
    alert("Error analyzing text: " + (err.message || err));
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = ORIGINAL_BTN_HTML;
  }
}

/* keyboard shortcut: Ctrl+Enter (or Cmd+Enter on mac) to submit */
reviewEl.addEventListener('keydown', (e) => {
  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  const submitKey = isMac ? e.metaKey && e.key === 'Enter' : e.ctrlKey && e.key === 'Enter';
  if (submitKey) {
    e.preventDefault();
    analyze();
  }
});

/* click handler */
analyzeBtn.addEventListener('click', analyze);
