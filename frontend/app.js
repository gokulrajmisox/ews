/**
 * app.js - SilentWindow Dashboard Client Logic.
 * Fully interactive, connects to FastAPI backend endpoints.
 */

// Global State
let currentTab = "overview";
let wardPatients = [];
let selectedPatientId = 138123;
let currentTimelineData = null;
let currentExplanationData = null;
let simInterval = null;
let isSimPlaying = false;
let simSpeed = 1;
let currentSimHour = 48.0;
let maxSimHour = 48.0;

// Chart Instances
const charts = {
  vitals: null,
  risk: null,
  evidence: null,
  roc: null,
  pr: null,
  tradeoff: null,
  noise: null
};

// --- Initialization ---
document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) lucide.createIcons();
  await loadOverview();
  await loadWardMonitor();
  await loadAblationData();
  await loadPerformanceData();
  await loadNoiseData();
});

// --- Tab Switching ---
function switchTab(tabId) {
  currentTab = tabId;
  document.querySelectorAll("main > section").forEach(sec => sec.classList.add("hidden"));
  const activeSec = document.getElementById(`view-${tabId}`);
  if (activeSec) activeSec.classList.remove("hidden");

  document.querySelectorAll(".tab-btn").forEach(btn => {
    if (btn.dataset.tab === tabId) {
      btn.classList.add("active", "border-b-2", "border-cyan-400", "text-cyan-400");
      btn.classList.remove("text-slate-300");
    } else {
      btn.classList.remove("active", "border-b-2", "border-cyan-400", "text-cyan-400");
      btn.classList.add("text-slate-300");
    }
  });

  if (window.lucide) lucide.createIcons();

  if (tabId === "timeline") {
    loadPatientTimeline(selectedPatientId);
  } else if (tabId === "explanation") {
    loadPatientExplanation(selectedPatientId, currentSimHour);
  }
}

// --- PAGE 1: OVERVIEW ---
async function loadOverview() {
  try {
    const res = await fetch("/api/overview");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("kpi-total-patients").innerText = data.total_patients;
    document.getElementById("kpi-stable-count").innerText = `${data.current_states.stable} Stable`;
    document.getElementById("kpi-watch-count").innerText = `${data.current_states.watch} Watch`;
    document.getElementById("kpi-alert-count").innerText = `${data.current_states.alert} Alert`;

    const sw = data.systems.silent_window;
    if (sw) {
      document.getElementById("kpi-alerts-pt-day").innerHTML = `${sw.alerts_per_patient_day} <span class="text-xs font-normal text-slate-500">alerts/pt-day</span>`;
      if (sw.median_lead_time_hours) {
        document.getElementById("kpi-median-lead-time").innerHTML = `${sw.median_lead_time_hours} <span class="text-xs font-normal text-slate-500">hours</span>`;
      }
    }
  } catch (err) {
    console.error("Error loading overview:", err);
  }
}

// --- PAGE 2: WARD MONITOR ---
async function loadWardMonitor() {
  try {
    const res = await fetch("/api/ward");
    if (!res.ok) return;
    const data = await res.json();
    wardPatients = data.patients || [];

    // Populate patient selector dropdown in timeline tab
    const selector = document.getElementById("patient-selector");
    if (selector) {
      selector.innerHTML = "";
      wardPatients.slice(0, 50).forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.patient_id;
        const icon = p.state === "ALERT" ? "🔴" : (p.state === "WATCH" ? "🟡" : "🟢");
        opt.text = `${icon} Patient ${p.patient_id} (Risk: ${(p.risk * 100).toFixed(0)}%)`;
        selector.appendChild(opt);
      });
      if (wardPatients.length > 0) {
        selectedPatientId = wardPatients[0].patient_id;
        selector.value = selectedPatientId;
      }
    }

    renderWardTable(wardPatients);
  } catch (err) {
    console.error("Error loading ward monitor:", err);
  }
}

function renderWardTable(patients) {
  const tbody = document.getElementById("ward-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  patients.forEach(p => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-50 transition cursor-pointer";
    tr.onclick = (e) => {
      // Don't trigger if action button was clicked directly
      if (e.target.tagName !== "BUTTON" && !e.target.closest("button")) {
        selectPatient(p.patient_id);
      }
    };

    let stateBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800">🟢 Stable</span>`;
    if (p.state === "WATCH") {
      stateBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800">🟡 Watch</span>`;
    } else if (p.state === "ALERT") {
      stateBadge = `<span class="px-2 py-0.5 rounded-full text-[10px] font-bold bg-red-100 text-red-800">🔴 Alert</span>`;
    }

    let trendIcon = `<span class="text-slate-400">→</span>`;
    if (p.trend === "increasing") trendIcon = `<span class="text-red-500 font-bold">↗</span>`;
    if (p.trend === "decreasing") trendIcon = `<span class="text-emerald-500 font-bold">↘</span>`;

    tr.innerHTML = `
      <td class="px-4 py-3 font-mono font-medium text-slate-900">${p.patient_id}</td>
      <td class="px-4 py-3">${stateBadge}</td>
      <td class="px-4 py-3">
        <div class="flex items-center gap-2">
          <div class="w-16 bg-slate-100 rounded-full h-1.5 overflow-hidden">
            <div class="bg-cyan-600 h-1.5 rounded-full" style="width: ${Math.min(100, p.risk * 100)}%"></div>
          </div>
          <span class="font-medium">${(p.risk * 100).toFixed(0)}%</span>
        </div>
      </td>
      <td class="px-4 py-3 font-semibold ${p.evidence_score >= 0.75 ? 'text-red-600' : (p.evidence_score >= 0.40 ? 'text-amber-600' : 'text-slate-700')}">
        ${p.evidence_score.toFixed(2)}
      </td>
      <td class="px-4 py-3 text-base">${trendIcon}</td>
      <td class="px-4 py-3 text-slate-500">${p.last_update_hours.toFixed(1)}h</td>
      <td class="px-4 py-3 font-semibold ${p.total_alerts > 0 ? 'text-red-600' : 'text-slate-500'}">${p.total_alerts}</td>
      <td class="px-4 py-3">
        <span class="px-2 py-0.5 rounded text-[10px] font-bold ${p.mean_credibility < 0.8 ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-700'}">
          ${(p.mean_credibility * 100).toFixed(0)}%
        </span>
      </td>
      <td class="px-4 py-3 text-right flex items-center justify-end gap-2">
        ${p.state === 'ALERT' ? `<button onclick="sendTelegramAlert(${p.patient_id})" class="px-2.5 py-1 text-xs font-semibold bg-red-100 hover:bg-red-600 hover:text-white text-red-700 rounded transition" title="Send Telegram Alert"><i data-lucide="send" class="w-3 h-3 inline"></i> Telegram</button>` : ''}
        <button onclick="selectPatient(${p.patient_id})\" class="px-2.5 py-1 text-xs font-semibold bg-slate-100 hover:bg-cyan-600 hover:text-white text-slate-700 rounded transition">
          Timeline & Sim →
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterWardTable() {
  const q = document.getElementById("ward-search").value.toLowerCase().trim();
  const filtered = wardPatients.filter(p => p.patient_id.toString().includes(q));
  renderWardTable(filtered);
}

function setWardFilter(filterState) {
  document.querySelectorAll(".ward-filter-btn").forEach(btn => {
    if (btn.dataset.filter === filterState) {
      btn.className = "ward-filter-btn px-2.5 py-1 rounded bg-slate-900 text-white font-medium";
    } else {
      btn.className = "ward-filter-btn px-2.5 py-1 rounded text-slate-700 hover:bg-slate-100";
    }
  });

  if (filterState === "ALL") {
    renderWardTable(wardPatients);
  } else {
    renderWardTable(wardPatients.filter(p => p.state === filterState));
  }
}

function selectPatient(patientId) {
  selectedPatientId = patientId;
  const sel = document.getElementById("patient-selector");
  if (sel) sel.value = patientId;
  switchTab("timeline");
}

function onPatientChange(newId) {
  selectedPatientId = parseInt(newId);
  loadPatientTimeline(selectedPatientId);
}

// --- PAGE 3: PATIENT TIMELINE & SIMULATION ---
async function loadPatientTimeline(patientId) {
  try {
    resetSimulation();
    const res = await fetch(`/api/patient/${patientId}/timeline`);
    if (!res.ok) return;
    currentTimelineData = await res.json();

    maxSimHour = currentTimelineData.total_monitoring_hours || 48.0;
    currentSimHour = maxSimHour;

    // Update scrubber controls
    const scrubber = document.getElementById("sim-scrubber");
    if (scrubber) {
      scrubber.min = 2;
      scrubber.max = maxSimHour;
      scrubber.value = currentSimHour;
    }

    // Update Patient Badges
    const bContainer = document.getElementById("patient-badges");
    if (bContainer) {
      const outcomeBadge = currentTimelineData.actual_outcome === 1
        ? `<span class="px-2 py-0.5 rounded bg-red-100 text-red-800 font-bold border border-red-300">Deteriorated (Outcome 1)</span>`
        : `<span class="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold border border-emerald-300">Discharged Stable (Outcome 0)</span>`;
      bContainer.innerHTML = `
        <span>SOFA: <strong>${currentTimelineData.sofa}</strong></span>
        <span>SAPS: <strong>${currentTimelineData.saps}</strong></span>
        ${outcomeBadge}
        <span>SilentWindow Alerts: <strong class="text-cyan-700">${currentTimelineData.total_alerts}</strong></span>
        <span>Threshold Alerts: <strong class="text-slate-600">${currentTimelineData.threshold_baseline_alerts}</strong></span>
      `;
    }

    renderSynchronizedCharts(currentSimHour);
  } catch (err) {
    console.error("Error loading patient timeline:", err);
  }
}

function onScrubberInput(hour) {
  currentSimHour = parseFloat(hour);
  updateSimTimeLabel(currentSimHour);
  renderSynchronizedCharts(currentSimHour);
}

function updateSimTimeLabel(hour) {
  document.getElementById("sim-time-display").innerText = `Data available up to: ${hour.toFixed(1)}h`;
  document.getElementById("scrubber-hour-label").innerText = `${hour.toFixed(1)}h / ${maxSimHour.toFixed(1)}h`;
}

function togglePlaySimulation() {
  if (isSimPlaying) {
    pauseSimulation();
  } else {
    playSimulation();
  }
}

function playSimulation() {
  if (currentSimHour >= maxSimHour) currentSimHour = 2.0;
  isSimPlaying = true;
  document.getElementById("sim-play-icon").setAttribute("data-lucide", "pause");
  if (window.lucide) lucide.createIcons();

  const stepInterval = simSpeed === 1 ? 700 : (simSpeed === 2 ? 350 : 150);
  simInterval = setInterval(() => {
    currentSimHour += 1.0;
    if (currentSimHour > maxSimHour) {
      currentSimHour = maxSimHour;
      pauseSimulation();
    }
    const scrubber = document.getElementById("sim-scrubber");
    if (scrubber) scrubber.value = currentSimHour;
    updateSimTimeLabel(currentSimHour);
    renderSynchronizedCharts(currentSimHour);
  }, stepInterval);
}

function pauseSimulation() {
  isSimPlaying = false;
  clearInterval(simInterval);
  document.getElementById("sim-play-icon").setAttribute("data-lucide", "play");
  if (window.lucide) lucide.createIcons();
}

function resetSimulation() {
  pauseSimulation();
  currentSimHour = maxSimHour;
  const scrubber = document.getElementById("sim-scrubber");
  if (scrubber) scrubber.value = currentSimHour;
  updateSimTimeLabel(currentSimHour);
  if (currentTimelineData) renderSynchronizedCharts(currentSimHour);
}

function setSimSpeed(speed) {
  simSpeed = speed;
  document.querySelectorAll(".sim-speed-btn").forEach(b => {
    if (parseInt(b.dataset.speed) === speed) {
      b.className = "sim-speed-btn px-1.5 py-0.5 rounded bg-cyan-600 text-white font-bold";
    } else {
      b.className = "sim-speed-btn px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 hover:bg-slate-700";
    }
  });
  if (isSimPlaying) {
    pauseSimulation();
    playSimulation();
  }
}

// --- RENDER SYNCHRONIZED TIMELINE CHARTS ---
function renderSynchronizedCharts(revealedHour) {
  if (!currentTimelineData || !currentTimelineData.timeline) return;

  // Filter revealed points strictly <= revealedHour (No Future Leakage)
  const revealed = currentTimelineData.timeline.filter(t => t.time_hours <= revealedHour + 1e-4);
  const labels = revealed.map(t => `${t.time_hours.toFixed(0)}h`);

  // --- Track 1: Vital Signs Chart ---
  const ctxVitals = document.getElementById("chart-vitals")?.getContext("2d");
  if (ctxVitals) {
    const hrData = revealed.map(t => t.vitals.HR ?? null);
    const sysData = revealed.map(t => t.vitals.SysABP ?? null);
    const diasData = revealed.map(t => t.vitals.DiasABP ?? null);
    const rrData = revealed.map(t => t.vitals.RespRate ?? null);

    // Identify downweighted points
    const downweightedPoints = revealed.map((t, idx) => {
      if (t.suspicious_count > 0 && t.mean_credibility < 0.70) {
        return hrData[idx] || sysData[idx] || 100;
      }
      return null;
    });

    if (charts.vitals) charts.vitals.destroy();
    charts.vitals = new Chart(ctxVitals, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { label: "Heart Rate (bpm)", data: hrData, borderColor: "#EF4444", backgroundColor: "#EF4444", borderWidth: 2, pointRadius: 2, tension: 0.2 },
          { label: "Systolic BP (mmHg)", data: sysData, borderColor: "#0284C7", backgroundColor: "#0284C7", borderWidth: 2, pointRadius: 2, tension: 0.2 },
          { label: "Diastolic BP (mmHg)", data: diasData, borderColor: "#38BDF8", backgroundColor: "#38BDF8", borderWidth: 1.5, borderDash: [4, 4], pointRadius: 0, tension: 0.2 },
          { label: "Resp Rate (bpm)", data: rrData, borderColor: "#10B981", backgroundColor: "#10B981", borderWidth: 2, pointRadius: 2, tension: 0.2, yAxisID: "y1" },
          { label: "Downweighted Artifact (Trust Layer)", data: downweightedPoints, borderColor: "#8B5CF6", backgroundColor: "#8B5CF6", pointRadius: 6, pointStyle: "rectRot", showLine: false }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { grid: { display: false } },
          y: { title: { display: true, text: "BP & HR" }, min: 30, max: 220 },
          y1: { position: "right", title: { display: true, text: "RR" }, min: 5, max: 45, grid: { display: false } }
        }
      }
    });
  }

  // --- Track 2: Calibrated Risk Probability ---
  const ctxRisk = document.getElementById("chart-risk")?.getContext("2d");
  if (ctxRisk) {
    const riskData = revealed.map(t => t.calibrated_risk);
    const rawRiskData = revealed.map(t => t.raw_risk);

    if (charts.risk) charts.risk.destroy();
    charts.risk = new Chart(ctxRisk, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { label: "Calibrated Deterioration Risk", data: riskData, borderColor: "#0891B2", backgroundColor: "rgba(8, 145, 178, 0.1)", fill: true, borderWidth: 2.5, pointRadius: 2, tension: 0.2 },
          { label: "Raw Model Risk", data: rawRiskData, borderColor: "#94A3B8", borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
          { label: "Neutral Threshold (0.22)", data: labels.map(() => 0.22), borderColor: "#64748B", borderDash: [2, 2], pointRadius: 0, borderWidth: 1 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { grid: { display: false } },
          y: { min: 0, max: 1.0, title: { display: true, text: "Risk Probability" } }
        }
      }
    });
  }

  // --- Track 3: Evidence Accumulator & Alert Markers ---
  const ctxEvidence = document.getElementById("chart-evidence")?.getContext("2d");
  if (ctxEvidence) {
    const evidenceData = revealed.map(t => t.evidence_score);

    // Watch (yellow) and Alert (red) event markers
    const alertMarkers = revealed.map(t => t.alert_fired ? t.evidence_score : null);
    const watchMarkers = revealed.map(t => (t.state === "WATCH" && !t.alert_fired) ? t.evidence_score : null);
    
    // Baseline comparators
    const thresholdMarkers = revealed.map(t => t.baseline_threshold_alert ? 0.30 : null);
    const plainMLMarkers = revealed.map(t => t.baseline_plain_ml_alert ? 0.20 : null);

    if (charts.evidence) charts.evidence.destroy();
    charts.evidence = new Chart(ctxEvidence, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { label: "Accumulated Evidence (Et)", data: evidenceData, borderColor: "#10B981", backgroundColor: "rgba(16, 185, 129, 0.1)", fill: true, borderWidth: 2.5, pointRadius: 2, tension: 0.2 },
          { label: "🔴 High-Confidence Alert Fired", data: alertMarkers, borderColor: "#EF4444", backgroundColor: "#EF4444", pointRadius: 7, pointStyle: "triangle", showLine: false },
          { label: "🟡 Watch Advisory Active", data: watchMarkers, borderColor: "#F59E0B", backgroundColor: "#F59E0B", pointRadius: 5, pointStyle: "circle", showLine: false },
          { label: "Simplified Threshold Alert", data: thresholdMarkers, borderColor: "#94A3B8", backgroundColor: "#94A3B8", pointRadius: 4, pointStyle: "crossRot", showLine: false },
          { label: "Plain ML Instant Alert", data: plainMLMarkers, borderColor: "#64748B", backgroundColor: "#64748B", pointRadius: 4, pointStyle: "star", showLine: false },
          { label: "Alert Threshold (0.75)", data: labels.map(() => 0.75), borderColor: "#EF4444", borderDash: [4, 4], pointRadius: 0, borderWidth: 1.5 },
          { label: "Watch Threshold (0.40)", data: labels.map(() => 0.40), borderColor: "#F59E0B", borderDash: [4, 4], pointRadius: 0, borderWidth: 1.5 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { grid: { display: false } },
          y: { min: 0, max: Math.max(1.5, ...evidenceData) + 0.2, title: { display: true, text: "Evidence Score" } }
        }
      }
    });
  }
}

// --- PAGE 4: ALERT EXPLANATION (SHAP & TRUST) ---
async function loadPatientExplanation(patientId, evalHour) {
  try {
    document.getElementById("explanation-patient-badge").innerText = `Patient ${patientId}`;
    const url = evalHour ? `/api/patient/${patientId}/explanation?eval_time=${evalHour}` : `/api/patient/${patientId}/explanation`;
    const res = await fetch(url);
    if (!res.ok) return;
    currentExplanationData = await res.json();

    document.getElementById("exp-risk-val").innerText = `${(currentExplanationData.calibrated_risk * 100).toFixed(0)}%`;
    document.getElementById("exp-hour-val").innerText = `${currentExplanationData.eval_time_hours.toFixed(1)}h`;

    // Retrieve evidence and lead time from timeline if available
    let leadTime = "--";
    let evidence = "--";
    if (currentTimelineData && currentTimelineData.timeline) {
      const matchPoint = currentTimelineData.timeline.find(t => Math.abs(t.time_hours - currentExplanationData.eval_time_hours) < 0.5) || currentTimelineData.timeline[currentTimelineData.timeline.length - 1];
      if (matchPoint) {
        evidence = matchPoint.evidence_score.toFixed(2);
        if (matchPoint.estimated_lead_time_hours !== null && matchPoint.estimated_lead_time_hours !== undefined) {
          leadTime = `${matchPoint.estimated_lead_time_hours.toFixed(1)}h`;
        }
      }
    }
    document.getElementById("exp-evidence-val").innerText = evidence;
    document.getElementById("exp-lead-val").innerText = leadTime;

    // Bullets
    const bContainer = document.getElementById("explanation-bullets");
    bContainer.innerHTML = "";
    currentExplanationData.summary_bullets.forEach(b => {
      const p = document.createElement("p");
      p.innerText = b;
      bContainer.appendChild(p);
    });

    // SHAP factors
    const fContainer = document.getElementById("shap-factors-container");
    fContainer.innerHTML = "";
    const topFactors = currentExplanationData.top_factors || [];
    const maxShap = topFactors.length > 0 ? Math.max(...topFactors.map(f => f.shap_impact)) : 1.0;

    topFactors.forEach(f => {
      const pct = Math.min(100, Math.max(5, (f.shap_impact / maxShap) * 100));
      const div = document.createElement("div");
      div.className = "p-3 bg-slate-50 border border-slate-200 rounded-lg text-xs";
      div.innerHTML = `
        <div class="flex justify-between items-center mb-1">
          <span class="font-semibold text-slate-800">${f.display_name}</span>
          <span class="font-mono text-cyan-700 font-bold">+${f.shap_impact.toFixed(3)} SHAP</span>
        </div>
        <div class="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
          <div class="bg-cyan-600 h-1.5 rounded-full" style="width: ${pct}%"></div>
        </div>
        <div class="text-[11px] text-slate-500 mt-1">Measured Value: <strong>${f.current_value}</strong></div>
      `;
      fContainer.appendChild(div);
    });

    // Suspicious readings table
    const sBody = document.getElementById("suspicious-readings-body");
    sBody.innerHTML = "";
    const suspicious = currentExplanationData.suspicious_readings || [];
    if (suspicious.length === 0) {
      sBody.innerHTML = `<tr><td colspan="5" class="px-3 py-4 text-center text-slate-400">No downweighted readings detected. Telemetry is fully credible.</td></tr>`;
    } else {
      suspicious.forEach(s => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="px-3 py-2 font-mono text-slate-600">${s.time_hours.toFixed(1)}h</td>
          <td class="px-3 py-2 font-semibold text-slate-800">${s.parameter}</td>
          <td class="px-3 py-2 font-mono font-bold text-red-600">${s.value}</td>
          <td class="px-3 py-2"><span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-100 text-amber-800">${(s.credibility_score * 100).toFixed(0)}%</span></td>
          <td class="px-3 py-2 text-slate-600">${s.reason}</td>
        `;
        sBody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Error loading explanation:", err);
  }
}

// --- PAGE 5: PERFORMANCE ---
async function loadPerformanceData() {
  try {
    const res = await fetch("/api/performance");
    if (!res.ok) return;
    const data = await res.json();

    // 1. ROC Curve
    const ctxRoc = document.getElementById("chart-roc")?.getContext("2d");
    if (ctxRoc && data.curves && data.curves.roc) {
      if (charts.roc) charts.roc.destroy();
      charts.roc = new Chart(ctxRoc, {
        type: "line",
        data: {
          labels: data.curves.roc.map(p => p.fpr),
          datasets: [
            { label: "SilentWindow ROC (AUC: 0.6990)", data: data.curves.roc.map(p => p.tpr), borderColor: "#0891B2", borderWidth: 2, pointRadius: 0, tension: 0.1 },
            { label: "Chance Diagonal", data: [0, 1], borderColor: "#94A3B8", borderDash: [4, 4], pointRadius: 0, borderWidth: 1 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: "False Positive Rate (FPR)" }, min: 0, max: 1 },
            y: { title: { display: true, text: "True Positive Rate (Sensitivity)" }, min: 0, max: 1 }
          }
        }
      });
    }

    // 2. PR Curve
    const ctxPr = document.getElementById("chart-pr")?.getContext("2d");
    if (ctxPr && data.curves && data.curves.pr) {
      if (charts.pr) charts.pr.destroy();
      charts.pr = new Chart(ctxPr, {
        type: "line",
        data: {
          labels: data.curves.pr.map(p => p.recall),
          datasets: [
            { label: "Precision-Recall Curve (PR-AUC: 0.2803)", data: data.curves.pr.map(p => p.precision), borderColor: "#10B981", borderWidth: 2, pointRadius: 0, tension: 0.1 }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: "Recall" }, min: 0, max: 1 },
            y: { title: { display: true, text: "Precision" }, min: 0, max: 1 }
          }
        }
      });
    }

    // 3. Trade-off Chart
    const ctxTradeoff = document.getElementById("chart-tradeoff")?.getContext("2d");
    if (ctxTradeoff) {
      if (charts.tradeoff) charts.tradeoff.destroy();
      charts.tradeoff = new Chart(ctxTradeoff, {
        type: "bar",
        data: {
          labels: ["SilentWindow", "Plain ML Baseline", "Simplified Threshold"],
          datasets: [
            { label: "False Alarms across Cohort", data: [51, 24, 120], backgroundColor: ["#0891B2", "#64748B", "#EF4444"] },
            { label: "Alerts per Patient-Day", data: [0.203, 0.203, 0.504], backgroundColor: ["rgba(8, 145, 178, 0.4)", "rgba(100, 116, 139, 0.4)", "rgba(239, 68, 68, 0.4)"], yAxisID: "y1" }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { title: { display: true, text: "False Alarms (Counts)" } },
            y1: { position: "right", title: { display: true, text: "Alerts / Pt-Day" }, grid: { display: false } }
          }
        }
      });
    }
  } catch (err) {
    console.error("Error loading performance data:", err);
  }
}

// --- PAGE 6: ABLATION STUDY ---
async function loadAblationData() {
  try {
    const res = await fetch("/api/ablation");
    if (!res.ok) return;
    const data = await res.json();
    const tbody = document.getElementById("ablation-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.results_table.forEach(r => {
      const tr = document.createElement("tr");
      const isFull = r.variant_id === "full_silentwindow";
      tr.className = isFull ? "bg-cyan-50/40 font-semibold" : "hover:bg-slate-50";
      tr.innerHTML = `
        <td class="px-4 py-3">
          <div class="text-slate-900">${r.system_name} ${isFull ? '<span class="text-[10px] px-1.5 py-0.5 rounded bg-cyan-600 text-white ml-1">FULL</span>' : ''}</div>
          <div class="text-[11px] text-slate-500 font-normal">${r.description}</div>
        </td>
        <td class="px-4 py-3 font-mono">${(r.sensitivity * 100).toFixed(1)}%</td>
        <td class="px-4 py-3 font-mono ${r.false_alarms > 50 ? 'text-slate-800' : 'text-slate-600'}">${r.false_alarms}</td>
        <td class="px-4 py-3 font-mono">${r.alerts_per_patient_day.toFixed(3)}</td>
        <td class="px-4 py-3 font-mono">${r.median_lead_time ? r.median_lead_time.toFixed(1) + 'h' : '--'}</td>
        <td class="px-4 py-3 font-mono font-bold">${r.total_alerts}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading ablation data:", err);
  }
}

// --- PAGE 7: NOISE LAB ---
async function loadNoiseData() {
  try {
    const res = await fetch("/api/noise-lab");
    if (!res.ok) return;
    const data = await res.json();
    renderNoiseResults(data);
  } catch (err) {
    console.error("Error loading noise data:", err);
  }
}

function updateNoiseSlider(val) {
  document.getElementById("noise-intensity-label").innerText = `${val}%`;
}

async function triggerNoiseStressTest() {
  const btn = document.getElementById("noise-run-btn");
  btn.disabled = true;
  btn.innerHTML = `<i data-lucide="loader" class="w-4 h-4 animate-spin"></i> Running Stress Test...`;
  if (window.lucide) lucide.createIcons();

  const intensity = parseFloat(document.getElementById("noise-intensity-slider").value) / 100.0;
  const injectMissing = document.getElementById("noise-toggle-missing").checked;
  const injectSpikes = document.getElementById("noise-toggle-spikes").checked;
  const injectJitter = document.getElementById("noise-toggle-jitter").checked;

  try {
    const res = await fetch("/api/noise-lab/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        intensity: intensity,
        inject_missingness: injectMissing,
        inject_spikes: injectSpikes,
        inject_jitter: injectJitter,
        max_patients: 25
      })
    });
    if (res.ok) {
      const data = await res.json();
      renderNoiseResults(data);
    }
  } catch (err) {
    console.error("Error running noise test:", err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i data-lucide="zap" class="w-4 h-4"></i> Run Stress Test`;
    if (window.lucide) lucide.createIcons();
  }
}

function renderNoiseResults(data) {
  const container = document.getElementById("noise-resilience-cards");
  if (!container || !data.comparison) return;
  container.innerHTML = "";

  data.comparison.forEach(c => {
    const isSW = c.system === "SilentWindow";
    const card = document.createElement("div");
    card.className = `p-4 rounded-xl border ${isSW ? 'border-cyan-500 bg-cyan-50/20' : 'border-slate-200 bg-white'}`;
    
    let changeBadge = `<span class="text-emerald-700 font-bold bg-emerald-100 px-2 py-0.5 rounded text-xs">+0.0% false alarms</span>`;
    if (c.false_alert_increase_pct > 0) {
      changeBadge = `<span class="text-red-700 font-bold bg-red-100 px-2 py-0.5 rounded text-xs">+${c.false_alert_increase_pct}% false alarms!</span>`;
    }

    card.innerHTML = `
      <div class="flex justify-between items-start mb-2">
        <h4 class="font-bold text-sm text-slate-900">${c.system}</h4>
        ${changeBadge}
      </div>
      <div class="space-y-1.5 text-xs text-slate-700 mt-2">
        <div class="flex justify-between"><span>Clean False Alerts:</span> <strong>${c.clean_false_alerts}</strong></div>
        <div class="flex justify-between"><span>Noisy False Alerts:</span> <strong>${c.noisy_false_alerts}</strong></div>
        <div class="flex justify-between"><span>Clean Alert Rate:</span> <span>${c.clean_alerts_per_day.toFixed(3)} /day</span></div>
        <div class="flex justify-between"><span>Noisy Alert Rate:</span> <span>${c.noisy_alerts_per_day.toFixed(3)} /day</span></div>
      </div>
    `;
    container.appendChild(card);
  });

  // Render Bar Chart
  const ctx = document.getElementById("chart-noise-comparison")?.getContext("2d");
  if (ctx) {
    if (charts.noise) charts.noise.destroy();
    charts.noise = new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.comparison.map(c => c.system),
        datasets: [
          { label: "Original Clean Telemetry (False Alerts)", data: data.comparison.map(c => c.clean_false_alerts), backgroundColor: "#0284C7" },
          { label: "Corrupted Telemetry (False Alerts)", data: data.comparison.map(c => c.noisy_false_alerts), backgroundColor: "#EF4444" }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { title: { display: true, text: "False Alarms (Counts)" } }
        }
      }
    });
  }
}

// --- AI Chat Logic ---
let chatCsvFile = null;

document.getElementById('chat-csv-upload')?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        chatCsvFile = file;
        document.getElementById('csv-upload-status').textContent = file.name;
    } else {
        chatCsvFile = null;
        document.getElementById('csv-upload-status').textContent = 'No file selected';
    }
});

function appendChatMessage(sender, text) {
    const container = document.getElementById('chat-history');
    if(!container) return;
    const isBot = sender === 'bot';
    
    const div = document.createElement('div');
    div.className = 'flex gap-3 ' + (isBot ? '' : 'flex-row-reverse');
    
    const iconDiv = document.createElement('div');
    iconDiv.className = 'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ' + 
        (isBot ? 'bg-cyan-100 text-cyan-700' : 'bg-slate-200 text-slate-700');
    iconDiv.innerHTML = isBot ? '<i data-lucide="bot" class="w-4 h-4"></i>' : '<i data-lucide="user" class="w-4 h-4"></i>';
    
    const msgDiv = document.createElement('div');
    msgDiv.className = 'bg-white border border-slate-200 rounded-2xl px-4 py-2 text-sm max-w-[85%] shadow-sm ' + 
        (isBot ? 'rounded-tl-none text-slate-800 whitespace-pre-wrap' : 'rounded-tr-none text-slate-800 whitespace-pre-wrap');
    msgDiv.textContent = text;
    
    div.appendChild(iconDiv);
    div.appendChild(msgDiv);
    container.appendChild(div);
    
    // Re-render icons for dynamically added content
    if(window.lucide) lucide.createIcons();
    
    container.scrollTop = container.scrollHeight;
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg && !chatCsvFile) return;
    
    if (msg) appendChatMessage('user', msg);
    input.value = '';
    
    const formData = new FormData();
    if (msg) formData.append('message', msg);
    if (chatCsvFile) formData.append('file', chatCsvFile);
    
    try {
        // Clear file after sending so we don't keep uploading it
        if(chatCsvFile) {
            chatCsvFile = null;
            document.getElementById('chat-csv-upload').value = '';
            document.getElementById('csv-upload-status').textContent = 'No file selected';
        }

        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if(response.ok) {
            appendChatMessage('bot', data.reply);
        } else {
            appendChatMessage('bot', 'Error: ' + data.detail);
        }
    } catch (err) {
        appendChatMessage('bot', 'Failed to communicate with AI server.');
    }
}

// --- Telegram Alert Logic ---
async function sendTelegramAlert(patientId) {
    try {
        const response = await fetch(`/api/telegram_alert?patient_id=${patientId}`, { method: 'POST' });
        const data = await response.json();
        if(data.success) {
            alert(`Telegram Alert Sent for Patient ${patientId}!`);
        } else {
            alert(`Failed to send Telegram alert: ${data.message}`);
        }
    } catch (err) {
        alert("Failed to reach server to send Telegram alert.");
    }
}
