// State Store
let state = {
  bestConclusions: null,
  allRuns: [],
  curves: null,
  activeTab: 'detector-tab'
};

// Global Chart References
let metricsChartInstance = null;
let lossChartInstance = null;
let f1ChartInstance = null;

// DOM Elements
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const newsInput = document.getElementById('news-input');
const btnDetect = document.getElementById('btn-detect');
const predictionResults = document.getElementById('prediction-results');
const serverDeviceInfo = document.getElementById('server-device-info');

// Tab Switching Logic
tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const tabId = btn.getAttribute('data-tab');
    
    tabButtons.forEach(b => b.classList.remove('active'));
    tabContents.forEach(c => c.classList.remove('active'));
    
    btn.classList.add('active');
    document.getElementById(tabId).classList.add('active');
    
    state.activeTab = tabId;
    
    // Resize charts if switching to analytics
    if (tabId === 'analytics-tab') {
      setTimeout(resizeCharts, 50);
    }
  });
});

// Resize helper for Chart.js
function resizeCharts() {
  if (metricsChartInstance) metricsChartInstance.resize();
  if (lossChartInstance) lossChartInstance.resize();
  if (f1ChartInstance) f1ChartInstance.resize();
}

// Format Numbers
function fmt(val, decimals = 4) {
  if (val === undefined || val === null || isNaN(val)) return '-';
  return Number(val).toFixed(decimals);
}

function fmtPct(val) {
  if (val === undefined || val === null || isNaN(val)) return '-';
  return (Number(val) * 100).toFixed(2) + '%';
}

// Fetch Conclusions & Summaries
async function loadConclusions() {
  try {
    const res = await fetch('/api/best_conclusions');
    if (!res.ok) throw new Error('Failed to load conclusions');
    const data = await res.json();
    state.bestConclusions = data;
    
    // Update Tab 1 Quick Summaries
    document.getElementById('summary-lstm-acc').innerText = fmtPct(data.best_lstm.test_accuracy);
    document.getElementById('summary-lstm-params').innerText = `LR: ${data.best_lstm.lr} | Drop: ${data.best_lstm.dropout} | BS: ${data.best_lstm.batch_size}`;
    
    document.getElementById('summary-bert-acc').innerText = fmtPct(data.best_distilbert.test_accuracy);
    document.getElementById('summary-bert-params').innerText = `LR: ${data.best_distilbert.lr} | Drop: ${data.best_distilbert.dropout} | BS: ${data.best_distilbert.batch_size}`;
    
    // Update Tab 2 Winner Card
    const winnerName = data.winner === 'distilbert' ? 'DistilBERT (Transformer)' : 'Bi-LSTM';
    const loserName = data.winner === 'distilbert' ? 'LSTM' : 'DistilBERT';
    const winMetric = data.winner === 'distilbert' ? data.best_distilbert : data.best_lstm;
    const loseMetric = data.winner === 'distilbert' ? data.best_lstm : data.best_distilbert;
    const diff = (winMetric.test_f1 - loseMetric.test_f1) * 100;
    
    document.getElementById('winner-title').innerText = `${winnerName} đạt hiệu quả cao nhất`;
    document.getElementById('winner-description').innerHTML = `
      Dự án đã thực hiện so sánh đánh giá chuyên sâu giữa mô hình học máy chuỗi thời gian <strong>Bi-LSTM</strong> 
      và mô hình học sâu Transformer <strong>DistilBERT</strong>. <br>
      Mô hình chiến thắng là <strong>${winnerName}</strong> (Test F1-score = <strong>${fmtPct(winMetric.test_f1)}</strong>, 
      Test Accuracy = <strong>${fmtPct(winMetric.test_accuracy)}</strong>), 
      vượt qua mô hình ${loserName} là <strong>${diff.toFixed(2)}%</strong> trên chỉ số F1-Score.
    `;
    
    // Update Tab 2 Metrics Table
    document.getElementById('best-lstm-acc').innerText = fmt(data.best_lstm.test_accuracy);
    document.getElementById('best-lstm-prec').innerText = fmt(data.best_lstm.test_precision);
    document.getElementById('best-lstm-rec').innerText = fmt(data.best_lstm.test_recall);
    document.getElementById('best-lstm-f1').innerText = fmt(data.best_lstm.test_f1);
    
    document.getElementById('best-bert-acc').innerText = fmt(data.best_distilbert.test_accuracy);
    document.getElementById('best-bert-prec').innerText = fmt(data.best_distilbert.test_precision);
    document.getElementById('best-bert-rec').innerText = fmt(data.best_distilbert.test_recall);
    document.getElementById('best-bert-f1').innerText = fmt(data.best_distilbert.test_f1);
    
    // Build Metric Bar Chart
    buildComparisonChart(data.best_lstm, data.best_distilbert);
    
  } catch (err) {
    console.error('Error loading conclusions:', err);
  }
}

// Build Metric Bar Chart
function buildComparisonChart(lstm, bert) {
  const ctx = document.getElementById('metricsCompareChart').getContext('2d');
  
  if (metricsChartInstance) {
    metricsChartInstance.destroy();
  }
  
  metricsChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
      datasets: [
        {
          label: 'Bi-LSTM (Best)',
          data: [lstm.test_accuracy, lstm.test_precision, lstm.test_recall, lstm.test_f1],
          backgroundColor: 'rgba(59, 130, 246, 0.45)',
          borderColor: 'rgb(59, 130, 246)',
          borderWidth: 1.5,
          borderRadius: 4
        },
        {
          label: 'DistilBERT (Best)',
          data: [bert.test_accuracy, bert.test_precision, bert.test_recall, bert.test_f1],
          backgroundColor: 'rgba(168, 85, 247, 0.45)',
          borderColor: 'rgb(168, 85, 247)',
          borderWidth: 1.5,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 1.0,
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#f8fafc', font: { family: 'Outfit' } }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1
        }
      }
    }
  });
}

// Fetch Curves & Draw Charts
async function loadCurves() {
  try {
    const res = await fetch('/api/curves');
    if (!res.ok) throw new Error('Failed to load training curves');
    const data = await res.json();
    state.curves = data;
    
    // Draw curves
    buildCurvesCharts(data.lstm, data.distilbert);
  } catch (err) {
    console.error('Error loading curves:', err);
  }
}

function buildCurvesCharts(lstmHistory, bertHistory) {
  const lossCtx = document.getElementById('lossCurvesChart').getContext('2d');
  const f1Ctx = document.getElementById('f1CurvesChart').getContext('2d');
  
  if (lossChartInstance) lossChartInstance.destroy();
  if (f1ChartInstance) f1ChartInstance.destroy();
  
  // Extract data arrays
  const lstmEpochs = lstmHistory ? lstmHistory.map(h => `Ep ${h.epoch}`) : [];
  const bertEpochs = bertHistory ? bertHistory.map(h => `Ep ${h.epoch}`) : [];
  const maxEpochs = Math.max(lstmEpochs.length, bertEpochs.length);
  const labels = Array.from({length: maxEpochs}, (_, i) => `Epoch ${i+1}`);
  
  // Loss Curves
  lossChartInstance = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'LSTM Train Loss',
          data: lstmHistory ? lstmHistory.map(h => h.train_loss) : [],
          borderColor: 'rgba(59, 130, 246, 0.4)',
          borderDash: [5, 5],
          backgroundColor: 'transparent',
          tension: 0.2
        },
        {
          label: 'LSTM Val Loss',
          data: lstmHistory ? lstmHistory.map(h => h.val_loss) : [],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'transparent',
          tension: 0.2,
          pointStyle: 'rectRot',
          pointRadius: 5
        },
        {
          label: 'DistilBERT Train Loss',
          data: bertHistory ? bertHistory.map(h => h.train_loss) : [],
          borderColor: 'rgba(168, 85, 247, 0.4)',
          borderDash: [5, 5],
          backgroundColor: 'transparent',
          tension: 0.2
        },
        {
          label: 'DistilBERT Val Loss',
          data: bertHistory ? bertHistory.map(h => h.val_loss) : [],
          borderColor: 'rgb(168, 85, 247)',
          backgroundColor: 'transparent',
          tension: 0.2,
          pointStyle: 'triangle',
          pointRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
        },
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: { labels: { color: '#f8fafc', font: { family: 'Outfit', size: 10 } } }
      }
    }
  });

  // F1 Curves
  f1ChartInstance = new Chart(f1Ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'LSTM Train F1',
          data: lstmHistory ? lstmHistory.map(h => h.train_f1) : [],
          borderColor: 'rgba(59, 130, 246, 0.4)',
          borderDash: [5, 5],
          backgroundColor: 'transparent',
          tension: 0.2
        },
        {
          label: 'LSTM Val F1',
          data: lstmHistory ? lstmHistory.map(h => h.val_f1) : [],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'transparent',
          tension: 0.2,
          pointStyle: 'rectRot',
          pointRadius: 5
        },
        {
          label: 'DistilBERT Train F1',
          data: bertHistory ? bertHistory.map(h => h.train_f1) : [],
          borderColor: 'rgba(168, 85, 247, 0.4)',
          borderDash: [5, 5],
          backgroundColor: 'transparent',
          tension: 0.2
        },
        {
          label: 'DistilBERT Val F1',
          data: bertHistory ? bertHistory.map(h => h.val_f1) : [],
          borderColor: 'rgb(168, 85, 247)',
          backgroundColor: 'transparent',
          tension: 0.2,
          pointStyle: 'triangle',
          pointRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
        },
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: { labels: { color: '#f8fafc', font: { family: 'Outfit', size: 10 } } }
      }
    }
  });
}

// Fetch All Runs & Bind Filters
async function loadRuns() {
  try {
    const res = await fetch('/api/history');
    if (!res.ok) throw new Error('Failed to load runs');
    const data = await res.json();
    state.allRuns = data;
    renderTable();
  } catch (err) {
    console.error('Error loading history:', err);
    document.getElementById('hyperparams-table-body').innerHTML = `
      <tr>
        <td colspan="10" style="text-align: center; color: var(--fake-color); padding: 2rem;">
          Không thể tải dữ liệu bảng hyperparameters. Vui lòng kiểm tra file all_runs_results.csv.
        </td>
      </tr>
    `;
  }
}

// Render hyperparameter comparison table
function renderTable() {
  const modelFilter = document.getElementById('filter-model').value;
  const dropoutFilter = document.getElementById('filter-dropout').value;
  const batchFilter = document.getElementById('filter-batch-size').value;
  const sortOption = document.getElementById('sort-metrics').value;
  
  // Filter runs
  let filtered = state.allRuns.filter(run => {
    const matchesModel = modelFilter === 'all' || run.model === modelFilter;
    const matchesDropout = dropoutFilter === 'all' || Number(run.dropout) === Number(dropoutFilter);
    const matchesBatch = batchFilter === 'all' || Number(run.batch_size) === Number(batchFilter);
    return matchesModel && matchesDropout && matchesBatch;
  });
  
  // Sort runs
  filtered.sort((a, b) => {
    if (sortOption === 'test_f1-desc') {
      return b.test_f1 - a.test_f1;
    } else if (sortOption === 'test_accuracy-desc') {
      return b.test_accuracy - a.test_accuracy;
    } else if (sortOption === 'val_f1-desc') {
      return b.val_f1 - a.val_f1;
    } else if (sortOption === 'train_loss-asc') {
      return a.train_loss - b.train_loss;
    }
    return 0;
  });
  
  const body = document.getElementById('hyperparams-table-body');
  if (filtered.length === 0) {
    body.innerHTML = `
      <tr>
        <td colspan="10" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          Không tìm thấy cấu hình nào khớp với điều kiện lọc.
        </td>
      </tr>
    `;
    return;
  }
  
  body.innerHTML = filtered.map(run => {
    const modelBadge = run.model === 'lstm' ? '<span class="badge-model lstm">LSTM</span>' : '<span class="badge-model distilbert">DistilBERT</span>';
    
    // Add custom coloring for test accuracy & F1 score to make comparison easy
    let testF1Class = '';
    if (run.test_f1 >= 0.8) testF1Class = 'val-high';
    else if (run.test_f1 >= 0.75) testF1Class = 'val-mid';
    else testF1Class = 'val-low';
    
    let testAccClass = '';
    if (run.test_accuracy >= 0.84) testAccClass = 'val-high';
    else if (run.test_accuracy >= 0.8) testAccClass = 'val-mid';
    else testAccClass = 'val-low';
    
    return `
      <tr>
        <td>${modelBadge}</td>
        <td><code>${run.learning_rate}</code></td>
        <td>${run.dropout}</td>
        <td>${run.batch_size}</td>
        <td>${run.epochs_trained}</td>
        <td>${run.best_epoch}</td>
        <td>${fmt(run.val_loss)}</td>
        <td>${fmt(run.val_f1)}</td>
        <td class="${testAccClass}">${fmt(run.test_accuracy)}</td>
        <td class="${testF1Class}">${fmt(run.test_f1)}</td>
      </tr>
    `;
  }).join('');
}

// Bind Filter Listeners
['filter-model', 'filter-dropout', 'filter-batch-size', 'sort-metrics'].forEach(id => {
  document.getElementById(id).addEventListener('change', renderTable);
});

// Run Real-time Detection
btnDetect.addEventListener('click', async () => {
  const text = newsInput.value.trim();
  if (!text) {
    alert('Vui lòng nhập nội dung tin tức cần kiểm tra!');
    return;
  }
  
  // Set Loading UI state
  btnDetect.disabled = true;
  const originalHtml = btnDetect.innerHTML;
  btnDetect.innerHTML = '<div class="spinner"></div> Đang phân tích...';
  predictionResults.style.display = 'none';
  
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    
    if (!res.ok) throw new Error('API Error');
    const result = await res.json();
    
    // Display results panel
    predictionResults.style.display = 'block';
    serverDeviceInfo.innerText = `Thiết bị xử lý inference: ${result.device.toUpperCase()}`;
    
    // Bind LSTM result
    const lstm = result.lstm;
    if (lstm.error) {
      document.getElementById('lstm-verdict').innerText = 'ERROR';
      document.getElementById('lstm-verdict').className = 'verdict-badge fake';
      document.getElementById('lstm-latency').innerText = '--';
    } else {
      const v = document.getElementById('lstm-verdict');
      v.innerText = lstm.label === 'Fake' ? 'Tin Giả' : 'Tin Thật';
      v.className = lstm.label === 'Fake' ? 'verdict-badge fake' : 'verdict-badge real';
      document.getElementById('lstm-latency').innerText = `${lstm.latency_ms} ms`;
      
      // Update bars
      document.getElementById('lstm-conf-real').innerText = fmtPct(lstm.prob_real);
      document.getElementById('lstm-bar-real').style.width = `${lstm.prob_real * 100}%`;
      document.getElementById('lstm-conf-fake').innerText = fmtPct(lstm.prob_fake);
      document.getElementById('lstm-bar-fake').style.width = `${lstm.prob_fake * 100}%`;
    }
    
    // Bind DistilBERT result
    const bert = result.distilbert;
    if (bert.error) {
      document.getElementById('bert-verdict').innerText = 'ERROR';
      document.getElementById('bert-verdict').className = 'verdict-badge fake';
      document.getElementById('bert-latency').innerText = '--';
    } else {
      const v = document.getElementById('bert-verdict');
      v.innerText = bert.label === 'Fake' ? 'Tin Giả' : 'Tin Thật';
      v.className = bert.label === 'Fake' ? 'verdict-badge fake' : 'verdict-badge real';
      document.getElementById('bert-latency').innerText = `${bert.latency_ms} ms`;
      
      // Update bars
      document.getElementById('bert-conf-real').innerText = fmtPct(bert.prob_real);
      document.getElementById('bert-bar-real').style.width = `${bert.prob_real * 100}%`;
      document.getElementById('bert-conf-fake').innerText = fmtPct(bert.prob_fake);
      document.getElementById('bert-bar-fake').style.width = `${bert.prob_fake * 100}%`;
    }
    
    // Scroll to results smoothly
    setTimeout(() => {
      predictionResults.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
    
  } catch (err) {
    console.error('Detection failed:', err);
    alert('Đã xảy ra lỗi trong quá trình phân loại. Vui lòng thử lại!');
  } finally {
    btnDetect.disabled = false;
    btnDetect.innerHTML = originalHtml;
  }
});

// App Entry Point
async function initApp() {
  await loadConclusions();
  await loadCurves();
  await loadRuns();
}

// Trigger load
window.addEventListener('DOMContentLoaded', initApp);
