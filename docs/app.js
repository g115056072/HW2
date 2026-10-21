const DATASETS = {
  "House Price Prediction": "./data/modified_data.csv",
  "Human-AI Interaction": "./data/Human-AI Interaction.csv",
};

const state = {
  rawRows: [],
  currentDataset: "House Price Prediction",
};

function shouldAutoAnalyze() {
  return state.currentDataset === "House Price Prediction" || state.currentDataset === "Human-AI Interaction";
}

const els = {
  datasetSelect: document.getElementById("datasetSelect"),
  targetSelect: document.getElementById("targetSelect"),
  dateSelect: document.getElementById("dateSelect"),
  customFile: document.getElementById("customFile"),
  uploadWrap: document.getElementById("uploadWrap"),
  runBtn: document.getElementById("runBtn"),
  status: document.getElementById("status"),
};

function setStatus(msg) {
  els.status.textContent = msg;
}

function round(v, d = 4) {
  if (!Number.isFinite(v)) return null;
  return Number(v.toFixed(d));
}

function parseCSVText(csvText) {
  const parsed = Papa.parse(csvText, { header: true, dynamicTyping: false, skipEmptyLines: true });
  return parsed.data;
}

async function loadDataset(name) {
  if (name === "Custom Dataset") {
    state.rawRows = [];
    fillColumnSelectors([]);
    setStatus("Upload a CSV for custom dataset.");
    return;
  }

  setStatus("Loading dataset...");
  const res = await fetch(DATASETS[name]);
  const text = await res.text();
  state.rawRows = parseCSVText(text);
  fillColumnSelectors(state.rawRows);
  setStatus(`Loaded ${state.rawRows.length.toLocaleString()} rows.`);
}

function getColumns(rows) {
  if (!rows || rows.length === 0) return [];
  return Object.keys(rows[0]);
}

function fillColumnSelectors(rows) {
  const cols = getColumns(rows);
  els.targetSelect.innerHTML = "";
  els.dateSelect.innerHTML = "";

  for (const col of cols) {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    els.targetSelect.appendChild(opt);
  }

  const noneOpt = document.createElement("option");
  noneOpt.value = "";
  noneOpt.textContent = "None";
  els.dateSelect.appendChild(noneOpt);

  for (const col of cols) {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    els.dateSelect.appendChild(opt);
  }

  if (cols.length > 0) {
    const defaultTarget =
      state.currentDataset === "House Price Prediction" && cols.includes("price")
        ? "price"
        : state.currentDataset === "Human-AI Interaction" && cols.includes("latent_trust_score")
        ? "latent_trust_score"
        : cols[cols.length - 1];
    els.targetSelect.value = defaultTarget;

    const dateCandidate = ["date", "Date", "datetime", "timestamp"].find((x) => cols.includes(x));
    els.dateSelect.value = dateCandidate || "";
  }

  renderTable("previewTable", rows.slice(0, 15));
}

function isNumericColumn(rows, col) {
  let seen = 0;
  for (const r of rows) {
    const v = r[col];
    if (v === null || v === undefined || v === "") continue;
    const n = Number(v);
    if (!Number.isFinite(n)) return false;
    seen += 1;
    if (seen > 20) break;
  }
  return seen > 0;
}

function median(arr) {
  if (arr.length === 0) return null;
  const a = [...arr].sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 === 0 ? (a[m - 1] + a[m]) / 2 : a[m];
}

function mode(arr) {
  const map = new Map();
  for (const x of arr) map.set(x, (map.get(x) || 0) + 1);
  let best = null;
  let bestCount = -1;
  for (const [k, c] of map.entries()) {
    if (c > bestCount) {
      best = k;
      bestCount = c;
    }
  }
  return best;
}

function shuffleSplit(rows, testSize = 0.2, seed = 42) {
  const arr = [...rows];
  let x = seed;
  function rand() {
    x = (1664525 * x + 1013904223) % 4294967296;
    return x / 4294967296;
  }
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  const testN = Math.max(1, Math.floor(arr.length * testSize));
  return { train: arr.slice(0, arr.length - testN), test: arr.slice(arr.length - testN) };
}

function pearson(x, y) {
  const n = x.length;
  if (n < 2) return 0;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] - mx;
    const b = y[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  if (dx === 0 || dy === 0) return 0;
  return num / Math.sqrt(dx * dy);
}

function transpose(A) {
  return A[0].map((_, j) => A.map((row) => row[j]));
}

function matMul(A, B) {
  const out = Array.from({ length: A.length }, () => Array(B[0].length).fill(0));
  for (let i = 0; i < A.length; i++) {
    for (let k = 0; k < B.length; k++) {
      const aik = A[i][k];
      for (let j = 0; j < B[0].length; j++) {
        out[i][j] += aik * B[k][j];
      }
    }
  }
  return out;
}

function matVecMul(A, v) {
  return A.map((row) => row.reduce((s, x, i) => s + x * v[i], 0));
}

function solveLinearSystem(A, b) {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) continue;
    [M[col], M[pivot]] = [M[pivot], M[col]];

    const div = M[col][col];
    for (let j = col; j <= n; j++) M[col][j] /= div;

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let j = col; j <= n; j++) M[r][j] -= factor * M[col][j];
    }
  }
  return M.map((row) => row[n]);
}

function fitRidge(X, y, lambda = 1e-6) {
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  for (let i = 0; i < XtX.length; i++) XtX[i][i] += lambda;
  const Xty = Xt.map((row) => row.reduce((s, x, idx) => s + x * y[idx], 0));
  return solveLinearSystem(XtX, Xty);
}

function predictLinear(X, beta) {
  return matVecMul(X, beta);
}

function mae(y, p) {
  return y.reduce((s, v, i) => s + Math.abs(v - p[i]), 0) / y.length;
}
function rmse(y, p) {
  return Math.sqrt(y.reduce((s, v, i) => s + (v - p[i]) ** 2, 0) / y.length);
}
function r2(y, p) {
  const m = y.reduce((a, b) => a + b, 0) / y.length;
  const ssRes = y.reduce((s, v, i) => s + (v - p[i]) ** 2, 0);
  const ssTot = y.reduce((s, v) => s + (v - m) ** 2, 0);
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

function renderTable(elId, rows) {
  const el = document.getElementById(elId);
  if (!rows || rows.length === 0) {
    el.innerHTML = "<p style='padding:10px'>No data.</p>";
    return;
  }
  const cols = Object.keys(rows[0]);
  const thead = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const tbody = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`)
    .join("");
  el.innerHTML = `<table><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
}

function prepareRows(rawRows, target, datasetName) {
  let rows = rawRows.map((r) => ({ ...r }));

  if (datasetName === "House Price Prediction") {
    rows = rows.filter((r) => Number(r[target]) > 0);
    for (const r of rows) {
      const d = new Date(r.date);
      if (!Number.isNaN(d.getTime())) {
        r.year = d.getFullYear();
        r.month = d.getMonth() + 1;
        r.day_of_month = d.getDate();
      }
      delete r.price_per_sqft;
      delete r.street;
    }
  }

  rows = rows.filter((r) => Number.isFinite(Number(r[target])));
  return rows;
}

function buildFeatureMaps(rows, target) {
  const cols = Object.keys(rows[0]).filter((c) => c !== target && c !== "date");
  const numericCols = cols.filter((c) => isNumericColumn(rows, c));
  const catCols = cols.filter((c) => !numericCols.includes(c));

  const numMedian = {};
  for (const c of numericCols) {
    const vals = rows.map((r) => Number(r[c])).filter((v) => Number.isFinite(v));
    numMedian[c] = median(vals) ?? 0;
  }

  const catMode = {};
  const catLevels = {};
  for (const c of catCols) {
    const vals = rows.map((r) => (r[c] === "" || r[c] == null ? null : String(r[c]))).filter((v) => v != null);
    catMode[c] = mode(vals) ?? "Unknown";
    catLevels[c] = Array.from(new Set(vals.length ? vals : ["Unknown"]));
  }

  return { cols, numericCols, catCols, numMedian, catMode, catLevels };
}

function encodeRows(rows, target, fmap) {
  const featureNames = [...fmap.numericCols];
  for (const c of fmap.catCols) {
    for (const lvl of fmap.catLevels[c]) featureNames.push(`${c}=${lvl}`);
  }

  const X = [];
  const y = [];

  for (const r of rows) {
    const row = [];
    for (const c of fmap.numericCols) {
      const v = Number(r[c]);
      row.push(Number.isFinite(v) ? v : fmap.numMedian[c]);
    }
    for (const c of fmap.catCols) {
      const val = r[c] == null || r[c] === "" ? fmap.catMode[c] : String(r[c]);
      for (const lvl of fmap.catLevels[c]) row.push(val === lvl ? 1 : 0);
    }

    const yt = Number(r[target]);
    if (Number.isFinite(yt)) {
      X.push(row);
      y.push(yt);
    }
  }

  return { X, y, featureNames };
}

function withIntercept(X) {
  return X.map((r) => [1, ...r]);
}

function fScoreFromCorrelation(r, n) {
  const rr = Math.min(0.999999, Math.abs(r) ** 2);
  return (rr / Math.max(1e-12, 1 - rr)) * Math.max(1, n - 2);
}

function fitModels(rawRows, target, datasetName) {
  const rows = prepareRows(rawRows, target, datasetName);
  if (rows.length < 30) throw new Error("Dataset has too few usable rows after cleaning.");

  const split = shuffleSplit(rows, 0.2, 42);
  const fmap = buildFeatureMaps(split.train, target);

  if (fmap.numericCols.length === 0) {
    throw new Error("Need at least one numeric feature for simple regression.");
  }

  const trainSimplePairs = fmap.numericCols
    .map((c) => {
      const xs = [];
      const ys = [];
      for (const r of split.train) {
        const xv = Number(r[c]);
        const yv = Number(r[target]);
        if (Number.isFinite(xv) && Number.isFinite(yv)) {
          xs.push(xv);
          ys.push(yv);
        }
      }
      const corr = Math.abs(pearson(xs, ys));
      return { feature: c, corr };
    })
    .sort((a, b) => b.corr - a.corr);

  const bestSingleFeature = trainSimplePairs[0].feature;

  const xTrain = [];
  const yTrain = [];
  for (const r of split.train) {
    const xv = Number(r[bestSingleFeature]);
    const yv = Number(r[target]);
    if (Number.isFinite(xv) && Number.isFinite(yv)) {
      xTrain.push([1, xv]);
      yTrain.push(yv);
    }
  }
  const bSimple = fitRidge(xTrain, yTrain);

  const xTestSimple = [];
  const yTestSimple = [];
  for (const r of split.test) {
    const xv = Number(r[bestSingleFeature]);
    const yv = Number(r[target]);
    if (Number.isFinite(xv) && Number.isFinite(yv)) {
      xTestSimple.push([1, xv]);
      yTestSimple.push(yv);
    }
  }
  const predSimple = predictLinear(xTestSimple, bSimple);

  const encTrain = encodeRows(split.train, target, fmap);
  const encTest = encodeRows(split.test, target, fmap);

  const featureScores = encTrain.featureNames.map((f, j) => {
    const col = encTrain.X.map((r) => r[j]);
    const rxy = pearson(col, encTrain.y);
    return { feature: f, f_score: fScoreFromCorrelation(rxy, encTrain.y.length), abs_corr: Math.abs(rxy), idx: j };
  });

  featureScores.sort((a, b) => b.f_score - a.f_score);
  const k = Math.min(20, featureScores.length);
  const selected = featureScores.slice(0, k);
  const selectedIdx = selected.map((s) => s.idx);

  const XtrSel = encTrain.X.map((r) => selectedIdx.map((j) => r[j]));
  const XteSel = encTest.X.map((r) => selectedIdx.map((j) => r[j]));

  const bMulti = fitRidge(withIntercept(XtrSel), encTrain.y);
  const predMulti = predictLinear(withIntercept(XteSel), bMulti);

  const simpleMetrics = {
    MAE: round(mae(yTestSimple, predSimple), 4),
    RMSE: round(rmse(yTestSimple, predSimple), 4),
    R2: round(r2(yTestSimple, predSimple), 4),
  };

  const multiMetrics = {
    MAE: round(mae(encTest.y, predMulti), 4),
    RMSE: round(rmse(encTest.y, predMulti), 4),
    R2: round(r2(encTest.y, predMulti), 4),
  };

  return {
    rows,
    bestSingleFeature,
    singleCorrTop: trainSimplePairs.slice(0, 20).map((x) => ({ feature: x.feature, "|correlation|": round(x.corr, 6) })),
    selectedFeatures: selected.map((s) => ({ feature: s.feature, f_score: round(s.f_score, 6) })),
    yTestSimple,
    predSimple,
    yTestMulti: encTest.y,
    predMulti,
    simpleResiduals: yTestSimple.map((v, i) => v - predSimple[i]),
    multiResiduals: encTest.y.map((v, i) => v - predMulti[i]),
    simpleMetrics,
    multiMetrics,
  };
}

function runAutoregression(rawRows, target, dateCol) {
  if (!dateCol) return null;

  const rows = rawRows
    .map((r) => ({ d: new Date(r[dateCol]), y: Number(r[target]) }))
    .filter((x) => Number.isFinite(x.y) && !Number.isNaN(x.d.getTime()));

  if (rows.length < 40) return null;

  const byDay = new Map();
  for (const r of rows) {
    const key = r.d.toISOString().slice(0, 10);
    if (!byDay.has(key)) byDay.set(key, []);
    byDay.get(key).push(r.y);
  }

  const series = Array.from(byDay.entries())
    .map(([day, vals]) => ({ date: day, value: median(vals) }))
    .sort((a, b) => a.date.localeCompare(b.date));

  if (series.length < 30) return null;

  const n = series.length;
  const trainEnd = Math.floor(n * 0.8);
  const development = series.slice(0, trainEnd);
  const test = series.slice(trainEnd);
  const valStart = Math.floor(development.length * 0.8);
  const tr = development.slice(0, valStart);
  const val = development.slice(valStart);

  function fitAR(trainSeries, lag) {
    const X = [];
    const y = [];
    for (let t = lag; t < trainSeries.length; t++) {
      const row = [1, t];
      for (let j = 1; j <= lag; j++) row.push(trainSeries[t - j].value);
      X.push(row);
      y.push(trainSeries[t].value);
    }
    const beta = fitRidge(X, y);
    return beta;
  }

  function forecastAR(trainSeries, horizonSeries, lag, beta) {
    const history = trainSeries.map((x) => x.value);
    const out = [];
    for (let i = 0; i < horizonSeries.length; i++) {
      const t = history.length;
      const row = [1, t];
      for (let j = 1; j <= lag; j++) row.push(history[history.length - j]);
      const yhat = row.reduce((s, x, idx) => s + x * beta[idx], 0);
      out.push(yhat);
      history.push(yhat);
    }
    return out;
  }

  const lagRows = [];
  for (let lag = 1; lag <= 7; lag++) {
    if (tr.length <= lag + 2) continue;
    const beta = fitAR(tr, lag);
    const pred = forecastAR(tr, val, lag, beta);
    const yTrue = val.map((x) => x.value);
    lagRows.push({
      lags: lag,
      MAE: round(mae(yTrue, pred), 4),
      RMSE: round(rmse(yTrue, pred), 4),
      R2: round(r2(yTrue, pred), 4),
    });
  }

  if (lagRows.length === 0) return null;
  lagRows.sort((a, b) => a.RMSE - b.RMSE);
  const bestLag = lagRows[0].lags;

  const betaFinal = fitAR(development, bestLag);
  const predFinal = forecastAR(development, test, bestLag, betaFinal);
  const yTest = test.map((x) => x.value);

  return {
    bestLag,
    lagRows,
    metrics: { MAE: round(mae(yTest, predFinal), 4), RMSE: round(rmse(yTest, predFinal), 4), R2: round(r2(yTest, predFinal), 4) },
    development,
    test,
    pred: predFinal,
  };
}

function plotActualPred(elId, yTrue, yPred, title) {
  const low = Math.min(Math.min(...yTrue), Math.min(...yPred));
  const high = Math.max(Math.max(...yTrue), Math.max(...yPred));
  Plotly.newPlot(
    elId,
    [
      { x: yTrue, y: yPred, mode: "markers", type: "scatter", name: "points", opacity: 0.55 },
      { x: [low, high], y: [low, high], mode: "lines", name: "ideal", line: { dash: "dash" } },
    ],
    { title, margin: { t: 50 } },
    { responsive: true }
  );
}

function runAnalysis() {
  if (state.rawRows.length === 0) {
    setStatus("No dataset loaded.");
    return;
  }
  const target = els.targetSelect.value;
  const dateCol = els.dateSelect.value || null;

  try {
    setStatus("Running analysis...");
    const r = fitModels(state.rawRows, target, state.currentDataset);
    const ar = runAutoregression(state.rawRows, target, dateCol);

    const evalRows = [
      { Model: "Simple Linear Regression", ...r.simpleMetrics },
      { Model: "Multiple Linear Regression", ...r.multiMetrics },
    ];
    if (ar) evalRows.push({ Model: "Autoregression", ...ar.metrics });
    evalRows.sort((a, b) => a.RMSE - b.RMSE);
    renderTable("metricsTable", evalRows);

    Plotly.newPlot(
      "plotTarget",
      [{ x: r.rows.map((x) => Number(x[target])), type: "histogram", nbinsx: 60 }],
      { title: `Distribution of ${target}`, margin: { t: 50 } },
      { responsive: true }
    );

    Plotly.newPlot(
      "plotRmse",
      [{ x: evalRows.map((x) => x.Model), y: evalRows.map((x) => x.RMSE), type: "bar" }],
      { title: "RMSE Comparison", margin: { t: 50 } },
      { responsive: true }
    );

    plotActualPred("plotSimple", r.yTestSimple, r.predSimple, `Simple LR (feature: ${r.bestSingleFeature})`);
    plotActualPred("plotMulti", r.yTestMulti, r.predMulti, "Multiple Linear Regression");

    Plotly.newPlot(
      "plotResiduals",
      [
        { x: r.simpleResiduals, type: "histogram", name: "Simple LR", opacity: 0.55, nbinsx: 60 },
        { x: r.multiResiduals, type: "histogram", name: "Multiple LR", opacity: 0.55, nbinsx: 60 },
      ],
      {
        title: "Residual Distributions",
        barmode: "overlay",
        shapes: [{ type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper", line: { dash: "dash" } }],
        margin: { t: 50 },
      },
      { responsive: true }
    );

    renderTable("singleCorrTable", r.singleCorrTop);
    renderTable("multiFeatTable", r.selectedFeatures);

    if (!ar) {
      document.getElementById("plotAr").innerHTML = "<p style='padding:10px'>Autoregression unavailable (need valid date column and enough observations).</p>";
      renderTable("lagTable", []);
    } else {
      const xDev = ar.development.map((d) => d.date);
      const yDev = ar.development.map((d) => d.value);
      const xTest = ar.test.map((d) => d.date);
      const yTest = ar.test.map((d) => d.value);

      Plotly.newPlot(
        "plotAr",
        [
          { x: xDev, y: yDev, type: "scatter", mode: "lines", name: "Development data" },
          { x: xTest, y: yTest, type: "scatter", mode: "lines", name: "Actual test" },
          { x: xTest, y: ar.pred, type: "scatter", mode: "lines", name: `AR(${ar.bestLag}) forecast` },
        ],
        {
          title: "Autoregression on Daily Median Target",
          shapes: [{ type: "line", x0: xTest[0], x1: xTest[0], y0: 0, y1: 1, yref: "paper", line: { dash: "dash" } }],
          margin: { t: 50 },
        },
        { responsive: true }
      );
      renderTable("lagTable", ar.lagRows);
    }

    setStatus("Done.");
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  }
}

function wireTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

els.datasetSelect.addEventListener("change", async (e) => {
  state.currentDataset = e.target.value;
  els.uploadWrap.classList.toggle("hidden", state.currentDataset !== "Custom Dataset");
  await loadDataset(state.currentDataset);
  if (shouldAutoAnalyze()) runAnalysis();
});

els.targetSelect.addEventListener("change", () => {
  if (shouldAutoAnalyze()) runAnalysis();
});

els.dateSelect.addEventListener("change", () => {
  if (shouldAutoAnalyze()) runAnalysis();
});

els.customFile.addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  const text = await f.text();
  state.rawRows = parseCSVText(text);
  fillColumnSelectors(state.rawRows);
  setStatus(`Loaded custom dataset (${state.rawRows.length.toLocaleString()} rows).`);
});

els.runBtn.addEventListener("click", runAnalysis);

wireTabs();
loadDataset(state.currentDataset).then(() => {
  if (shouldAutoAnalyze()) runAnalysis();
});
