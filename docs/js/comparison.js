/* Researcher Population Pyramids — Comparison (inflow vs gender gap) page */

(function () {
  const YEAR_MIN = 1970;
  const YEAR_MAX = 2050;
  const HISTORICAL_END = 2023;
  const DEFAULT_YEAR = 2023;
  const FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", '
                    + 'Roboto, "Helvetica Neue", Arial, sans-serif';
  const AXIS_LINE_WIDTH = 2.0;
  const TICK_WIDTH = 2.0;
  const TICK_LEN = 8;

  // One color per UN-style region. Uses Okabe & Ito (2008) — a palette
  // designed to remain distinguishable under common color-vision differences
  // (protanopia, deuteranopia, tritanopia). See https://jfly.uni-koeln.de/color/.
  const REGION_COLORS = {
    'Africa':                          '#E69F00',  // orange
    'Asia':                            '#0072B2',  // blue
    'Europe':                          '#009E73',  // bluish green
    'Latin America and the Caribbean': '#CC79A7',  // reddish purple
    'North America':                   '#D55E00',  // vermillion
    'Oceania':                         '#56B4E9',  // sky blue
    'Unknown':                         '#7f7f7f',  // gray
  };

  const state = {
    year: DEFAULT_YEAR,
    speed: 1,
    playing: false,
    playTimer: null,
    axisMode: 'auto',   // 'auto' = per-year, 'fixed' = all years
    countries: null,    // [{code, name, region}, ...]
    scatter: null,      // {years, data: {CC: [{year, inflow_ratio, gender_gap}, ...]}}
    fixedAxisRanges: null,
  };

  // ------- URL -------
  function readStateFromURL() {
    const params = new URLSearchParams(window.location.search);
    const y = parseInt(params.get('year'), 10);
    if (!isNaN(y) && y >= YEAR_MIN && y <= YEAR_MAX) state.year = y;
    const m = params.get('axis');
    if (m === 'auto' || m === 'fixed') state.axisMode = m;
  }
  function writeStateToURL() {
    const params = new URLSearchParams();
    params.set('year', state.year);
    params.set('axis', state.axisMode);
    window.history.replaceState({}, '', window.location.pathname + '?' + params.toString());
  }

  // ------- Axis ranges -------
  function rangesFromPoints(pts) {
    const xs = pts.map(p => p.gender_gap);
    const ys = pts.map(p => p.inflow_ratio);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = Math.max(0.02, (xMax - xMin) * 0.08);
    const yPad = Math.max(0.005, (yMax - yMin) * 0.08);
    return {
      x: [xMin - xPad, xMax + xPad],
      y: [0, yMax + yPad],
    };
  }

  function computeFixedAxisRanges() {
    const pts = [];
    for (const cc in state.scatter.data) {
      for (const p of state.scatter.data[cc]) pts.push(p);
    }
    state.fixedAxisRanges = rangesFromPoints(pts);
  }

  function axisRangesForYear(year) {
    if (state.axisMode === 'fixed') return state.fixedAxisRanges;
    const pts = [];
    for (const cc in state.scatter.data) {
      const p = state.scatter.data[cc].find(x => x.year === year);
      if (p) pts.push(p);
    }
    if (!pts.length) return state.fixedAxisRanges;
    return rangesFromPoints(pts);
  }

  // ------- Build per-year points -------
  function pointsForYear(year) {
    const byRegion = {};
    for (const cc in state.scatter.data) {
      const meta = countryMeta(cc);
      const region = meta.region || 'Unknown';
      const p = state.scatter.data[cc].find(x => x.year === year);
      if (!p) continue;
      (byRegion[region] = byRegion[region] || []).push({
        code: cc,
        name: meta.name || cc,
        x: p.gender_gap,
        y: p.inflow_ratio,
      });
    }
    return byRegion;
  }

  function countryMeta(cc) {
    if (!state.countries) return { name: cc, region: 'Unknown' };
    return state.countries.find(c => c.code === cc) || { name: cc, region: 'Unknown' };
  }

  function median(arr) {
    if (!arr.length) return null;
    const s = arr.slice().sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  }

  function mediansForYear(year) {
    const gaps = [], inflows = [];
    for (const cc in state.scatter.data) {
      const p = state.scatter.data[cc].find(x => x.year === year);
      if (!p) continue;
      gaps.push(p.gender_gap);
      inflows.push(p.inflow_ratio);
    }
    return { x: median(gaps), y: median(inflows) };
  }

  // ------- Render -------
  function renderPlot() {
    const byRegion = pointsForYear(state.year);
    const regions = Object.keys(REGION_COLORS).filter(r => r !== 'Unknown');
    if (byRegion['Unknown']) regions.push('Unknown');
    const ranges = axisRangesForYear(state.year);

    const traces = [];
    for (const region of regions) {
      const pts = byRegion[region];
      if (!pts || !pts.length) continue;
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: pts.map(p => p.x),
        y: pts.map(p => p.y),
        text: pts.map(p => `${p.name} (${p.code})`),
        customdata: pts.map(p => [p.x, p.y]),
        hovertemplate:
          '<b>%{text}</b><br>' +
          'Gender gap: %{customdata[0]:.3f}<br>' +
          'Researcher inflow: %{customdata[1]:.3f}<extra></extra>',
        name: region,
        marker: {
          size: 12,
          color: REGION_COLORS[region],
          line: { color: '#000', width: 1.2 },
          opacity: 0.85,
        },
      });
    }

    const tickFont = { size: 12, family: FONT_FAMILY, weight: 700 };
    const xAxisCommon = {
      ticks: 'outside',
      tickwidth: TICK_WIDTH,
      ticklen: TICK_LEN,
      tickcolor: '#000',
      tickfont: tickFont,
      linewidth: AXIS_LINE_WIDTH,
      linecolor: '#000',
      showgrid: false,
      mirror: false,
      zeroline: true,
      zerolinecolor: '#aaa',
      zerolinewidth: 1,
    };

    const layout = {
      font: { family: FONT_FAMILY },
      margin: { l: 70, r: 30, t: 30, b: 60 },
      xaxis: Object.assign({}, xAxisCommon, {
        title: { text: '<b>Gender gap in cumulative productivity</b>',
                 font: { size: 14, family: FONT_FAMILY } },
        range: ranges.x,
        tickformat: '.2f',
      }),
      yaxis: Object.assign({}, xAxisCommon, {
        title: { text: '<b>Researcher inflow</b>',
                 font: { size: 14, family: FONT_FAMILY } },
        range: ranges.y,
        tickformat: '.2f',
        zeroline: false,
      }),
      showlegend: true,
      legend: {
        orientation: 'v',
        x: 1.02, y: 1, xanchor: 'left', yanchor: 'top',
        font: { size: 12, family: FONT_FAMILY },
        bgcolor: 'rgba(255,255,255,0)',
      },
      hoverlabel: { font: { size: 13, family: FONT_FAMILY } },
      plot_bgcolor: state.year > HISTORICAL_END ? '#f7f7fc' : 'white',
      paper_bgcolor: 'white',
      shapes: [],
    };

    // Median lines across the 58 countries at the current year.
    const meds = mediansForYear(state.year);
    if (meds.x != null) {
      layout.shapes.push({
        type: 'line',
        x0: meds.x, x1: meds.x,
        y0: 0, y1: 1, yref: 'paper',
        line: { color: '#888', width: 1.2, dash: 'dash' },
      });
    }
    if (meds.y != null) {
      layout.shapes.push({
        type: 'line',
        x0: 0, x1: 1, xref: 'paper',
        y0: meds.y, y1: meds.y,
        line: { color: '#888', width: 1.2, dash: 'dash' },
      });
    }

    const config = { displayModeBar: false, responsive: true };
    Plotly.react('comparison-plot', traces, layout, config);
  }

  // ------- Year label -------
  function updateYearLabel() {
    const label = document.getElementById('year-label');
    const projection = state.year > HISTORICAL_END;
    label.classList.toggle('is-projection', projection);
    label.innerHTML = projection
      ? `${state.year} <span class="projection-tag">PROJECTION</span>`
      : `${state.year}`;
  }

  // ------- Animation -------
  const SPEED_TO_MS = { 0.5: 800, 1: 400, 2: 200, 4: 100 };
  function play() {
    stop();
    state.playing = true;
    document.getElementById('btn-play').classList.add('is-active');
    state.playTimer = setInterval(() => {
      let y = state.year + 1;
      if (y > YEAR_MAX) y = YEAR_MIN;
      state.year = y;
      document.getElementById('year-slider').value = y;
      updateYearLabel();
      renderPlot();
      writeStateToURL();
    }, SPEED_TO_MS[state.speed] || 400);
  }
  function stop() {
    state.playing = false;
    document.getElementById('btn-play').classList.remove('is-active');
    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = null;
  }

  // ------- Events -------
  function wireEvents() {
    document.getElementById('axis-mode-select').addEventListener('change', (e) => {
      state.axisMode = e.target.value;
      renderPlot();
      writeStateToURL();
    });
    document.getElementById('year-slider').addEventListener('input', (e) => {
      stop();
      state.year = parseInt(e.target.value, 10);
      updateYearLabel();
      renderPlot();
      writeStateToURL();
    });
    document.getElementById('btn-play').addEventListener('click', () => {
      state.playing ? stop() : play();
    });
    document.getElementById('btn-reset').addEventListener('click', () => {
      stop();
      state.year = DEFAULT_YEAR;
      document.getElementById('year-slider').value = DEFAULT_YEAR;
      updateYearLabel();
      renderPlot();
      writeStateToURL();
    });
    document.querySelectorAll('[data-speed]').forEach(btn => {
      btn.addEventListener('click', () => {
        state.speed = parseFloat(btn.dataset.speed);
        document.querySelectorAll('[data-speed]').forEach(b =>
          b.classList.toggle('is-active', parseFloat(b.dataset.speed) === state.speed));
        if (state.playing) { stop(); play(); }
      });
    });
    document.getElementById('btn-download').addEventListener('click', downloadCSV);
  }

  // CSV: long format covering every (year, country) pair available in the
  // data so the file is a complete dump of the scatter dataset, independent
  // of the slider position or axis mode.
  function downloadCSV() {
    const meta = {};
    if (state.countries) for (const c of state.countries) meta[c.code] = c;

    const rows = [['year', 'country_code', 'country_name', 'region',
                   'gender_gap_in_cumulative_productivity', 'researcher_inflow']];
    const codes = Object.keys(state.scatter.data).sort();
    for (const cc of codes) {
      const m = meta[cc] || { name: cc, region: 'Unknown' };
      for (const p of state.scatter.data[cc]) {
        rows.push([
          p.year, cc, csvEscape(m.name), csvEscape(m.region),
          p.gender_gap, p.inflow_ratio,
        ]);
      }
    }
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `comparison_all_countries_${YEAR_MIN}-${YEAR_MAX}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function csvEscape(s) {
    if (s == null) return '';
    const str = String(s);
    return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
  }

  // ------- Init -------
  async function init() {
    readStateFromURL();
    const [countries, scatter] = await Promise.all([
      DataLoader.countries(),
      DataLoader.scatter(),
    ]);
    state.countries = countries;
    state.scatter = scatter;
    computeFixedAxisRanges();
    document.getElementById('axis-mode-select').value = state.axisMode;
    document.getElementById('year-slider').value = state.year;
    updateYearLabel();
    renderPlot();
    wireEvents();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
