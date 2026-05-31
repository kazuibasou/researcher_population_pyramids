/* Researcher Population Pyramids — main pyramid page logic */

(function () {
  const HISTORICAL_RANGE = [1970, 2023];
  const PROJECTION_RANGE   = [2024, 2050];
  const MIN_YEAR = HISTORICAL_RANGE[0];
  const MAX_YEAR = PROJECTION_RANGE[1];
  const DEFAULT_YEAR = 2023;
  const DEFAULT_DOMAIN = 'overall';
  const DEFAULT_COUNTRY = 'US';
  const DEFAULT_YMAX = 100;          // paper's max_productivity = 100
  const DEFAULT_LOG_XMAX = 6;        // 10^6, bigger than every country's max
  // Exact matplotlib named colors used in the paper.
  const COLOR_FEMALE = '#006400';    // 'darkgreen'
  const COLOR_MALE   = '#FFA500';    // 'orange'
  // Paper plotting parameters mirrored from old/make_figs.ipynb rcParams.
  const AXIS_LINE_WIDTH = 2.0;
  const TICK_WIDTH = 2.0;
  const TICK_LEN = 8;
  // Font stack reused for chart text — matches docs/css/style.css.
  const FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", '
                    + 'Roboto, "Helvetica Neue", Arial, sans-serif';

  const DOMAIN_LABELS = {
    overall:           'Overall',
    health_sciences:   'Health Sciences',
    life_sciences:     'Life Sciences',
    physical_sciences: 'Physical Sciences',
    social_sciences:   'Social Sciences',
  };

  // ------- State -------
  const state = {
    country: DEFAULT_COUNTRY,
    domain: DEFAULT_DOMAIN,
    year: DEFAULT_YEAR,
    yMax: DEFAULT_YMAX,
    logXMax: DEFAULT_LOG_XMAX,
    pyramid: null,
    projection: null,
    countries: null,
    playing: false,
    speed: 1,
    playTimer: null,
  };

  // ------- URL state ------
  function readStateFromURL() {
    const params = new URLSearchParams(window.location.search);
    const c = params.get('country');
    const d = params.get('domain');
    const y = parseInt(params.get('year'), 10);
    if (c) state.country = c.toUpperCase();
    if (d && DOMAIN_LABELS[d]) state.domain = d;
    if (!isNaN(y) && y >= MIN_YEAR && y <= MAX_YEAR) state.year = y;
  }

  function writeStateToURL() {
    const params = new URLSearchParams();
    params.set('country', state.country);
    params.set('domain', state.domain);
    params.set('year', state.year);
    const newURL = window.location.pathname + '?' + params.toString();
    window.history.replaceState({}, '', newURL);
  }

  // ------- Country selector -------
  function buildCountrySelector(countries) {
    const select = document.getElementById('country-select');
    select.innerHTML = '';
    // Group by region
    const byRegion = {};
    for (const c of countries) {
      (byRegion[c.region] = byRegion[c.region] || []).push(c);
    }
    const regions = Object.keys(byRegion).sort();
    for (const region of regions) {
      const group = document.createElement('optgroup');
      group.label = region;
      for (const c of byRegion[region]) {
        const opt = document.createElement('option');
        opt.value = c.code;
        opt.textContent = `${c.name} (${c.code})`;
        if (c.code === state.country) opt.selected = true;
        group.appendChild(opt);
      }
      select.appendChild(group);
    }
    // Initialize Choices.js for search
    if (window.Choices) {
      // Avoid re-initialization
      if (select._choices) select._choices.destroy();
      select._choices = new Choices(select, {
        searchEnabled: true,
        searchFields: ['label', 'value'],
        itemSelectText: '',
        shouldSort: false,
        searchResultLimit: 20,
      });
    }
  }

  function buildDomainSelector() {
    const select = document.getElementById('domain-select');
    select.innerHTML = '';
    for (const key of Object.keys(DOMAIN_LABELS)) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = DOMAIN_LABELS[key];
      if (key === state.domain) opt.selected = true;
      select.appendChild(opt);
    }
  }

  function syncAxisSelectors() {
    const yEl = document.getElementById('ymax-select');
    if (yEl) yEl.value = String(state.yMax);
    const xEl = document.getElementById('xmax-select');
    if (xEl) xEl.value = String(state.logXMax);
  }

  function buildYearSlider() {
    const slider = document.getElementById('year-slider');
    slider.min = MIN_YEAR;
    slider.max = MAX_YEAR;
    slider.step = 1;
    slider.value = state.year;
  }

  // ------- Data fetch -------
  async function loadCountryData(cc) {
    const [p, f] = await Promise.all([
      DataLoader.pyramid(cc),
      DataLoader.projection(cc),
    ]);
    state.pyramid = p;
    state.projection = f;
  }

  // ------- Pyramid building -------
  function isProjection(year) { return year >= PROJECTION_RANGE[0]; }

  function getYearData(year, domain) {
    if (!isProjection(year)) {
      const d = state.pyramid && state.pyramid.data[domain];
      return d ? d[String(year)] : null;
    }
    const fc = state.projection && state.projection.data[domain];
    if (!fc) return null;
    const baseKey = 'base_2023';
    const yd = fc[baseKey] && fc[baseKey][String(year)];
    return yd || null;
  }

  function totalsFor(year, domain) {
    const d = getYearData(year, domain);
    if (!d) return { f: 0, m: 0 };
    return { f: d.total_female || 0, m: d.total_male || 0 };
  }

  // ------- Plotly render -------
  // Two-subplot layout mirroring the paper's plot_productive_people_pyramid:
  //   - Left subplot: female counts on a log x-axis, x-range reversed so
  //     small values sit at the center divide.
  //   - Right subplot: male counts on a log x-axis, normal orientation.
  //   - Shared y-axis (cumulative productivity) with ticks placed in the
  //     central gap (left subplot's right side) and a "Cumulative
  //     productivity" axis title in the same central column.
  function renderPyramid() {
    const year = state.year;
    const domain = state.domain;
    const yd = getYearData(year, domain);
    const totals = totalsFor(year, domain);
    const projection = isProjection(year);
    const approx = projection ? '≈' : '=';
    const fmt = (n) => Math.round(n).toLocaleString();
    const logMax = state.logXMax;
    const xMax = Math.pow(10, logMax);
    // Auto dtick for the y-axis so the number of ticks stays roughly stable
    // as the y-max changes (e.g., yMax=100 → 10, yMax=500 → 50).
    const yDtick = Math.max(10, Math.round(state.yMax / 10));

    function buildTrace(g, color, xref, yref, hoverLabel) {
      // Paper convention is count=0 → 0.1 to make the bar invisible on a log
      // axis. With true per-side log axes (Plotly type='log'), we instead
      // omit bins whose count is < 1: such bars cannot be drawn on a log
      // axis anyway, and omitting avoids spurious crossings of the centerline.
      const xs = [];
      const ys = [];
      const customdata = [];
      const hist = yd ? (yd[g] || {}) : {};
      for (let n = 1; n <= state.yMax; n++) {
        const v = hist[String(n)];
        if (v && v >= 1) {
          xs.push(v);
          ys.push(n);
          customdata.push(Math.round(v));
        }
      }
      return {
        type: 'bar',
        orientation: 'h',
        x: xs,
        y: ys,
        customdata,
        marker: { color, line: { width: 0 } },
        hovertemplate: `${hoverLabel}<br>Cumulative productivity: %{y}<br>Authors: %{customdata:,}<extra></extra>`,
        xaxis: xref,
        yaxis: yref,
        showlegend: false,
      };
    }

    const traceF = buildTrace('female', COLOR_FEMALE, 'x',  'y',  'Female');
    const traceM = buildTrace('male',   COLOR_MALE,   'x2', 'y2', 'Male');

    // Axes use only the primary line (no outer box), with outward ticks
    // and bold tick labels. X axes show 10^k at every decade.
    const tickFont = { size: 12, family: FONT_FAMILY, weight: 700 };
    const xAxisCommon = {
      type: 'log',
      ticks: 'outside',
      tickwidth: TICK_WIDTH,
      ticklen: TICK_LEN,
      tickcolor: '#000',
      tickfont: tickFont,
      linewidth: AXIS_LINE_WIDTH,
      linecolor: '#000',
      showgrid: false,
      mirror: false,
      dtick: 1,                                  // every decade
      exponentformat: 'power',                   // render as 10^k
    };
    const yAxisCommon = {
      range: [0, state.yMax + 5],               // small top padding so the
                                                  // top bar is not crowded by
                                                  // the Female/Male title
      dtick: yDtick,
      ticks: 'outside',
      tickwidth: TICK_WIDTH,
      ticklen: TICK_LEN,
      tickcolor: '#000',
      tickfont: tickFont,
      linewidth: AXIS_LINE_WIDTH,
      linecolor: '#000',
      showgrid: false,
      mirror: false,
      fixedrange: true,
    };

    const layout = {
      font: { family: FONT_FAMILY },
      bargap: 0.5,                                // paper barh height=0.5
      margin: { l: 40, r: 40, t: 50, b: 60 },
      xaxis: Object.assign({}, xAxisCommon, {
        domain: [0, 0.44],
        range: [logMax, 0],                       // reversed: max_n on left
        anchor: 'y',
      }),
      xaxis2: Object.assign({}, xAxisCommon, {
        domain: [0.56, 1],
        range: [0, logMax],
        anchor: 'y2',
      }),
      yaxis: Object.assign({}, yAxisCommon, {
        domain: [0, 1],
        side: 'right',                            // labels in central column
        anchor: 'x',
      }),
      yaxis2: Object.assign({}, yAxisCommon, {
        domain: [0, 1],
        showticklabels: false,                    // paper: set_yticklabels([])
        anchor: 'x2',
      }),
      plot_bgcolor: projection ? '#f7f7fc' : 'white',
      paper_bgcolor: 'white',
      annotations: [
        // Subplot titles (paper: ax[0/1].set_title)
        {
          x: 0.23, y: 1.06, xref: 'paper', yref: 'paper',
          text: `<b>Female (N ${approx} ${fmt(totals.f)})</b>`,
          showarrow: false, font: { size: 17, family: FONT_FAMILY },
        },
        {
          x: 0.77, y: 1.06, xref: 'paper', yref: 'paper',
          text: `<b>Male (N ${approx} ${fmt(totals.m)})</b>`,
          showarrow: false, font: { size: 17, family: FONT_FAMILY },
        },
        // Central "Cumulative productivity" label
        // (paper: fig.text(0.51, 0.5, ..., rotation='vertical')).
        {
          x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
          text: '<b>Cumulative productivity</b>',
          textangle: -90, showarrow: false,
          font: { size: 15, family: FONT_FAMILY },
        },
        // Shared x-axis label
        {
          x: 0.5, y: -0.13, xref: 'paper', yref: 'paper',
          text: '<b>Number of authors (log scale)</b>',
          showarrow: false, font: { size: 14, family: FONT_FAMILY },
        },
      ],
      hoverlabel: { font: { size: 13, family: FONT_FAMILY } },
    };

    const config = {
      displayModeBar: false,
      responsive: true,
    };
    Plotly.react('pyramid', [traceF, traceM], layout, config);
  }

  // ------- Year label -------
  function updateYearLabel() {
    const label = document.getElementById('year-label');
    const projection = isProjection(state.year);
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
      if (y > MAX_YEAR) y = MIN_YEAR;
      state.year = y;
      document.getElementById('year-slider').value = y;
      updateYearLabel();
      renderPyramid();
      writeStateToURL();
    }, SPEED_TO_MS[state.speed] || 400);
  }

  function stop() {
    state.playing = false;
    document.getElementById('btn-play').classList.remove('is-active');
    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = null;
  }

  // ------- Event wiring -------
  function wireEvents() {
    document.getElementById('country-select').addEventListener('change', async (e) => {
      stop();
      state.country = e.target.value.toUpperCase();
      await loadCountryData(state.country);
      renderPyramid();
      writeStateToURL();
    });
    document.getElementById('domain-select').addEventListener('change', (e) => {
      state.domain = e.target.value;
      renderPyramid();
      writeStateToURL();
    });
    document.getElementById('ymax-select').addEventListener('change', (e) => {
      state.yMax = parseInt(e.target.value, 10);
      renderPyramid();
    });
    document.getElementById('xmax-select').addEventListener('change', (e) => {
      state.logXMax = parseInt(e.target.value, 10);
      renderPyramid();
    });
    document.getElementById('year-slider').addEventListener('input', (e) => {
      stop();
      state.year = parseInt(e.target.value, 10);
      updateYearLabel();
      renderPyramid();
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
      renderPyramid();
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
    document.getElementById('btn-download').addEventListener('click', () => {
      downloadCurrentCSV();
    });
  }

  // CSV covers every cumulative-productivity bin from 1 to CSV_MAX_BIN
  // regardless of the on-screen axis settings, so the file structure is
  // consistent across countries and years.
  const CSV_MAX_BIN = 500;

  function downloadCurrentCSV() {
    const yd = getYearData(state.year, state.domain);
    if (!yd) return;
    const fHist = yd.female || {};
    const mHist = yd.male   || {};

    const rows = [['cumulative_productivity', 'female', 'male']];
    for (let n = 1; n <= CSV_MAX_BIN; n++) {
      rows.push([n, fHist[String(n)] || 0, mHist[String(n)] || 0]);
    }
    rows.push(['# country', state.country]);
    rows.push(['# domain', state.domain]);
    rows.push(['# year', state.year]);
    rows.push(['# total_female', yd.total_female || 0]);
    rows.push(['# total_male',   yd.total_male   || 0]);
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `pyramid_${state.country}_${state.domain}_${state.year}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ------- Init -------
  async function init() {
    readStateFromURL();
    state.countries = await DataLoader.countries();
    buildCountrySelector(state.countries);
    buildDomainSelector();
    syncAxisSelectors();
    buildYearSlider();
    await loadCountryData(state.country);
    updateYearLabel();
    renderPyramid();
    wireEvents();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
