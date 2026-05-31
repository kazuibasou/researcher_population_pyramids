/* Researcher Population Pyramids — Trends page logic */

(function () {
  const DATA_YEAR_MIN = 1970;
  const DATA_YEAR_MAX = 2050;
  const HISTORICAL_END = 2023;
  const DEFAULT_YEAR_MIN = 2000;
  const DEFAULT_YEAR_MAX = 2050;
  // Default selection: paper's reference set + the Arab set (fig 1a–d).
  const DEFAULT_COUNTRIES = [
    'AU', 'CA', 'DE', 'ES', 'FR', 'GB', 'IT', 'JP', 'SE', 'US',
  ];
  const FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", '
                    + 'Roboto, "Helvetica Neue", Arial, sans-serif';
  const AXIS_LINE_WIDTH = 2.0;
  const TICK_WIDTH = 2.0;
  const TICK_LEN = 8;

  // Cycle of distinguishable colors for multi-country lines. 12 colors based
  // on the d3 / tableau palettes, picked to be readable at line widths of 2.
  const COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78',
  ];

  const METRIC_INFO = {
    women_ratio: {
      label: 'Proportion of female active authors',
      compute: (yd) => {
        const f = yd[0], m = yd[1];
        const t = f + m;
        return t > 0 ? f / t : null;
      },
      isRatio: true,
    },
    total: {
      label: 'Number of active authors',
      compute: (yd) => yd[0] + yd[1],
      isRatio: false,
    },
    women: {
      label: 'Number of female active authors',
      compute: (yd) => yd[0],
      isRatio: false,
    },
    men: {
      label: 'Number of male active authors',
      compute: (yd) => yd[1],
      isRatio: false,
    },
  };

  const state = {
    countries: null,
    timeline: null,
    selected: DEFAULT_COUNTRIES.slice(),
    metric: 'women_ratio',
    yScale: 'log',
    yearMin: DEFAULT_YEAR_MIN,
    yearMax: DEFAULT_YEAR_MAX,
    countryChoices: null,   // Choices.js instance
  };

  // ------- URL state ------
  function readStateFromURL() {
    const params = new URLSearchParams(window.location.search);
    const cs = params.get('countries');
    const m = params.get('metric');
    const y = params.get('yscale');
    const ymin = parseInt(params.get('ymin'), 10);
    const ymax = parseInt(params.get('ymax'), 10);
    if (cs) {
      const parsed = cs.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
      if (parsed.length) state.selected = parsed;
    }
    if (m && METRIC_INFO[m]) state.metric = m;
    if (y === 'log' || y === 'linear') state.yScale = y;
    if (!isNaN(ymin) && ymin >= DATA_YEAR_MIN && ymin <= DATA_YEAR_MAX) state.yearMin = ymin;
    if (!isNaN(ymax) && ymax >= DATA_YEAR_MIN && ymax <= DATA_YEAR_MAX) state.yearMax = ymax;
    if (state.yearMin > state.yearMax) {
      [state.yearMin, state.yearMax] = [state.yearMax, state.yearMin];
    }
  }

  function writeStateToURL() {
    const params = new URLSearchParams();
    params.set('countries', state.selected.join(','));
    params.set('metric', state.metric);
    params.set('yscale', state.yScale);
    params.set('ymin', state.yearMin);
    params.set('ymax', state.yearMax);
    window.history.replaceState({}, '', window.location.pathname + '?' + params.toString());
  }

  // ------- Selector builders -------
  function buildCountrySelector(countries) {
    const select = document.getElementById('countries-select');
    select.innerHTML = '';
    const byRegion = {};
    for (const c of countries) (byRegion[c.region] = byRegion[c.region] || []).push(c);
    const regions = Object.keys(byRegion).sort();
    for (const region of regions) {
      const group = document.createElement('optgroup');
      group.label = region;
      for (const c of byRegion[region]) {
        const opt = document.createElement('option');
        opt.value = c.code;
        opt.textContent = `${c.name} (${c.code})`;
        if (state.selected.includes(c.code)) opt.selected = true;
        group.appendChild(opt);
      }
      select.appendChild(group);
    }
    if (window.Choices) {
      if (state.countryChoices) state.countryChoices.destroy();
      state.countryChoices = new Choices(select, {
        searchEnabled: true,
        searchFields: ['label', 'value'],
        removeItemButton: true,
        itemSelectText: '',
        shouldSort: false,
        searchResultLimit: 20,
        placeholderValue: 'Add a country…',
        placeholder: true,
      });
    }
  }

  function buildMetricSelector() {
    const select = document.getElementById('metric-select');
    select.value = state.metric;
  }

  function buildYScaleSelector() {
    const select = document.getElementById('yscale-select');
    select.value = state.yScale;
    select.disabled = METRIC_INFO[state.metric].isRatio;
  }

  function buildYearInputs() {
    document.getElementById('year-min').value = state.yearMin;
    document.getElementById('year-max').value = state.yearMax;
    updateYearRangeText();
  }

  function updateYearRangeText() {
    const el = document.getElementById('year-range-text');
    if (!el) return;
    const lo = Math.min(state.yearMin, state.yearMax);
    const hi = Math.max(state.yearMin, state.yearMax);
    el.textContent = lo === hi ? `${lo}` : `${lo} and ${hi}`;
  }

  function clampYear(v) {
    if (isNaN(v)) return null;
    return Math.max(DATA_YEAR_MIN, Math.min(DATA_YEAR_MAX, Math.round(v)));
  }

  // ------- Plot -------
  function colorFor(idx) { return COLOR_PALETTE[idx % COLOR_PALETTE.length]; }

  function chooseYearDtick(span) {
    if (span <= 10) return 1;
    if (span <= 30) return 5;
    return 10;
  }

  function renderPlot() {
    const metric = METRIC_INFO[state.metric];
    const isRatio = metric.isRatio;
    const yScale = isRatio ? 'linear' : state.yScale;

    const traces = [];
    const codeToName = {};
    if (state.countries) {
      for (const c of state.countries) codeToName[c.code] = c.name;
    }

    let i = 0;
    let maxValue = 0;
    const yMinYr = Math.min(state.yearMin, state.yearMax);
    const yMaxYr = Math.max(state.yearMin, state.yearMax);
    for (const cc of state.selected) {
      const series = state.timeline[cc];
      if (!series) continue;
      const x = [];
      const y = [];
      for (let yr = yMinYr; yr <= yMaxYr; yr++) {
        const yd = series[String(yr)];
        if (!yd) continue;
        const v = metric.compute(yd);
        if (v == null) continue;
        x.push(yr);
        y.push(v);
        if (v > maxValue) maxValue = v;
      }
      traces.push({
        type: 'scatter',
        mode: 'lines+markers',
        x: x,
        y: y,
        name: `${codeToName[cc] || cc} (${cc})`,
        line: { color: colorFor(i), width: 2 },
        marker: { size: 6 },
        hovertemplate: `<b>${codeToName[cc] || cc} (${cc})</b><br>%{x}: %{y${isRatio ? ':.1%' : ':,'}}<extra></extra>`,
      });
      i++;
    }

    const tickFont = { size: 12, family: FONT_FAMILY, weight: 700 };
    const xAxis = {
      title: { text: '<b>Publication year</b>', font: { size: 14, family: FONT_FAMILY } },
      ticks: 'outside',
      tickwidth: TICK_WIDTH,
      ticklen: TICK_LEN,
      tickcolor: '#000',
      tickfont: tickFont,
      linewidth: AXIS_LINE_WIDTH,
      linecolor: '#000',
      showgrid: false,
      mirror: false,
      dtick: chooseYearDtick(yMaxYr - yMinYr),
      range: [yMinYr - 0.5, yMaxYr + 0.5],
    };
    const yAxis = {
      title: { text: `<b>${metric.label}</b>`, font: { size: 14, family: FONT_FAMILY } },
      ticks: 'outside',
      tickwidth: TICK_WIDTH,
      ticklen: TICK_LEN,
      tickcolor: '#000',
      tickfont: tickFont,
      linewidth: AXIS_LINE_WIDTH,
      linecolor: '#000',
      showgrid: false,
      mirror: false,
    };
    if (isRatio) {
      yAxis.tickformat = '.0%';
      yAxis.range = [0, Math.min(1, Math.max(0.05, maxValue * 1.1))];
      yAxis.dtick = 0.1;
    } else if (yScale === 'log') {
      yAxis.type = 'log';
      yAxis.exponentformat = 'power';
      yAxis.dtick = 1;
    }

    const layout = {
      font: { family: FONT_FAMILY },
      margin: { l: 70, r: 30, t: 30, b: 60 },
      xaxis: xAxis,
      yaxis: yAxis,
      showlegend: true,
      legend: {
        orientation: 'v',
        x: 1.02,
        y: 1,
        xanchor: 'left',
        yanchor: 'top',
        font: { size: 12, family: FONT_FAMILY },
        bgcolor: 'rgba(255,255,255,0)',
      },
      hoverlabel: { font: { size: 13, family: FONT_FAMILY } },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      shapes: [],
    };

    // Shade the projection portion (years > HISTORICAL_END) of the plot.
    if (yMaxYr > HISTORICAL_END) {
      const x0 = Math.max(yMinYr, HISTORICAL_END) + 0.5;
      const x1 = yMaxYr + 0.5;
      if (x0 < x1) {
        layout.shapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: x0, x1: x1,
          y0: 0, y1: 1,
          fillcolor: '#f7f7fc',
          line: { width: 0 },
          layer: 'below',
        });
      }
    }

    const config = { displayModeBar: false, responsive: true };
    Plotly.react('trends-plot', traces, layout, config);
  }

  // ------- Events -------
  function wireEvents() {
    document.getElementById('countries-select').addEventListener('change', () => {
      const choices = state.countryChoices;
      const values = choices
        ? choices.getValue().map(v => v.value)
        : Array.from(document.getElementById('countries-select').selectedOptions).map(o => o.value);
      state.selected = values.map(v => v.toUpperCase());
      renderPlot();
      writeStateToURL();
    });
    document.getElementById('metric-select').addEventListener('change', (e) => {
      state.metric = e.target.value;
      buildYScaleSelector();
      renderPlot();
      writeStateToURL();
    });
    document.getElementById('yscale-select').addEventListener('change', (e) => {
      state.yScale = e.target.value;
      renderPlot();
      writeStateToURL();
    });
    document.getElementById('year-min').addEventListener('change', (e) => {
      const v = clampYear(parseInt(e.target.value, 10));
      if (v == null) { e.target.value = state.yearMin; return; }
      state.yearMin = v;
      if (state.yearMin > state.yearMax) state.yearMax = state.yearMin;
      buildYearInputs();
      renderPlot();
      writeStateToURL();
    });
    document.getElementById('year-max').addEventListener('change', (e) => {
      const v = clampYear(parseInt(e.target.value, 10));
      if (v == null) { e.target.value = state.yearMax; return; }
      state.yearMax = v;
      if (state.yearMax < state.yearMin) state.yearMin = state.yearMax;
      buildYearInputs();
      renderPlot();
      writeStateToURL();
    });
    document.getElementById('btn-download').addEventListener('click', downloadCSV);
  }

  // CSV: per-country, per-year raw counts of female and male authors for the
  // currently selected countries. Years span the full data range (1950–2023)
  // regardless of the on-screen 2000–2023 window so the file is complete.
  function downloadCSV() {
    const codeToName = {};
    if (state.countries) for (const c of state.countries) codeToName[c.code] = c.name;

    // Determine year range from the data itself.
    let minYear = Infinity, maxYear = -Infinity;
    for (const cc of state.selected) {
      const s = state.timeline[cc];
      if (!s) continue;
      for (const yr of Object.keys(s)) {
        const n = parseInt(yr, 10);
        if (n < minYear) minYear = n;
        if (n > maxYear) maxYear = n;
      }
    }
    if (!isFinite(minYear)) return;

    const rows = [['year', 'country_code', 'country_name', 'female_authors', 'male_authors']];
    for (const cc of state.selected) {
      const s = state.timeline[cc];
      if (!s) continue;
      for (let yr = minYear; yr <= maxYear; yr++) {
        const yd = s[String(yr)];
        if (!yd) continue;
        rows.push([yr, cc, csvEscape(codeToName[cc] || cc), yd[0], yd[1]]);
      }
    }
    rows.push(['# metric_shown', METRIC_INFO[state.metric].label]);
    rows.push(['# countries_shown', state.selected.join(';')]);

    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `trends_${state.selected.join('-')}.csv`;
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
    const [countries, timeline] = await Promise.all([
      DataLoader.countries(),
      DataLoader.timeline(),
    ]);
    state.countries = countries;
    state.timeline = timeline;
    buildCountrySelector(countries);
    buildMetricSelector();
    buildYScaleSelector();
    buildYearInputs();
    renderPlot();
    wireEvents();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
