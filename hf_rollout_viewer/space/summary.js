'use strict';
// Summary and Analysis tabs: aggregate statistics over every rollout, read from index/summary.json
// (built by build_summary.py) and drawn as inline SVG. Every chart has a table view and every mark
// a hover/focus tooltip. Mark colors come from the validated data-viz palette.
window.RolloutSummary = (() => {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const C = {blue: '#2a78d6', orange: '#eb6834', aqua: '#1baf7a', gray: '#c3c2b7', ink: '#1d2529', muted: '#5d6b72'};
  const RAMP3 = ['#86b6ef', '#3987e5', '#184f95'];
  const RAMP4 = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b'];
  const FAMILIES = [
    {key: 'gvr', label: 'GVR family', color: C.blue},
    {key: 'value_tool', label: 'Value tool', color: C.orange},
    {key: 'other', label: 'Direct, plan-work-review', color: C.aqua},
  ];
  const END_STATES = [
    {key: 'accepted', label: 'Verifier accepted', color: C.blue, statuses: ['accepted']},
    {key: 'cycle_limit', label: 'Hit the candidate limit', color: C.orange, statuses: ['cycle_limit']},
    {key: 'completed', label: 'Solver returned an answer', color: C.aqua, statuses: ['completed']},
    {key: 'error', label: 'Error or exhausted budget', color: C.gray, statuses: ['protocol_error', 'budget_exhausted', 'context_exhausted', 'failed']},
  ];
  const BUCKETS = [
    {key: 'generation', label: 'Generation', color: C.blue},
    {key: 'verification', label: 'Verification', color: C.orange},
    {key: 'subagents', label: 'Subagents', color: C.aqua},
  ];
  const CALLS = [
    {key: '1', label: '1 verifier call', color: RAMP3[0]},
    {key: '2', label: '2 calls', color: RAMP3[1]},
    {key: '3', label: '3 calls', color: RAMP3[2]},
    {key: '0', label: 'No verdict (error)', color: C.gray},
  ];
  const QUERIES = [0, 1, 2, 3].map(k => ({key: String(k), label: k === 1 ? '1 query' : `${k} queries`, color: RAMP4[k]}));
  const STRATA = [
    {key: 'never', label: 'Direct never solves'},
    {key: 'sometimes', label: 'Direct sometimes solves'},
    {key: 'always', label: 'Direct always solves'},
  ];
  const FIRST = [
    {key: 'fixed', label: 'Fixed by revision', sub: 'sent back, final 7/7'},
    {key: 'right_first', label: 'Right first try', sub: 'accepted as is, 7/7'},
    {key: 'revised_wrong', label: 'Revised, still wrong', sub: 'sent back, final < 7'},
    {key: 'kept_wrong', label: 'Kept wrong', sub: 'accepted as is, < 7'},
    {key: 'no_output', label: 'No judged output', sub: 'error or budget'},
  ];
  const CAL_COLORS = [C.blue, C.orange, C.aqua];
  const view = {data: null, meta: null, page: 'summary', exp: null, scope: 'all', sort: 'rank', root: null, cards: [], onChange: null};

  const node = (tag, cls, text) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  };
  const svg = (tag, attrs = {}, text) => {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (text !== undefined) el.textContent = String(text);
    return el;
  };
  const pct = (x, digits = 1) => (x === null || x === undefined || !Number.isFinite(x) ? '–' : `${(100 * x).toFixed(digits)}%`);
  const pctTick = t => `${+(100 * t).toFixed(1)}%`;
  const ci = pair => (pair ? `${pct(pair[0])}–${pct(pair[1])}` : '–');
  const int = n => (n === null || n === undefined ? '–' : Math.round(n).toLocaleString('en-US'));
  const kfmt = n => (n >= 1000 ? `${+(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : String(Math.round(n)));
  const rateOf = pair => (pair && pair[0] ? pair[1] / pair[0] : null);
  const signedPts = x => `${x >= 0 ? '+' : '−'}${Math.abs(100 * x).toFixed(1)} pts`;
  const sum = values => values.reduce((a, b) => a + b, 0);
  const fixed = (x, digits) => (x === null || x === undefined ? '–' : x.toFixed(digits));
  const ciFixed = (pair, digits) => (pair ? `${pair[0].toFixed(digits)}–${pair[1].toFixed(digits)}` : '–');
  const wilson = (wins, n) => {
    const z = 1.96, p = wins / n, d = 1 + z * z / n;
    const center = (p + z * z / (2 * n)) / d, half = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.max(0, center - half), Math.min(1, center + half)];
  };

  const measure = document.createElement('canvas').getContext('2d');
  const textWidth = (text, size = 12, weight = 400) => {
    measure.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    return measure.measureText(text).width;
  };
  const luminance = hex => {
    const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map(v => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const contrast = (a, b) => {
    const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  };
  // Text set inside a filled segment takes white or ink, whichever contrasts more with the fill.
  const inkOn = fill => (contrast(fill, '#ffffff') >= contrast(fill, C.ink) ? '#ffffff' : C.ink);

  // One shared tooltip. Labels are data, so they only ever become text nodes.
  let tipEl = null;
  function tip() {
    if (!tipEl) {
      tipEl = node('div', 'viz-tip');
      tipEl.setAttribute('role', 'tooltip');
      tipEl.hidden = true;
      document.body.append(tipEl);
    }
    return tipEl;
  }
  function showTip(content, x, y) {
    const el = tip();
    const head = node('div', 'viz-tip-head');
    if (content.color) {
      const key = node('span', 'viz-tip-key');
      key.style.background = content.color;
      head.append(key);
    }
    head.append(node('strong', '', content.value), node('span', 'viz-tip-label', content.label));
    el.replaceChildren(head, ...(content.lines || []).map(line => node('div', 'viz-tip-line', line)));
    el.hidden = false;
    moveTip(x, y);
  }
  function moveTip(x, y) {
    const el = tip();
    const {width, height} = el.getBoundingClientRect();
    el.style.left = `${Math.max(8, Math.min(window.innerWidth - width - 8, x + 14))}px`;
    el.style.top = `${Math.max(8, y + height + 24 > window.innerHeight ? y - height - 12 : y + 16)}px`;
  }
  function hideTip() { if (tipEl) tipEl.hidden = true; }
  // focus() fires before the browser scrolls the element into view, so a focus tooltip is placed on
  // the next frame and then follows its element while the page scrolls; pointer tooltips just hide.
  let tipAnchor = null;
  const anchorTip = content => {
    const r = tipAnchor.getBoundingClientRect();
    if (content) showTip(content, r.left + Math.min(r.width / 2, 160), r.top + r.height / 2);
    else moveTip(r.left + Math.min(r.width / 2, 160), r.top + r.height / 2);
  };
  window.addEventListener('scroll', () => {
    if (!tipEl || tipEl.hidden) return;
    if (tipAnchor && tipAnchor.isConnected) anchorTip(null);
    else hideTip();
  }, {passive: true});
  function bindTip(el, content) {
    el.setAttribute('tabindex', '0');
    el.setAttribute('aria-label', [content.label, content.value, ...(content.lines || [])].join('. '));
    el.addEventListener('pointerenter', event => showTip(content, event.clientX, event.clientY));
    el.addEventListener('pointermove', event => moveTip(event.clientX, event.clientY));
    el.addEventListener('pointerleave', hideTip);
    el.addEventListener('focus', () => requestAnimationFrame(() => {
      if (document.activeElement !== el) return;
      tipAnchor = el;
      anchorTip(content);
    }));
    el.addEventListener('blur', () => { tipAnchor = null; hideTip(); });
  }

  function niceStep(span, count) {
    const raw = span / count, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * mag;
  }
  function ticksBetween(min, max, step) {
    const ticks = [];
    for (let i = 0; min + i * step <= max + step / 1000; i++) ticks.push(+(min + i * step).toPrecision(12));
    return ticks;
  }
  // Zero-based scale for lengths (bars).
  function lengthScale(max, count = 5) {
    const step = niceStep(max > 0 ? max : 1, count);
    const top = Math.ceil((max > 0 ? max : 1) / step - 1e-9) * step;
    return {min: 0, max: top, ticks: ticksBetween(0, top, step)};
  }
  // Padded scale for positions (scatter), clamped at zero.
  function positionScale(lo, hi, count = 5) {
    const pad = Math.max((hi - lo) * 0.12, Math.abs(hi) * 0.02, 1e-6);
    const step = niceStep(hi - lo + 2 * pad, count);
    const min = Math.max(0, Math.floor((lo - pad) / step) * step), max = Math.ceil((hi + pad) / step) * step;
    return {min, max, ticks: ticksBetween(min, max, step)};
  }
  // Bar with a 4px rounded data end, square at the baseline.
  function barPath(x0, x1, y, h, r = 4) {
    if (x1 - x0 <= 0.5) return '';
    const rr = Math.min(r, x1 - x0, h / 2);
    return `M${x0},${y}H${x1 - rr}A${rr},${rr} 0 0 1 ${x1},${y + rr}V${y + h - rr}A${rr},${rr} 0 0 1 ${x1 - rr},${y + h}H${x0}Z`;
  }
  const labelWidth = (rows, width) => Math.min(Math.max(200, width * 0.34), Math.max(80, ...rows.map(r => textWidth(r.label))) + 16);
  // Middle ellipsis: method names differ at both ends ("GVR · … · gold").
  function fit(text, max, size = 12, weight = 400) {
    if (textWidth(text, size, weight) <= max) return text;
    let keep = text.length;
    while (keep > 4 && textWidth(`${text.slice(0, Math.ceil(keep / 2))}…${text.slice(text.length - Math.floor(keep / 2))}`, size, weight) > max) keep--;
    return `${text.slice(0, Math.ceil(keep / 2)).trimEnd()}…${text.slice(text.length - Math.floor(keep / 2)).trimStart()}`;
  }
  function rowLabel(label, x, y, max) {
    const shown = fit(label, max);
    const t = svg('text', {x, y, class: 'viz-label', 'text-anchor': 'end', 'dominant-baseline': 'central'}, shown);
    if (shown !== label) t.append(svg('title', {}, label));
    return t;
  }

  // Horizontal single-series bars, one row per method, in one or more side-by-side panels that
  // share the row labels but keep their own x-scale. A panel: {title, get(row) -> {value, lo, hi},
  // tip(row), ref: {value, label}}.
  function barPanels(rows, panels, width) {
    const band = 26, barH = 14, gap = 28, right = 58, bottom = 26;
    const labelW = labelWidth(rows, width);
    const top = panels.some(p => p.title) ? 42 : panels.some(p => p.ref) ? 24 : 6;
    const panelW = Math.max(150, (width - labelW - panels.length * right - (panels.length - 1) * gap) / panels.length);
    const plotBottom = top + rows.length * band;
    const root = svg('svg', {width: labelW + panels.length * (panelW + right) + (panels.length - 1) * gap, height: plotBottom + bottom, class: 'viz-svg'});
    rows.forEach((row, i) => root.append(rowLabel(row.label, labelW - 10, top + i * band + band / 2, labelW - 14)));
    panels.forEach((panel, p) => {
      const x0 = labelW + p * (panelW + right + gap);
      const values = rows.map(row => panel.get(row));
      const scale = lengthScale(Math.max(...values.map(v => v.hi ?? v.value ?? 0), panel.ref?.value ?? 0), panelW < 260 ? 4 : 5);
      const x = v => x0 + (Math.max(0, v) / scale.max) * panelW;
      if (panel.title) {
        const room = panelW + right - 8;
        const title = textWidth(panel.title, 12, 600) <= room ? panel.title : fit(panel.short || panel.title, room, 12, 600);
        root.append(svg('text', {x: x0, y: 14, class: 'viz-panel-title'}, title));
      }
      for (const t of scale.ticks) {
        root.append(svg('line', {x1: x(t), x2: x(t), y1: top, y2: plotBottom, class: t === 0 ? 'viz-axis' : 'viz-grid'}));
        root.append(svg('text', {x: x(t), y: plotBottom + 16, class: 'viz-tick', 'text-anchor': 'middle'}, pctTick(t)));
      }
      rows.forEach((row, i) => {
        const v = values[i], y = top + i * band, cy = y + band / 2;
        const g = svg('g', {class: 'viz-row'});
        g.append(svg('rect', {x: x0 - 6, y, width: panelW + right, height: band, class: 'viz-hit'}));
        if (v.value !== null && v.value !== undefined) {
          g.append(svg('path', {d: barPath(x(0), x(v.value), cy - barH / 2, barH), fill: panel.color || C.blue, class: 'viz-mark'}));
          if (v.lo !== undefined && v.lo !== null) {
            g.append(svg('line', {x1: x(v.lo), x2: x(v.hi), y1: cy, y2: cy, class: 'viz-whisker'}));
            for (const end of [v.lo, v.hi]) g.append(svg('line', {x1: x(end), x2: x(end), y1: cy - 4, y2: cy + 4, class: 'viz-whisker'}));
          }
          g.append(svg('text', {x: Math.max(x(v.value), v.hi != null ? x(v.hi) : 0) + 6, y: cy, class: 'viz-value', 'dominant-baseline': 'central'}, pct(v.value)));
        }
        bindTip(g, panel.tip(row));
        root.append(g);
      });
      if (panel.ref) {
        const rx = x(panel.ref.value), label = `${panel.ref.label} ${pct(panel.ref.value)}`;
        const flip = rx + 4 + textWidth(label, 11) > x0 + panelW + right;
        root.append(svg('line', {x1: rx, x2: rx, y1: top - 8, y2: plotBottom, class: 'viz-ref'}));
        root.append(svg('text', {x: flip ? rx - 4 : rx + 4, y: top - 10, class: 'viz-ref-label', 'text-anchor': flip ? 'end' : 'start'}, label));
      }
    });
    return root;
  }

  // Horizontal stacked bars with a 2px surface gap between segments. 'share' normalizes each row to
  // 100%; 'abs' puts all rows on one scale and labels each total. rows: [{label, values, tip(series,
  // value, total)}].
  function stackedBars(rows, series, width, {mode = 'share'} = {}) {
    const band = 28, barH = 18, top = 4, bottom = 26, right = mode === 'abs' ? 56 : 24;
    const labelW = labelWidth(rows, width);
    const plotW = Math.max(220, width - labelW - right);
    const totals = rows.map(row => sum(series.map(s => row.values[s.key] || 0)));
    const scale = mode === 'share' ? {max: 1, ticks: [0, 0.25, 0.5, 0.75, 1]} : lengthScale(Math.max(...totals));
    const x = v => labelW + (v / scale.max) * plotW;
    const plotBottom = top + rows.length * band;
    const root = svg('svg', {width: labelW + plotW + right, height: plotBottom + bottom, class: 'viz-svg'});
    for (const t of scale.ticks) {
      root.append(svg('line', {x1: x(t), x2: x(t), y1: top, y2: plotBottom, class: t === 0 ? 'viz-axis' : 'viz-grid'}));
      root.append(svg('text', {x: x(t), y: plotBottom + 16, class: 'viz-tick', 'text-anchor': 'middle'}, mode === 'share' ? pctTick(t) : kfmt(t)));
    }
    rows.forEach((row, i) => {
      const y = top + i * band, by = y + (band - barH) / 2, total = totals[i];
      root.append(rowLabel(row.label, labelW - 10, y + band / 2, labelW - 14));
      const parts = series.map(s => ({s, v: row.values[s.key] || 0})).filter(part => part.v > 0);
      let acc = 0;
      parts.forEach((part, j) => {
        const v = mode === 'share' ? part.v / total : part.v;
        const a = x(acc), b = x(acc + v);
        acc += v;
        const a2 = a + (j > 0 ? 1 : 0), b2 = b - (j < parts.length - 1 ? 1 : 0);
        const g = svg('g', {class: 'viz-seg'});
        g.append(svg('rect', {x: a, y, width: Math.max(0, b - a), height: band, class: 'viz-hit'}));
        const d = j === parts.length - 1 ? barPath(a2, b2, by, barH) : b2 > a2 ? `M${a2},${by}H${b2}V${by + barH}H${a2}Z` : '';
        if (d) g.append(svg('path', {d, fill: part.s.color, class: 'viz-mark'}));
        const text = mode === 'share' ? pct(v, 0) : kfmt(part.v);
        if (textWidth(text, 11) + 10 <= b2 - a2) {
          g.append(svg('text', {x: (a2 + b2) / 2, y: by + barH / 2, class: 'viz-inlabel', style: `fill:${inkOn(part.s.color)}`, 'text-anchor': 'middle', 'dominant-baseline': 'central'}, text));
        }
        bindTip(g, row.tip(part.s, part.v, total));
        root.append(g);
      });
      if (mode === 'abs') root.append(svg('text', {x: x(total) + 6, y: by + barH / 2, class: 'viz-value', 'dominant-baseline': 'central'}, kfmt(total)));
    });
    return root;
  }

  // Scatter of methods: x = mean tokens, y = 7/7 rate. Dots carry a 2px surface ring and a 24px hit
  // area; a hollow reference dot marks an out-of-experiment baseline.
  function scatter(points, width, {ref = null, labelIds = []} = {}) {
    const left = 58, right = 24, top = 24, bottom = 44, height = 340;
    const plotW = Math.max(240, width - left - right), plotH = height - top - bottom;
    const all = ref ? [...points, ref] : points;
    const xs = positionScale(Math.min(...all.map(p => p.x)), Math.max(...all.map(p => p.x)));
    const ys = positionScale(Math.min(...all.map(p => p.y)), Math.max(...all.map(p => p.y)));
    const x = v => left + ((v - xs.min) / (xs.max - xs.min)) * plotW;
    const y = v => top + plotH - ((v - ys.min) / (ys.max - ys.min)) * plotH;
    const root = svg('svg', {width: left + plotW + right, height, class: 'viz-svg'});
    for (const t of xs.ticks) {
      root.append(svg('line', {x1: x(t), x2: x(t), y1: top, y2: top + plotH, class: 'viz-grid'}));
      root.append(svg('text', {x: x(t), y: top + plotH + 16, class: 'viz-tick', 'text-anchor': 'middle'}, kfmt(t)));
    }
    for (const t of ys.ticks) {
      root.append(svg('line', {x1: left, x2: left + plotW, y1: y(t), y2: y(t), class: t === ys.min ? 'viz-axis' : 'viz-grid'}));
      root.append(svg('text', {x: left - 8, y: y(t), class: 'viz-tick', 'text-anchor': 'end', 'dominant-baseline': 'central'}, pctTick(t)));
    }
    root.append(svg('text', {x: left, y: 12, class: 'viz-axis-title'}, '7/7 rate'));
    root.append(svg('text', {x: left + plotW / 2, y: height - 8, class: 'viz-axis-title', 'text-anchor': 'middle'}, 'Mean generated tokens per run'));
    const label = labeler(root, [left, top - 6, left + plotW, top + plotH], all.map(p => [x(p.x), y(p.y)]));
    if (ref) {
      const g = svg('g', {class: 'viz-dot'});
      g.append(svg('circle', {cx: x(ref.x), cy: y(ref.y), r: 12, class: 'viz-hit'}));
      g.append(svg('circle', {cx: x(ref.x), cy: y(ref.y), r: 4.5, class: 'viz-mark viz-ref-dot'}));
      bindTip(g, ref.tip);
      root.append(g);
      label(ref.label, x(ref.x), y(ref.y));
    }
    for (const p of points) {
      const g = svg('g', {class: 'viz-dot'});
      g.append(svg('circle', {cx: x(p.x), cy: y(p.y), r: 12, class: 'viz-hit'}));
      g.append(svg('circle', {cx: x(p.x), cy: y(p.y), r: 5, fill: p.color, class: 'viz-mark viz-ringed'}));
      bindTip(g, p.tip);
      root.append(g);
    }
    for (const p of points) if (labelIds.includes(p.id)) label(p.label, x(p.x), y(p.y));
    return root;
  }

  // Places a point label right, left, above or below its point, skipping any spot that would cover
  // another point, a line, a label or a reserved box, or leave the plot. A label with no free spot is
  // left to the legend, the tooltip and the table.
  function labeler(root, [x0, y0, x1, y1], dots, {reserved = [], lines = []} = {}) {
    const placed = [...reserved];
    const overlaps = (a, b) => a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3];
    const trail = [];
    for (const line of lines) for (let i = 1; i < line.length; i++) {
      const [ax, ay] = line[i - 1], [bx, by] = line[i], steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 3));
      for (let k = 0; k <= steps; k++) trail.push([ax + (bx - ax) * k / steps - 2, ay + (by - ay) * k / steps - 2, 4, 4]);
    }
    return (text, cx, cy) => {
      const w = textWidth(text, 11.5), h = 14;
      for (const [lx, ly, anchor, bx] of [[cx + 9, cy, 'start', cx + 9], [cx - 9, cy, 'end', cx - 9 - w],
        [cx, cy - 14, 'middle', cx - w / 2], [cx, cy + 15, 'middle', cx - w / 2]]) {
        const box = [bx, ly - h / 2, w, h];
        if (bx < x0 || bx + w > x1 || box[1] < y0 || box[1] + h > y1) continue;
        if (dots.some(([px, py]) => (px !== cx || py !== cy) && overlaps(box, [px - 7, py - 7, 14, 14]))) continue;
        if (placed.some(other => overlaps(box, other)) || trail.some(seg => overlaps(box, seg))) continue;
        placed.push(box);
        root.append(svg('text', {x: lx, y: ly, class: 'viz-point-label', 'text-anchor': anchor, 'dominant-baseline': 'central'}, text));
        return true;
      }
      return false;
    };
  }

  // Reliability diagram: the observed 7/7 rate against the mean predicted probability in each bin;
  // the diagonal is perfect calibration. series: [{label, color, bins: [[runs, sum of p, wins, lowest
  // p, highest p]]}], bins holding equal numbers of runs.
  function reliability(series, side) {
    const left = 50, top = 26, right = 16, bottom = 46;
    const x = v => left + v * side, y = v => top + side - v * side;
    const root = svg('svg', {width: left + side + right, height: top + side + bottom, class: 'viz-svg'});
    for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      root.append(svg('line', {x1: x(t), x2: x(t), y1: top, y2: top + side, class: t === 0 ? 'viz-axis' : 'viz-grid'}));
      root.append(svg('line', {x1: left, x2: left + side, y1: y(t), y2: y(t), class: t === 0 ? 'viz-axis' : 'viz-grid'}));
      root.append(svg('text', {x: x(t), y: top + side + 16, class: 'viz-tick', 'text-anchor': 'middle'}, pctTick(t)));
      root.append(svg('text', {x: left - 8, y: y(t), class: 'viz-tick', 'text-anchor': 'end', 'dominant-baseline': 'central'}, pctTick(t)));
    }
    root.append(svg('text', {x: left, y: 12, class: 'viz-axis-title'}, 'Observed 7/7 rate'));
    root.append(svg('text', {x: left + side / 2, y: top + side + 38, class: 'viz-axis-title', 'text-anchor': 'middle'}, 'Predicted success_probability (bin mean)'));
    root.append(svg('line', {x1: x(0), y1: y(0), x2: x(1), y2: y(1), class: 'viz-ref'}));
    const diagonal = 'calibrated', dw = textWidth(diagonal, 11);
    root.append(svg('text', {x: x(0.97), y: y(0.97) + 16, class: 'viz-ref-label', 'text-anchor': 'end'}, diagonal));
    const points = series.map(s => s.bins.map(([n, total, wins, lo, hi]) => ({n, wins, lo, hi, p: total / n, rate: wins / n})));
    const polylines = points.map(pts => pts.map(pt => [x(pt.p), y(pt.rate)]));
    const label = labeler(root, [left, top - 4, left + side + right, top + side], polylines.flat(),
      {reserved: [[x(0.97) - dw, y(0.97) + 8, dw, 14]], lines: [...polylines, [[x(0), y(0)], [x(1), y(1)]]]});
    series.forEach((s, k) => {
      const pts = points[k];
      if (pts.length > 1) root.append(svg('path', {d: pts.map((pt, j) => `${j ? 'L' : 'M'}${x(pt.p)},${y(pt.rate)}`).join(''), class: 'viz-line', stroke: s.color}));
      for (const pt of pts) {
        const g = svg('g', {class: 'viz-dot'});
        g.append(svg('circle', {cx: x(pt.p), cy: y(pt.rate), r: 12, class: 'viz-hit'}));
        g.append(svg('circle', {cx: x(pt.p), cy: y(pt.rate), r: 4.5, fill: s.color, class: 'viz-mark viz-ringed'}));
        const [lo, hi] = wilson(pt.wins, pt.n);
        bindTip(g, {value: `${pct(pt.rate)} scored 7/7`, label: `${s.label} · p ${pt.lo.toFixed(2)}–${pt.hi.toFixed(2)}`, color: s.color,
          lines: [`${int(pt.wins)} of ${int(pt.n)} runs`, `Mean predicted p ${pct(pt.p)}`, `95% CI ${pct(lo)}–${pct(hi)}`]});
        root.append(g);
      }
    });
    series.forEach((s, k) => { const last = points[k].at(-1); if (last) label(s.label, x(last.p), y(last.rate)); });
    return root;
  }

  function calibrationMetrics(series) {
    const table = node('table', 'viz-table');
    const thead = node('thead'), head = node('tr');
    ['Series', 'Runs', 'Mean p', 'Observed 7/7', 'AUROC', 'Brier', 'Constant guess']
      .forEach((text, i) => head.append(node('th', i ? 'num' : '', text)));
    thead.append(head);
    const tbody = node('tbody');
    for (const s of series) {
      const tr = node('tr'), name = node('td');
      const key = node('span', 'cal-key');
      key.style.background = s.color;
      name.append(key, document.createTextNode(s.label));
      tr.append(name);
      // Intervals sit under their value so the panel fits beside the chart.
      for (const [text, interval] of [[int(s.runs)], [pct(s.mean_p)], [pct(s.rate)], [fixed(s.auroc, 2), ciFixed(s.auroc_ci, 2)],
        [fixed(s.brier, 3), ciFixed(s.brier_ci, 3)], [fixed(s.brier_base, 3)]]) {
        const td = node('td', 'num', text);
        if (interval && interval !== '–') td.append(node('span', 'cal-ci', interval));
        tr.append(td);
      }
      tbody.append(tr);
    }
    table.append(thead, tbody);
    const box = node('div', 'cal-metrics table-scroll');
    box.append(table);
    return box;
  }

  function legend(items, shape = '') {
    const box = node('div', 'viz-legend');
    for (const item of items) {
      const entry = node('span', 'viz-legend-item');
      const swatch = node('span', `viz-swatch ${item.shape || shape}`);
      if (item.color) swatch.style.background = item.color;
      entry.append(swatch, document.createTextNode(item.label));
      box.append(entry);
    }
    return box;
  }

  function dataTable(head, rows) {
    const table = node('table', 'viz-table');
    const thead = node('thead'), tr = node('tr');
    head.forEach((text, i) => tr.append(node('th', i ? 'num' : '', text)));
    thead.append(tr);
    const tbody = node('tbody');
    for (const row of rows) {
      const r = node('tr');
      row.forEach((text, i) => r.append(node('td', i ? 'num' : '', text)));
      tbody.append(r);
    }
    table.append(thead, tbody);
    const scroll = node('div', 'table-scroll');
    scroll.append(table);
    return scroll;
  }

  // A table whose cells carry inline bars on one shared 0-100% scale: every column is a single
  // series, so the header names it and no legend is needed. cells: [{value, title}].
  function barTable(columns, groups) {
    const table = node('table', 'bar-table');
    const thead = node('thead'), head = node('tr');
    head.append(node('th', 'bt-label', 'Method'));
    for (const column of columns) {
      const th = node('th');
      th.append(node('span', '', column.label));
      if (column.sub) th.append(node('span', 'bt-sub', column.sub));
      head.append(th);
    }
    thead.append(head);
    const tbody = node('tbody');
    for (const group of groups) {
      if (group.title) {
        const tr = node('tr', 'bt-group');
        const td = node('td', '', group.title);
        td.colSpan = columns.length + 1;
        tr.append(td);
        tbody.append(tr);
      }
      for (const row of group.rows) {
        const tr = node('tr');
        tr.append(node('td', 'bt-label', row.label));
        for (const cell of row.cells) {
          const td = node('td');
          if (cell && cell.value !== null && cell.value !== undefined) {
            const wrap = node('div', 'bt'), track = node('span', 'bt-track'), bar = node('span', 'bt-bar');
            bar.style.width = `${Math.min(100, 100 * cell.value)}%`;
            track.append(bar);
            wrap.append(track, node('span', 'bt-val', pct(cell.value)));
            if (cell.title) td.title = cell.title;
            td.append(wrap);
          } else {
            td.append(node('span', 'hint', '–'));
          }
          tr.append(td);
        }
        tbody.append(tr);
      }
    }
    table.append(thead, tbody);
    const scroll = node('div', 'table-scroll');
    scroll.append(table);
    return scroll;
  }

  function card(title, subtitle, {draw, table = null, legendEl = null, note = null}) {
    const box = node('section', 'viz-card');
    const head = node('header', 'viz-card-head');
    const text = node('div');
    text.append(node('h3', '', title));
    if (subtitle) text.append(node('p', 'viz-sub', subtitle));
    head.append(text);
    const chart = node('div', 'viz-chart'), body = node('div', 'viz-body');
    if (legendEl) chart.append(legendEl);
    chart.append(body);
    let tableBox = null;
    const redraw = () => { if (!chart.hidden) body.replaceChildren(draw(Math.max(360, body.clientWidth))); };
    if (table) {
      const toggle = node('button', 'button secondary viz-toggle', 'Table view');
      toggle.type = 'button';
      toggle.setAttribute('aria-pressed', 'false');
      toggle.addEventListener('click', () => {
        const showTable = toggle.getAttribute('aria-pressed') !== 'true';
        toggle.setAttribute('aria-pressed', String(showTable));
        toggle.textContent = showTable ? 'Chart view' : 'Table view';
        if (!tableBox) {
          tableBox = node('div', 'viz-table-wrap');
          tableBox.append(table());
          chart.after(tableBox);
        }
        tableBox.hidden = !showTable;
        chart.hidden = showTable;
        redraw();
      });
      head.append(toggle);
    }
    box.append(head, chart);
    if (note) box.append(node('p', 'viz-note', note));
    view.cards.push(redraw);
    return box;
  }

  function tiles(items) {
    const row = node('div', 'stat-tiles');
    for (const [label, value, sub] of items) {
      const tile = node('div', 'stat-tile');
      tile.append(node('div', 'stat-label', label), node('div', 'stat-value', value));
      if (sub) tile.append(node('div', 'stat-sub', sub));
      row.append(tile);
    }
    return row;
  }

  function segmented(label, options, current, pick) {
    const group = node('div', 'seg-group');
    const seg = node('div', 'seg');
    seg.setAttribute('role', 'radiogroup');
    seg.setAttribute('aria-label', label);
    for (const [value, text] of options) {
      const b = node('button', `seg-btn${value === current ? ' active' : ''}`, text);
      b.type = 'button';
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', String(value === current));
      b.addEventListener('click', () => { if (value !== current) pick(value); });
      seg.append(b);
    }
    group.append(node('span', 'seg-label', label), seg);
    return group;
  }

  const expMeta = id => view.meta.experiments.find(e => e.id === id);
  const armLabel = id => expMeta(view.exp)?.arms.find(a => a.id === id)?.label || id;
  const directLabel = scope => {
    if (scope.direct.source === view.exp) return 'Direct';
    const source = expMeta(scope.direct.source);
    return `Direct (${source.model.split('/').pop()}, seeds ${source.seeds[0]}–${source.seeds.at(-1)})`;
  };

  // Methods in the experiment's canonical order, or ranked by the page's headline metric.
  function order(scope, metric) {
    const ids = expMeta(view.exp).arms.map(a => a.id).filter(id => scope.arms[id]);
    if (view.sort === 'rank') ids.sort((a, b) => (metric(scope.arms[b]) ?? -1) - (metric(scope.arms[a]) ?? -1));
    return ids;
  }

  function filterRow() {
    const row = node('div', 'filter-row');
    const exps = view.meta.experiments.filter(e => view.data.experiments[e.id]);
    row.append(segmented('Experiment', exps.map(e => [e.id, e.label]), view.exp, value => update({exp: value})));
    const scopes = view.data.experiments[view.exp].scopes;
    const benches = [['all', `Both benchmarks · ${scopes.all.problems}`],
      ...view.meta.benchmarks.filter(b => scopes[b.id]).map(b => [b.id, `${b.label} · ${scopes[b.id].problems}`])];
    row.append(segmented('Problems', benches, view.scope, value => update({scope: value})));
    row.append(segmented('Order', [['rank', view.page === 'summary' ? 'By 7/7 rate' : 'By fix rate'], ['method', 'Method order']],
      view.sort, value => update({sort: value})));
    return row;
  }

  function update(changes) {
    Object.assign(view, changes);
    render();
    if (view.onChange) view.onChange();
  }

  function contextLine(scope) {
    const meta = expMeta(view.exp);
    const direct = scope.direct.source === view.exp ? '' : ` Direct baseline: ${expMeta(scope.direct.source)?.label}, seeds ${expMeta(scope.direct.source)?.seeds[0]}–${expMeta(scope.direct.source)?.seeds.at(-1)}.`;
    return `${meta.model.split('/').pop()} · ${scope.problems} problems × ${meta.seeds.length} seeds (${meta.seeds[0]}–${meta.seeds.at(-1)}) = ${int(scope.problems * meta.seeds.length)} runs per method. ` +
      `Judge: gpt-oss-20b; strict success is 7/7, and failed or unjudged runs count as 0. Intervals are 95% bootstraps over problems.${direct}`;
  }

  function summaryPage(page, scope) {
    const ids = order(scope, a => a.success);
    const rowsFor = list => list.map(id => ({id, label: armLabel(id), arm: scope.arms[id]}));
    page.append(node('h2', 'stats-title', 'How the methods compare'), node('p', 'stats-lede', contextLine(scope)));

    page.append(card('Strict success by method', `Share of runs the judge scored 7/7, with 95% intervals. The line marks ${directLabel(scope)}.`, {
      draw: width => barPanels(rowsFor(ids), [{
        get: r => ({value: r.arm.success, lo: r.arm.ci[0], hi: r.arm.ci[1]}),
        ref: {value: scope.direct.success, label: directLabel(scope)},
        tip: r => ({value: pct(r.arm.success), label: r.label, color: C.blue, lines: [
          `7/7 in ${int(r.arm.wins)} of ${int(r.arm.runs)} runs`, `95% CI ${ci(r.arm.ci)}`, `Mean grade ${r.arm.mean_grade.toFixed(2)} of 7`]}),
      }], width),
      table: () => dataTable(['Method', '7/7 rate', '95% CI', '7/7 runs', 'Runs', 'Mean grade (0–7)', 'Judged runs'],
        rowsFor(ids).map(r => [r.label, pct(r.arm.success), ci(r.arm.ci), int(r.arm.wins), int(r.arm.runs), r.arm.mean_grade.toFixed(2), int(r.arm.judged)])),
    }));

    const buckets = BUCKETS.filter(b => ids.some(id => scope.arms[id].tokens.buckets[b.key]));
    page.append(card('Generated tokens per run', 'Mean completion tokens per run, split by role. Generation covers solver, generator, reviser, planner and worker calls; verification covers verifier, value-verifier and reviewer calls. The table adds medians, prompt tokens and model calls.', {
      legendEl: legend(buckets),
      draw: width => stackedBars(rowsFor(ids).map(r => ({
        label: r.label, values: r.arm.tokens.buckets,
        tip: (s, v, total) => ({value: `${int(v)} tokens`, label: `${s.label} · ${r.label}`, color: s.color,
          lines: [`${pct(v / total)} of this method's mean ${int(total)} tokens per run`]}),
      })), buckets, width, {mode: 'abs'}),
      table: () => dataTable(['Method', 'Mean', 'Median', 'p90', ...buckets.map(b => b.label), 'Prompt tokens', 'Calls per run', 'Most calls'],
        rowsFor(ids).map(r => [r.label, int(r.arm.tokens.mean), int(r.arm.tokens.median), int(r.arm.tokens.p90),
          ...buckets.map(b => int(r.arm.tokens.buckets[b.key] || 0)), int(r.arm.tokens.prompt), r.arm.calls ? r.arm.calls.mean.toFixed(2) : '–', r.arm.calls ? r.arm.calls.max : '–'])),
    }));

    const families = FAMILIES.filter(f => ids.some(id => scope.arms[id].family === f.key));
    const external = scope.direct.source !== view.exp;
    const ref = external ? {id: 'direct-ref', label: directLabel(scope), x: scope.direct.tokens_mean, y: scope.direct.success,
      tip: {value: pct(scope.direct.success), label: directLabel(scope), lines: [`${int(scope.direct.tokens_mean)} tokens per run`, 'Reference from another run']}} : null;
    const best = ids.slice().sort((a, b) => scope.arms[b].success - scope.arms[a].success)[0];
    page.append(card('7/7 rate against tokens', 'One dot per method: mean generated tokens per run against strict success. Higher and further left is better.', {
      legendEl: legend([...families.map(f => ({...f, shape: 'dot'})), ...(ref ? [{label: ref.label, shape: 'ring'}] : [])]),
      draw: width => scatter(rowsFor(ids).map(r => {
        const family = FAMILIES.find(f => f.key === r.arm.family);
        return {id: r.id, label: r.label, x: r.arm.tokens.mean, y: r.arm.success, color: family.color,
          tip: {value: pct(r.arm.success), label: r.label, color: family.color, lines: [`${int(r.arm.tokens.mean)} tokens per run`, `95% CI ${ci(r.arm.ci)}`]}};
      }), width, {ref, labelIds: ['direct', best]}),
      table: () => dataTable(['Method', 'Family', 'Mean tokens', '7/7 rate'], rowsFor(ids).map(r => [r.label,
        FAMILIES.find(f => f.key === r.arm.family).label, int(r.arm.tokens.mean), pct(r.arm.success)])),
    }));

    const endRows = rowsFor(ids).map(r => {
      const counts = {}, wins = {};
      for (const state of END_STATES) {
        counts[state.key] = sum(state.statuses.map(s => r.arm.status[s] || 0));
        wins[state.key] = sum(state.statuses.map(s => (r.arm.success_by_status[s] || [0, 0])[1]));
      }
      return {...r, counts, wins};
    });
    const endStates = END_STATES.filter(s => endRows.some(r => r.counts[s.key]));
    page.append(card('How runs end', 'GVR stops at the first "correct" verdict or after its third candidate; plan-work-review stops when a review approves or its retakes run out. Direct and the value tool stop when the solver answers. Errors and exhausted budgets score 0.', {
      legendEl: legend(endStates),
      draw: width => stackedBars(endRows.map(r => ({
        label: r.label, values: r.counts,
        tip: (s, n, total) => ({value: pct(n / total), label: `${s.label} · ${r.label}`, color: s.color,
          lines: [`${int(n)} of ${int(total)} runs`, `7/7 in ${pct(r.wins[s.key] / n)} of these`]}),
      })), endStates, width),
      table: () => {
        const statuses = [...new Set(endRows.flatMap(r => Object.keys(r.arm.status)))];
        return dataTable(['Method', ...statuses.map(s => s.replace(/_/g, ' ')), '7/7 when accepted', '7/7 at candidate limit'],
          endRows.map(r => [r.label, ...statuses.map(s => int(r.arm.status[s] || 0)),
            pct(rateOf(r.arm.success_by_status.accepted)), pct(rateOf(r.arm.success_by_status.cycle_limit))]));
      },
    }));

    const gvrRows = rowsFor(ids.filter(id => scope.arms[id].gvr));
    if (gvrRows.length) {
      const calls = CALLS.filter(c => gvrRows.some(r => r.arm.gvr.verdict_counts[c.key]));
      const meanCalls = arm => sum(Object.entries(arm.gvr.verdict_counts).map(([k, n]) => k * n)) / arm.runs;
      page.append(card('GVR: verifier calls per run', 'The verifier checks each candidate once, so a run makes 1–3 verifier calls. Accepted runs stop early; three calls means the third candidate was accepted or the run hit the candidate limit.', {
        legendEl: legend(calls),
        draw: width => stackedBars(gvrRows.map(r => ({
          label: r.label, values: r.arm.gvr.verdict_counts,
          tip: (s, n, total) => {
            const accepted = r.arm.gvr.accepted_at[s.key] || 0;
            const lines = [`${int(n)} of ${int(total)} runs`];
            if (s.key !== '0') lines.push(`Accepted at call ${s.key}: ${int(accepted)}`);
            if (s.key === '3') lines.push(`Stopped at the candidate limit: ${int(r.arm.status.cycle_limit || 0)}`);
            if (s.key !== '0' && s.key !== '3' && n - accepted) lines.push(`Ended in an error: ${int(n - accepted)}`);
            return {value: pct(n / total), label: `${s.label} · ${r.label}`, color: s.color, lines};
          },
        })), calls, width),
        table: () => dataTable(['Method', 'Mean verifier calls', 'Accepted at call 1', 'at call 2', 'at call 3', 'Candidate limit',
          'First verdict: correct', 'minor fix', 'critical flaw', '7/7 when accepted', '7/7 at limit'],
        gvrRows.map(r => {
          const g = r.arm.gvr;
          return [r.label, meanCalls(r.arm).toFixed(2), int(g.accepted_at['1'] || 0), int(g.accepted_at['2'] || 0), int(g.accepted_at['3'] || 0),
            int(r.arm.status.cycle_limit || 0), int(g.first_verdict.correct || 0), int(g.first_verdict.minor_fix || 0), int(g.first_verdict.critical_flaw || 0),
            pct(rateOf(r.arm.success_by_status.accepted)), pct(rateOf(r.arm.success_by_status.cycle_limit))];
        })),
      }));
    }

    const valueRows = rowsFor(ids.filter(id => scope.arms[id].value_tool));
    if (valueRows.length) {
      const withQuery = arm => Object.entries(arm.value_tool.success_by_queries).filter(([k]) => k !== '0')
        .reduce((acc, [, v]) => [acc[0] + v[0], acc[1] + v[1]], [0, 0]);
      page.append(card('Value tool: queries per run', 'The solver decides when to call query_success_probability, at most 3 times; the run ends the first time it answers without a tool call. The probability never stops a run by itself.', {
        legendEl: legend(QUERIES),
        draw: width => stackedBars(valueRows.map(r => ({
          label: r.label, values: r.arm.value_tool.query_counts,
          tip: (s, n, total) => ({value: pct(n / total), label: `${s.label} · ${r.label}`, color: s.color,
            lines: [`${int(n)} of ${int(total)} runs`, `7/7 in ${pct(rateOf(r.arm.value_tool.success_by_queries[s.key]))} of these`]}),
        })), QUERIES, width),
        table: () => dataTable(['Method', 'Mean queries', '0', '1', '2', '3', 'Runs with a query', '7/7 without a query', '7/7 with a query'],
          valueRows.map(r => {
            const v = r.arm.value_tool, queried = withQuery(r.arm);
            return [r.label, v.mean_queries.toFixed(2), ...['0', '1', '2', '3'].map(k => int(v.query_counts[k] || 0)),
              pct(queried[0] / r.arm.runs), pct(rateOf(v.success_by_queries['0'])), pct(rateOf(queried))];
          })),
      }));
      calibrationCard(page, scope, valueRows);
    }
  }

  function calibrationCard(page, scope, valueRows) {
    const series = (scope.calibration || []).map((c, k) => ({...c, color: CAL_COLORS[k]}));
    if (!series.length) return;
    const {max, min_runs: minRuns} = view.data.calibration_bins;
    const pooled = series.some(s => s.arms.length > 1);
    page.append(card('Value tool: are its probabilities calibrated?', `Each run's last success_probability against whether its final answer scored 7/7, for runs that queried at least once (failed or unjudged runs count as 0). Runs are sorted by p and split into up to ${max} equal bins of at least ${minRuns} runs. Points on the diagonal are calibrated; points below it are overconfident.${pooled ? ' Lines pool arms by the evidence the verifier saw; the table view lists every arm.' : ''}`, {
      legendEl: legend(series.map(s => ({label: s.label, color: s.color, shape: 'line'}))),
      draw: width => {
        const wrap = node('div', 'cal-wrap');
        wrap.append(reliability(series, width >= 900 ? 340 : Math.max(220, Math.min(340, width - 80))), calibrationMetrics(series));
        return wrap;
      },
      table: () => {
        const box = node('div');
        const arms = valueRows.filter(r => r.arm.value_tool.calibration);
        box.append(node('h4', 'viz-table-title', 'Every value-tool arm'), dataTable(
          ['Method', 'Runs with a query', 'Mean p', 'Observed 7/7', 'AUROC', '95% CI', 'Brier', '95% CI', 'Constant guess'],
          arms.map(r => {
            const c = r.arm.value_tool.calibration;
            return [r.label, int(c.runs), pct(c.mean_p), pct(c.rate), fixed(c.auroc, 2), ciFixed(c.auroc_ci, 2), fixed(c.brier, 3), ciFixed(c.brier_ci, 3), fixed(c.brier_base, 3)];
          })));
        box.append(node('h4', 'viz-table-title', 'Bins'), dataTable(['Series', 'Predicted p range', 'Runs', 'Mean predicted', 'Observed 7/7', '95% CI'],
          series.flatMap(s => s.bins.map(([n, total, wins, low, high]) => {
            const [lo, hi] = wilson(wins, n);
            return [s.label, `${low.toFixed(2)}–${high.toFixed(2)}`, int(n), pct(total / n), pct(wins / n), `${pct(lo)}–${pct(hi)}`];
          }))));
        return box;
      },
      note: 'AUROC is the chance that a run ending 7/7 got a higher p than one that did not (0.5 means no signal). Brier is the mean squared error of p, lower is better; the constant guess always predicts the observed 7/7 rate. Small numbers under AUROC and Brier are 95% bootstrap intervals over problems; bin intervals in the table view are Wilson intervals.',
    }));
  }

  function analysisPage(page, scope) {
    const direct = scope.direct;
    const n = direct.runs_per_problem.join('/');
    const ids = order(scope, a => a.vs_direct?.fix).filter(id => scope.arms[id].vs_direct);
    const rows = ids.map(id => ({id, label: armLabel(id), arm: scope.arms[id], v: scope.arms[id].vs_direct}));
    page.append(node('h2', 'stats-title', 'When Direct is wrong, does verification fix it?'));
    page.append(node('p', 'stats-lede', `Direct makes one solver call with no verifier. For each method: on problems where a Direct run is wrong, how often does the method reach 7/7 (fix rate), and where Direct is right, how often does it fall short (break rate)? Direct and the methods never share a sample, so every Direct run is compared with every method run on the same problem. ${contextLine(scope)}`));
    page.append(tiles([
      [`${directLabel(scope)} 7/7 rate`, pct(direct.success), `95% CI ${ci(direct.ci)} · ${int(direct.runs)} runs`],
      ['Direct never solves', `${direct.problems.never} problems`, `0 of ${n} Direct runs scored 7/7`],
      ['Direct sometimes solves', `${direct.problems.sometimes} problems`, `some but not all of ${n} runs`],
      ['Direct always solves', `${direct.problems.always} problems`, `all ${n} runs scored 7/7`],
    ]));

    const resample = direct.resample;
    page.append(card('Fix rate and break rate against Direct', 'Fix rate: P(method 7/7 | Direct run below 7/7). Break rate: P(method below 7/7 | Direct run 7/7). The line marks Direct compared with another Direct run on the same problem, i.e. the flips that re-sampling alone produces.', {
      draw: width => barPanels(rows, [
        {title: 'Fix rate: 7/7 when Direct is wrong', short: 'Fix rate', get: r => ({value: r.v.fix, lo: r.v.fix_ci[0], hi: r.v.fix_ci[1]}),
          ref: {value: resample.fix, label: 'Direct re-sampled'},
          tip: r => ({value: pct(r.v.fix), label: `${r.label} · fix rate`, color: C.blue,
            lines: [`95% CI ${ci(r.v.fix_ci)}`, `Direct re-sampled: ${pct(resample.fix)}`, `Net vs Direct: ${signedPts(r.v.net)}`]})},
        {title: 'Break rate: below 7/7 when Direct is right', short: 'Break rate', get: r => ({value: r.v.break, lo: r.v.break_ci[0], hi: r.v.break_ci[1]}),
          ref: {value: resample.break, label: 'Direct re-sampled'},
          tip: r => ({value: pct(r.v.break), label: `${r.label} · break rate`, color: C.blue,
            lines: [`95% CI ${ci(r.v.break_ci)}`, `Direct re-sampled: ${pct(resample.break)}`, `Net vs Direct: ${signedPts(r.v.net)}`]})},
      ], width),
      table: () => dataTable(['Method', 'Fix rate', '95% CI', 'Break rate', '95% CI', 'Net vs Direct', '95% CI'], [
        ['Direct re-sampled', pct(resample.fix), ci(resample.fix_ci), pct(resample.break), ci(resample.break_ci), '–', '–'],
        ...rows.map(r => [r.label, pct(r.v.fix), ci(r.v.fix_ci), pct(r.v.break), ci(r.v.break_ci), signedPts(r.v.net),
          `${signedPts(r.v.net_ci[0])} to ${signedPts(r.v.net_ci[1])}`]),
      ]),
      note: 'Net vs Direct (in the tooltip and table) is the method\'s 7/7 rate minus Direct\'s: fixes minus breaks, as a share of all pairs.',
    }));

    page.append(card('7/7 rate by how often Direct solves the problem', `Problems are grouped by their ${n} Direct runs. The first column is the share of runs that solve a problem Direct never solves; the last shows how often a method keeps a problem Direct always solves.`, {
      draw: () => barTable(STRATA.map(s => ({label: s.label, sub: `${direct.problems[s.key]} problems`})), [{rows: rows.map(r => ({
        label: r.label,
        cells: STRATA.map(s => {
          const [runs, wins] = r.arm.strata[s.key];
          return runs ? {value: wins / runs, title: `${int(wins)} of ${int(runs)} runs scored 7/7`} : null;
        }),
      }))}]),
    }));

    const gvr = rows.filter(r => r.arm.gvr);
    if (gvr.length) {
      const groups = [...STRATA, {key: 'all', label: 'All problems'}].map(s => ({
        title: `${s.label} · ${s.key === 'all' ? scope.problems : direct.problems[s.key]} problems`,
        rows: gvr.map(r => {
          const counts = r.arm.gvr.first_candidate[s.key], total = sum(Object.values(counts));
          return {label: r.label, cells: FIRST.map(f => (total ? {value: counts[f.key] / total, title: `${int(counts[f.key])} of ${int(total)} runs`} : null))};
        }),
      })).filter(g => g.rows.some(r => r.cells.some(Boolean)));
      page.append(card('GVR: what the verifier does with the first candidate', 'The verifier either accepts the first candidate as is or sends it back for revision. Kept wrong is a false accept: the verifier kept an answer the judge scored below 7/7. Fixed by revision means the candidate was sent back at least once and the final answer scored 7/7.', {
        draw: () => barTable(FIRST.map(f => ({label: f.label, sub: f.sub})), groups),
        note: 'The judge grades only final answers, so the grade of a first candidate that was sent back is unknown. GVR\'s first candidate is its own sample, not Direct\'s answer.',
      }));
    }

    const caveats = node('ul', 'caveats');
    for (const text of [
      'Strata use the official scoring: a failed or unjudged Direct run counts as wrong. PB-Basic-001 (7 of 8 plus one budget-exhausted run) is therefore "sometimes solved" here, while the evidence-bank filter in Rollouts lists it as 7/7.',
      scope.direct.source === view.exp
        ? 'Direct and the methods ran on the same seeds, but different prompts give different samples, so seed-matched pairs are no more related than any two runs on the problem.'
        : 'This experiment has no Direct arm: Direct comes from the 9B harness run (seeds 0–7), the same runs that fill the attempt-conditioned verifiers\' evidence bank, while the methods ran on seeds 8–15.',
    ]) caveats.append(node('li', '', text));
    page.append(caveats);
  }

  function render() {
    hideTip();
    view.cards = [];
    const scope = view.data.experiments[view.exp].scopes[view.scope];
    const page = node('div', 'stats-page');
    page.append(filterRow());
    (view.page === 'analysis' ? analysisPage : summaryPage)(page, scope);
    view.root.replaceChildren(page);
    for (const redraw of view.cards) redraw();
  }

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (view.root && !view.root.hidden) for (const redraw of view.cards) redraw(); }, 150);
  });

  async function show(root, page, params, {fetchJSON, meta, onChange}) {
    Object.assign(view, {root, page, meta, onChange});
    const ids = meta.experiments.map(e => e.id);
    if (ids.includes(params.get('x'))) view.exp = params.get('x');
    if (['all', ...meta.benchmarks.map(b => b.id)].includes(params.get('sc'))) view.scope = params.get('sc');
    if (['rank', 'method'].includes(params.get('so'))) view.sort = params.get('so');
    if (!view.data) {
      root.replaceChildren(node('div', 'empty-state', 'Loading summary statistics…'));
      try {
        view.data = await fetchJSON('index/summary.json');
      } catch (error) {
        root.replaceChildren(node('p', 'error', `Could not load summary statistics: ${error.message}`));
        return;
      }
    }
    if (!view.data.experiments[view.exp]) view.exp = ids.find(id => view.data.experiments[id]);
    if (!view.data.experiments[view.exp].scopes[view.scope]) view.scope = 'all';
    render();
  }

  const params = () => [['x', view.exp], ['sc', view.scope], ['so', view.sort]];

  return {show, params, hide: hideTip};
})();
